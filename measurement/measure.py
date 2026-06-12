"""
Pixel-to-MM Measurement Pipeline
=================================
Computes real-world metric measurements (width & height in mm) from
segmented phone instances using calibrated camera parameters.

Measurement Methodology:
    1. Undistort image using intrinsic calibration parameters
    2. Detect reference object (checkerboard) to compute pixels_per_mm ratio
    3. Run segmentation model to get phone mask
    4. Extract mask contour and compute oriented bounding box
    5. Convert pixel dimensions to mm using the calibrated ratio

Usage:
    python measurement/measure.py --input <image_path>
    python measurement/demo.py --input <image_path>   (end-to-end demo)
"""

import os
import sys
import json
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CALIBRATION_PARAMS = os.path.join(PROJECT_ROOT, "calibration", "calibration_params.json")


def load_calibration():
    """Load camera calibration parameters."""
    with open(CALIBRATION_PARAMS, 'r') as f:
        params = json.load(f)
    
    camera_matrix = np.array(params["camera_matrix"])
    dist_coeffs = np.array(params["dist_coeffs"])
    square_size_mm = params["checkerboard"]["square_size_mm"]
    board_rows = params["checkerboard"]["inner_corners_rows"]
    board_cols = params["checkerboard"]["inner_corners_cols"]
    
    return camera_matrix, dist_coeffs, square_size_mm, board_rows, board_cols


def undistort_image(image, camera_matrix, dist_coeffs):
    """Apply lens correction."""
    h, w = image.shape[:2]
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0, (w, h)
    )
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_cam_mtx), new_cam_mtx


def compute_pixels_per_mm_from_calibration(camera_matrix, dist_coeffs, square_size_mm, 
                                            image, board_rows, board_cols):
    """
    Compute pixels_per_mm ratio using checkerboard detection in the image.
    
    If checkerboard is visible in the measurement image, we detect it and use
    the known square size. Otherwise, we fall back to focal-length estimation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board_size = (board_cols, board_rows)
    
    ret, corners = cv2.findChessboardCorners(gray, board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    
    if ret:
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Calculate average pixel distance between adjacent corners
        corners = corners.reshape(-1, 2)
        
        # Horizontal distances
        h_dists = []
        for row in range(board_rows):
            for col in range(board_cols - 1):
                idx1 = row * board_cols + col
                idx2 = row * board_cols + col + 1
                dist = np.linalg.norm(corners[idx2] - corners[idx1])
                h_dists.append(dist)
        
        # Vertical distances
        v_dists = []
        for row in range(board_rows - 1):
            for col in range(board_cols):
                idx1 = row * board_cols + col
                idx2 = (row + 1) * board_cols + col
                dist = np.linalg.norm(corners[idx2] - corners[idx1])
                v_dists.append(dist)
        
        avg_pixel_dist = (np.mean(h_dists) + np.mean(v_dists)) / 2.0
        pixels_per_mm = avg_pixel_dist / square_size_mm
        
        print(f"  Checkerboard detected! Using reference-based calibration.")
        print(f"  Average square size in pixels: {avg_pixel_dist:.2f}")
        print(f"  Pixels per mm: {pixels_per_mm:.4f}")
        
        return pixels_per_mm, "checkerboard_detected"
    
    else:
        # Fallback: Use focal length and estimated distance
        # For known object distance, px/mm = focal_length_px / distance_mm
        # We'll use the camera matrix focal length as reference
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        focal_px = (fx + fy) / 2.0
        
        # Estimate working distance from image (approximate)
        # This is a fallback — document that checkerboard reference is preferred
        estimated_distance_mm = 165.0  # ~16.5cm working distance to match close-up dataset images
        pixels_per_mm = focal_px / estimated_distance_mm
        
        print(f"  No checkerboard in image — using focal length estimation.")
        print(f"  Focal length: {focal_px:.2f} px")
        print(f"  Estimated distance: {estimated_distance_mm} mm")
        print(f"  Pixels per mm: {pixels_per_mm:.4f}")
        
        return pixels_per_mm, "focal_length_estimate"


def extract_mask_dimensions(mask):
    """
    Extract width and height from segmentation mask using oriented bounding box.
    
    Uses cv2.minAreaRect for rotation-invariant measurement, which gives
    the tightest-fitting rotated rectangle around the mask contour.
    
    Returns:
        width_px: width in pixels (shorter dimension)
        height_px: height in pixels (longer dimension)
        contour: the mask contour points
        rect: the oriented bounding box (center, size, angle)
    """
    # Find contours
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, None, None
    
    # Get largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Minimum area rotated rectangle
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect
    
    # Ensure width < height (phone is taller than wide)
    width_px = min(w, h)
    height_px = max(w, h)
    
    return width_px, height_px, largest_contour, rect


def measure_object(mask, pixels_per_mm):
    """
    Convert mask pixel dimensions to real-world mm measurements.
    
    Returns:
        dict with width_mm, height_mm, width_px, height_px, area_px, etc.
    """
    width_px, height_px, contour, rect = extract_mask_dimensions(mask)
    
    if contour is None:
        return None
    
    width_mm = width_px / pixels_per_mm
    height_mm = height_px / pixels_per_mm
    area_px = cv2.contourArea(contour)
    
    return {
        "width_px": round(float(width_px), 2),
        "height_px": round(float(height_px), 2),
        "width_mm": round(float(width_mm), 2),
        "height_mm": round(float(height_mm), 2),
        "area_px": int(area_px),
        "area_mm2": round(float(area_px / (pixels_per_mm ** 2)), 2),
        "pixels_per_mm": round(float(pixels_per_mm), 4),
        "aspect_ratio": round(float(height_px / width_px) if width_px > 0 else 0, 3),
        "contour": contour,
        "rect": rect
    }


def annotate_measurement(image, measurement, label="Phone"):
    """
    Draw measurement annotations on the image.
    Includes: oriented bounding box, dimension labels, contour outline.
    """
    annotated = image.copy()
    
    if measurement is None:
        return annotated
    
    rect = measurement["rect"]
    contour = measurement["contour"]
    
    # Draw contour
    cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
    
    # Draw oriented bounding box
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)
    cv2.drawContours(annotated, [box_points], 0, (0, 0, 255), 3)
    
    # Add measurement labels
    (cx, cy), (w, h), angle = rect
    cx, cy = int(cx), int(cy)
    
    # Background rectangle for text
    text_w = f"W: {measurement['width_mm']:.1f} mm"
    text_h = f"H: {measurement['height_mm']:.1f} mm"
    text_conf = f"{label}"
    
    y_offset = cy - 80
    for text in [text_conf, text_w, text_h]:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(annotated, (cx - tw // 2 - 10, y_offset - th - 10),
                     (cx + tw // 2 + 10, y_offset + 10), (0, 0, 0), -1)
        cv2.putText(annotated, text, (cx - tw // 2, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        y_offset += th + 25
    
    return annotated
