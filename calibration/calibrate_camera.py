"""
Camera Calibration Script
=========================
Performs intrinsic camera calibration using checkerboard pattern images.
Detects checkerboard corners, computes camera matrix and distortion coefficients,
and saves calibration parameters for use in the measurement pipeline.

Usage:
    python calibration/calibrate_camera.py

Output:
    - calibration/calibration_params.json  (intrinsic matrix + distortion coefficients)
    - calibration/undistorted_samples/     (before/after comparison images)
"""

import os
import sys
import json
import glob
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================
CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_IMAGES = os.path.join(CALIBRATION_DIR, "*.jpg")
OUTPUT_PARAMS = os.path.join(CALIBRATION_DIR, "calibration_params.json")
UNDISTORTED_DIR = os.path.join(CALIBRATION_DIR, "undistorted_samples")

# Checkerboard inner corners (rows x cols)
# Standard checkerboard: count INNER corners, not squares
# Adjust these values to match YOUR checkerboard pattern
CHECKERBOARD_ROWS = 6   # number of inner corner rows
CHECKERBOARD_COLS = 9   # number of inner corner columns
SQUARE_SIZE_MM = 25.0   # size of each square in mm (measure your printed board)


def detect_corners(image_paths, board_size):
    """
    Detect checkerboard corners in all calibration images.
    
    Returns:
        obj_points: 3D points in real-world space
        img_points: 2D points in image plane
        img_shape:  (width, height) of images
        used_images: list of image paths where corners were detected
    """
    # Prepare 3D object points (0,0,0), (1,0,0), (2,0,0) ... scaled by square size
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[1], 0:board_size[0]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    
    obj_points = []  # 3D points in world
    img_points = []  # 2D points in image
    used_images = []
    img_shape = None
    
    # Criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    print(f"Looking for {board_size[1]}x{board_size[0]} inner corners...")
    print(f"Processing {len(image_paths)} calibration images...\n")
    
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{i+1}] SKIP - Cannot read: {os.path.basename(img_path)}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = (gray.shape[1], gray.shape[0])  # (width, height)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, (board_size[1], board_size[0]),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corner locations to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            used_images.append(img_path)
            print(f"  [{i+1}] OK    - {os.path.basename(img_path)}")
        else:
            print(f"  [{i+1}] FAIL  - {os.path.basename(img_path)} (corners not found)")
    
    print(f"\nCorners detected in {len(used_images)}/{len(image_paths)} images")
    return obj_points, img_points, img_shape, used_images


def calibrate(obj_points, img_points, img_shape):
    """
    Run camera calibration using detected corner correspondences.
    
    Returns:
        camera_matrix: 3x3 intrinsic camera matrix
        dist_coeffs:   distortion coefficients [k1, k2, p1, p2, k3]
        rvecs:         rotation vectors for each image
        tvecs:         translation vectors for each image
        reproj_error:  overall reprojection error (pixels)
    """
    print("\nRunning camera calibration...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )
    
    # Calculate per-image reprojection error
    total_error = 0
    per_image_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(error)
        total_error += error
    
    mean_error = total_error / len(obj_points)
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, mean_error, per_image_errors


def save_params(camera_matrix, dist_coeffs, reproj_error, img_shape, num_images):
    """Save calibration parameters to JSON file."""
    params = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "reprojection_error_px": round(reproj_error, 6),
        "image_size": {"width": img_shape[0], "height": img_shape[1]},
        "num_calibration_images_used": num_images,
        "checkerboard": {
            "inner_corners_rows": CHECKERBOARD_ROWS,
            "inner_corners_cols": CHECKERBOARD_COLS,
            "square_size_mm": SQUARE_SIZE_MM
        }
    }
    
    with open(OUTPUT_PARAMS, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"\nCalibration parameters saved to: {OUTPUT_PARAMS}")
    return params


def generate_comparison_images(used_images, camera_matrix, dist_coeffs, img_shape, num_samples=5):
    """Generate before/after undistortion comparison images."""
    os.makedirs(UNDISTORTED_DIR, exist_ok=True)
    
    # Get optimal new camera matrix
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_shape, 1, img_shape
    )
    
    # Select evenly spaced samples
    step = max(1, len(used_images) // num_samples)
    sample_images = used_images[::step][:num_samples]
    
    print(f"\nGenerating {len(sample_images)} comparison images...")
    
    for img_path in sample_images:
        img = cv2.imread(img_path)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Undistort
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mtx)
        
        # Crop to ROI
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted_cropped = undistorted[y:y+h, x:x+w]
        else:
            undistorted_cropped = undistorted
        
        # Save side-by-side comparison (resize for manageable file size)
        scale = 0.25
        orig_small = cv2.resize(img, None, fx=scale, fy=scale)
        undist_small = cv2.resize(undistorted_cropped, (orig_small.shape[1], orig_small.shape[0]))
        
        # Add labels
        cv2.putText(orig_small, "ORIGINAL (Distorted)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(undist_small, "UNDISTORTED (Corrected)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        comparison = np.hstack([orig_small, undist_small])
        
        out_path = os.path.join(UNDISTORTED_DIR, f"comparison_{basename}.jpg")
        cv2.imwrite(out_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  Saved: {out_path}")
    
    return new_cam_mtx, roi


def print_report(camera_matrix, dist_coeffs, reproj_error, per_image_errors, num_images):
    """Print a formatted calibration report to console."""
    print("\n" + "=" * 60)
    print("          CAMERA CALIBRATION REPORT")
    print("=" * 60)
    
    print(f"\nImages used: {num_images}")
    print(f"Checkerboard: {CHECKERBOARD_COLS}×{CHECKERBOARD_ROWS} inner corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    
    print(f"\n--- Intrinsic Camera Matrix (K) ---")
    print(f"  fx = {camera_matrix[0][0]:.4f}")
    print(f"  fy = {camera_matrix[1][1]:.4f}")
    print(f"  cx = {camera_matrix[0][2]:.4f}")
    print(f"  cy = {camera_matrix[1][2]:.4f}")
    print(f"\n  Full matrix:")
    for row in camera_matrix:
        print(f"    [{row[0]:12.4f}  {row[1]:12.4f}  {row[2]:12.4f}]")
    
    dc = dist_coeffs.flatten()
    print(f"\n--- Distortion Coefficients ---")
    print(f"  k1 = {dc[0]:.8f}")
    print(f"  k2 = {dc[1]:.8f}")
    print(f"  p1 = {dc[2]:.8f}  (tangential)")
    print(f"  p2 = {dc[3]:.8f}  (tangential)")
    print(f"  k3 = {dc[4]:.8f}")
    
    print(f"\n--- Reprojection Error ---")
    print(f"  Mean: {reproj_error:.6f} px")
    print(f"  Min:  {min(per_image_errors):.6f} px")
    print(f"  Max:  {max(per_image_errors):.6f} px")
    
    if reproj_error < 0.3:
        quality = "EXCELLENT (< 0.3 px)"
    elif reproj_error < 0.5:
        quality = "GOOD (< 0.5 px)"
    elif reproj_error < 1.0:
        quality = "ACCEPTABLE (< 1.0 px)"
    else:
        quality = "POOR (>= 1.0 px) — consider re-capturing calibration images"
    
    print(f"  Quality: {quality}")
    print("=" * 60)


def main():
    # Find all calibration images
    image_paths = sorted(glob.glob(CALIBRATION_IMAGES))
    
    if len(image_paths) == 0:
        print(f"ERROR: No .jpg images found in {CALIBRATION_DIR}")
        sys.exit(1)
    
    if len(image_paths) < 10:
        print(f"WARNING: Only {len(image_paths)} images found. 20+ recommended for good calibration.")
    
    board_size = (CHECKERBOARD_ROWS, CHECKERBOARD_COLS)
    
    # Step 1: Detect corners
    obj_points, img_points, img_shape, used_images = detect_corners(image_paths, board_size)
    
    if len(used_images) < 5:
        print(f"\nERROR: Only {len(used_images)} images had corners detected. Need at least 5.")
        print("Try adjusting CHECKERBOARD_ROWS and CHECKERBOARD_COLS at the top of this script.")
        sys.exit(1)
    
    # Step 2: Calibrate
    camera_matrix, dist_coeffs, rvecs, tvecs, reproj_error, per_image_errors = calibrate(
        obj_points, img_points, img_shape
    )
    
    # Step 3: Save parameters
    save_params(camera_matrix, dist_coeffs, reproj_error, img_shape, len(used_images))
    
    # Step 4: Print report
    print_report(camera_matrix, dist_coeffs.flatten().tolist() if isinstance(dist_coeffs, np.ndarray) else dist_coeffs,
                 reproj_error, per_image_errors, len(used_images))
    # Fix: pass the numpy arrays for proper formatting
    print_report_arr = camera_matrix  # already printed above
    
    # Step 5: Generate comparison images
    generate_comparison_images(used_images, camera_matrix, dist_coeffs, img_shape)
    
    print("\nCalibration complete! Next step: run undistort_dataset.py")


if __name__ == "__main__":
    main()
