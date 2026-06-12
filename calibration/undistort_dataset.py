"""
Dataset Undistortion Script
============================
Applies camera calibration parameters to undistort all dataset images.
This is a mandatory step — distorted images produce incorrect measurements.

Usage:
    python calibration/undistort_dataset.py

Requires:
    - calibration/calibration_params.json (run calibrate_camera.py first)
"""

import os
import sys
import json
import glob
import cv2
import numpy as np


CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CALIBRATION_DIR)
PARAMS_FILE = os.path.join(CALIBRATION_DIR, "calibration_params.json")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")


def load_calibration_params():
    """Load saved calibration parameters."""
    if not os.path.exists(PARAMS_FILE):
        print(f"ERROR: Calibration parameters not found at {PARAMS_FILE}")
        print("Run calibrate_camera.py first!")
        sys.exit(1)
    
    with open(PARAMS_FILE, 'r') as f:
        params = json.load(f)
    
    camera_matrix = np.array(params["camera_matrix"])
    dist_coeffs = np.array(params["dist_coeffs"])
    img_size = (params["image_size"]["width"], params["image_size"]["height"])
    
    return camera_matrix, dist_coeffs, img_size


def undistort_images(image_dir, camera_matrix, dist_coeffs, img_size):
    """
    Undistort all images in a directory, overwriting originals.
    
    We overwrite to keep the pipeline simple — all downstream processing
    (training, inference, measurement) uses undistorted images.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    if not image_paths:
        print(f"  No images found in {image_dir}")
        return 0
    
    # Get optimal new camera matrix (keeps all pixels, alpha=1)
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, 0, img_size
    )
    
    count = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Cannot read {os.path.basename(img_path)}")
            continue
        
        # Undistort using calibration parameters
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mtx)
        
        # Overwrite original with undistorted version
        cv2.imwrite(img_path, undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    
    return count


def main():
    camera_matrix, dist_coeffs, img_size = load_calibration_params()
    
    print("=" * 50)
    print("  DATASET UNDISTORTION")
    print("=" * 50)
    print(f"\nUsing calibration from: {PARAMS_FILE}")
    
    # Process each split
    splits = ["train", "val", "test"]
    total = 0
    
    for split in splits:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        if os.path.exists(img_dir):
            count = undistort_images(img_dir, camera_matrix, dist_coeffs, img_size)
            print(f"  {split:>5}: undistorted {count} images")
            total += count
        else:
            print(f"  {split:>5}: directory not found — skipping")
    
    # Also undistort calibration images for documentation
    cal_count = undistort_images(CALIBRATION_DIR, camera_matrix, dist_coeffs, img_size)
    
    print(f"\nTotal: {total} dataset images undistorted")
    print(f"       {cal_count} calibration images undistorted")
    print("\nAll images are now lens-corrected and ready for training/measurement.")


if __name__ == "__main__":
    main()
