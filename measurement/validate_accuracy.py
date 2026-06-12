"""
Accuracy Validation Script
==========================
Runs the measurement pipeline on all 12 held-out test images.
Compares the system measurements against the ground truth physical
dimensions of the Techno Spark 6 Go with its case (168.6 mm x 78.8 mm).
Calculates and reports:
  - System output for each image
  - Width and height absolute error
  - Width and height percentage error
  - Mean Absolute Error (MAE)
  - Mean Percentage Error (MPE)

Usage:
    python measurement/validate_accuracy.py
"""

import os
import sys
import json
import glob
import cv2
import numpy as np
import torch
from tabulate import tabulate

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from measurement.measure import load_calibration, undistort_image, compute_pixels_per_mm_from_calibration, measure_object
from measurement.demo import build_predictor

# Ground Truth dimensions (Techno Spark 6 Go with protective case)
GT_WIDTH_MM = 78.8
GT_HEIGHT_MM = 168.6

TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "test", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "measurement", "outputs")


def main():
    print("=" * 70)
    print("  XIS COMPUTER VISION ASSESSMENT — SYSTEM ACCURACY VALIDATION")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Calibration
    camera_matrix, dist_coeffs, square_size_mm, board_rows, board_cols = load_calibration()
    
    # 2. Build Predictor
    print("Loading Mask R-CNN model...")
    predictor = build_predictor(threshold=0.5)
    
    # Get all test images
    image_paths = sorted(
        glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
    )
    
    if not image_paths:
        print(f"ERROR: No test images found in {TEST_IMAGES_DIR}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} test images. Running measurement validation...\n")
    
    results = []
    width_errors = []
    height_errors = []
    width_pct_errors = []
    height_pct_errors = []
    
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Undistort
        img_undistorted, new_cam_mtx = undistort_image(img, camera_matrix, dist_coeffs)
        
        # Pixels per mm
        pixels_per_mm, method = compute_pixels_per_mm_from_calibration(
            new_cam_mtx, dist_coeffs, square_size_mm, img_undistorted, board_rows, board_cols
        )
        
        # Segmentation
        outputs = predictor(img_undistorted)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) == 0:
            print(f"  [{filename}] No phone segmented! Skipping.")
            continue
            
        # Get highest score detection
        scores = instances.scores.numpy()
        best_idx = np.argmax(scores)
        best_mask = instances.pred_masks[best_idx].numpy()
        
        # Measure
        meas = measure_object(best_mask, pixels_per_mm)
        if meas is None:
            print(f"  [{filename}] Measurement extraction failed! Skipping.")
            continue
            
        w_sys = meas["width_mm"]
        h_sys = meas["height_mm"]
        
        # Calculate errors
        w_err = w_sys - GT_WIDTH_MM
        h_err = h_sys - GT_HEIGHT_MM
        
        w_err_abs = abs(w_err)
        h_err_abs = abs(h_err)
        
        w_pct = (w_err_abs / GT_WIDTH_MM) * 100
        h_pct = (h_err_abs / GT_HEIGHT_MM) * 100
        
        results.append({
            "index": idx + 1,
            "filename": filename,
            "gt_width": GT_WIDTH_MM,
            "sys_width": round(w_sys, 2),
            "width_err": round(w_err, 2),
            "width_pct": round(w_pct, 2),
            "gt_height": GT_HEIGHT_MM,
            "sys_height": round(h_sys, 2),
            "height_err": round(h_err, 2),
            "height_pct": round(h_pct, 2),
            "pixels_per_mm": round(pixels_per_mm, 4),
            "method": method
        })
        
        width_errors.append(w_err_abs)
        height_errors.append(h_err_abs)
        width_pct_errors.append(w_pct)
        height_pct_errors.append(h_pct)
        
        print(f"  [{filename}] Measured: {w_sys:.2f}x{h_sys:.2f} mm | Errors: W={w_err:+.2f}mm, H={h_err:+.2f}mm")
        
    # Calculate summary metrics
    mae_width = np.mean(width_errors)
    mae_height = np.mean(height_errors)
    mpe_width = np.mean(width_pct_errors)
    mpe_height = np.mean(height_pct_errors)
    
    # Save results to JSON
    summary = {
        "individual_results": results,
        "metrics": {
            "ground_truth": {
                "width_mm": GT_WIDTH_MM,
                "height_mm": GT_HEIGHT_MM
            },
            "mae": {
                "width_mm": round(float(mae_width), 4),
                "height_mm": round(float(mae_height), 4)
            },
            "mpe": {
                "width_pct": round(float(mpe_width), 4),
                "height_pct": round(float(mpe_height), 4)
            },
            "num_validated_instances": len(results)
        }
    }
    
    json_path = os.path.join(OUTPUT_DIR, "accuracy_validation.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Print tabulate table
    headers = [
        "No.", "Image", "GT W", "Sys W", "W Err", "W Pct", "GT H", "Sys H", "H Err", "H Pct", "Ratio", "Method"
    ]
    rows = []
    for r in results:
        rows.append([
            r["index"], r["filename"][:15] + "..", r["gt_width"], r["sys_width"],
            f"{r['width_err']:+.2f}", f"{r['width_pct']:.2f}%",
            r["gt_height"], r["sys_height"], f"{r['height_err']:+.2f}", f"{r['height_pct']:.2f}%",
            r["pixels_per_mm"], r["method"]
        ])
        
    print("\n" + "=" * 80)
    print("  ACCURACY VALIDATION REPORT TABLE")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 80)
    print(f"  Mean Absolute Error (MAE):   Width = {mae_width:.3f} mm, Height = {mae_height:.3f} mm")
    print(f"  Mean Percentage Error (MPE): Width = {mpe_width:.3f}%, Height = {mpe_height:.3f}%")
    print(f"  Validated instances count:   {len(results)}")
    print(f"  Accuracy report saved to:    {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
