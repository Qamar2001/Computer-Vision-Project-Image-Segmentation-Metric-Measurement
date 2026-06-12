"""
End-to-End Measurement Demo Script
===================================
Takes a single raw input image, applies camera undistortion, runs Mask R-CNN
instance segmentation, converts segmented phone mask to real-world metric
dimensions (width & height in mm), and saves the annotated demonstration image.

Usage:
    python measurement/demo.py --input <image_path>
    python measurement/demo.py --input <image_path> --output <output_path>
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from measurement.measure import load_calibration, undistort_image, compute_pixels_per_mm_from_calibration, measure_object, annotate_measurement

MODEL_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "output", "model_final.pth")
DEMO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "measurement", "outputs")


def build_predictor(threshold=0.5):
    """Load Mask R-CNN model predictor."""
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"ERROR: Model weights not found at: {MODEL_WEIGHTS}")
        print("Please run models/train.py first!")
        sys.exit(1)
        
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return DefaultPredictor(cfg)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Phone Measurement Demo")
    parser.add_argument("--input", type=str, required=True, help="Path to input raw image")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Model detection score threshold")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  XIS COMPUTER VISION ASSESSMENT — MEASUREMENT DEMO")
    print("=" * 60)
    
    # 1. Load Calibration params
    camera_matrix, dist_coeffs, square_size_mm, board_rows, board_cols = load_calibration()
    print("Camera calibration parameters loaded successfully")
    
    # 2. Build Predictor
    print("Loading Mask R-CNN model...")
    predictor = build_predictor(args.threshold)
    print(f"Model loaded. Inference device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    
    # 3. Read raw image
    img = cv2.imread(args.input)
    if img is None:
        print(f"ERROR: Cannot read image at {args.input}")
        sys.exit(1)
    print(f"Loaded input image: {os.path.basename(args.input)} ({img.shape[1]}x{img.shape[0]} px)")
    
    # 4. Undistort image
    print("Applying lens distortion correction (undistort)...")
    img_undistorted, new_cam_mtx = undistort_image(img, camera_matrix, dist_coeffs)
    
    # 5. Compute pixels per mm ratio
    # Detect if checkerboard is in image, otherwise fallback
    pixels_per_mm, method = compute_pixels_per_mm_from_calibration(
        new_cam_mtx, dist_coeffs, square_size_mm, img_undistorted, board_rows, board_cols
    )
    
    # 6. Run Segmentation Model Inference
    print("Running Mask R-CNN segmenter...")
    outputs = predictor(img_undistorted)
    instances = outputs["instances"].to("cpu")
    
    num_phones = len(instances)
    print(f"Detected {num_phones} phone instance(s) in image")
    
    if num_phones == 0:
        print("ERROR: No phone segmented in image!")
        sys.exit(1)
        
    # Get highest confidence detection
    scores = instances.scores.numpy()
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_mask = instances.pred_masks[best_idx].numpy()
    
    print(f"Segmented phone instance with confidence: {best_score:.3f}")
    
    # 7. Compute Metric Measurements
    print("Calculating real-world metric dimensions...")
    measurement = measure_object(best_mask, pixels_per_mm)
    
    if measurement is None:
        print("ERROR: Could not compute measurements from mask contour.")
        sys.exit(1)
        
    print("\n--- MEASUREMENT RESULTS ---")
    print(f"Width:  {measurement['width_mm']:.1f} mm  (pixels: {measurement['width_px']:.1f} px)")
    print(f"Height: {measurement['height_mm']:.1f} mm  (pixels: {measurement['height_px']:.1f} px)")
    print(f"Area:   {measurement['area_mm2']:.1f} mm²  (pixels: {measurement['area_px']} px)")
    print(f"Pixels per mm ratio: {pixels_per_mm:.4f} px/mm (Method: {method})")
    print("---------------------------\n")
    
    # 8. Annotate Measurement Results on Image
    label_text = f"Phone (Conf: {best_score:.2%})"
    annotated_img = annotate_measurement(img_undistorted, measurement, label=label_text)
    
    # 9. Save output
    if args.output is None:
        os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(DEMO_OUTPUT_DIR, f"demo_measured_{basename}.jpg")
        
    cv2.imwrite(args.output, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Demo result successfully saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
