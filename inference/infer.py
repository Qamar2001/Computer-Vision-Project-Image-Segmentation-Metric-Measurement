"""
Inference Script
================
Runs Mask R-CNN inference on new images with camera undistortion applied.

Usage:
    python inference/infer.py --input <image_path>
    python inference/infer.py --input <image_path> --output <output_path>
    python inference/infer.py --input_dir <directory>  # Process all images in directory

Output:
    Annotated image with segmentation mask overlay saved to inference/outputs/
"""

import os
import sys
import argparse
import json
import glob
import cv2
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ============================================================
# CONFIGURATION
# ============================================================
INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(INFERENCE_DIR, "outputs")
MODEL_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "output", "model_final.pth")
CALIBRATION_PARAMS = os.path.join(PROJECT_ROOT, "calibration", "calibration_params.json")


def load_calibration():
    """Load camera calibration parameters for undistortion."""
    if not os.path.exists(CALIBRATION_PARAMS):
        print("WARNING: Calibration params not found — skipping undistortion")
        return None, None
    
    with open(CALIBRATION_PARAMS, 'r') as f:
        params = json.load(f)
    
    camera_matrix = np.array(params["camera_matrix"])
    dist_coeffs = np.array(params["dist_coeffs"])
    return camera_matrix, dist_coeffs


def undistort_image(image, camera_matrix, dist_coeffs):
    """Apply lens undistortion to an image."""
    if camera_matrix is None:
        return image
    
    h, w = image.shape[:2]
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0, (w, h)
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_cam_mtx)
    return undistorted


def build_predictor(threshold=0.5):
    """Build Detectron2 predictor with trained weights."""
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        print("Run models/train.py first!")
        sys.exit(1)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set metadata for visualization
    metadata = MetadataCatalog.get("phone_inference")
    metadata.set(thing_classes=["phone"])
    
    predictor = DefaultPredictor(cfg)
    return predictor, metadata


def run_inference(image_path, predictor, metadata, camera_matrix, dist_coeffs, output_path=None):
    """
    Run inference on a single image.
    
    Pipeline:
    1. Load image
    2. Undistort using calibration parameters
    3. Run Mask R-CNN inference
    4. Visualize predictions
    5. Save annotated output
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return None
    
    # Step 1: Undistort
    img_undistorted = undistort_image(img, camera_matrix, dist_coeffs)
    
    # Step 2: Run inference
    outputs = predictor(img_undistorted)
    instances = outputs["instances"].to("cpu")
    
    # Step 3: Extract results
    num_detections = len(instances)
    results = {
        "image": os.path.basename(image_path),
        "num_detections": num_detections,
        "detections": []
    }
    
    if num_detections > 0:
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()
        
        for i in range(num_detections):
            det = {
                "class": "phone",
                "confidence": float(scores[i]),
                "bbox": boxes[i].tolist(),
                "mask_area_pixels": int(masks[i].sum())
            }
            results["detections"].append(det)
            print(f"  Detection {i+1}: phone (confidence={scores[i]:.3f}, "
                  f"bbox_area={int((boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]))} px)")
    else:
        print(f"  No detections in {os.path.basename(image_path)}")
    
    # Step 4: Visualize
    v = Visualizer(
        img_undistorted[:, :, ::-1],
        metadata=metadata,
        scale=0.6,
        instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(instances)
    annotated = out.get_image()[:, :, ::-1]
    
    # Step 5: Save output
    if output_path is None:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUTS_DIR, f"result_{basename}.jpg")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Output saved: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN Phone Segmentation Inference")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--input_dir", type=str, help="Path to directory of images")
    parser.add_argument("--output", type=str, default=None, help="Output path (optional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.print_help()
        sys.exit(1)
    
    print("=" * 50)
    print("  MASK R-CNN INFERENCE — Phone Segmentation")
    print("=" * 50)
    
    # Load calibration
    camera_matrix, dist_coeffs = load_calibration()
    if camera_matrix is not None:
        print("Camera calibration loaded — undistortion enabled")
    
    # Build predictor
    predictor, metadata = build_predictor(args.threshold)
    print(f"Model loaded from: {MODEL_WEIGHTS}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Process images
    all_results = []
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    if args.input:
        print(f"\nProcessing: {args.input}")
        result = run_inference(args.input, predictor, metadata, 
                              camera_matrix, dist_coeffs, args.output)
        if result:
            all_results.append(result)
    
    elif args.input_dir:
        image_paths = sorted(
            glob.glob(os.path.join(args.input_dir, "*.jpg")) +
            glob.glob(os.path.join(args.input_dir, "*.png"))
        )
        print(f"\nProcessing {len(image_paths)} images from: {args.input_dir}\n")
        
        for img_path in image_paths:
            result = run_inference(img_path, predictor, metadata,
                                  camera_matrix, dist_coeffs)
            if result:
                all_results.append(result)
    
    # Save results summary
    summary_path = os.path.join(OUTPUTS_DIR, "inference_results.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processed {len(all_results)} images")
    print(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
