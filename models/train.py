"""
Mask R-CNN Training Script (Detectron2)
=======================================
Trains a Mask R-CNN model with ResNet-50-FPN backbone for phone instance
segmentation using a custom COCO-format dataset.

Why Mask R-CNN?
--------------
1. Instance segmentation (per-pixel masks) — essential for accurate contour-based
   measurement, unlike bounding-box-only detectors.
2. Proven architecture on COCO benchmark with well-documented performance.
3. Detectron2 provides robust COCO-format dataset integration out of the box.
4. Transfer learning from COCO pretrained weights enables training with small datasets.
5. Outputs masks, bounding boxes, and confidence scores simultaneously.
6. NOT YOLO/Ultralytics and NOT Roboflow — satisfies assessment constraints.

Usage:
    python models/train.py

Output:
    - models/output/model_final.pth   (trained weights)
    - models/output/metrics.json      (training metrics)
    - models/output/loss_curves.png   (loss visualization)
"""

import os
import sys
import json
import copy
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

setup_logger()

# ============================================================
# CONFIGURATION
# ============================================================
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODELS_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(MODELS_DIR, "output")

# Dataset paths
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_DIR, "train", "labels", "labels.json")
VAL_IMAGES = os.path.join(DATASET_DIR, "val", "images")
VAL_LABELS = os.path.join(DATASET_DIR, "val", "labels", "labels.json")
TEST_IMAGES = os.path.join(DATASET_DIR, "test", "images")
TEST_LABELS = os.path.join(DATASET_DIR, "test", "labels", "labels.json")


class LossLogger(HookBase):
    """Custom hook to log training losses for plotting."""
    
    def __init__(self):
        super().__init__()
        self.losses = []
    
    def after_step(self):
        if self.trainer.iter % 20 == 0:
            loss_dict = {}
            for k, v in self.trainer.storage.latest().items():
                if 'loss' in k.lower():
                    val = v[0]
                    loss_dict[k] = val.item() if hasattr(val, 'item') else val
            total_loss = sum(loss_dict.values())
            self.losses.append({
                "iteration": self.trainer.iter,
                "total_loss": total_loss,
                **loss_dict
            })


def register_datasets():
    """Register train/val/test datasets with Detectron2."""
    datasets = {
        "phone_train": (TRAIN_LABELS, TRAIN_IMAGES),
        "phone_val": (VAL_LABELS, VAL_IMAGES),
        "phone_test": (TEST_LABELS, TEST_IMAGES),
    }
    
    for name, (json_path, img_dir) in datasets.items():
        if name not in DatasetCatalog.list():
            register_coco_instances(name, {}, json_path, img_dir)
            MetadataCatalog.get(name).set(thing_classes=["phone"])
    
    print(f"Registered datasets: {list(datasets.keys())}")
    return datasets


def build_config():
    """
    Build Detectron2 configuration for Mask R-CNN training.
    
    Architecture: Mask R-CNN with ResNet-50 backbone + FPN (Feature Pyramid Network)
    - ResNet-50-FPN provides multi-scale feature extraction
    - Pretrained on COCO for transfer learning
    - FPN enables detection of objects at different scales
    """
    cfg = get_cfg()
    
    # Base architecture: Mask R-CNN R50-FPN
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Use COCO pretrained weights for transfer learning
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Dataset
    cfg.DATASETS.TRAIN = ("phone_train",)
    cfg.DATASETS.TEST = ("phone_val",)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = 0
    
    # Solver / Training hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 2          # Batch size (2 for 6GB VRAM)
    cfg.SOLVER.BASE_LR = 0.0025           # Learning rate (lower than default for fine-tuning)
    cfg.SOLVER.MAX_ITER = 3000            # Total training iterations
    cfg.SOLVER.STEPS = (2000, 2500)       # LR decay schedule
    cfg.SOLVER.GAMMA = 0.1                # LR decay factor
    cfg.SOLVER.WARMUP_ITERS = 200         # Warmup iterations
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 500    # Save checkpoint every 500 iters
    
    # Model head — single class (phone)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # ROI batch size
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5     # Inference threshold
    
    # Input augmentation
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Output directory
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Device
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("WARNING: Training on CPU — this will be very slow!")
    
    return cfg


def plot_loss_curves(losses, output_path):
    """Generate and save training loss curve plots."""
    if not losses:
        print("No losses recorded — skipping plot")
        return
    
    iterations = [l["iteration"] for l in losses]
    total_loss = [l["total_loss"] for l in losses]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(iterations, total_loss, 'b-', linewidth=1.5, label='Total Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Individual losses
    loss_keys = [k for k in losses[0].keys() if k not in ('iteration', 'total_loss')]
    for key in loss_keys:
        values = [l.get(key, 0) for l in losses]
        axes[1].plot(iterations, values, linewidth=1, label=key)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Individual Loss Components')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {output_path}")


def evaluate_model(cfg, dataset_name="phone_test"):
    """Run COCO evaluation on test set."""
    print(f"\n{'='*50}")
    print(f"  EVALUATION ON {dataset_name}")
    print(f"{'='*50}")
    
    cfg_eval = cfg.clone()
    cfg_eval.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    cfg_eval.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg_eval.DATASETS.TEST = (dataset_name,)
    
    predictor = DefaultPredictor(cfg_eval)
    
    evaluator = COCOEvaluator(dataset_name, output_dir=OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg_eval, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    return results


def visualize_predictions(cfg, dataset_name="phone_test", num_samples=6):
    """Visualize model predictions on test images."""
    cfg_vis = cfg.clone()
    cfg_vis.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    cfg_vis.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg_vis)
    metadata = MetadataCatalog.get(dataset_name)
    
    dataset = DatasetCatalog.get(dataset_name)
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, d in enumerate(dataset[:num_samples]):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        out_path = os.path.join(vis_dir, f"prediction_{i+1}.jpg")
        cv2.imwrite(out_path, out.get_image()[:, :, ::-1])
        print(f"  Saved visualization: {out_path}")
    
    print(f"\n{num_samples} prediction visualizations saved to: {vis_dir}")


def main():
    print("=" * 60)
    print("  MASK R-CNN TRAINING — Phone Segmentation")
    print("=" * 60)
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Step 1: Register datasets
    register_datasets()
    
    # Step 2: Build config
    cfg = build_config()
    
    # Print training configuration summary
    print(f"\n--- Training Configuration ---")
    print(f"  Architecture: Mask R-CNN R50-FPN (pretrained on COCO)")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  LR schedule: decay at {cfg.SOLVER.STEPS}")
    print(f"  Device: {cfg.MODEL.DEVICE}")
    print(f"  Output: {cfg.OUTPUT_DIR}")
    
    # Step 3: Train
    print(f"\nStarting training...\n")
    
    loss_logger = LossLogger()
    trainer = DefaultTrainer(cfg)
    trainer.register_hooks([loss_logger])
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Step 4: Plot loss curves
    loss_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plot_loss_curves(loss_logger.losses, loss_path)
    
    # Save losses as JSON for documentation
    with open(os.path.join(OUTPUT_DIR, "training_losses.json"), 'w') as f:
        json.dump(loss_logger.losses, f, indent=2)
    
    # Step 5: Evaluate on validation set
    val_results = evaluate_model(cfg, "phone_val")
    
    # Step 6: Evaluate on test set
    test_results = evaluate_model(cfg, "phone_test")
    
    # Step 7: Save evaluation results
    all_results = {
        "validation": val_results,
        "test": test_results,
        "config": {
            "architecture": "Mask R-CNN R50-FPN",
            "pretrained": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x",
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "learning_rate": cfg.SOLVER.BASE_LR,
            "max_iterations": cfg.SOLVER.MAX_ITER,
            "num_classes": 1,
            "class_names": ["phone"]
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Step 8: Visualize predictions
    visualize_predictions(cfg, "phone_test")
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nModel saved to: {os.path.join(OUTPUT_DIR, 'model_final.pth')}")
    print(f"Results saved to: {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")
    print(f"\nNext step: Run inference with inference/infer.py")


if __name__ == "__main__":
    main()
