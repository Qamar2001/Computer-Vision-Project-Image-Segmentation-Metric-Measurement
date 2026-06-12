# Environment Setup & Installation Guide

This guide explains how to set up the environment and run the computer vision pipeline on a Windows machine.

---

## 1. Prerequisites & Hardware Requirements

### Hardware Checklist
*   **CPU:** Ryzen 5 2600 (6 Cores, 12 Threads) or equivalent
*   **RAM:** 16 GB DDR4
*   **GPU:** NVIDIA GTX 1660 Super (6 GB VRAM)
*   **OS:** Windows 10/11 (x64)

### Software Prerequisites
*   **Python:** 3.10.x (recommended) or 3.9.x
*   **CUDA Toolkit:** CUDA 12.x or CUDA 11.x (driver support up to 12.x/13.x is fine)
*   **Compiler:** Visual Studio Build Tools (C++ compiler) — needed if compiling Detectron2 from source.

---

## 2. Installation Steps

### Step 2.1: Clone the Repository
Open a terminal (e.g., PowerShell) and navigate to the project directory:
```bash
cd d:\MSCS\Project
```

### Step 2.2: Set Up Virtual Environment (Recommended)
Create and activate a virtual environment to keep dependencies isolated:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### Step 2.3: Upgrade pip and Install Setuptools
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 2.4: Install PyTorch with GPU Support (CUDA 12.6/12.1)
To utilize the GTX 1660 Super for fast deep learning training, we install PyTorch built with CUDA support:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --upgrade
```

To verify if PyTorch detects your GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
Expected output:
```text
CUDA Available: True
Device Name: NVIDIA GeForce GTX 1660 SUPER
```

### Step 2.5: Install Detectron2 on Windows
Detectron2 does not officially support Windows, but precompiled wheels are available, or it can be compiled from source.

**Option A (Precompiled wheels for Windows):**
```powershell
pip install detectron2 -f https://dl.fbaipubliccloud.com/detectron2/wheels/cu121/torch2.1/index.html
```

**Option B (From Source):**
1.  Install **Visual Studio Build Tools** with the "Desktop development with C++" workload.
2.  Run:
    ```powershell
    pip install git+https://github.com/facebookresearch/detectron2.git
    ```

### Step 2.6: Install Remaining Dependencies
Install pycocotools (COCO dataset parser) and the rest of the libraries:
```powershell
pip install -r requirements.txt
```

---

## 3. Pipeline Execution Guide

Each step below builds on the outputs of the previous step.

### Step 3.1: Camera Calibration
Runs intrinsic camera calibration using checkerboard images in the `calibration/` directory.
```powershell
python calibration/calibrate_camera.py
```
*   **Output:** `calibration/calibration_params.json` (intrinsic matrix and distortion coefficients)

### Step 3.2: Undistort Dataset
Applies calibration parameters to lens-correct all training, validation, test, and calibration images.
```powershell
python calibration/undistort_dataset.py
```
*   **Output:** Images in `dataset/` splits are corrected in-place.

### Step 3.3: Train the Segmentation Model
Trains a Mask R-CNN model on the undistorted dataset.
```powershell
python models/train.py
```
*   **Output:**
    *   `models/output/model_final.pth` (trained weights)
    *   `models/output/loss_curves.png` (loss plots)
    *   `models/output/evaluation_results.json` (COCO evaluation metrics)

### Step 3.4: Run Segmentation Inference
Runs Mask R-CNN prediction on a single test image or input folder.
```powershell
# Single image
python inference/infer.py --input dataset/test/images/1781284063548.jpg

# Whole directory
python inference/infer.py --input_dir dataset/test/images
```
*   **Output:** Marked images under `inference/outputs/`.

### Step 3.5: Run Metric Measurement
Computes width and height of the phone in millimeters from raw images.
```powershell
python measurement/measure.py --input dataset/test/images/1781284063548.jpg
```
*   **Output:** Printed measurement values and annotated image with oriented bounding box and dimensions.
