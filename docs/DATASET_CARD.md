# Dataset Card: Phone Instance Segmentation Dataset

This document provides a comprehensive overview of the self-collected and self-labelled dataset used for training the Mask R-CNN phone segmentation model.

## 1. Dataset Overview

*   **Target Object:** Mobile Phone (specifically a Techno Spark 6 Go with a protective phone case)
*   **Target Class:** `phone` (ID: 1)
*   **Total Images:** 116 images
*   **Image Resolution:** $4080 \times 3072$ pixels (captured using a **Redmi 14 Pro** smartphone camera)
*   **Annotation Type:** Polygon segmentation masks (per-pixel contours)

---

## 2. Collection Strategy & Diversity

To ensure the model is robust to varied environmental factors and generalizes well for measurements, the data was collected with the following variations:

*   **Varied Angles:** Images were captured from directly overhead (top-down), slight angles (perspectives), and tilted orientations.
*   **Varied Lighting Conditions:** Captured under indoor LED lighting, natural window daylight, shadow overlays, and high/low brightness environments.
*   **Varied Backgrounds:** Phones were placed on wooden tables, solid desks, tiled floors, paper sheets, and alongside clutter to prevent background-association bias.
*   **Occlusions & Clutter:** Some images include mild occlusion (e.g., cables, hands, or nearby office items) to test the model's segmentation boundary precision.

---

## 3. Data Split Statistics

The dataset is divided into training, validation, and test splits following a standard 70% / 20% / 10% ratio:

| Split | Number of Images | Description |
|---|---|---|
| **Train** | 81 | Used for parameter updating and gradient descent optimization. |
| **Validation** | 23 | Used for hyperparameter tuning and model selection. |
| **Test** | 12 | Held-out set used solely for final evaluation and prediction visualization. |
| **Total** | **116** | |

---

## 4. Labelling & Annotation Details

*   **Labelling Tool:** CVAT (Computer Vision Annotation Tool)
*   **Format:** COCO 1.0 Instance Segmentation
*   **Annotation Quality:** High-density polygon boundaries tracing the exact outer edge of the phone (including the protective case).
*   **Files:**
    *   `dataset/train/labels/labels.json`
    *   `dataset/val/labels/labels.json`
    *   `dataset/test/labels/labels.json`

---

## 5. Physical Dimensions & Reference Specifications

The model segments the phone, and the measurement pipeline converts the segmented pixels to metric units.

### Physical Dimensions of the Phone
*   **Model:** Techno Spark 6 Go
*   **Official Dimensions:** $165.6 \times 76.3 \times 9.1\text{ mm}$ (Height × Width × Thickness)
*   **With Case:** A protective cover/case is installed. Since a vernier caliper is not available, the protective case is assumed to add approximately $2.5\text{ mm}$ to the width and $3.0\text{ mm}$ to the height.
*   **Estimated Physical Boundary Dimensions:**
    *   **Width (with case):** $\approx 78.8\text{ mm}$
    *   **Height (with case):** $\approx 168.6\text{ mm}$
    *   **Reference Ratio:** Used for metric conversion checks and Mean Absolute Error (MAE) validation.
