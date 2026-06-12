# Camera Calibration Report

This report documents the camera calibration process performed for the XIS Technical Assessment. Camera calibration is a necessary prerequisite for accurate metric measurement because it computes the camera's intrinsic parameters and lens distortion coefficients, allowing us to mathematically correct (undistort) the captured imagery.

## 1. Methodology

We used the standard camera calibration model proposed by Zhang (2000) and implemented in OpenCV. 

### Checkerboard Specifications
*   **Pattern Type:** Checkerboard (Chessboard)
*   **Dimensions:** 7 rows × 10 columns of squares
*   **Inner Corners:** 6 rows × 9 columns (54 inner corners)
*   **Square Size:** 25.0 mm (standard size on a calibration target screen/page)

### Process
1.  **Image Collection:** Collected 34 images of the checkerboard pattern from various distances, orientations, and angles. Out of these, 17 images with clear, fully visible patterns were successfully detected and used for calibration.
2.  **Corner Detection:** Corner points were detected using `cv2.findChessboardCorners()` and refined to sub-pixel accuracy using `cv2.cornerSubPix()` to minimize localization error.
3.  **Parameter Optimization:** Computed the intrinsic camera matrix, distortion coefficients, and reprojection errors using `cv2.calibrateCamera()`.

---

## 2. Calibration Parameters

### Intrinsic Camera Matrix ($K$)
The camera matrix represents the optical center and focal lengths of the camera in pixel units:

$$
K = \begin{bmatrix} 
f_x & 0 & c_x \\ 
0 & f_y & c_y \\ 
0 & 0 & 1 
\end{bmatrix}
$$

From our calibration, the values are:

```json
[
    [2866.8645554840896, 0.0, 2038.9470405569791],
    [0.0, 2866.8405384223383, 1515.9795261596641],
    [0.0, 0.0, 1.0]
]
```

*   **Focal Lengths:** $f_x \approx 2866.86\text{ px}$, $f_y \approx 2866.84\text{ px}$ (nearly identical, indicating square pixels)
*   **Principal Point (Optical Center):** $c_x \approx 2038.95\text{ px}$, $c_y \approx 1515.98\text{ px}$ (very close to the physical image center of $2040 \times 1536$)

### Distortion Coefficients
The lens distortion is modeled using 5 coefficients ($k_1, k_2, p_1, p_2, k_3$):
*   **Radial Distortion ($k_1, k_2, k_3$):** Corrects for barrel and pincushion distortion.
*   **Tangential Distortion ($p_1, p_2$):** Corrects for lens-sensor misalignment.

Our computed coefficients are:

```json
[
    [0.09002579782345324, -0.4376937043286679, -0.00008335713681173696, -0.00010713550143323614, 0.5052271672309571]
]
```

*   $k_1 \approx 0.0900$
*   $k_2 \approx -0.4377$
*   $p_1 \approx -0.000083$
*   $p_2 \approx -0.000107$
*   $k_3 \approx 0.5052$

---

## 3. Reprojection Error

The quality of calibration is evaluated using the **Mean Reprojection Error**, which is the average Euclidean distance (in pixels) between the detected corners in the calibration images and their corresponding projected 3D points.

*   **Reprojection Error:** **0.1626 pixels**
*   **Assessment:** Excellent (the threshold for "excellent" is $< 0.30$ pixels; ours is nearly half of that, demonstrating extremely high geometric precision).
*   **Calibration Image Size:** $4080 \times 3072$ pixels
*   **Images Used:** 17 out of the collected set.

---

## 4. Undistortion Effect

The computed intrinsic parameters are used to undistort the captured images using:
1.  `cv2.getOptimalNewCameraMatrix()` to calculate a new camera matrix that accounts for the scaling of the corrected image.
2.  `cv2.undistort()` to re-map the pixels and remove radial/tangential distortion.

This process straightens out curved lines near the borders of the image, making sure that lines that are straight in the physical world are straight in pixel space, which is critical for making accurate metric measurements of object dimensions.
