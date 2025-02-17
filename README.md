# MyAutoPano - Panorama Image Stitching
RBE/CS Fall 2022: Classical and Deep Learning Approaches for Geometric Computer Vision

## Team Members - Group 19
- Prasham Soni
- Sarthak Mehta

## Project Structure
```
Group19_p1/
└── Phase1/
    └── Code/
        └── wrapper.py
    └── Phase2/
        ├── Code/
        │   ├── Wrapper.py                 # Main wrapper script with switch case for Supervised & Unsupervised
        │   ├── predict_supervised.py       # Supervised model pipeline for homography prediction
        │   ├── predict_unsupervised.py     # Unsupervised model pipeline for homography prediction
        │   ├── CustomDataset.py            # Dataset loading and preprocessing
        │   ├── Train.py                    # Training script for models
        │   ├── Test.py                     # Testing script for evaluation
        ├── Network/
        │   ├── HomographyNet.py             # Supervised deep learning model architecture
        │   ├── Unsupervised_HomographyNet.py # Unsupervised model architecture
        ├── Misc/                            # Additional utility files
        ├── TxtFiles/                        # Configuration and experiment logs

```

## Overview
---

## **Overview**
This project explores **Deep Learning-based Panorama Stitching**, extending **classical feature-based homography estimation (Phase 1)** with **Supervised and Unsupervised Learning Approaches**. Instead of relying on **corner detection and feature matching**, deep neural networks are trained to estimate homographies between overlapping images.

---

## **Phase 1: Classical Feature-Based Panorama Stitching**
1. **Corner Detection**  
   - Uses **Harris corner detection** to identify key points in images.

2. **ANMS (Adaptive Non-Maximal Suppression)**  
   - Selects the best corner points while maintaining spatial distribution.

3. **Feature Descriptor & Matching**  
   - Generates **SIFT feature descriptors** for keypoints.
   - Matches features using **the ratio test**.

4. **Homography Estimation using RANSAC**  
   - Eliminates outliers and computes the **global homography matrix**.

5. **Image Warping & Blending**  
   - Warps images using **perspective transformation**.
   - Blends the images to generate the panorama.

---

## **Phase 2: Deep Learning-Based Homography Estimation**
### ✅ **Supervised Learning Approach**
- The **Supervised Homography Network (HomographyNet.py)** is trained on **128×128 patches**.
- The model **predicts 4-point displacement vectors (H4Pt)** instead of directly computing homographies.
- Uses **Direct Linear Transformation (DLT)** to convert **H4Pt** into a **homography matrix**.
- Predicted homographies are applied **patch-wise**, leading to **local alignment inconsistencies**.

### 🤖 **Unsupervised Learning Approach**
- The **Unsupervised Homography Network (Unsupervised_HomographyNet.py)** learns to predict the **homography matrix (H_pred)** end-to-end.
- Uses a **self-supervised loss function**, minimizing the difference between warped images.
- The model struggled to **converge**, leading to **overlapping but misaligned images**.
- The loss function **did not stabilize**, making training unreliable.

---

## **Algorithm Pipeline**
### **Supervised Learning Pipeline**
1. **Patch Extraction**  
   - Extracts **128×128 patches** from images and computes their transformations.
   
2. **CNN-Based Homography Prediction**  
   - The model predicts **H4Pt displacements** between patches.
   - Uses **DLT** to compute the **homography matrix**.

3. **Warping & Blending**  
   - Warps patches using predicted homographies.
   - Aggregates local transformations to construct the final panorama.

### **Unsupervised Learning Pipeline**
1. **Feature Learning Without Ground Truth**  
   - Directly estimates the **homography matrix** using CNN.
   - Uses a **loss function** to compare the warped and target images.

2. **Image Warping**  
   - The predicted homography is applied to **warp one image onto another**.

3. **Blending & Panorama Stitching**  
   - Final images are merged using **weighted blending**.
   - Unsupervised training struggled with **alignment stability**.

---

## **Requirements**
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy  
- PyTorch  
- Matplotlib  

---

## **Usage**
1. Place input images in `Data/Test/`
2. Run the **wrapper script** to choose between **Supervised or Unsupervised** methods:
   ```bash
   python Wrapper.py
