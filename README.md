#  Leaf and Fruits Disease Detection using Color Segmentation and Morphological Operations


## Problem Statement

In agriculture, timely detection of diseases in leaves and fruits is essential to prevent crop loss and improve yield.  
This project provides an ML-based solution that analyzes uploaded images to:

- Detect whether the sample is healthy or diseased.
- Highlight potentially affected regions visually.
- Estimate the percentage of the sample area impacted by visible symptoms.
- Report disease class predictions with confidence-aware interpretation.

---

## Techniques Used

- **Color Segmentation (HSV)**: Detects visually suspicious regions based on color variation.
- **Morphological Operations**: Cleans segmented masks using `open` and `close` operations.
- **Feature Engineering**:
  - HOG (shape/gradient texture)
  - LBP (local texture patterns)
  - Color statistics (BGR/HSV/LAB)
  - HSV histogram
  - Shape features (edge density + Hu moments)
- **Machine Learning Classifier**:
  - `StandardScaler + PCA + SVC (RBF)` for disease classification.
- **Confidence-Aware Reporting**:
  - Low-confidence class outputs are marked as **tentative** instead of final diagnosis.

---

## Program Flow

1. **Upload Input**
   - User uploads a leaf or fruit image in the Streamlit app.

2. **Preprocessing**
   - Convert image to HSV.
   - Segment plant region and isolate primary sample.
   - Generate cleaned disease mask via morphological operations.

3. **Feature Extraction & Prediction**
   - Extract multi-feature descriptor from isolated sample.
   - Predict class using trained SVC pipeline.
   - Derive overall health status (`Healthy` / `Diseased`) and confidence.

4. **Result Dashboard**
   - Show prediction, confidence, severity, and affected-area percentage.
   - Visualize original image, disease mask, detected overlay, and isolation mask.
   - Display written summary with confidence-aware class interpretation.

5. **User Interpretation**
   - If class confidence is low, result is shown as a **low-confidence estimate**.

---

## Evaluation Metrics

| Metric | Value |
|---|---|
| Disease Detection Method | HSV-based segmentation + ML classification |
| Morphological Operations | Open and Close for mask cleaning |
| Inference Output | Healthy/Diseased + disease class (confidence-aware) |
| Affected Area Displayed | Yes (percentage of affected pixels) |
| Confidence Gating | Yes (low-confidence class marked tentative) |

---

## Sample Results

![Screenshot 2025-05-05 141627](https://github.com/user-attachments/assets/fe205c66-99ee-41fa-aac6-d7f69c4695db)

![Screenshot 2025-05-05 141026](https://github.com/user-attachments/assets/936fcf88-9bf8-412c-af1e-c8b60a07d210)


## Sample Results

---
![Screenshot 2025-05-05 141627](https://github.com/user-attachments/assets/fe205c66-99ee-41fa-aac6-d7f69c4695db)

![Screenshot 2025-05-05 141026](https://github.com/user-attachments/assets/936fcf88-9bf8-412c-af1e-c8b60a07d210)

