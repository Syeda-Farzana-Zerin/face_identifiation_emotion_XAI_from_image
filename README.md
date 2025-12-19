# 🧠 Face Identification & Emotion Recognition with Grad-CAM (Streamlit)

This repository presents an **interactive Streamlit application** for **face identification and emotion recognition**, enhanced with **explainable AI (XAI)** using **Grad-CAM visualizations**.

The project demonstrates how **convolutional neural networks (CNNs)** make predictions on facial images and provides visual explanations highlighting **which facial regions influence identity and emotion predictions**.

---

## ✨ Features

- **Face Identification** using a CNN trained on the **Labeled Faces in the Wild (LFW)** dataset  
- **Emotion Recognition** using a **pretrained CNN model**  
- **Explainability with Grad-CAM**
  - Identity Grad-CAM (grayscale saliency overlay)
  - Emotion Grad-CAM (colored heatmap overlay)
  - Identity Grad-CAM (heatmap-only view)
- **Interactive Streamlit Interface**
  - Step-through testing using a **Next image** button
  - Side-by-side comparison of explanations
- **Visualization-safe preprocessing**
  - Contrast enhancement (CLAHE) applied **only for display**
  - Model inputs remain unchanged for correct learning and inference

---

## 🖼️ Application Output

For each test image, the application displays:

1. **Identity Grad-CAM (grayscale overlay)**  
2. **Emotion Grad-CAM (colored overlay)**  
3. **Identity Grad-CAM (heatmap only)**  

This layout allows inspection of:
- Model attention patterns
- Differences between identity and emotion cues
- Explainability on a per-sample basis

---

## 📊 Datasets

### Labeled Faces in the Wild (LFW)
- Used for **face identification**
- Grayscale benchmark dataset
- Automatically downloaded using `scikit-learn`

### Emotion Recognition Model
- Pretrained CNN (FER-style architecture)
- Loaded locally from:
  ```text
  emotion_model.h5
