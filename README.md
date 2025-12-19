# 🧠 Face Identification & Emotion Recognition with Grad-CAM (Streamlit)

An interactive Streamlit application for **face identification and emotion recognition**, enhanced with **explainable AI (XAI)** using **Grad-CAM** visualizations.

---

## 🧠 Models

```text
Identity Classification Model

- Custom CNN trained from scratch

Architecture:
- Convolution + MaxPooling layers
- Fully connected layers with Dropout

Optimizer: Adam  
Loss: Categorical Cross-Entropy


Emotion Recognition Model

- Pretrained CNN
- Predicts 7 emotion classes:
  Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Used only for inference
🔍 Explainability (Grad-CAM)
text
Copy code
Gradient-weighted Class Activation Mapping (Grad-CAM) is used to:

- Highlight important facial regions influencing predictions
- Compare attention for identity vs emotion
- Provide both overlay and standalone heatmap views

The visualization strategy follows best practices used in explainable AI research.
🚀 Installation
text
Copy code
Install dependencies:
pip install streamlit tensorflow scikit-learn opencv-python numpy


Prepare the emotion model:
Place the pretrained emotion model in the project root:

emotion_model.h5


Run the application:
streamlit run app.py
📁 Repository Structure
text
Copy code
.
├── app.py                 # Main Streamlit application
├── emotion_model.h5       # Pretrained emotion recognition model
├── README.md              # Project documentation
🧪 How to Use
text
Copy code
1. Launch the Streamlit app
2. Click "Train Identity Model"
3. Navigate through test samples using "Next image"
4. Observe:
   - Identity prediction
   - Emotion prediction
   - Grad-CAM explanations
🎓 Intended Use
text
Copy code
This repository is suitable for:

- Explainable AI (XAI) demonstrations
- Computer vision education
- Research prototypes
- Academic projects and theses
⚠️ Limitations
text
Copy code
- LFW is a benchmark dataset with limited visual quality
- Identity recognition performance is constrained by dataset size
- Emotion recognition depends on pretrained model quality
