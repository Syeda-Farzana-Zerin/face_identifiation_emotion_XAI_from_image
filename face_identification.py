import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# =================================================
# Utilities
# =================================================
def to_uint8_gray3(img):
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def enhance_gray_for_display(img):
    """
    CLAHE-enhanced grayscale image for visualization ONLY
    """
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def overlay_cam_gray(img, heatmap):
    """
    Identity Grad-CAM as grayscale saliency map
    """
    base = enhance_gray_for_display(img.squeeze())

    heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    heatmap_gray = np.uint8(255 * heatmap)
    heatmap_gray = cv2.cvtColor(heatmap_gray, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(base, 0.6, heatmap_gray, 0.4, 0)


# =================================================
# Streamlit setup
# =================================================
st.set_page_config(page_title="Face ID + Emotion + Grad-CAM", layout="centered")
st.title("Face Identification + Emotion Recognition + Grad-CAM")

# =================================================
# Load LFW dataset
# =================================================
@st.cache_data
def load_lfw():
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
    X = lfw.images / 255.0
    y = lfw.target
    names = lfw.target_names
    X = X[..., np.newaxis]
    y_cat = to_categorical(y)
    return X, y, y_cat, names


X, y, y_cat, target_names = load_lfw()

# =================================================
# Train / Test split
# =================================================
X_train, X_test, y_train_cat, y_test_cat, y_train_lbl, y_test_lbl = train_test_split(
    X, y_cat, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =================================================
# Identity CNN
# =================================================
@st.cache_resource
def train_identity_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               input_shape=X_train.shape[1:], name="id_conv1"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu', name="id_conv2"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu', name="id_conv3"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(y_train_cat.shape[1], activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train_cat,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    return model


# =================================================
# Grad-CAM
# =================================================
def compute_gradcam(model, image, class_idx, layer_name):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = np.power(heatmap, 0.5)

    return heatmap


def overlay_cam(img, heatmap):
    img_gray = to_uint8_gray3(img.squeeze())
    heatmap = cv2.resize(heatmap, (img_gray.shape[1], img_gray.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_gray, 0.5, heatmap, 0.5, 0)


# =================================================
# Emotion CNN (weights-only)
# =================================================
@st.cache_resource
def load_emotion_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               input_shape=(48, 48, 1), name="emo_conv1"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu', name="emo_conv2"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu', name="emo_conv3"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu', name="emo_fc1"),
        Dropout(0.5),
        Dense(7, activation='softmax', name="emo_predictions")
    ])

    model.load_weights("emotion_model.h5", by_name=True, skip_mismatch=True)
    return model


emotion_model = load_emotion_model()
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def predict_emotion(face):
    img = cv2.resize(face.squeeze(), (48, 48)) / 255.0
    img = img.reshape(1, 48, 48, 1)
    preds = emotion_model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return idx, emotion_labels[idx], np.max(preds)


# =================================================
# Training
# =================================================
st.header("🔹 Part 1: Train Identity CNN")

if st.button("Train Identity Model"):
    st.session_state["id_model"] = train_identity_model()
    st.session_state["idx"] = 0
    st.success("Training completed")

# =================================================
# Testing
# =================================================
st.header("🔹 Part 2: Testing (Identity + Emotion + Grad-CAM)")

if "id_model" in st.session_state:
    model = st.session_state["id_model"]

    if "idx" not in st.session_state:
        st.session_state["idx"] = 0

    idx = st.session_state["idx"]

    face = X_test[idx:idx + 1]
    true_id = y_test_lbl[idx]

    pred_id = np.argmax(model.predict(face, verbose=0))
    id_cam = compute_gradcam(model, face, pred_id, "id_conv2")

    emo_idx, emo_label, emo_conf = predict_emotion(face[0])
    emo_face = cv2.resize(face[0].squeeze(), (48, 48)).reshape(1, 48, 48, 1)
    emo_cam = compute_gradcam(emotion_model, emo_face, emo_idx, "emo_conv2")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            overlay_cam(face[0], id_cam),
            caption="Identity Grad-CAM",
            use_container_width=True
        )

    with col2:
        st.image(
            overlay_cam(face[0], emo_cam),
            caption="Emotion Grad-CAM",
            use_container_width=True
        )

    with col3:
        st.image(
            overlay_cam_gray(face[0], id_cam),
            caption="Original Face",
            use_container_width=True
        )

    st.markdown(
        f"""
        **True Identity:** {target_names[true_id]}  
        **Predicted Identity:** {target_names[pred_id]}  
        **Emotion:** {emo_label} ({emo_conf*100:.1f}%)
        """
    )

    if st.button("➡️ Next image"):
        st.session_state["idx"] += 1
        if st.session_state["idx"] >= len(X_test):
            st.session_state["idx"] = 0

else:
    st.info("Please train the identity model first.")

