import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------
# 1️⃣ Load your trained model
# -----------------------
model = load_model("mask_detector_model.h5")
IMG_SIZE = 224

# -----------------------
# 2️⃣ Streamlit App Layout
# -----------------------
st.title("Mask Detection App")
st.write("Upload an image or use your webcam to check if a person is wearing a mask.")

# Tabs for upload or webcam
tab1, tab2 = st.tabs(["Upload Image", "Webcam (Optional)"])

# -----------------------
# 3️⃣ Image Upload Prediction
# -----------------------
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image with PIL
        image = Image.open(uploaded_file).convert('RGB')
        frame = np.array(image)

        # Preprocess
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_normalized = frame_resized / 255.0
        frame_input = np.expand_dims(frame_normalized, axis=0)

        # Prediction
        prediction = model.predict(frame_input, verbose=0)
        label = "Without Mask" if prediction[0][0] > 0.5 else "Mask"

        # Display image and prediction
        st.image(frame, caption='Uploaded Image', use_column_width=True)
        st.write(f"Prediction: **{label}**")

# -----------------------
# 4️⃣ Optional: Webcam Prediction
# -----------------------
with tab2:
    st.write("Webcam support requires installing `streamlit-webrtc`.")
    st.write("Run `pip install streamlit-webrtc` to enable live webcam detection.")

    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

        class MaskDetector(VideoTransformerBase):
            def __init__(self):
                self.model = model
                self.img_size = IMG_SIZE

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                # Preprocess
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                img_normalized = img_resized / 255.0
                img_input = np.expand_dims(img_normalized, axis=0)
                # Predict
                pred = self.model.predict(img_input, verbose=0)
                label = "Mask" if pred[0][0] > 0.5 else "No Mask"
                # Put label on frame
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                return img

        webrtc_streamer(key="mask-detection", video_transformer_factory=MaskDetector)

    except ImportError:
        st.warning("streamlit-webrtc is not installed. Webcam feature is disabled.")
