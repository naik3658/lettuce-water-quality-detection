import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
# Load your transfer learning model
model = tf.keras.models.load_model("../models/lettuce_transfer_classifier.h5")
class_names = ["Fresh Water", "Contaminated"]

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    1. Upload a clear image of a lettuce leaf (JPG/PNG).
    2. The model will predict if it was grown with fresh or contaminated water.
    3. You will see the prediction, confidence, and the uploaded image.
    """
)

st.title("🥬 Lettuce Water Quality Classifier")
st.write("Upload a lettuce image to predict if it was grown with **fresh** or **contaminated** water.")

uploaded_file = st.file_uploader("Choose a lettuce image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Uploaded Image', use_column_width=True)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    st.markdown(f"### Prediction: **{class_names[class_idx]}**")
    st.progress(confidence)
    if class_idx == 0:
        st.success(f"This lettuce is likely grown with **Fresh Water**. (Confidence: {confidence:.2f})")
    else:
        st.warning(f"This lettuce may be grown with **Contaminated Water**. (Confidence: {confidence:.2f})")

# Footer
st.markdown("---")
st.markdown("<center>Made by Karthik Naik for Lettuce Water Quality Detection | Powered by Streamlit & TensorFlow</center>", unsafe_allow_html=True) 