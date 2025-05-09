import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import cv2

# Load the trained model
model = tf.keras.models.load_model('parkinsons_model.h5')  # Adjust this path to your model

# Define the image size expected by the model (224x224 for MobileNet)
image_size = (224, 224)

# Define class names
class_names = ['normal', 'parkinson']

# Function to preprocess the uploaded image
def preprocess_image(img):
    # Check if the image is in grayscale or color
    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB image
        img_resized = cv2.resize(img, image_size)  # Resize image to match model input
    else:  # Grayscale image
        img_resized = cv2.resize(img, image_size)  # Resize image to match model input
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    img_normalized = img_resized / 255.0  # Normalize image
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Expand dimensions to match model input shape
    return img_expanded

# Streamlit interface
st.title("Parkinson's Disease Classification")
st.markdown("Upload an image to check if it's a normal person or has Parkinson's disease.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image as a PIL image
    img = Image.open(uploaded_file)

    # Convert the PIL image to an OpenCV image (numpy array)
    img_array = np.array(img)

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img_processed = preprocess_image(img_array)

    # Predict the class of the image
    predictions = model.predict(img_processed)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]
    confidence = predictions[0][predicted_class[0]]

    # Display prediction result
    st.write(f"Prediction: **{predicted_label}**")
    st.write(f"Confidence: {confidence * 100:.2f}%")
