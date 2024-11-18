import streamlit as st
from PIL import Image
from fastai.learner import load_learner
import torch
import torchvision.transforms as transforms

# Set the title of the app
st.title("FastAI Model Image Prediction App")

# Load the FastAI model
@st.cache_resource
def load_model():
    model = load_learner("Skin_disease.pkl")
    return model

model = load_model()

# Define image transformations (optional, depending on your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# File uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction using FastAI's model
    pred, pred_idx, probs = model.predict(image)

    # Display the prediction
    st.write(f"Prediction: **{pred}**")
    st.write(f"Probability: **{probs[pred_idx]:.4f}**")
else:
    st.write("Please upload an image.")
