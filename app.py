import streamlit as st
import numpy as np
import pickle
from PIL import Image
import io

# Load the trained model
MODEL_PATH = "Skin_disease.pkl"

# Function to load the model
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# Function to preprocess the image
def preprocess_image(image):
    # Resize image (example size: 224x224 for many neural networks)
    image = image.resize((224, 224))
    # Convert image to numpy array and scale to [0, 1]
    image = np.array(image) / 255.0
    # Add a batch dimension if needed (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    return image

# Main Streamlit app
def main():
    st.title("Skin disease Classification App")
    st.write("Upload an image and the model will classify it.")
    st.write("By Teerapat Sittichottithikun")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load the model
        model = load_model()

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Display the result
        st.write(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
