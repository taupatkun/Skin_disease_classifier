import streamlit as st
from fastai.learner import load_learner
from fastai.vision.core import PILImage
import pathlib
import sys
import torch
import fastai
import torchvision
import PIL
import os

#*st.write("Python:", sys.version)
#st.write("PyTorch:", torch.__version__)
#st.write("Torchvision:", torchvision.__version__)
#st.write("fastai:", fastai.__version__)
#st.write("Pillow:", PIL.__version__)
# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Derma - AI Skin Disease Screening",
    page_icon="🩺",
    layout="centered"
)
# =========================================================
# TITLE
# =========================================================
st.title("🩺 Derma")
st.subheader(
    "AI-Based Preliminary Skin Disease Screening"
)
st.write(
    "By Ruengsit Matachaiyasit & Teerapat Sittichottithikun"
)
st.write(
    "Upload an image of a skin condition to receive "
    "an AI-based preliminary screening result."
)
st.warning(
    "⚠️ This result is an AI-based preliminary "
    "screening and is NOT a medical diagnosis. "
    "The model may produce incorrect results. "
    "Please consult a qualified healthcare "
    "professional for proper evaluation."
)
# =========================================================
# DISEASE INFORMATION
# =========================================================
disease_info = {
    "BA- cellulitis": {
        "name": "Cellulitis",
        "description":
            "A bacterial infection affecting the deeper layers of the skin.",
        "causes": [
            "Usually caused by bacteria entering through a break in the skin.",
            "Cuts, wounds, insect bites, or other skin injuries can provide an entry point for bacteria."
        ],
        "what_to_do": [
            "Keep the affected area clean.",
            "Avoid touching or scratching the affected area.",
            "Seek medical evaluation, especially if redness or swelling is spreading.",
            "Follow treatment recommended by a healthcare professional."
        ]
    },
    "BA-impetigo": {
        "name": "Impetigo",
        "description":
            "A contagious bacterial skin infection that commonly affects the surface of the skin.",
        "causes": [
            "Usually caused by Staphylococcus or Streptococcus bacteria.",
            "Can spread through direct contact with an infected person.",
            "Can also spread through contaminated personal items."
        ],
        "what_to_do": [
            "Keep the affected area clean.",
            "Avoid scratching or touching the affected area.",
            "Avoid sharing towels, clothing, or other personal items.",
            "Seek medical advice for appropriate treatment."
        ]
    },
    "FU-nail-fungus": {
        "name": "Nail Fungus",
        "description":
            "A fungal infection affecting the fingernails or toenails.",
        "causes": [
            "Caused by a fungal infection.",
            "Can spread through contact with contaminated surfaces or objects.",
            "Warm and moist environments can promote fungal growth."
        ],
        "what_to_do": [
            "Keep nails clean and dry.",
            "Avoid sharing nail clippers or other personal items.",
            "Keep feet dry if toenails are affected.",
            "Seek medical advice for appropriate treatment."
        ]
    },
    "FU-ringworm": {
        "name": "Ringworm",
        "description":
            "A fungal infection that can cause circular or ring-shaped skin lesions.",
        "causes": [
            "Caused by a fungal infection.",
            "Can spread through contact with infected people or animals.",
            "Can also spread through contaminated clothing, towels, or other objects."
        ],
        "what_to_do": [
            "Keep the affected area clean and dry.",
            "Avoid scratching the affected area.",
            "Avoid sharing towels, clothing, or personal items.",
            "Seek medical advice if the condition does not improve."
        ]
    },
    "FU-athlete-foot": {
        "name": "Athlete's Foot",
        "description":
            "A fungal infection that commonly affects the skin of the feet.",
        "causes": [
            "Caused by a fungal infection.",
            "Warm and moist environments can promote fungal growth.",
            "Can spread through contaminated floors, shoes, socks, or towels."
        ],
        "what_to_do": [
            "Keep the feet clean and dry.",
            "Change socks regularly.",
            "Avoid sharing towels, socks, or footwear.",
            "Seek medical advice if symptoms persist or worsen."
        ]
    },
    "PA-cutaneous-larva-migrans": {
        "name": "Cutaneous Larva Migrans",
        "description":
            "A skin condition caused by larvae that migrate through the skin.",
        "causes": [
            "Usually associated with contact with soil or sand contaminated with animal hookworm larvae.",
            "The larvae can enter the skin through direct contact with contaminated ground."
        ],
        "what_to_do": [
            "Avoid further contact with potentially contaminated soil or sand.",
            "Keep the affected skin clean.",
            "Seek medical evaluation for appropriate treatment."
        ]
    },
    "VI-shingles": {
        "name": "Shingles",
        "description":
            "A viral infection that can cause a painful skin rash.",
        "causes": [
            "Caused by reactivation of the varicella-zoster virus.",
            "The virus can remain inactive in the body after a previous chickenpox infection."
        ],
        "what_to_do": [
            "Seek medical evaluation promptly.",
            "Avoid scratching or touching the rash.",
            "Avoid close contact with people who may be vulnerable to varicella infection.",
            "Follow medical advice and prescribed treatment."
        ]
    },
    "VI-chickenpox": {
        "name": "Chickenpox",
        "description":
            "A contagious viral infection that commonly causes an itchy rash and fluid-filled spots.",
        "causes": [
            "Caused by the varicella-zoster virus.",
            "Can spread through respiratory droplets and close contact."
        ],
        "what_to_do": [
            "Avoid close contact with others while contagious.",
            "Avoid scratching the rash.",
            "Keep the affected skin clean.",
            "Seek medical advice when necessary."
        ]
    }
}
# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = load_learner(
        "Skin_disease.pkl",
        cpu=True
    )
    return model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
# =========================================================
# IMAGE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)
# =========================================================
# IMAGE PROCESSING
# =========================================================
if uploaded_file is not None:
    # Convert uploaded image into FastAI-compatible image
    image = PILImage.create(uploaded_file)
    # =====================================================
    # DISPLAY IMAGE
    # =====================================================
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )
    # =====================================================
    # ANALYZE BUTTON
    # =====================================================
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image..."):
            try:
                # =================================================
                # FASTAI PREDICTION
                # =================================================
                pred, pred_idx, probs = model.predict(image)
                # Convert tensor index to Python integer
                pred_idx = pred_idx.item()
                # Convert model confidence to percentage
                confidence = float(
                    probs[pred_idx]
                ) * 100
                # Convert prediction to string
                pred_key = str(pred).strip()
                # Find disease information
                info = disease_info.get(
                    pred_key
                )
                # =================================================
                # RESULT
                # =================================================
                st.divider()
                st.subheader(
                    "🔎 Screening Result"
                )
                # -------------------------------------------------
                # MAIN PREDICTION
                # -------------------------------------------------
                if info:
                    display_name = info["name"]
                else:
                    display_name = pred_key
                st.write(
                    f"### Prediction: **{display_name}**"
                )
                st.write(
                    f"**Model Confidence: {confidence:.2f}%**"
                )
                # =================================================
                # OTHER MODEL PREDICTIONS
                # =================================================
                st.subheader(
                    "🔄 Other Model Predictions"
                )
                # Get class names from FastAI
                class_names = model.dls.vocab
                # Sort probabilities from highest to lowest
                sorted_indices = probs.argsort(
                    descending=True
                )
                # Number of alternative predictions
                num_alternatives = 3
                shown = 0
                for idx in sorted_indices:
                    idx = int(idx)
                    # Skip the main prediction
                    if idx == pred_idx:
                        continue
                    # Get class name
                    alternative_key = str(
                        class_names[idx]
                    ).strip()
                    # Get confidence
                    alternative_confidence = float(
                        probs[idx]
                    ) * 100
                    # Find disease information
                    alternative_info = disease_info.get(
                        alternative_key
                    )
                    # Get readable disease name
                    if alternative_info:
                        alternative_name = (
                            alternative_info["name"]
                        )
                    else:
                        alternative_name = alternative_key
                    # Display alternative prediction
                    st.write(
                        f"**{shown + 1}. "
                        f"{alternative_name} — "
                        f"{alternative_confidence:.2f}%**"
                    )
                    # Progress bar
                    st.progress(
                        min(
                            alternative_confidence / 100,
                            1.0
                        )
                    )
                    shown += 1
                    # Only show top 3 alternatives
                    if shown >= num_alternatives:
                        break
                # =================================================
                # DISEASE INFORMATION
                # =================================================
                if info:
                    st.divider()
                    # -------------------------------------------------
                    # DESCRIPTION
                    # -------------------------------------------------
                    st.subheader(
                        "📖 About this condition"
                    )
                    st.write(
                        info["description"]
                    )
                    # -------------------------------------------------
                    # POSSIBLE CAUSES
                    # -------------------------------------------------
                    st.subheader(
                        "🧬 Possible causes"
                    )
                    for cause in info["causes"]:
                        st.write(
                            f"- {cause}"
                        )
                    # -------------------------------------------------
                    # WHAT TO DO
                    # -------------------------------------------------
                    st.subheader(
                        "💡 What you can do"
                    )
                    for action in info["what_to_do"]:
                        st.write(
                            f"- {action}"
                        )
                else:
                    st.info(
                        "Additional information for this "
                        "prediction is not currently available."
                    )
                # =================================================
                # DISCLAIMER
                # =================================================
                st.divider()
                st.warning(
                    "⚠️ This result is an AI-based preliminary "
                    "screening and is NOT a medical diagnosis. "
                    "The model may produce incorrect results. "
                    "Please consult a qualified healthcare "
                    "professional for proper evaluation."
                )
            except Exception as e:
                st.error(
                    "An error occurred while analyzing the image."
                )
                st.exception(e)
