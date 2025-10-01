# NoiseLab.py

# install dependencies
# pip install streamlit opencv-python-headless numpy Pillow scikit-image matplotlib reportlab scipy


import streamlit as st
from noise_descriptions import run_noise_descriptions
from noise_Identification import run_noise_identification_removal
from noise_addition_filtering import run_noise_addition_filtering

st.set_page_config(page_title="NoiseLab", layout="wide")

st.title("🔬 NoiseLab: Interactive Image Noise Playground")

tab1, tab2, tab3 = st.tabs([
    "🔎 Noise Identification & Removal",
    "🧪 Noise Addition & Filtering",
    "📖 Noise Descriptions & Education"
])

with tab1:
    run_noise_identification_removal()
with tab2:
    run_noise_addition_filtering()
with tab3:
    run_noise_descriptions()
