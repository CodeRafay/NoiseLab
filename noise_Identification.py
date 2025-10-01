import streamlit as st
import cv2
import numpy as np
from skimage import restoration
from skimage.restoration import denoise_wavelet, denoise_nl_means
from scipy.signal import wiener
from skimage.filters import median
from skimage.morphology import disk

# --- Filtering Functions ---


def apply_median(img, size=3):
    return median(img, disk(size))


def apply_wiener(img, size=5):
    return wiener(img, (size, size))


def apply_gaussian_blur(img, ksize=5, sigma=1):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_nl_means(img, h=0.1):
    return denoise_nl_means(img, h=h, fast_mode=True)


def apply_wavelet(img):
    return denoise_wavelet(img, mode="soft", channel_axis=-1, rescale_sigma=True)

# --- Utility for display ---


def display_comparison(original, processed, noise_type):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Noisy Image", width=300)
    with col2:
        st.image(
            processed, caption=f"Denoised Image ({noise_type})", width=300)

# --- Noise Identification & Removal UI ---


def run_noise_identification_removal():
    st.header("🔎 Noise Identification & Removal")

    uploaded_file = st.file_uploader(
        "Upload a noisy image", type=["jpg", "png", "jpeg"])
    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Noise type selection
    noise_type = st.selectbox(
        "Select the type of noise to remove",
        [
            "Gaussian",
            "Poisson",
            "Gamma (Speckle)",
            "Salt-and-Pepper",
            "Exponential",
            "Rayleigh",
            "Uniform",
            "White",
            "Colored",
            "Additive",
            "Multiplicative",
            "Quantization",
            "Photon"
        ]
    )

    denoised = None

    # Parameters + Filtering
    if noise_type == "Salt-and-Pepper":
        size = st.slider("Median filter size", 1, 9, 3, step=2)
        denoised = apply_median(img_rgb, size)

    elif noise_type in ["Gaussian", "Uniform", "White"]:
        ksize = st.slider("Gaussian kernel size", 3, 15, 5, step=2)
        sigma = st.slider("Sigma", 0.1, 5.0, 1.0)
        denoised = apply_gaussian_blur(img_rgb, ksize, sigma)

    elif noise_type in ["Poisson", "Photon"]:
        h = st.slider("Non-local means h", 0.05, 0.5, 0.1, step=0.05)
        denoised = apply_nl_means(img_rgb, h)

    elif noise_type in ["Gamma (Speckle)", "Rayleigh"]:
        st.caption("Applying advanced denoising (Wavelet shrinkage)")
        denoised = apply_wavelet(img_rgb)

    elif noise_type in ["Exponential", "Colored"]:
        st.caption("Applying Non-local means for correlated noise")
        denoised = apply_nl_means(img_rgb, h=0.15)

    elif noise_type in ["Additive", "Multiplicative"]:
        st.caption("Applying Wiener filter for additive/multiplicative noise")
        denoised = apply_wiener(img_rgb)

    elif noise_type == "Quantization":
        st.caption("Applying smoothing for quantization noise")
        ksize = st.slider("Smoothing kernel size", 3, 15, 5, step=2)
        denoised = cv2.blur(img_rgb, (ksize, ksize))

    # Display results side by side
    if denoised is not None:
        st.subheader("Comparison")
        display_comparison(img_rgb, denoised, noise_type)
