import streamlit as st
import cv2
import numpy as np
from skimage.util import random_noise
from skimage.restoration import denoise_nl_means, denoise_wavelet, denoise_tv_chambolle
from scipy.signal import wiener

# --------------------
# Noise Generators
# --------------------


def add_gaussian_noise(image, sigma=25):
    var = (sigma / 255.0) ** 2
    out = random_noise(image, mode='gaussian', var=var)
    return (out * 255).astype(np.uint8)


def add_salt_pepper(image, amount=0.02):
    out = random_noise(image, mode='s&p', amount=amount)
    return (out * 255).astype(np.uint8)


def add_poisson_noise(image):
    out = random_noise(image, mode='poisson')
    return (out * 255).astype(np.uint8)


def add_speckle_noise(image, var=0.01):
    out = random_noise(image, mode='speckle', var=var)
    return (out * 255).astype(np.uint8)


def add_uniform_noise(image, low=-0.1, high=0.1):
    noise = np.random.uniform(low, high, image.shape)
    out = np.clip(image / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)


def add_exponential_noise(image, scale=0.05):
    noise = np.random.exponential(scale, image.shape)
    out = np.clip(image / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)


def add_rayleigh_noise(image, scale=0.05):
    noise = np.random.rayleigh(scale, image.shape)
    out = np.clip(image / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)


def add_white_noise(image, sigma=0.05):
    noise = np.random.normal(0, sigma, image.shape)
    out = np.clip(image / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)


def add_colored_noise(image, alpha=0.1):
    noise = np.random.randn(*image.shape)
    # Apply simple correlation by blurring noise
    noise = cv2.GaussianBlur(noise, (5, 5), 0) * alpha
    out = np.clip(image / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)


def add_additive_noise(image, sigma=0.05):
    noise = np.random.normal(0, sigma, image.shape)
    out = np.clip(image / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)


def add_multiplicative_noise(image, sigma=0.05):
    noise = 1 + np.random.normal(0, sigma, image.shape)
    out = np.clip((image / 255.0) * noise, 0, 1)
    return (out * 255).astype(np.uint8)


def add_quantization_noise(image, levels=16):
    img = image.astype(np.float32) / 255.0
    quantized = np.round(img * levels) / levels
    return (quantized * 255).astype(np.uint8)


def add_photon_noise(image, scale=0.01):
    img = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(img / scale) * scale
    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)


# --------------------
# Filters
# --------------------


def median_filter(img, k=3): return cv2.medianBlur(img, k)


def gaussian_filter(img, k=5, sigma=1): return cv2.GaussianBlur(
    img, (k, k), sigma)


def nl_means(img, h=0.1):
    arr = img.astype(np.float32) / 255.0
    den = denoise_nl_means(arr, h=h, fast_mode=True, channel_axis=-1)
    return (np.clip(den, 0, 1) * 255).astype(np.uint8)


def wavelet_filter(img):
    arr = img.astype(np.float32) / 255.0
    den = denoise_wavelet(
        arr, mode="soft", channel_axis=-1, rescale_sigma=True)
    return (np.clip(den, 0, 1) * 255).astype(np.uint8)


def wiener_filter(img):
    arr = img.astype(np.float32)
    out = wiener(arr)
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_filter(img, k=5): return cv2.blur(img, (k, k))

# --------------------
# Display Utility
# --------------------


def display_triplet(original, noisy, denoised, t1="Original", t2="Noisy", t3="Denoised"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original, caption=t1, width=250)
    with col2:
        st.image(noisy, caption=t2, width=250)
    with col3:
        st.image(denoised, caption=t3, width=250)

# --------------------
# Main UI
# --------------------


def run_noise_addition_filtering():
    st.header("🧪 Noise Addition & Filtering")

    uploaded_file = st.file_uploader("Upload a clean image", type=[
                                     "jpg", "png", "jpeg"], key="noise_add")
    if uploaded_file:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        noise_choice = st.selectbox("Select Noise to Add", [
            "Gaussian", "Salt-and-Pepper", "Poisson", "Gamma (Speckle)",
            "Exponential", "Rayleigh", "Uniform",
            "White", "Colored",
            "Additive", "Multiplicative",
            "Quantization", "Photon"
        ], key="noise_type")

        noisy = None
        if noise_choice == "Gaussian":
            sigma = st.slider("Sigma", 1, 100, 25)
            noisy = add_gaussian_noise(img, sigma)
            denoised = gaussian_filter(noisy)
        elif noise_choice == "Salt-and-Pepper":
            amt = st.slider("Amount", 0.0, 0.1, 0.02, step=0.01)
            noisy = add_salt_pepper(img, amt)
            denoised = median_filter(noisy)
        elif noise_choice == "Poisson":
            noisy = add_poisson_noise(img)
            denoised = nl_means(noisy)
        elif noise_choice == "Gamma (Speckle)":
            var = st.slider("Variance", 0.0, 0.1, 0.01)
            noisy = add_speckle_noise(img, var)
            denoised = wavelet_filter(noisy)
        elif noise_choice == "Exponential":
            scale = st.slider("Scale", 0.01, 0.5, 0.05)
            noisy = add_exponential_noise(img, scale)
            denoised = nl_means(noisy)
        elif noise_choice == "Rayleigh":
            scale = st.slider("Scale", 0.01, 0.5, 0.05)
            noisy = add_rayleigh_noise(img, scale)
            denoised = wavelet_filter(noisy)
        elif noise_choice == "Uniform":
            noisy = add_uniform_noise(img)
            denoised = gaussian_filter(noisy)
        elif noise_choice == "White":
            sigma = st.slider("Sigma", 0.0, 1.0, 0.05)
            noisy = add_white_noise(img, sigma)
            denoised = gaussian_filter(noisy)
        elif noise_choice == "Colored":
            alpha = st.slider("Alpha", 0.0, 1.0, 0.1)
            noisy = add_colored_noise(img, alpha)
            denoised = nl_means(noisy)
        elif noise_choice == "Additive":
            noisy = add_additive_noise(img)
            denoised = wiener_filter(noisy)
        elif noise_choice == "Multiplicative":
            noisy = add_multiplicative_noise(img)
            denoised = wiener_filter(noisy)
        elif noise_choice == "Quantization":
            levels = st.slider("Levels", 2, 64, 16)
            noisy = add_quantization_noise(img, levels)
            denoised = smooth_filter(noisy)
        elif noise_choice == "Photon":
            scale = st.slider("Scale (photon level)", 0.001, 0.1, 0.01)
            noisy = add_photon_noise(img, scale)
            denoised = nl_means(noisy)

        if noisy is not None:
            st.subheader("Comparison")
            display_triplet(img, noisy, denoised, "Original",
                            f"Noisy ({noise_choice})", "Denoised")
