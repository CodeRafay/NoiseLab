# noise_Identification.py for "Noise Identification and removal" tab on app
import streamlit as st
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, denoise_nl_means, estimate_sigma
from scipy.signal import wiener
from skimage.filters import median  # kept for reference if needed
from skimage.morphology import disk
import math

# --------------------
# Helpers
# --------------------


def ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def resize_for_preview(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img.copy()
    scale = max_side / side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# --------------------
# Filtering Functions (robust to color images)
# --------------------


def apply_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Use OpenCV medianBlur which supports 1- and 3-channel 8-bit images.
    size must be odd and >= 1.
    """
    k = ensure_odd(max(1, int(size)))
    # OpenCV expects uint8
    src = img.astype(np.uint8)
    return cv2.medianBlur(src, k)


def apply_wiener(img: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Apply Wiener filter per-channel. Accepts color images.
    Uses scipy.signal.wiener which operates on 2D arrays.
    """
    # convert to float
    img_f = img.astype(np.float32)
    if img_f.ndim == 2:
        out = wiener(img_f, (size, size))
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        channels = []
        for c in range(img_f.shape[2]):
            ch = wiener(img_f[..., c], (size, size))
            channels.append(np.clip(ch, 0, 255).astype(np.uint8))
        return np.stack(channels, axis=-1)


def apply_gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    k = ensure_odd(max(1, int(ksize)))
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_nl_means(img: np.ndarray, h: float = 0.8, patch_size: int = 5, patch_distance: int = 6, use_full_res: bool = False) -> np.ndarray:
    """
    Apply Non-local Means denoising using skimage.
    Accepts uint8 color image; converts to float [0,1] internally.
    For preview use scaled-down version unless use_full_res is True.
    """
    # Normalize to [0,1]
    arr = img.astype(np.float32) / 255.0

    # Choose preview/resolution
    preview = False
    if not use_full_res:
        # downscale for speed
        preview = True
        preview_img = resize_for_preview(
            (arr * 255.0).astype(np.uint8), max_side=512)
        arr_proc = preview_img.astype(np.float32) / 255.0
    else:
        arr_proc = arr

    # estimate sigma robustly
    try:
        sigma_est = np.mean(estimate_sigma(arr_proc, channel_axis=-1))
    except TypeError:
        sigma_est = np.mean(estimate_sigma(arr_proc, multichannel=True))

    # h parameter is scaled by sigma estimate (common practice)
    h_param = h * max(sigma_est, 1e-6)

    den = denoise_nl_means(arr_proc, h=h_param, fast_mode=True,
                           patch_size=patch_size, patch_distance=patch_distance, channel_axis=-1)
    den_clipped = np.clip(den, 0.0, 1.0)

    if preview and not use_full_res:
        # upsample back to original size
        # den is floats in [0,1], convert to uint8 then resize
        den_uint8 = (den_clipped * 255.0).astype(np.uint8)
        den_up = cv2.resize(
            den_uint8, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return den_up
    else:
        return (den_clipped * 255.0).astype(np.uint8)


def apply_wavelet(img: np.ndarray, use_full_res: bool = False) -> np.ndarray:
    """
    Wavelet denoising (skimage). Normalize to [0,1] before processing.
    Downscale for preview if requested.
    """
    arr = img.astype(np.float32) / 255.0
    if not use_full_res:
        preview_img = resize_for_preview(
            (arr * 255.0).astype(np.uint8), max_side=512)
        arr_proc = preview_img.astype(np.float32) / 255.0
    else:
        arr_proc = arr

    try:
        den = denoise_wavelet(arr_proc, mode="soft",
                              channel_axis=-1, rescale_sigma=True)
    except TypeError:
        den = denoise_wavelet(arr_proc, mode="soft",
                              multichannel=True, rescale_sigma=True)

    den = np.clip(den, 0.0, 1.0)
    if den.shape[:2] != img.shape[:2]:
        den_uint8 = (den * 255.0).astype(np.uint8)
        den_up = cv2.resize(
            den_uint8, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return den_up
    return (den * 255.0).astype(np.uint8)

# --------------------
# Display helper
# --------------------


def display_comparison(original, processed, noise_type):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Noisy Image", width=300)
    with col2:
        st.image(
            processed, caption=f"Denoised Image ({noise_type})", width=300)

# --------------------
# UI
# --------------------


def run_noise_identification_removal():
    st.header("🔎 Noise Identification & Removal")

    uploaded_file = st.file_uploader(
        "Upload a noisy image", type=["jpg", "png", "jpeg"])
    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    # Load image (uint8 BGR -> RGB)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Couldn't read the uploaded image. Try another file.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

    # Option to force full-resolution processing (slower)
    process_full = st.checkbox("Process full resolution (slower)", value=False)

    denoised = None

    # Parameters + Filtering
    if noise_type == "Salt-and-Pepper":
        size = st.slider("Median filter size (odd)", 1, 15, 3, step=2)
        # use cv2 median which supports color images
        denoised = apply_median(img_rgb, size)

    elif noise_type in ["Gaussian", "Uniform", "White"]:
        ksize = st.slider("Gaussian kernel size (odd)", 3, 31, 5, step=2)
        sigma = st.slider("Sigma", 0.1, 10.0, 1.0)
        denoised = apply_gaussian_blur(img_rgb, ksize, sigma)

    elif noise_type in ["Poisson", "Photon"]:
        st.caption("Non-local means (fast preview, upscale for full-res)")
        h = st.slider("NLM strength (h)", 0.1, 2.0, 0.8, step=0.1)
        with st.spinner("Applying NLM (may be slow on full resolution)..."):
            denoised = apply_nl_means(
                img_rgb, h=h, patch_size=5, patch_distance=6, use_full_res=process_full)

    elif noise_type in ["Gamma (Speckle)", "Rayleigh"]:
        st.caption("Wavelet shrinkage (fast preview)")
        with st.spinner("Applying wavelet denoising..."):
            denoised = apply_wavelet(img_rgb, use_full_res=process_full)

    elif noise_type in ["Exponential", "Colored"]:
        st.caption("Non-local means for correlated noise")
        h = st.slider("NLM strength (h)", 0.1, 2.0, 0.8, step=0.1)
        with st.spinner("Applying NLM (preview or full-res based on checkbox)..."):
            denoised = apply_nl_means(
                img_rgb, h=h, patch_size=5, patch_distance=6, use_full_res=process_full)

    elif noise_type in ["Additive", "Multiplicative"]:
        st.caption("Applying Wiener filter (per-channel)")
        size = st.slider("Wiener neighborhood size", 3, 15, 5, step=2)
        with st.spinner("Applying Wiener filter..."):
            denoised = apply_wiener(img_rgb, size=size)

    elif noise_type == "Quantization":
        st.caption("Smoothing for quantization noise")
        ksize = st.slider("Smoothing kernel size", 3, 31, 5, step=2)
        denoised = cv2.blur(img_rgb, (ksize, ksize))

    # Display results side by side
    if denoised is not None:
        st.subheader("Comparison")
        display_comparison(img_rgb, denoised, noise_type)

