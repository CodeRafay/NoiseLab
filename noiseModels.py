# streamlit_image_noise_app_fixed.py
# Noise models in image processing (fixed)
# Comprehensive Streamlit app for image noise processing
# Features: Noise Identification & Removal, Noise Addition & Filtering,
# Noise Description & Education with dynamic PDF export

import io
import math
from typing import Tuple, Dict

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from skimage import util, restoration, exposure
from skimage.restoration import denoise_wavelet, denoise_nl_means, estimate_sigma
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime

# -------------------------------
# Utility and caching
# -------------------------------

st.set_page_config(page_title="Image Noise Lab", layout="wide")


@st.cache_data(show_spinner=False)
def load_image_to_array(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert('RGB')
    arr = np.array(img)
    return arr


@st.cache_data
def pil_from_array(arr: np.ndarray) -> Image.Image:
    arr_clamped = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_clamped)


@st.cache_data
def to_8bit_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def ensure_odd(k: int) -> int:
    if k % 2 == 0:
        return k + 1
    return k


# -------------------------------
# Noise generation functions
# -------------------------------


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    # sigma provided in [0..255] units
    var = (sigma / 255.0) ** 2
    out = random_noise(image, mode='gaussian', var=var, seed=None)
    return (out * 255).astype(np.uint8)


def add_salt_pepper(image: np.ndarray, amount: float) -> np.ndarray:
    out = random_noise(image, mode='s&p', amount=amount, seed=None)
    return (out * 255).astype(np.uint8)


def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    out = random_noise(image, mode='poisson', seed=None)
    return (out * 255).astype(np.uint8)


def add_speckle_gamma(image: np.ndarray, var: float) -> np.ndarray:
    # approximate multiplicative gamma speckle noise: image + image*noise
    noise = np.random.gamma(shape=1.0, scale=var, size=image.shape)
    out = image.astype(float) + image.astype(float) * noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_exponential_noise(image: np.ndarray, scale: float) -> np.ndarray:
    noise = np.random.exponential(scale=scale, size=image.shape)
    out = image.astype(float) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_rayleigh_noise(image: np.ndarray, scale: float) -> np.ndarray:
    noise = np.random.rayleigh(scale=scale, size=image.shape)
    out = image.astype(float) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_uniform_noise(image: np.ndarray, low: float, high: float) -> np.ndarray:
    noise = np.random.uniform(low, high, size=image.shape)
    out = image.astype(float) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_quantization_noise(image: np.ndarray, levels: int) -> np.ndarray:
    # simple uniform quantization to 'levels' levels
    arr = image.astype(np.float32) / 255.0
    q = np.floor(arr * levels) / levels
    return (q * 255).astype(np.uint8)


def add_colored_noise(image: np.ndarray, intensity: float, freqs: Tuple[float, float] = (5, 5)) -> np.ndarray:
    # generate low-frequency colored noise using gaussian blur on white noise
    h, w = image.shape[:2]
    noise = np.random.randn(h, w, 1).astype(np.float32)
    # sigmaX/Y should be positive and not zero
    sigma_x = max(1.0, float(freqs[0]))
    sigma_y = max(1.0, float(freqs[1]))
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    noise = noise / (np.std(noise) + 1e-9) * intensity
    if image.ndim == 3:
        noise = np.repeat(noise, 3, axis=2)
    out = image.astype(float) + noise * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def add_photon_noise(image: np.ndarray, scale: float) -> np.ndarray:
    # model photon as Poisson with scaling; scale should be small for realistic behaviour
    img_float = image.astype(np.float32) / 255.0
    # avoid division by zero; scale controls intensity of poisson sampling
    lam = np.clip(img_float * (1.0 / max(scale, 1e-6)), 0.0, 1e6)
    vals = np.random.poisson(lam) * max(scale, 1e-6)
    return np.clip(vals * 255.0, 0, 255).astype(np.uint8)


# Wrapper mapping
NOISE_ADD_FUNCTIONS = {
    'Gaussian': add_gaussian_noise,
    'Poisson': lambda img, param: add_poisson_noise(img),
    'Gamma (Speckle)': lambda img, param: add_speckle_gamma(img, param),
    'Salt-and-Pepper': lambda img, param: add_salt_pepper(img, param),
    'Exponential': lambda img, param: add_exponential_noise(img, param),
    'Rayleigh': lambda img, param: add_rayleigh_noise(img, param),
    'Uniform': lambda img, param: add_uniform_noise(img, -param, param),
    'White noise': lambda img, param: add_uniform_noise(img, -param, param),
    'Colored noise': lambda img, param: add_colored_noise(img, param),
    'Additive noise': lambda img, param: add_gaussian_noise(img, param),
    'Multiplicative noise': lambda img, param: add_speckle_gamma(img, param),
    'Quantization noise': lambda img, param: add_quantization_noise(img, max(2, int(param))),
    'Photon noise': lambda img, param: add_photon_noise(img, param),
}

# -------------------------------
# Denoising filter functions
# -------------------------------


def apply_median(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ensure_odd(max(1, ksize))
    # cv2.medianBlur accepts single-channel or 3-channel 8-bit images
    return cv2.medianBlur(img, k)


def apply_adaptive_median(img: np.ndarray, max_ksize: int) -> np.ndarray:
    # naive adaptive median: try increasing kernel sizes until median reduces impulses
    gray = to_8bit_gray(img)
    out = gray.copy()
    for k in range(3, max_ksize + 1, 2):
        med = cv2.medianBlur(out, k)
        diff = np.abs(out - med)
        threshold = 0
        mask = diff > threshold
        out[mask] = med[mask]
    if img.ndim == 3:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out


def apply_gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    k = ensure_odd(max(1, ksize))
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_wiener(img: np.ndarray, mysize: Tuple[int, int] = (5, 5)) -> np.ndarray:
    # Use Wiener as a per-channel operation; fallback to Gaussian if Wiener fails
    try:
        arr = img.astype(np.float32) / 255.0
        if arr.ndim == 3:
            channels = []
            for c in range(3):
                ch = restoration.wiener(
                    arr[..., c], np.ones(mysize), balance=0.1)
                channels.append(ch)
            out = np.stack(channels, axis=-1)
        else:
            out = restoration.wiener(arr, np.ones(mysize), balance=0.1)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        # safe fallback: small gaussian blur
        return apply_gaussian_blur(img, ksize=max(mysize), sigma=1.0)


def apply_nlm(img: np.ndarray, h: float = 0.8, patch_size: int = 7, patch_distance: int = 11) -> np.ndarray:
    arr = img.astype(np.float32) / 255.0
    # estimate sigma with channel_axis
    try:
        sigma_est = np.mean(estimate_sigma(arr, channel_axis=-1))
    except TypeError:
        # older API fallback (if installed)
        sigma_est = np.mean(estimate_sigma(arr, multichannel=True))
    patch_kw = dict(patch_size=patch_size,
                    patch_distance=patch_distance, channel_axis=-1)
    den = denoise_nl_means(arr, h=h * sigma_est, fast_mode=True, **patch_kw)
    return (np.clip(den, 0, 1) * 255).astype(np.uint8)


def apply_wavelet_denoise(img: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    arr = img.astype(np.float32) / 255.0
    try:
        den = denoise_wavelet(
            arr, sigma=sigma, channel_axis=-1, convert2ycbcr=True)
    except TypeError:
        den = denoise_wavelet(
            arr, sigma=sigma, multichannel=True, convert2ycbcr=True)
    return (np.clip(den, 0, 1) * 255).astype(np.uint8)


# Lee filter implementation for multiplicative noise (speckle)
def lee_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    img = img.astype(np.float32)
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(3):
            out[..., c] = lee_filter(img[..., c], size)
        return np.clip(out, 0, 255).astype(np.uint8)
    mean = cv2.blur(img, (size, size))
    mean_sq = cv2.blur(img * img, (size, size))
    var = mean_sq - mean * mean
    overall_var = np.mean(var)
    k = var / (var + overall_var + 1e-9)
    result = mean + k * (img - mean)
    return np.clip(result, 0, 255).astype(np.uint8)


# Frost filter approximation using exponential weighting
def frost_filter(img: np.ndarray, size: int = 7, damping: int = 2) -> np.ndarray:
    img = img.astype(np.float32)
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(3):
            out[..., c] = frost_filter(img[..., c], size, damping)
        return np.clip(out, 0, 255).astype(np.uint8)
    h, w = img.shape
    pad = size // 2
    padded = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            local = padded[i:i + size, j:j + size]
            local_mean = np.mean(local)
            local_var = np.var(local)
            center = padded[i + pad, j + pad]
            if local_var < 1e-9:
                out[i, j] = center
            else:
                wts = np.exp(-damping * np.abs(local - center) /
                             (local_var + 1e-9))
                wts = wts / (np.sum(wts) + 1e-9)
                out[i, j] = np.sum(wts * local)
    return np.clip(out, 0, 255).astype(np.uint8)


# Mapping of recommended filters per noise type
FILTER_MAP = {
    'Salt-and-Pepper': ('Median / Adaptive median', apply_median),
    'Gaussian': ("Gaussian / Wiener / NLM", apply_wiener),
    'Uniform': ("Gaussian / Wiener", apply_wiener),
    'Poisson': ("Anscombe + Wavelet / NLM", apply_wavelet_denoise),
    'Photon noise': ("Variance-stabilizing + NLM", apply_nlm),
    'Gamma (Speckle)': ("Lee / Kuan", lee_filter),
    'Rayleigh': ("Frost / Lee", frost_filter),
    'Colored noise': ("Band-stop or adaptive filters", apply_nlm),
    'Exponential': ("Median / Wavelet", apply_median),
    'White noise': ("Low-pass / Wiener", apply_wiener),
    'Additive noise': ("Wiener / Gaussian", apply_wiener),
    'Multiplicative noise': ("Lee / Log-transform + NLM", lee_filter),
    'Quantization noise': ("Dithering + Smoothing", apply_gaussian_blur),
}

# -------------------------------
# PDF generation
# -------------------------------


def generate_pdf_report(images: Dict[str, np.ndarray], summary_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Image Noise Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 12))
    for title, img_arr in images.items():
        story.append(Paragraph(title, styles['Heading2']))
        img_pil = pil_from_array(img_arr)
        bio = io.BytesIO()
        img_pil.save(bio, format='PNG')
        bio.seek(0)
        rlimg = RLImage(bio, width=400, height=int(
            400 * img_arr.shape[0] / img_arr.shape[1]))
        story.append(rlimg)
        story.append(Spacer(1, 12))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# -------------------------------
# Educational content
# -------------------------------


NOISE_DESCRIPTIONS = {
    'Gaussian': {
        'desc': 'Additive zero-mean Gaussian noise with standard deviation sigma, common in electronic noise.',
        'model': 'I(x) + N(0, sigma^2)',
        'sources': 'Sensor read noise, thermal noise',
        'filters': 'Gaussian smoothing, Wiener filter, non-local means',
    },
    'Poisson': {
        'desc': 'Noise following Poisson distribution, variance equals mean, significant for low-intensity signals.',
        'model': 'Poisson(I(x))',
        'sources': 'Photon counting processes in low-light imaging',
        'filters': 'Anscombe transform + wavelet denoising, variance-stabilizing transforms',
    },
    'Gamma': {
        'desc': 'Multiplicative noise modeled using Gamma distribution, often appears as speckle noise.',
        'model': 'I(x) * Gamma(k, θ)',
        'sources': 'Radar imaging, SAR, ultrasound',
        'filters': 'Lee filter, Frost filter, adaptive speckle reduction',
    },
    'Salt-and-Pepper': {
        'desc': 'Impulse noise where random pixels are set to min or max values.',
        'model': 'I(x) with probability p replaced by 0 or 255',
        'sources': 'Transmission errors, faulty memory',
        'filters': 'Median, adaptive median',
    },
    'Exponential': {
        'desc': 'Noise following exponential distribution, typically positive with decaying probability.',
        'model': 'Exp(λ)',
        'sources': 'Photon arrival times, queuing systems, random event arrivals',
        'filters': 'Median, wavelet denoising, variance stabilizing transforms',
    },
    'Rayleigh': {
        'desc': 'Noise following Rayleigh distribution, often arises in scattered signals.',
        'model': 'Rayleigh(σ)',
        'sources': 'Radar backscatter, wireless communication channels',
        'filters': 'Adaptive thresholding, Frost/Lee, wavelet shrinkage',
    },
    'Uniform': {
        'desc': 'Noise with equal probability across a given range.',
        'model': 'U(a, b)',
        'sources': 'Quantization error, rounding, analog-to-digital conversion',
        'filters': 'Averaging filters, low-pass filters',
    },
    'White': {
        'desc': 'Noise with flat power spectrum, equal intensity across all frequencies.',
        'model': 'Uncorrelated, uniform across spectrum',
        'sources': 'Thermal noise, quantization error',
        'filters': 'Low-pass filtering, Wiener filter',
    },
    'Colored': {
        'desc': 'Noise with non-flat power spectrum, e.g. pink noise, brown noise.',
        'model': 'Correlated spectrum-shaped noise',
        'sources': 'Biological signals, natural systems, communication channels',
        'filters': 'Spectral subtraction, adaptive filtering',
    },
    'Additive': {
        'desc': 'Noise that simply adds to the original image.',
        'model': 'I(x) + N(x)',
        'sources': 'Sensor electronics, thermal fluctuations',
        'filters': 'Averaging, Wiener, non-local means',
    },
    'Multiplicative': {
        'desc': 'Noise that multiplies the original image signal.',
        'model': 'I(x) * N(x)',
        'sources': 'Speckle in radar, ultrasound imaging',
        'filters': 'Lee filter, Frost filter, homomorphic filtering',
    },
    'Quantization': {
        'desc': 'Noise from rounding during digitization, limited precision.',
        'model': 'Q(I(x)) - I(x)',
        'sources': 'Analog-to-digital conversion',
        'filters': 'Dithering, oversampling, error diffusion',
    },
    'Photon': {
        'desc': 'Noise due to statistical variation in photon arrival (shot noise).',
        'model': 'Poisson-distributed photon counts',
        'sources': 'Low-light photography, astronomy, microscopy',
        'filters': 'Variance-stabilizing transforms, denoising autoencoders',
    },
}

# -------------------------------
# UI Components
# -------------------------------


def show_image_columns(images: list, captions: list):
    n = len(images)
    cols = st.columns(n)
    for i, col in enumerate(cols):
        # use new width parameter API: 'stretch' mimics previous use_container_width=True
        col.image(images[i], caption=captions[i], width='stretch')


# Sidebar navigation
st.sidebar.title("Image Noise Lab")
mode = st.sidebar.radio("Choose mode", [
    "Noise Identification & Removal", "Noise Addition & Filtering", "Noise Descriptions & Education"
])

# -------------------------------
# Mode: Noise Identification & Removal
# -------------------------------
if mode == "Noise Identification & Removal":
    st.header("Noise Identification & Removal")
    uploaded = st.file_uploader("Upload a noisy image", type=[
                                'png', 'jpg', 'jpeg', 'bmp'])
    if uploaded is not None:
        img = load_image_to_array(uploaded)
        st.sidebar.markdown("---")
        noise_type = st.sidebar.selectbox(
            "Select suspected noise type", list(FILTER_MAP.keys()))
        st.sidebar.markdown("Adjust filter parameters")
        ksize = st.sidebar.slider("Filter size (odd)", 1, 31, 5, step=2)
        sigma = st.sidebar.slider(
            "Gaussian sigma / intensity", 0.0, 100.0, 10.0)
        nlm_h = st.sidebar.slider("NLM strength", 0.1, 2.0, 0.8)
        preview = st.checkbox("Realtime preview", value=True)

        st.write("Original noisy image")
        col1, col2 = st.columns(2)
        col1.image(img, caption='Noisy input', width='stretch')

        filter_name, filter_func = FILTER_MAP.get(
            noise_type, ("Wiener", apply_wiener))
        st.write(f"Recommended filters: {filter_name}")

        apply_now = st.button("Apply Filter")
        if apply_now or preview:
            with st.spinner("Applying filter..."):
                # choose filter based on mapping heuristics
                if filter_func == apply_median:
                    den = apply_median(img, ksize if ksize %
                                       2 == 1 else ksize + 1)
                elif filter_func == apply_wavelet_denoise:
                    den = apply_wavelet_denoise(img, sigma / 255.0)
                elif filter_func == apply_nlm:
                    den = apply_nlm(img, h=nlm_h, patch_size=7,
                                    patch_distance=11)
                elif filter_func == lee_filter:
                    den = lee_filter(img, size=ksize)
                elif filter_func == frost_filter:
                    den = frost_filter(img, size=ksize)
                elif filter_func == apply_gaussian_blur:
                    den = apply_gaussian_blur(img, ksize, sigma)
                else:
                    den = apply_wiener(img, mysize=(ksize, ksize))

            col2.image(
                den, caption=f'Denoised ({filter_name})', width='stretch')

            # metrics
            try:
                orig = img.astype(np.float32)
                score = psnr(orig, den)
                st.write(f"PSNR between noisy and denoised: {score:.2f} dB")
            except Exception:
                pass

            if st.button("Download PDF report"):
                summary = f"Noise type: {noise_type}. Filter used: {filter_name}."
                pdf_bytes = generate_pdf_report(
                    {'Noisy': img, 'Denoised': den}, summary)
                st.download_button("Download report PDF", data=pdf_bytes,
                                   file_name="noise_report.pdf", mime='application/pdf')

# -------------------------------
# Mode: Noise Addition & Filtering
# -------------------------------
elif mode == "Noise Addition & Filtering":
    st.header("Noise Addition & Filtering")
    uploaded = st.file_uploader("Upload a clean image", type=[
                                'png', 'jpg', 'jpeg', 'bmp'])
    if uploaded is not None:
        img = load_image_to_array(uploaded)
        st.sidebar.markdown("---")
        noise_choice = st.sidebar.selectbox(
            "Noise to add", list(NOISE_ADD_FUNCTIONS.keys()))
        st.sidebar.markdown("Noise parameters")

        # Show only relevant sliders based on selection
        # Default parameter placeholders
        gaussian_sigma = None
        sp_amount = None
        quant_levels = None
        uniform_param = None
        colored_intensity = None
        photon_scale = None
        gamma_var = None
        exponential_scale = None
        rayleigh_scale = None

        if noise_choice == 'Salt-and-Pepper':
            sp_amount = st.sidebar.slider(
                "Salt & pepper amount", 0.0, 0.5, 0.05)
        elif noise_choice == 'Gaussian' or noise_choice == 'Additive noise':
            gaussian_sigma = st.sidebar.slider(
                "Gaussian sigma (0-100)", 0.0, 100.0, 10.0)
        elif noise_choice == 'Poisson':
            st.sidebar.write(
                "Poisson noise has no continuous intensity slider (statistical).")
        elif noise_choice == 'Gamma (Speckle)' or noise_choice == 'Multiplicative noise':
            gamma_var = st.sidebar.slider(
                "Gamma speckle variance (scale)", 0.0, 5.0, 0.5)
        elif noise_choice == 'Exponential':
            exponential_scale = st.sidebar.slider(
                "Exponential scale", 0.0, 100.0, 10.0)
        elif noise_choice == 'Rayleigh':
            rayleigh_scale = st.sidebar.slider(
                "Rayleigh scale", 0.0, 100.0, 10.0)
        elif noise_choice == 'Uniform' or noise_choice == 'White noise':
            uniform_param = st.sidebar.slider(
                "Uniform amplitude (+/-)", 0.0, 100.0, 10.0)
        elif noise_choice == 'Quantization noise':
            quant_levels = st.sidebar.slider("Quantization levels", 2, 256, 32)
        elif noise_choice == 'Colored noise':
            colored_intensity = st.sidebar.slider(
                "Colored noise intensity", 0.0, 100.0, 10.0)
        elif noise_choice == 'Photon noise':
            photon_scale = st.sidebar.slider(
                "Photon noise scale (smaller = stronger effect)", 1e-4, 1.0, 0.01, format="%.4f")

        # Add noise button
        add_btn = st.button("Add noise")
        if add_btn:
            with st.spinner("Adding noise..."):
                if noise_choice == 'Salt-and-Pepper':
                    noisy = add_salt_pepper(img, float(sp_amount))
                elif noise_choice == 'Gaussian' or noise_choice == 'Additive noise':
                    noisy = add_gaussian_noise(img, float(gaussian_sigma))
                elif noise_choice == 'Poisson':
                    noisy = add_poisson_noise(img)
                elif noise_choice == 'Gamma (Speckle)' or noise_choice == 'Multiplicative noise':
                    noisy = add_speckle_gamma(img, float(gamma_var))
                elif noise_choice == 'Exponential':
                    noisy = add_exponential_noise(
                        img, float(exponential_scale))
                elif noise_choice == 'Rayleigh':
                    noisy = add_rayleigh_noise(img, float(rayleigh_scale))
                elif noise_choice == 'Uniform' or noise_choice == 'White noise':
                    noisy = add_uniform_noise(
                        img, -float(uniform_param), float(uniform_param))
                elif noise_choice == 'Quantization noise':
                    noisy = add_quantization_noise(img, int(quant_levels))
                elif noise_choice == 'Colored noise':
                    noisy = add_colored_noise(img, float(colored_intensity))
                elif noise_choice == 'Photon noise':
                    noisy = add_photon_noise(img, float(photon_scale))
                else:
                    noisy = add_gaussian_noise(img, 10.0)

            # show images (use width='stretch' for responsive columns)
            st.write("Preview")
            show_image_columns([img, noisy], ['Original clean', 'Noisy'])

            # apply default denoiser for that noise
            filter_name, filter_func = FILTER_MAP.get(
                noise_choice, ("Wiener", apply_wiener))
            st.write(f"Applying recommended filter: {filter_name}")
            with st.spinner("Denoising..."):
                if filter_func == apply_median:
                    den = apply_median(noisy, 5)
                elif filter_func == apply_wavelet_denoise:
                    den = apply_wavelet_denoise(noisy, 0.1)
                elif filter_func == apply_nlm:
                    den = apply_nlm(noisy, h=0.8)
                elif filter_func == lee_filter:
                    den = lee_filter(noisy, size=7)
                elif filter_func == frost_filter:
                    den = frost_filter(noisy, size=7)
                elif filter_func == apply_gaussian_blur:
                    # Gaussian blur is a simple smoothing fallback
                    den = apply_gaussian_blur(noisy, 5, 1.0)
                elif filter_func == apply_wiener:
                    # For Gaussian noise, use a stronger gaussian blur fallback if Wiener misbehaves
                    if noise_choice == 'Gaussian':
                        # try Wiener first, fallback to gaussian blur
                        try:
                            den = apply_wiener(noisy, mysize=(5, 5))
                        except Exception:
                            den = apply_gaussian_blur(noisy, 5, 1.5)
                    else:
                        den = apply_wiener(noisy, mysize=(5, 5))
                else:
                    den = apply_wiener(noisy, mysize=(5, 5))

            show_image_columns([img, noisy, den], [
                               'Original', 'Noisy', f'Denoised ({filter_name})'])

            # allow downloading PDF report
            if st.button("Download PDF with examples"):
                summary = f"Noise added: {noise_choice}. Filter: {filter_name}. Generated: {datetime.utcnow().isoformat()}Z"
                pdf_bytes = generate_pdf_report(
                    {'Original': img, 'Noisy': noisy, 'Denoised': den}, summary)
                st.download_button("Download generated PDF", data=pdf_bytes,
                                   file_name="noise_examples.pdf", mime='application/pdf')

# -------------------------------
# Mode: Noise Descriptions & Education
# -------------------------------
else:
    st.header("Noise Descriptions & Education")
    st.write(
        "Learn about common image noise models and recommended filtering strategies")
    for key in FILTER_MAP.keys():
        with st.expander(key):
            info = NOISE_DESCRIPTIONS.get(key, None)
            if info is None:
                st.write(
                    "Description: A common noise type encountered in imaging.")
                st.write("Mathematical model: See references")
                st.write(
                    "Sources: sensors, transmission, atmosphere, compression, etc.")
                st.write("Recommended filters: " + FILTER_MAP[key][0])
            else:
                st.write("Description: ", info.get('desc', ''))
                st.write("Model: ", info.get('model', ''))
                st.write("Sources: ", info.get('sources', ''))
                st.write("Best filters: ", info.get('filters', ''))

    if st.button("Download full PDF summary"):
        st.info("Generating PDF, this may take a few seconds")
        images_for_pdf = {}
        summary = "Summary of noise types and filters generated by Image Noise Lab"
        pdf_bytes = generate_pdf_report(images_for_pdf, summary)
        st.download_button("Download PDF summary", data=pdf_bytes,
                           file_name='noise_summary.pdf', mime='application/pdf')

# Footer notes
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with OpenCV, NumPy, scikit-image and ReportLab for education and prototyping.")
st.sidebar.caption("© 2024 Image Noise Lab")
st.sidebar.caption("Developed by Rafay Adeel")
