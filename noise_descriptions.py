# noise_descriptions.py
import streamlit as st


def run_noise_descriptions():
    st.header("📖 Noise Descriptions & Education")

    st.markdown("""
    Explore different types of **image noise**, their **probability density functions (PDFs)**,
    sources, and **best filtering techniques**.
    """)

    # -------------------------------
    # 1. Distribution-Based Noises
    # -------------------------------
    st.subheader("1. Distribution-Based Noises")

    with st.expander("Gaussian Noise"):
        st.write(
            "Random variations following a normal distribution, often modeled as additive noise.")
        st.latex(
            r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
        st.markdown("""
        **Causes:** Sensor thermal noise, electronic circuits, camera sensors  
        **Best Filters:** Gaussian smoothing, Wiener filter, Non-Local Means, BM3D  
        **Remarks:** Most widely used noise model in imaging.
        """)

    with st.expander("Poisson Noise (Shot Noise)"):
        st.write(
            "Signal-dependent noise caused by random arrival of photons/electrons. Variance = mean.")
        st.latex(r"P(k;\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}")
        st.markdown("""
        **Causes:** Low-light imaging, astronomy, medical imaging  
        **Best Filters:** Variance-stabilizing transform (Anscombe), Wavelet denoising, NLM  
        **Remarks:** Cannot be reduced by longer exposure if photons are very few.
        """)

    with st.expander("Gamma Distribution Noise (Speckle Noise)"):
        st.write(
            "Multiplicative noise following a gamma distribution, appears grainy (common in coherent imaging).")
        st.latex(r"f(x) = \frac{x^{k-1} e^{-x/\theta}}{\Gamma(k)\theta^k}")
        st.markdown("""
        **Causes:** Ultrasound, radar, SAR (Synthetic Aperture Radar)  
        **Best Filters:** Lee, Kuan, Frost filters, Wavelet denoising  
        **Remarks:** Strongly affects coherent imaging systems.
        """)

    with st.expander("Impulse Noise (Salt-and-Pepper)"):
        st.write(
            "Randomly replaces pixels with extreme values (0 or 255). Appears as black and white dots.")
        st.latex(
            r"P(x) = \begin{cases} \frac{p}{2}, & x=0 \\ \frac{p}{2}, & x=1 \\ 1-p, & \text{otherwise} \end{cases}")
        st.markdown("""
        **Causes:** Transmission errors, faulty sensors, memory corruption  
        **Best Filters:** Median filter, Adaptive median filter  
        **Remarks:** Easy to detect and remove compared to Gaussian.
        """)

    with st.expander("Exponential Noise"):
        st.write(
            "Noise following exponential distribution, models sudden bursts or delays.")
        st.latex(r"f(x;\lambda) = \lambda e^{-\lambda x}, \quad x \geq 0")
        st.markdown("""
        **Causes:** Random delays, impulse-like channel errors  
        **Best Filters:** Nonlinear filters, Wavelet denoising  
        **Remarks:** Less common in natural imaging, but important in communication.
        """)

    with st.expander("Rayleigh Noise"):
        st.write(
            "Arises when noise magnitude follows Rayleigh distribution, often due to scattered signals.")
        st.latex(
            r"f(x;\sigma) = \frac{x}{\sigma^2} e^{-x^2 / (2\sigma^2)}, \quad x \geq 0")
        st.markdown("""
        **Causes:** Radar systems, wireless fading channels, multipath scattering  
        **Best Filters:** Frost, Lee, Adaptive filters  
        **Remarks:** Common in radar/sonar imaging.
        """)

    with st.expander("Uniform Noise"):
        st.write("Noise values equally likely within a range [a, b].")
        st.latex(r"f(x) = \frac{1}{b-a}, \quad a \leq x \leq b")
        st.markdown("""
        **Causes:** Quantization error approximation, synthetic noise for testing  
        **Best Filters:** Gaussian smoothing, Averaging, Wiener filter  
        **Remarks:** Artificially introduced in simulations.
        """)

    # -------------------------------
    # 2. Correlation-Based Noises
    # -------------------------------
    st.subheader("2. Correlation-Based Noises")

    with st.expander("White Noise"):
        st.write(
            "Uncorrelated noise with flat power spectral density (all frequencies equally represented).")
        st.markdown("""
        **PDF:** Typically Gaussian with mean 0 and constant variance  
        **Causes:** Thermal noise, background electronic noise  
        **Best Filters:** Low-pass filters, Wiener filter  
        **Remarks:** Called 'white' because it spreads energy across all frequencies.
        """)

    with st.expander("Colored Noise"):
        st.write(
            "Correlated noise with non-flat spectral density (e.g., pink noise, brown noise).")
        st.markdown("""
        **PDF:** Varies depending on type (not a single closed form)  
        **Causes:** Natural phenomena, electrical interference, correlated processes  
        **Best Filters:** Adaptive filters, Band-stop filters, Wavelet filtering  
        **Remarks:** Unlike white noise, has frequency-dependent characteristics.
        """)

    # -------------------------------
    # 3. Nature-Based Noises
    # -------------------------------
    st.subheader("3. Nature-Based Noises")

    with st.expander("Additive Noise"):
        st.write("Noise that adds directly to the original signal.")
        st.latex(r"g(x) = f(x) + n(x)")
        st.markdown("""
        **Causes:** Electronic thermal noise, Gaussian noise  
        **Best Filters:** Wiener filter, Gaussian smoothing  
        **Remarks:** Simplest and most widely used model.
        """)

    with st.expander("Multiplicative Noise"):
        st.write("Noise that scales with the original signal (signal-dependent).")
        st.latex(r"g(x) = f(x) \cdot n(x)")
        st.markdown("""
        **Causes:** Speckle noise, Poisson noise  
        **Best Filters:** Log transforms + linear filtering, Lee/Kuan filters  
        **Remarks:** Harder to remove compared to additive noise.
        """)

    # -------------------------------
    # 4. Source-Based Noises
    # -------------------------------
    st.subheader("4. Source-Based Noises")

    with st.expander("Quantization Noise"):
        st.write(
            "Error introduced when continuous signals are quantized into discrete levels.")
        st.latex(r"\epsilon \sim U(-\Delta/2, \Delta/2)")
        st.markdown("""
        **Causes:** ADC conversion, image compression  
        **Best Filters:** Dithering, Smoothing filters, Oversampling  
        **Remarks:** Fundamental limitation of digital representation.
        """)

    with st.expander("Photon Noise"):
        st.write("Inherent randomness in photon arrival, modeled as Poisson process.")
        st.latex(r"P(k;\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}")
        st.markdown("""
        **Causes:** Low-light imaging, astronomy, microscopy  
        **Best Filters:** Anscombe transform + Wavelet denoising, Non-Local Means  
        **Remarks:** Cannot be fully eliminated due to quantum nature.
        """)
