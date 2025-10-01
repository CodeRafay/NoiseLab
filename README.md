# NoiseLab 🎛️

An interactive **Streamlit application** for exploring, adding, and removing image noise. This project bridges **image processing** with **education**, enabling real-time experimentation with noise models, filters, and theoretical concepts.

---

## 🚀 Features

### 1. Noise Identification & Removal  
- Upload noisy images.  
- Choose from 13+ noise types (Gaussian, Poisson, Salt-and-Pepper, Gamma, Rayleigh, Uniform, White, Colored, Exponential, Photon, Quantization, Additive, Multiplicative).  
- Apply specialized denoising filters:  
  - Median & adaptive median for impulse noise  
  - Gaussian blur & Wiener filter for Gaussian/uniform noise  
  - Non-local means, Anscombe transforms for Poisson & photon noise  
  - Lee, Frost, Kuan filters for speckle (Gamma, Rayleigh)  
- Adjustable parameters with sliders.  
- Side-by-side comparison of noisy vs denoised images.

### 2. Noise Addition & Filtering  
- Upload clean images.  
- Add selected noise type with adjustable intensity.  
- Immediately apply matching denoising filters.  
- View clean, noisy, and filtered images side-by-side.

### 3. Noise Description & Education  
- Detailed descriptions, models, and sources for each noise type.  
- Best filtering strategies included.  
- Organized in expandable sections for easy navigation.  
- Downloadable **PDF summaries** with text and example images.

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **Streamlit** for the UI  
- **OpenCV** & **scikit-image** for noise modeling & filtering  
- **NumPy** for numerical operations  
- **Matplotlib** for visualizations  
- **ReportLab** for PDF generation  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/CodeRafay/NoiseLab.git
cd NoiseLab
````

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📖 Educational Value

NoiseLab is ideal for:

* **Students** learning image processing & computer vision
* **Researchers** experimenting with denoising techniques
* **Educators** needing interactive demos for teaching

---

## 🔑 Keywords

`image-processing`, `denoising`, `noise-removal`, `streamlit`, `computer-vision`, `image-noise`, `machine-learning`, `image-filters`, `scikit-image`, `opencv`, `educational-tool`, `data-visualization`

---

## 🤝 Contributing

Contributions are welcome! Open issues or submit pull requests with enhancements, bug fixes, or additional noise models.  
---

## Developed by RAFAY ADEEL

---

## 📜 License

Apache 2.0 License © 2025
