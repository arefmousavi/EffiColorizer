# EffiColorizer: An Efficient Image Colorization Framework for Low-Power Devices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Paper Coming Soon](https://img.shields.io/badge/Paper-coming%20soon-lightgray)

<p align="center">
  <img src="images/teaser.png" alt="colorization_preview" style="max-width: 100%; height: auto;" />
  <em>EffiColorizer in Action — Input grayscale images (top) and their colorized outputs (bottom)</em>
  <br><br>
</p>

> **📌 Note:** This repository provides a **demo implementation** of our upcoming paper: _**"Efficient Image Colorization for Low-Power Devices via Spectrally Normalized GAN with scSE Attention and EfficientNet Encoder"**_. It omits some components (e.g., scSE attention) and uses a simplified loss function compared to the final version.


## 🔍 Overview

**EffiColorizer** is a lightweight, real-time image colorization framework built with **PyTorch** and optimized for **mobile and embedded devices**.

It supports applications such as:
- Colorizing outputs from **night-vision and industrial monochrome cameras** for better visualization or use in downstream vision tasks
- **On-device** colorization of historical images, avoiding cloud processing to improve latency, cost-efficiency, and privacy

While state-of-the-art models are often too large for low-power systems, EffiColorizer offers a **highly compact yet accurate** alternative for vivid, semantically correct colorizations in real time.


## 🚀 Key Contributions (Demo Version)

- **Spectrally Normalized GAN** for stable adversarial training
- **EfficientNet-B3 encoder integrated into U-Net generator** for high semantic understanding with minimal computational cost
- **Novel hybrid training strategy** that alternates between joint and decoupled generator/discriminator updates


## 🧠 Architecture Summary

- **Generator:** U-Net with EfficientNet-B3 encoder (predicts a/b in CIELAB space)  
- **Discriminator:** PatchGAN  
- **Input:** 320×320 grayscale  
- **Data:** 16k train / 4k val image pairs from COCO
- **Training:** ~100 epochs


## ⚙️ Getting Started

### Clone and install dependencies

### Setup

```bash
git clone https://github.com/aref-mousavi-eng/EffiColorizer.git
cd EffiColorizer
conda env create -f environment.yml
conda env list  # Verify 'EffiColorizer-pytorch' exists
```

### Training

1. Download a dataset (e.g., [COCO](https://cocodataset.org)).
2. Open `training.ipynb` and set the kernel to `EffiColorizer-pytorch`.
3. Update the dataset path and run the cells.

### Evaluation

1. Download pretrained weights from [this link](https://drive.google.com/drive/folders/1gCsAj0PQFZwtKqX3hk4UOPakIZu9l4yL?usp=sharing) and place them in the root directory.
2. Launch `evaluation.ipynb` using the `EffiColorizer-pytorch` kernel.
3. Run the cells to visualize colorization results on sample images.


## 📊 Benchmark Results

We evaluate our model against the widely cited **Colorful Image Colorization (ECCV 2016)** baseline ([official repo](https://github.com/richzhang/colorization)).  
While not a state-of-the-art model, the baseline is lightweight and widely adopted, making it a practical benchmark for low-resource scenarios.

### Quality Evaluation

- Image quality is evaluated using:
  - **FID** (distributional realism, most perceptually aligned)
  - **SSIM** (structural fidelity)
  - **PSNR** (pixel-level accuracy)
  - **LPIPS** (deep perceptual similarity)
- FID is the most indicative of visual quality
- Our method achieves a **2.4× improvement** in FID over the baseline

| Model                        | FID ↓    | SSIM ↑    | PSNR ↑       | LPIPS ↓   |
|-----------------------------|----------|-----------|--------------|-----------|
| **EffiColorizer (Ours)**    | **9.29** | **0.922** | **24.93 dB** | **0.111** |
| Colorful Image Colorization | 22.50    | 0.911     | 21.21 dB     | 0.181     |

### Computational Efficiency

- Models compared based on FLOPs and parameter count.
- FLOPs normalized to 256×256 input resolution for fair comparison (EffiColorizer uses 320×320, baseline 256×256).
- FLOPs reported per image.
- EffiColorizer requires **~10.8× fewer FLOPs** and **2.4× fewer parameters** than baseline.

| Model                        | Params ↓ | FLOPs @ 256×256 ↓ |
|-----------------------------|----------|--------------------|
| **EffiColorizer (Ours)**    | **13.16 M** | **~3.87 GFLOPs**   |
| Colorful Image Colorization | 32.24 M  | 41.78 GFLOPs       |

## 📄 Citation

Citation details will be provided upon publication.  
Please check back later for the official reference.


## ✉️ Contact

Interested in collaborating, extending this framework, or using it in your own work?  
Feel free to reach out via [GitHub Issues](https://github.com/aref-mousavi-eng/EffiColorizer/issues).  
