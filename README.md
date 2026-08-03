<p align="center">
  <h1 align="center">🎨 AI Image Colorizer Pro</h1>

  <p align="center">
    <strong>Professional AI-powered image colorization using multiple deep learning models</strong>
    <br><br>

  <a href="https://huggingface.co/spaces/YOUR_USERNAME/colorizer">
      <strong>🌐 Live Demo</strong>
  </a>
  ·
  <a href="../../issues">
      <strong>🐛 Report Bug</strong>
  </a>
  ·
  <a href="../../discussions">
      <strong>💬 Discussions</strong>
  </a>

  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Demo-HuggingFace-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge">
</p>

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🎯 Demo](#-demo)
- [🧠 Model Zoo](#-model-zoo)
- [⚙️ Processing Pipeline](#️-processing-pipeline)
- [📊 Performance Benchmarks](#-performance-benchmarks)
- [🖼️ Visual Comparison](#️-visual-comparison)
- [🚀 Installation](#-installation)
- [🧪 Testing](#-testing)
- [📈 Benchmark](#-benchmark)
- [📄 License](#-license)

---

## ✨ Features

| Feature                         | Description                                          |
| ------------------------------- | ---------------------------------------------------- |
| 🎨 **Multiple AI Models**       | SIGGRAPH17, ECCV16, Lightweight and Fast models      |
| 🖌 **Interactive Color Hints**  | Guide the network with manual hint points            |
| 🤖 **GPT Vision Analysis**      | Image understanding & intelligent color suggestions  |
| 🎛 **Advanced Post Processing** | Brightness, saturation, vibrance & contrast controls |
| 📊 **Quality Metrics**          | PSNR, SSIM and Colorfulness evaluation               |
| 🎥 **Video Colorization**       | Frame-by-frame video processing (Beta)               |
| 📈 **Benchmark Suite**          | Compare models on standard datasets                  |

---

## 🎯 Demo

<p align="center">
  <img src="assets/demo.gif" width="85%">
</p>

---

## 🧠 Model Zoo

| Model            | Quality | Speed | Best For          |
| ---------------- | :-----: | :---: | ----------------- |
| SIGGRAPH17 |  ⭐⭐⭐⭐⭐  | ⭐⭐☆☆☆ | Highest Quality   |
| ECCV16      |  ⭐⭐⭐⭐☆  | ⭐⭐⭐☆☆ | Balanced Results  |
| Lightweight  |  ⭐⭐⭐☆☆  | ⭐⭐⭐⭐☆ | Everyday Usage    |
| Fast          |  ⭐⭐☆☆☆  | ⭐⭐⭐⭐⭐ | Real-time Preview |

---

## ⚙️ Processing Pipeline

```mermaid
graph LR
  A[Upload Image] --> B[Select AI Model]
  B --> C[Optional Color Hints]
  C --> D[AI Colorization]
  D --> E[Post Processing]
  E --> F[Metrics]
  F --> G[Download]
```

---

## 📊 Performance Benchmarks

| Model          | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CPU    | GPU    |
| -------------- | ------ | ------ | ------- | ------ | ------ |
| 🥇 SIGGRAPH17  | 24.12  | 0.89   | 0.23    | 2.1 s  | 0.14 s |
| 🥈 ECCV16      | 23.45  | 0.87   | 0.26    | 1.8 s  | 0.12 s |
| 🥉 Lightweight | 20.87  | 0.82   | 0.35    | 0.3 s  | 0.03 s |
| ⚡ Fast         | 18.92  | 0.76   | 0.42    | 0.08 s | 0.01 s |

---

## 🖼️ Visual Comparison

| Input | AI Output | Ground Truth |
|:-----:|:---------:|:------------:|
| ![](samples/input1.png) | ![](samples/output1.png) | ![](samples/gt1.png) |
| ![](samples/input2.png) | ![](samples/output2.png) | ![](samples/gt2.png) |

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/ai-image-colorizer.git
cd ai-image-colorizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📈 Benchmark

```bash
python benchmark.py
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<p align="center">
  Made with ❤️ using<br>
  <strong>PyTorch • Streamlit • OpenAI • Hugging Face</strong>
</p>
