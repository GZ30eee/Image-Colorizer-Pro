# 🎨 AI Image Colorizer Pro v2.0

AI Image Colorizer Pro is a powerful, professional-grade tool designed to breathe life into black-and-white photos and videos. Built on **PyTorch** and wrapped in a sleek **Streamlit** interface, the application offers a suite of state-of-the-art deep learning models and advanced manual controls for high-fidelity color reconstruction.

## ✨ Key Features

**Multi-Model Support**: Choose between four specialized AI architectures:
* **SIGGRAPH17**: Best for high-quality, vibrant, and artistically accurate results.
* **ECCV16**: High-quality classic colorization.
* **Lightweight & Fast**: Optimized for quick inference and CPU-based environments.

**🎯 Interactive Color Hints**: Manually guide the AI by placing color points on the image to ensure accurate tones for specific areas like clothing or architecture (required for SIGGRAPH17).

**🤖 AI Image Analysis**: Integrated **OpenAI GPT-4 Vision** to analyze image content, estimate the historical era, and suggest appropriate color palettes.
 
**🌈 Advanced Post-Processing**: Fine-tune results with saturation, brightness, contrast, and vibrancy sliders.
 
**📊 Performance Analytics**: Real-time metrics including **PSNR, SSIM, and Colorfulness** to evaluate the quality of the generated images.

**📹 Video Support (Beta)**: Ability to upload video files and process frames.



## 🛠️ Technical Stack

**Framework**: Streamlit 
 
**Deep Learning**: PyTorch (TorchVision) 
 
**Image Processing**: OpenCV, Scikit-Image, PIL 

**LLM Integration**: OpenAI API (GPT-4o) 
 
**Data Visualization**: Plotly 



## 🚀 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ai-image-colorizer.git
cd ai-image-colorizer

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Environment Variables**:
Create a `.env` file in the root directory and add your OpenAI API key for image analysis features:
```env
OPENAI_API_KEY=your_sk_key_here

```


4. **Run the App**:
```bash
streamlit run app.py

```



## 📂 Project Structure

* `app.py`: The main Streamlit application interface and logic.


* `models.py`: Definitions for ECCV16, SIGGRAPH17, and lightweight colorization models.


* `utils.py`: Core processing logic, including the `ImageProcessor`, `ColorHintManager`, and `PerformanceMonitor`.


* `requirements.txt`: Necessary Python libraries.
