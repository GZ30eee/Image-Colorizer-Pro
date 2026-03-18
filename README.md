# AI Image Colorizer Pro v2.0

Professional AI-powered image colorization with SIGGRAPH17 model, OpenAI analysis, and advanced features.

## Features
- **Advanced Colorization Models**: ECCV16, Lightweight, Fast, and SIGGRAPH17.
- **Interactive Color Hints**: Manually guide the AI to use specific colors at specific points.
- **AI Analysis**: Vision-powered color suggestion and era estimation.
- **Video Support (Beta)**: Prepare first frame and ensure temporal consistency.
- **Model Comparison**: Side-by-side comparison of different models.
- **Performance Analytics**: Track PSNR, SSIM, and colorfulness metrics.
- **Export Options**: Download result as single image or multiple results in a ZIP file.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`.
3. Activate the virtual environment.
4. Install dependencies: `pip install -r requirements.txt`.
5. Create a `.env` file with your `GEMINI_API_KEY`.
6. Run the application: `streamlit run app.py`.

## Technology Stack
- Streamlit
- PyTorch
- OpenCV
- Scikit-Image
- Plotly
- Pandas
- Google Generative AI (Gemini)

## Credits
Based on research papers like ECCV 2016 and SIGGRAPH 2017.
