# benchmark.py
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, metrics
import lpips
import time
from pathlib import Path

from utils import ImageProcessor
from models import load_siggraph17, load_eccv16, load_lightweight, load_fast

# Configuration
TEST_IMAGE_DIR = "test_images"   # folder with ground truth images
OUTPUT_CSV = "benchmark_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path):
    return np.array(Image.open(path).convert('RGB')) / 255.0

def main():
    ip = ImageProcessor()
    lpips_fn = lpips.LPIPS(net='alex').eval().to(DEVICE)

    models = {
        'siggraph17': load_siggraph17().to(DEVICE),
        'eccv16': load_eccv16().to(DEVICE),
        'lightweight': load_lightweight().to(DEVICE),
        'fast': load_fast().to(DEVICE),
        'baseline': None
    }

    results = []
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]

    for fname in image_files:
        gt = load_image(os.path.join(TEST_IMAGE_DIR, fname))
        gray = color.rgb2gray(gt)
        gray_3ch = np.stack([gray]*3, axis=-1)

        for model_name, model in models.items():
            print(f"Processing {fname} with {model_name} ...")
            if model_name == 'baseline':
                pred = gray_3ch
            else:
                # Preprocess
                img_tensor_resized, img_tensor_original = ip.preprocess_for_model(gray_3ch)
                img_tensor_resized = img_tensor_resized.to(DEVICE)
                with torch.no_grad():
                    out_ab = model(img_tensor_resized)
                pred = ip.postprocess_output(img_tensor_original.cpu(), out_ab.cpu())
                pred = ip.apply_color_vibrancy(pred, 1.2)   # consistent enhancement

            # Ensure same size
            pred = ip.resize_to_match(pred, gt.shape[:2])

            # Metrics
            psnr = metrics.peak_signal_noise_ratio(gt, pred)
            ssim = metrics.structural_similarity(gt, pred, multichannel=True, channel_axis=2)

            # LPIPS
            gt_t = torch.tensor(gt).permute(2,0,1).unsqueeze(0).float().to(DEVICE) * 2 - 1
            pred_t = torch.tensor(pred).permute(2,0,1).unsqueeze(0).float().to(DEVICE) * 2 - 1
            lpips_val = lpips_fn(gt_t, pred_t).item()

            # Inference time (average over 3 runs)
            if model_name != 'baseline':
                times = []
                for _ in range(3):
                    start = time.time()
                    with torch.no_grad():
                        _ = model(img_tensor_resized)
                    times.append(time.time() - start)
                avg_time = np.mean(times)
            else:
                avg_time = 0

            results.append({
                'image': fname,
                'model': model_name,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips_val,
                'time_sec': avg_time
            })

    # Save and summarise
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n=== Average Metrics by Model ===")
    print(df.groupby('model').mean().round(4))
    print(f"\nDetailed results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
