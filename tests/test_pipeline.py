# tests/test_pipeline.py
import pytest
import torch
import numpy as np
from PIL import Image
import io

from utils import ImageProcessor
from models import load_fast, load_lightweight

def test_image_loading():
    ip = ImageProcessor()
    dummy = (np.random.rand(100,100,3)*255).astype(np.uint8)
    img_bytes = io.BytesIO()
    Image.fromarray(dummy).save(img_bytes, format='PNG')
    img = ip.load_image(img_bytes.getvalue())
    assert img.shape == (100,100,3)
    assert img.dtype == np.float32

def test_preprocess_postprocess():
    ip = ImageProcessor()
    dummy = np.random.rand(100,100,3).astype(np.float32)
    resized, orig = ip.preprocess_for_model(dummy, target_size=256)
    assert resized.shape == (1,1,256,256)
    out_ab = torch.randn(1,2,256,256)
    result = ip.postprocess_output(orig, out_ab)
    assert result.shape == dummy.shape

def test_model_forward():
    model = load_fast()
    dummy = torch.randn(1,1,256,256)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1,2,256,256)

def test_lightweight_forward():
    model = load_lightweight()
    dummy = torch.randn(1,1,256,256)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1,2,256,256)
