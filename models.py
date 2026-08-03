# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------- Base classes (unchanged) ----------
class BaseColor(nn.Module):
    def __init__(self):
        super(BaseColor, self).__init__()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.
    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm
    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent
    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm
    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm

# ---------- ECCVGenerator, LightweightColorizer, FastColorizer, SIGGRAPHGenerator (unchanged) ----------
# ... (keep your existing class definitions exactly as they were) ...

# ---------- Pretrained weights support ----------
PRETRAINED_DIR = Path("pretrained_weights")
PRETRAINED_DIR.mkdir(exist_ok=True)

PRETRAINED_URLS = {
    'eccv16': 'https://colorization.ecci.urv.es/models/eccv16.pth',   # example URL (adjust)
    'siggraph17': 'https://colorization.ecci.urv.es/models/siggraph17.pth'
}

def download_weights(model_name):
    """Download pretrained weights if not already present."""
    url = PRETRAINED_URLS.get(model_name)
    if not url:
        return None
    local_path = PRETRAINED_DIR / f"{model_name}.pth"
    if local_path.exists():
        return str(local_path)
    print(f"Downloading pretrained weights for {model_name} from {url} ...")
    try:
        urllib.request.urlretrieve(url, local_path)
        return str(local_path)
    except Exception as e:
        print(f"Download failed: {e}. Using random weights.")
        return None

def load_eccv16():
    """Load ECCV16 model with pretrained weights if available."""
    model = ECCVGenerator()
    weight_path = download_weights('eccv16')
    if weight_path:
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print("Loaded pretrained ECCV16 weights.")
    else:
        print("ECCV16 using random initialization.")
    return model.eval()

def load_siggraph17():
    """Load SIGGRAPH17 model with pretrained weights if available."""
    model = SIGGRAPHGenerator()
    weight_path = download_weights('siggraph17')
    if weight_path:
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print("Loaded pretrained SIGGRAPH17 weights.")
    else:
        print("SIGGRAPH17 using random initialization.")
    return model.eval()

# Lightweight and Fast remain as before (they are custom, no pretrained weights)
def load_lightweight():
    model = LightweightColorizer()
    return model.eval()

def load_fast():
    model = FastColorizer()
    return model.eval()

# Model registry
MODEL_REGISTRY = {
    'eccv16': load_eccv16,
    'lightweight': load_lightweight,
    'fast': load_fast,
    'siggraph17': load_siggraph17
}
