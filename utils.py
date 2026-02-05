# utils.py - COMPLETE VERSION
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from skimage import color, transform
import io
import zipfile
from datetime import datetime
import os
import tempfile
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import base64

class ImageProcessor:
    """Handles all image preprocessing and postprocessing"""
    
    @staticmethod
    def load_image(file_bytes, max_size=1024):
        """Load and resize image efficiently"""
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            # Resize if too large
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            img_np = np.array(img) / 255.0
            return img_np
        except Exception as e:
            raise ValueError(f"Image loading failed: {e}")
    
    @staticmethod
    def preprocess_for_model(img_rgb, target_size=256):
        """Prepare image for model input"""
        # Convert to LAB
        img_lab = color.rgb2lab(img_rgb)
        img_l_original = img_lab[:, :, 0]
        
        # Resize to target size for model
        img_rgb_resized = transform.resize(
            img_rgb, 
            (target_size, target_size), 
            mode='reflect',
            anti_aliasing=True
        )
        img_lab_resized = color.rgb2lab(img_rgb_resized)
        img_l_resized = img_lab_resized[:, :, 0]
        
        # Convert to tensors
        img_l_tensor_original = torch.FloatTensor(img_l_original)[None, None, :, :]
        img_l_tensor_resized = torch.FloatTensor(img_l_resized)[None, None, :, :]
        
        return img_l_tensor_resized, img_l_tensor_original
    
    @staticmethod
    def postprocess_output(img_l_tensor_orig, out_ab, original_size=None):
        """Convert model output back to RGB"""
        if original_size is None:
            original_size = img_l_tensor_orig.shape[2:]
        
        # Resize ab channels to match original L channel
        out_ab_resized = F.interpolate(
            out_ab, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Combine with original L channel
        out_lab = torch.cat((img_l_tensor_orig, out_ab_resized), dim=1)
        out_lab_np = out_lab.data.cpu().numpy()[0].transpose((1, 2, 0))
        
        # Convert to RGB
        img_rgb = color.lab2rgb(out_lab_np)
        return np.clip(img_rgb, 0, 1)
    
    @staticmethod
    def create_grayscale(img_rgb):
        """Create grayscale version"""
        img_lab = color.rgb2lab(img_rgb)
        img_l = img_lab[:, :, 0]
        img_lab_gray = np.stack([img_l, np.zeros_like(img_l), np.zeros_like(img_l)], axis=2)
        return np.clip(color.lab2rgb(img_lab_gray), 0, 1)
    
    @staticmethod
    def apply_color_enhancement(img_rgb, saturation=1.0, brightness=1.0, contrast=1.0):
        """Apply color enhancements to image"""
        enhanced = img_rgb.copy()
        
        # Adjust saturation in HSV space
        hsv = color.rgb2hsv(enhanced)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 1)
        enhanced = color.hsv2rgb(hsv)
        
        # Adjust brightness
        enhanced = np.clip(enhanced * brightness, 0, 1)
        
        # Adjust contrast
        mean = np.mean(enhanced)
        enhanced = np.clip((enhanced - mean) * contrast + mean, 0, 1)
        
        return enhanced
    
    @staticmethod
    def apply_color_vibrancy(img_rgb, vibrancy_factor=1.0):
        """Apply color vibrancy enhancement"""
        if vibrancy_factor == 1.0:
            return img_rgb
        
        # Convert to LAB color space
        lab = color.rgb2lab(img_rgb)
        
        # Enhance ab channels (color channels)
        lab[:, :, 1] = lab[:, :, 1] * vibrancy_factor
        lab[:, :, 2] = lab[:, :, 2] * vibrancy_factor
        
        # Clip to valid LAB range
        lab[:, :, 1] = np.clip(lab[:, :, 1], -128, 127)
        lab[:, :, 2] = np.clip(lab[:, :, 2], -128, 127)
        
        return color.lab2rgb(lab)

class PerformanceMonitor:
    """Tracks processing performance"""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        
    def start_timer(self):
        self.start_time = datetime.now()
        
    def stop_timer(self):
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.start_time = None
            return elapsed
        return 0
    
    def calculate_metrics(self, original, generated):
        """Calculate PSNR, SSIM, and colorfulness metrics"""
        try:
            # Ensure same size
            if original.shape != generated.shape:
                generated_resized = transform.resize(
                    generated, 
                    original.shape[:2], 
                    mode='reflect',
                    anti_aliasing=True
                )
            else:
                generated_resized = generated
            
            # PSNR calculation
            mse = np.mean((original - generated_resized) ** 2)
            if mse == 0:
                psnr = 100
            else:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            
            # SSIM calculation
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            mu_x = np.mean(original)
            mu_y = np.mean(generated_resized)
            sigma_x = np.std(original)
            sigma_y = np.std(generated_resized)
            sigma_xy = np.mean((original - mu_x) * (generated_resized - mu_y))
            
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
            
            # Colorfulness calculation
            colorfulness = self.calculate_colorfulness(generated_resized)
            
            return {
                'psnr': float(psnr),
                'ssim': float(ssim),
                'colorfulness': float(colorfulness),
                'mse': float(mse),
                'processing_time': self.stop_timer()
            }
        except:
            return {'psnr': 0, 'ssim': 0, 'colorfulness': 0, 'mse': 0, 'processing_time': 0}
    
    def calculate_colorfulness(self, image):
        """Calculate colorfulness metric"""
        # Split image into RGB components
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Calculate rg and yb
        rg = r - g
        yb = 0.5 * (r + g) - b
        
        # Calculate standard deviation and mean
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        
        # Calculate colorfulness
        std_root = np.sqrt(std_rg**2 + std_yb**2)
        mean_root = np.sqrt(mean_rg**2 + mean_yb**2)
        
        return std_root + 0.3 * mean_root
    
    def record_metrics(self, metrics, model_name):
        """Record metrics for dashboard"""
        self.metrics_history.append({
            'model': model_name,
            'timestamp': datetime.now(),
            **metrics
        })
        
        # Keep only last 50 entries
        if len(self.metrics_history) > 50:
            self.metrics_history.pop(0)

class ModelSelector:
    """Smart model selection based on image characteristics"""
    
    @staticmethod
    def analyze_image(img_rgb):
        """Analyze image to suggest best model"""
        analysis = {
            'type': 'general',
            'suggestion': 'siggraph17',
            'confidence': 0.8
        }
        
        # Simple analysis
        h, w = img_rgb.shape[:2]
        aspect_ratio = w / h
        
        # Determine image type
        if aspect_ratio > 1.5:
            analysis['type'] = 'landscape'
            analysis['suggestion'] = 'siggraph17'
        elif aspect_ratio < 0.7:
            analysis['type'] = 'portrait'
            analysis['suggestion'] = 'siggraph17'
        else:
            analysis['type'] = 'square'
            analysis['suggestion'] = 'siggraph17'
        
        return analysis
    
    @staticmethod
    def get_device_recommendation():
        """Recommend models based on available hardware"""
        if torch.cuda.is_available():
            return ['siggraph17', 'eccv16', 'lightweight', 'fast']
        else:
            return ['siggraph17', 'lightweight', 'fast']  # CPU-optimized models first

class ResultExporter:
    """Handles export of results"""
    
    @staticmethod
    def create_zip(results: Dict, filenames: List[str] = None):
        """Create ZIP file with all results"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for i, (name, img_array) in enumerate(results.items()):
                if name not in ['original', 'grayscale']:
                    # Convert to PIL Image
                    img_uint8 = (img_array * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8)
                    
                    # Save to bytes
                    img_buffer = io.BytesIO()
                    img_pil.save(img_buffer, format='PNG', optimize=True)
                    
                    # Create filename
                    if filenames and i < len(filenames):
                        filename = f"{filenames[i]}_{name}.png"
                    else:
                        filename = f"result_{name}_{datetime.now().strftime('%H%M%S')}.png"
                    
                    zip_file.writestr(filename, img_buffer.getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def save_single_image(img_array, filename="result.png"):
        """Save single image to bytes"""
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        
        return buffer

class ColorHintManager:
    """Manages interactive color hints"""
    
    def __init__(self):
        self.hints = []
    
    def add_hint(self, x, y, color_rgb, strength=1.0):
        """Add a color hint at position (x, y) with strength"""
        self.hints.append({
            'position': (int(x), int(y)),
            'color': color_rgb,
            'strength': float(strength)
        })
    
    def clear_hints(self):
        """Clear all hints"""
        self.hints = []
    
    def apply_hints_to_model(self, model_input_size=256, img_size=None):
        """Convert hints to model input format"""
        if not self.hints or img_size is None:
            return None, None
        
        img_h, img_w = img_size
        
        # Create hint tensors
        hints_ab = torch.zeros(1, 2, model_input_size, model_input_size)
        hints_mask = torch.zeros(1, 1, model_input_size, model_input_size)
        
        for hint in self.hints:
            x_orig, y_orig = hint['position']
            color_rgb = hint['color']
            strength = hint['strength']
            
            # Scale coordinates to model input size
            x_model = int(x_orig * model_input_size / img_w)
            y_model = int(y_orig * model_input_size / img_h)
            
            # Convert RGB to LAB
            rgb_norm = np.array([[color_rgb]]) / 255.0
            lab_color = color.rgb2lab(rgb_norm).squeeze()
            ab_norm = lab_color[1:] / 110.0  # Normalize ab channels
            
            # Apply hint
            if 0 <= x_model < model_input_size and 0 <= y_model < model_input_size:
                hints_ab[0, 0, y_model, x_model] = ab_norm[0]
                hints_ab[0, 1, y_model, x_model] = ab_norm[1]
                hints_mask[0, 0, y_model, x_model] = strength
        
        return hints_ab, hints_mask

class ColorPaletteManager:
    """Manages color palettes for different styles"""
    def __init__(self):
        self.palettes = {
            'natural': {
                'adjustments': {'saturation': 1.0, 'brightness': 1.0, 'contrast': 1.0}
            },
            'vintage': {
                'adjustments': {'saturation': 0.8, 'brightness': 0.9, 'contrast': 1.1},
                'color_shift': (10, -5, -10)  # Warm, slightly desaturated
            },
            'cinematic': {
                'adjustments': {'saturation': 0.7, 'brightness': 0.8, 'contrast': 1.2},
                'color_shift': (0, 5, -5)  # Cool, high contrast
            },
            'warm': {
                'adjustments': {'saturation': 1.1, 'brightness': 1.0, 'contrast': 1.0},
                'color_shift': (15, 0, -10)  # Warm orange tones
            },
            'cool': {
                'adjustments': {'saturation': 0.9, 'brightness': 1.0, 'contrast': 1.1},
                'color_shift': (-5, 5, 10)  # Cool blue tones
            },
            'vibrant': {
                'adjustments': {'saturation': 1.5, 'brightness': 1.1, 'contrast': 1.0},
                'color_shift': (10, 10, 0)  # Very vibrant
            }
        }
    
    def apply_palette(self, image, palette_name):
        """Apply color palette to image"""
        if palette_name not in self.palettes:
            return image
        
        palette = self.palettes[palette_name]
        result = image.copy()
        
        # Apply basic adjustments
        adjustments = palette.get('adjustments', {})
        if 'saturation' in adjustments:
            result = self.adjust_saturation(result, adjustments['saturation'])
        if 'brightness' in adjustments:
            result = self.adjust_brightness(result, adjustments['brightness'])
        if 'contrast' in adjustments:
            result = self.adjust_contrast(result, adjustments['contrast'])
        
        # Apply color shift if specified
        if 'color_shift' in palette:
            shift = palette['color_shift']
            result = self.apply_color_shift(result, shift)
        
        return np.clip(result, 0, 1)
    
    def adjust_saturation(self, image, factor):
        """Adjust image saturation"""
        hsv = color.rgb2hsv(image)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 1)
        return color.hsv2rgb(hsv)
    
    def adjust_brightness(self, image, factor):
        """Adjust image brightness"""
        return np.clip(image * factor, 0, 1)
    
    def adjust_contrast(self, image, factor):
        """Adjust image contrast"""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 1)
    
    def apply_color_shift(self, image, shift):
        """Apply RGB color shift"""
        r, g, b = shift
        result = image.copy()
        result[:, :, 0] = np.clip(result[:, :, 0] + r/255.0, 0, 1)
        result[:, :, 1] = np.clip(result[:, :, 1] + g/255.0, 0, 1)
        result[:, :, 2] = np.clip(result[:, :, 2] + b/255.0, 0, 1)
        return result

class TemporalConsistency:
    """Ensure temporal consistency in video colorization"""
    def __init__(self):
        self.prev_frame = None
        self.optical_flow = None

    def calculate_optical_flow(self, frame1, frame2):
        """Calculate dense optical flow between frames"""
        gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def warp_frame(self, frame, flow):
        """Warp frame using optical flow"""
        h, w = flow.shape[:2]
        flow_map = -flow.copy()
        flow_map[:,:,0] += np.arange(w)
        flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = cv2.remap((frame * 255).astype(np.uint8), flow_map, None, cv2.INTER_LINEAR)
        return warped.astype(np.float32) / 255.0

    def apply_temporal_consistency(self, current_frame, current_colorized, prev_colorized):
        """Blend current frame with warped previous frame for consistency"""
        if prev_colorized is None:
            return current_colorized

        flow = self.calculate_optical_flow(prev_colorized, current_colorized)
        warped_prev = self.warp_frame(prev_colorized, flow)

        # Blend based on flow confidence
        flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        confidence = np.exp(-flow_magnitude / 10.0)
        confidence = np.stack([confidence] * 3, axis=2)

        blended = confidence * warped_prev + (1 - confidence) * current_colorized
        return blended

class NeuralStyleTransfer:
    """Advanced neural style transfer"""
    def __init__(self):
        # Import torchvision models only if needed
        from torchvision import models
        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def get_features(self, x, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram

    def style_transfer(self, content_img, style_img, content_weight=1e4, style_weight=1e6, iterations=300):
        """Perform neural style transfer"""
        content_features = self.get_features(content_img)
        style_features = self.get_features(style_img)

        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}

        # Initialize with content image
        target = content_img.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=0.003)

        for i in range(iterations):
            target_features = self.get_features(target)
            content_loss = torch.mean((target_features['conv4_1'] - content_features['conv4_1']) ** 2)

            style_loss = 0
            for layer in style_grams:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (d * h * w)

            total_loss = content_weight * content_loss + style_weight * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return target

class ComprehensiveEvaluator:
    """Comprehensive evaluation suite"""
    def __init__(self):
        self.metrics = {
            'psnr': self.psnr_metric,
            'ssim': self.ssim_metric,
            'colorfulness': self.colorfulness_metric
        }

    def psnr_metric(self, img1, img2):
        """Calculate PSNR"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(1.0 / np.sqrt(mse))

    def ssim_metric(self, img1, img2):
        """Calculate SSIM"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))

    def colorfulness_metric(self, image):
        """Calculate colorfulness metric"""
        # Split image into RGB components
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Calculate rg and yb
        rg = r - g
        yb = 0.5 * (r + g) - b
        
        # Calculate standard deviation and mean
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        
        # Calculate colorfulness
        std_root = np.sqrt(std_rg**2 + std_yb**2)
        mean_root = np.sqrt(mean_rg**2 + mean_yb**2)
        
        return std_root + 0.3 * mean_root

    def evaluate(self, original, generated, reference=None):
        results = {}

        # Basic metrics
        results['psnr'] = self.metrics['psnr'](original, generated)
        results['ssim'] = self.metrics['ssim'](original, generated)
        results['colorfulness'] = self.metrics['colorfulness'](generated)

        return results