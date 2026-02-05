# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class BaseColor(nn.Module):
    '''Base class for colorization models.'''
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

class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()
        # Downsample layers
        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(64)]
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(128)]
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(256)]
        
        # Bottleneck layers
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        
        # Upsample and output
        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_l):
        # Fixed forward pass
        conv1_2 = self.model1(self.normalize_l(input_l))
        
        # Downsample
        conv2_2 = self.model2(conv1_2)
        
        # Downsample
        conv3_3 = self.model3(conv2_2)
        
        # Downsample
        conv4_3 = self.model4(conv3_3)
        
        # Bottleneck layers
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        
        # Upsample
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))
        
        return self.unnormalize_ab(self.upsample4(out_reg))

class LightweightColorizer(BaseColor):
    """Lightweight fallback model for CPU deployment"""
    def __init__(self):
        super(LightweightColorizer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_l):
        x = self.normalize_l(input_l)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.unnormalize_ab(x)

class FastColorizer(BaseColor):
    """Fast inference model optimized for CPU"""
    def __init__(self):
        super(FastColorizer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, input_l):
        x = self.normalize_l(input_l)
        x = self.net(x)
        return self.unnormalize_ab(x)

# Model loading functions (NO decorators here - they go in app.py)
def load_eccv16():
    """Load ECCV16 model"""
    try:
        model = ECCVGenerator()
        return model.eval()
    except Exception as e:
        print(f"ECCV16 load failed: {e}")
        return None

def load_lightweight():
    """Load lightweight fallback model"""
    try:
        model = LightweightColorizer()
        return model.eval()
    except Exception as e:
        print(f"Lightweight model load failed: {e}")
        return None

def load_fast():
    """Load fast CPU model"""
    try:
        model = FastColorizer()
        return model.eval()
    except Exception as e:
        print(f"Fast model load failed: {e}")
        return None

# Add to existing models.py file

class SIGGRAPHGenerator(BaseColor):
    """SIGGRAPH 2017 model - Produces vibrant, colorful results"""
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(SIGGRAPHGenerator, self).__init__()
        # Model takes 4 channels: L + ab hints + mask
        model1=[nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(64)]
        
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(128)]
        
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(256)]
        
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        
        # Bottleneck with dilation
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(512)]
        
        # Upsampling layers with skip connections
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        
        model8=[nn.ReLU(True), 
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(256)]
        
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        
        model9=[nn.ReLU(True), 
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True), 
                nn.ReLU(True), 
                norm_layer(128)]
        
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        
        model10=[nn.ReLU(True), 
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True), 
                nn.LeakyReLU(negative_slope=.2)]
        
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True), 
                  nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)
        self.model_out = nn.Sequential(*model_out)

    def forward(self, input_A, input_B=None, mask_B=None):
        """Forward pass with optional hints"""
        if input_B is None:
            input_B = torch.cat((input_A*0, input_A*0), dim=1)
        if mask_B is None:
            mask_B = input_A*0

        # Combine inputs: L channel + ab hints + mask
        conv1_2 = self.model1(torch.cat((self.normalize_l(input_A), 
                                         self.normalize_ab(input_B), 
                                         mask_B), dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        
        # Upsampling with skip connections
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return self.unnormalize_ab(out_reg)

def load_siggraph17():
    """Load SIGGRAPH17 model with pretrained weights if available"""
    try:
        model = SIGGRAPHGenerator()
        
        # Try to load pretrained weights (you would need to provide these)
        # For now, we'll use randomly initialized weights
        # model.load_state_dict(torch.load('siggraph17.pth'))
        
        return model.eval()
    except Exception as e:
        print(f"SIGGRAPH17 load failed: {e}")
        return None

# Update MODEL_REGISTRY
MODEL_REGISTRY = {
    'eccv16': load_eccv16,
    'lightweight': load_lightweight,
    'fast': load_fast,
    'siggraph17': load_siggraph17
}