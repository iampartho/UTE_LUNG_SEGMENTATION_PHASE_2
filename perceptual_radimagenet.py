import torch
import torch.nn as nn
from torch import Tensor
import os
import errno

# ===================================================================
#  YOUR PROVIDED ResNet50 MODEL CODE (UNCHANGED)
# ===================================================================
# This ensures perfect compatibility with your weight file.

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        width = int(planes)
        self.conv1 = conv1x1(inplanes, width, stride=stride)
        self.bn1 = nn.BatchNorm2d(width, eps=1.001e-5, momentum=0.99)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width, eps=1.001e-5, momentum=0.99)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=1.001e-5, momentum=0.99)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1.001e-5, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion, eps=1.001e-5, momentum=0.99),
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# ===================================================================
#  NEW FEATURE EXTRACTOR CLASS
# ===================================================================
# This class uses your ResNet50 as a backbone to extract features.

class RadImageNetFeatureExtractor(nn.Module):
    """
    Extracts multi-level features from a 3D tensor using the provided
    pre-trained 2D RadImageNet ResNet-50 implementation.
    """
    def __init__(self, pretrained_path: str, device: str = "cuda"):
        super().__init__()
        
        # 1. Instantiate the ResNet-50 model from your code
        loss_network = ResNet50()
        
        # 2. Load the pre-trained weights from your file
        # This will now work perfectly without any key errors.
        loss_network.load_state_dict(torch.load(pretrained_path, map_location=device))
        
        # 3. Store the network, set to eval mode, and disable gradients
        self.loss_network = loss_network.to(device)
        self.loss_network.eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Processes a 3D tensor to extract and return mean features.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, 1, D, H, W).
                              Example: (2, 1, 192, 192, 96)

        Returns:
            list[torch.Tensor]: A list containing the mean feature maps from 
                                each specified layer.
        """
        b, c, d, h, w = x.shape
        
        # ResNet-50 expects a 3-channel input, so we replicate the single channel.
        if c == 1:
            x = x.repeat(1, 3, 1, 1, 1) # Shape: (B, 3, D, H, W)
        
        # Permute and reshape the 3D input into a batch of 2D slices
        # (B, 3, D, H, W) -> (B, D, 3, H, W) -> (B*D, 3, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * d, 3, h, w)
        
        # --- Feature Extraction ---
        # We manually pass the input through each layer of the ResNet
        # to capture the intermediate outputs.
        
        # Initial layers
        out = self.loss_network.conv1(x_2d)
        out = self.loss_network.bn1(out)
        out = self.loss_network.relu(out)
        out = self.loss_network.maxpool(out)
        
        # ResNet blocks
        f1 = self.loss_network.layer1(out)
        f2 = self.loss_network.layer2(f1)
        f3 = self.loss_network.layer3(f2)
        f4 = self.loss_network.layer4(f3)
        
        features = [f1, f2, f3, f4]
            
        # --- Post-processing: Mean over Depth Dimension ---
        mean_features = []
        for f in features:
            # Reshape back to 5D: (B*D, C_feat, H_feat, W_feat) -> (B, D, C_feat, H_feat, W_feat)
            c_feat, h_feat, w_feat = f.shape[1:]
            f_5d = f.view(b, d, c_feat, h_feat, w_feat)
            
            # Take the mean across the depth dimension (dim=1)
            mean_f = torch.mean(f_5d, dim=1)
            mean_features.append(mean_f)
            
        return mean_features

# ===================================================================
#  EXAMPLE USAGE
# ===================================================================
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # IMPORTANT: Update this path to your actual weights file
    WEIGHTS_PATH = "RadImageNet-ResNet50_notop.pth" 

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weight file not found at '{WEIGHTS_PATH}'")
        print("Please ensure the path is correct.")
    else:
        # 1. Instantiate the feature extractor
        feature_extractor = RadImageNetFeatureExtractor(pretrained_path=WEIGHTS_PATH, device=DEVICE)
        print("Feature extractor created successfully! 🚀\n")

        # 2. Create a dummy 3D input tensor
        # Shape: (batch_size, channels, depth, height, width)
        dummy_input = torch.randn(2, 1, 192, 192, 96).to(DEVICE) # D, H, W
        print(f"Input tensor shape: {dummy_input.shape}\n")

        # 3. Get the features
        list_of_features = feature_extractor(dummy_input)

        # 4. Inspect the output
        print("--- Extracted Mean Features ---")
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        for name, feat in zip(layer_names, list_of_features):
            print(f"Shape of features from {name}: {feat.shape}")