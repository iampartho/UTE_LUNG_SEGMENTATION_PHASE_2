import torch
import torch.nn as nn

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(Discriminator3D, self).__init__()
        
        layers = []
        
        # Initial layer
        layers.append(
            nn.Conv3d(in_channels, features[0], kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Downsampling layers
        in_feat = features[0]
        for feat in features[1:]:
            layers.append(
                nn.Conv3d(in_feat, feat, kernel_size=4, stride=2, padding=1, bias=False)
            )
            layers.append(nn.InstanceNorm3d(feat, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_feat = feat
            
        # Final classification layer
        # Output 1 channel prediction map (PatchGAN)
        layers.append(
            nn.Conv3d(in_feat, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

