import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class BasicUNet(nn.Module):
    """
    Basic U-Net with gradient checkpointing for memory optimization.
    
    Gradient checkpointing trades computation for memory by:
    - Not storing intermediate activations during forward pass
    - Recomputing them during backward pass
    - Significantly reducing GPU memory usage at the cost of ~33% more computation time
    """
    def __init__(self):
        super(BasicUNet, self).__init__()
        
        # Define individual layers separately
        self.conv_0_0 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.norm_0_0 = nn.InstanceNorm3d(32, affine=True)
        self.norm_0_0_ct = nn.InstanceNorm3d(32, affine=True)
        self.act_0_0 = nn.LeakyReLU(0.1)
        
        self.conv_0_1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.norm_0_1 = nn.InstanceNorm3d(32, affine=True)
        self.norm_0_1_ct = nn.InstanceNorm3d(32, affine=True)
        self.act_0_1 = nn.LeakyReLU(0.1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv_1_0 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.norm_1_0 = nn.InstanceNorm3d(64, affine=True)
        self.norm_1_0_ct = nn.InstanceNorm3d(64, affine=True)
        self.act_1_0 = nn.LeakyReLU(0.1)
        
        self.conv_1_1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.norm_1_1 = nn.InstanceNorm3d(64, affine=True)
        self.norm_1_1_ct = nn.InstanceNorm3d(64, affine=True)
        self.act_1_1 = nn.LeakyReLU(0.1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_2_0 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.norm_2_0 = nn.InstanceNorm3d(128, affine=True)
        self.norm_2_0_ct = nn.InstanceNorm3d(128, affine=True)
        self.act_2_0 = nn.LeakyReLU(0.1)
        
        self.conv_2_1 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.norm_2_1 = nn.InstanceNorm3d(128, affine=True)
        self.norm_2_1_ct = nn.InstanceNorm3d(128, affine=True)
        self.act_2_1 = nn.LeakyReLU(0.1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_3_0 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.norm_3_0 = nn.InstanceNorm3d(256, affine=True)
        self.norm_3_0_ct = nn.InstanceNorm3d(256, affine=True)
        self.act_3_0 = nn.LeakyReLU(0.1)
        
        self.conv_3_1 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.norm_3_1 = nn.InstanceNorm3d(256, affine=True)
        self.norm_3_1_ct = nn.InstanceNorm3d(256, affine=True)
        self.act_3_1 = nn.LeakyReLU(0.1)

        self.up_4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        
        self.conv_up_4_0 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.norm_up_4_0 = nn.InstanceNorm3d(128, affine=True)
        self.norm_up_4_0_ct = nn.InstanceNorm3d(128, affine=True)
        self.act_up_4_0 = nn.LeakyReLU(0.1)

        self.conv_up_4_1 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.norm_up_4_1 = nn.InstanceNorm3d(128, affine=True)
        self.norm_up_4_1_ct = nn.InstanceNorm3d(128, affine=True)
        self.act_up_4_1 = nn.LeakyReLU(0.1)

        self.up_3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        
        self.conv_up_3_0 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.norm_up_3_0 = nn.InstanceNorm3d(64, affine=True)
        self.norm_up_3_0_ct = nn.InstanceNorm3d(64, affine=True)
        self.act_up_3_0 = nn.LeakyReLU(0.1)

        self.conv_up_3_1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.norm_up_3_1 = nn.InstanceNorm3d(64, affine=True)
        self.norm_up_3_1_ct = nn.InstanceNorm3d(64, affine=True)
        self.act_up_3_1 = nn.LeakyReLU(0.1)

        self.up_2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        
        self.conv_up_2_0 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.norm_up_2_0 = nn.InstanceNorm3d(32, affine=True)
        self.norm_up_2_0_ct = nn.InstanceNorm3d(32, affine=True)
        self.act_up_2_0 = nn.LeakyReLU(0.1)

        self.conv_up_2_1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.norm_up_2_1 = nn.InstanceNorm3d(32, affine=True)
        self.norm_up_2_1_ct = nn.InstanceNorm3d(32, affine=True)
        self.act_up_2_1 = nn.LeakyReLU(0.1)

        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder with gradient checkpointing
        # if isMRI:
        x1 = checkpoint(lambda x: self.act_0_0(self.norm_0_0(self.conv_0_0(x))), x)#, use_reentrant=False)
        x1 = checkpoint(lambda x: self.act_0_1(self.norm_0_1(self.conv_0_1(x))), x1)#, use_reentrant=False)
        
        x2 = self.pool_1(x1)
        x2 = checkpoint(lambda x: self.act_1_0(self.norm_1_0(self.conv_1_0(x))), x2)#, use_reentrant=False)
        x2 = checkpoint(lambda x: self.act_1_1(self.norm_1_1(self.conv_1_1(x))), x2)#, use_reentrant=False)
        
        x3 = self.pool_2(x2)
        x3 = checkpoint(lambda x: self.act_2_0(self.norm_2_0(self.conv_2_0(x))), x3)#, use_reentrant=False)
        x3 = checkpoint(lambda x: self.act_2_1(self.norm_2_1(self.conv_2_1(x))), x3)#, use_reentrant=False)

        x4 = self.pool_3(x3)
        x4 = checkpoint(lambda x: self.act_3_0(self.norm_3_0(self.conv_3_0(x))), x4)#, use_reentrant=False)
        x4 = checkpoint(lambda x: self.act_3_1(self.norm_3_1(self.conv_3_1(x))), x4)#, use_reentrant=False)

        # Decoder with gradient checkpointing
        x4 = self.up_4(x4)
        x4 = torch.cat([x4, x3], dim=1)
        x4 = checkpoint(lambda x: self.act_up_4_0(self.norm_up_4_0(self.conv_up_4_0(x))), x4)#, use_reentrant=False)
        x4 = checkpoint(lambda x: self.act_up_4_1(self.norm_up_4_1(self.conv_up_4_1(x))), x4)#, use_reentrant=False)

        x3 = self.up_3(x4)
        x3 = torch.cat([x3, x2], dim=1)
        x3 = checkpoint(lambda x: self.act_up_3_0(self.norm_up_3_0(self.conv_up_3_0(x))), x3)#, use_reentrant=False)
        x3 = checkpoint(lambda x: self.act_up_3_1(self.norm_up_3_1(self.conv_up_3_1(x))), x3)#, use_reentrant=False)

        x2 = self.up_2(x3)
        x2 = torch.cat([x2, x1], dim=1)
        x2 = checkpoint(lambda x: self.act_up_2_0(self.norm_up_2_0(self.conv_up_2_0(x))), x2)#, use_reentrant=False)
        x2 = checkpoint(lambda x: self.act_up_2_1(self.norm_up_2_1(self.conv_up_2_1(x))), x2)#, use_reentrant=False)

        # Final layer
        x_out = self.final_conv(x2)
        # else:
        #     x1 = checkpoint(lambda x: self.act_0_0(self.norm_0_0_ct(self.conv_0_0(x))), x, use_reentrant=False)
        #     x1 = checkpoint(lambda x: self.act_0_1(self.norm_0_1_ct(self.conv_0_1(x))), x1, use_reentrant=False)
            
        #     x2 = self.pool_1(x1)
        #     x2 = checkpoint(lambda x: self.act_1_0(self.norm_1_0_ct(self.conv_1_0(x))), x2, use_reentrant=False)
        #     x2 = checkpoint(lambda x: self.act_1_1(self.norm_1_1_ct(self.conv_1_1(x))), x2, use_reentrant=False)
            
        #     x3 = self.pool_2(x2)
        #     x3 = checkpoint(lambda x: self.act_2_0(self.norm_2_0_ct(self.conv_2_0(x))), x3, use_reentrant=False)
        #     x3 = checkpoint(lambda x: self.act_2_1(self.norm_2_1_ct(self.conv_2_1(x))), x3, use_reentrant=False)

        #     x4 = self.pool_3(x3)
        #     x4 = checkpoint(lambda x: self.act_3_0(self.norm_3_0_ct(self.conv_3_0(x))), x4, use_reentrant=False)
        #     x4 = checkpoint(lambda x: self.act_3_1(self.norm_3_1_ct(self.conv_3_1(x))), x4, use_reentrant=False)

        #     # Decoder with gradient checkpointing
        #     x4 = self.up_4(x4)
        #     x4 = torch.cat([x4, x3], dim=1)
        #     x4 = checkpoint(lambda x: self.act_up_4_0(self.norm_up_4_0_ct(self.conv_up_4_0(x))), x4, use_reentrant=False)
        #     x4 = checkpoint(lambda x: self.act_up_4_1(self.norm_up_4_1_ct(self.conv_up_4_1(x))), x4, use_reentrant=False)

        #     x3 = self.up_3(x4)
        #     x3 = torch.cat([x3, x2], dim=1)
        #     x3 = checkpoint(lambda x: self.act_up_3_0(self.norm_up_3_0_ct(self.conv_up_3_0(x))), x3, use_reentrant=False)
        #     x3 = checkpoint(lambda x: self.act_up_3_1(self.norm_up_3_1_ct(self.conv_up_3_1(x))), x3, use_reentrant=False)

        #     x2 = self.up_2(x3)
        #     x2 = torch.cat([x2, x1], dim=1)
        #     x2 = checkpoint(lambda x: self.act_up_2_0(self.norm_up_2_0_ct(self.conv_up_2_0(x))), x2, use_reentrant=False)
        #     x2 = checkpoint(lambda x: self.act_up_2_1(self.norm_up_2_1_ct(self.conv_up_2_1(x))), x2, use_reentrant=False)


        #     # Final layer
        #     x_out = self.final_conv(x2)

        return x_out
