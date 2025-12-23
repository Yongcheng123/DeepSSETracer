import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Double 3D convolution block with batch normalization and activation."""
    
    def __init__(self, in_channels, mid_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            activation,
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            activation
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.1)
    
    def forward(self, x):
        return self.conv(x)


class UpConvBlock3D(nn.Module):
    """Upsampling conv block with skip connection concatenation."""
    
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            activation,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            activation,
            nn.Dropout(p=dropout, inplace=True)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    3D U-Net for volumetric secondary structure element segmentation.
    
    Architecture:
        - 3 encoder levels: 64 -> 128 -> 256 channels
        - 2 decoder levels with skip connections
        - Output: 3 classes (background, helix, sheet)
    """
    
    def __init__(self):
        super().__init__()
        activation = nn.ReLU()
        
        # Encoder path
        self.down_1 = ConvBlock3D(1, 32, 64, activation)
        self.pool_1 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.down_2 = ConvBlock3D(64, 64, 128, activation)
        self.pool_2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.down_3 = ConvBlock3D(128, 128, 256, activation)
        
        # Decoder path
        self.up_conv_1 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.up_1 = UpConvBlock3D(256, 128, activation)
        self.up_conv_2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.up_2 = UpConvBlock3D(128, 64, activation)
        
        # Output layer
        self.out = nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.out.weight, std=0.1)
    
    def forward(self, x):
        # Encoder
        d1 = self.down_1(x)
        p1 = self.pool_1(d1)
        
        d2 = self.down_2(p1)
        p2 = self.pool_2(d2)
        
        d3 = self.down_3(p2)
        
        # Decoder with skip connections
        u1 = self.up_conv_1(d3)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_1(u1)
        
        u2 = self.up_conv_2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_2(u2)
        
        return self.out(u2)


class Gem_UNet(nn.Module):
    """Wrapper for UNet with configuration options."""
    
    def __init__(self, args):
        super().__init__()
        self.unet = UNet()
        self.gpu = args.cuda

    def forward(self, x):
        return self.unet(x)