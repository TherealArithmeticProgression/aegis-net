import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        # Load pretrained ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        # Encoder Layers
        # Input: 3 x H x W
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # Conv1, BN, ReLU -> 64 x H/2 x W/2
        self.layer0_1 = nn.Sequential(*self.base_layers[3:5]) # MaxPool, Layer1 -> 64 x H/4 x W/4
        self.layer1 = self.base_layers[5] # Layer2 -> 128 x H/8 x W/8
        self.layer2 = self.base_layers[6] # Layer3 -> 256 x H/16 x W/16
        self.layer3 = self.base_layers[7] # Layer4 -> 512 x H/32 x W/32

        # Decoder Layers
        self.up3 = self.up_block(512, 256) # 512 + 256 = 768 in -> 256
        self.up2 = self.up_block(256, 128) # 256 + 128 = 384 in -> 128
        self.up1 = self.up_block(128, 64)  # 128 + 64 = 192 in -> 64
        self.up0 = self.up_block(64, 64)   # 64 + 64 = 128 in -> 64

        # Final Convolution
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # H/2 -> H
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_class, kernel_size=1)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)       # 1/2
        x1 = self.layer0_1(x0)    # 1/4
        x2 = self.layer1(x1)      # 1/8
        x3 = self.layer2(x2)      # 1/16
        x4 = self.layer3(x3)      # 1/32

        # Decoder
        x = self.up3(torch.cat([x4, x3], dim=1)) # 1/16
        x = self.up2(torch.cat([x, x2], dim=1))  # 1/8
        x = self.up1(torch.cat([x, x1], dim=1))  # 1/4
        
        # Note: x0 is 1/2 size, but output of up1 is 1/4. We upsample in up0.
        # Wait, up1 took us to 1/4. up0 takes us to 1/2.
        
        x = self.up0(torch.cat([x, x0], dim=1))  # 1/2
        
        # Final upsample to original size
        x = self.final_conv(x) # 1/1
        
        return x
