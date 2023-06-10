"""Modified from https://github.com/zhoudaxia233/PyTorch-Unet/blob/master/resnet_unet.py"""

import torch.nn as nn
import torch
import torchxrayvision as xrv


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )
    
class ResNetAEUNet(nn.Module):
    def __init__(self, adaptor, pretrained=False, out_channels=1):
        super().__init__()
        
        resnet_ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
        self.encoder_layers = list(resnet_ae.children())[:8]
        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = nn.Sequential(*self.encoder_layers[5])
        self.block4 = nn.Sequential(*self.encoder_layers[6])
        self.block5 = nn.Sequential(*self.encoder_layers[7])
        
        self.adaptor = adaptor
        
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1, bias=False), 
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, bias=False), 
        )
        
        self.up_conv6 = up_conv(512, 512)
        self.conv6 = double_conv(512 + 1024, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 64, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = nn.Conv2d(32, out_channels, kernel_size=1)
        
        if not pretrained:
            self._weights_init()
    
    def _freeze_encoder(self):
        for name, block in self.named_children():
            if name.startswith('block'):
                for param in block.parameters():
                    param.requires_grad = False
    
    def _weights_init(self):
        for name, block in self.named_children():
            if name.startswith('block') or name == 'adaptor':
                continue
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        
        return [block1, block2, block3, block4, block5]
    
    def fusion(self, x):
        return self.adaptor(x)
    
    def decode(self, x, features):
        block1, block2, block3, block4, block5 = features
        neck = self.neck(x)
        
        x = self.up_conv6(neck)
        
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = self.conv10(x)

        return x
    
    def get_global_features(self, x):
        return torch.flatten(x, start_dim=2).permute((0, 2, 1)).mean(1) 
    
    def forward(self, x):
        encoder_features = self.encode(x)
        unimodal_features = encoder_features[-1]
        global_unimodal_features = self.get_global_features(unimodal_features)
        multimodal_features = self.fusion(global_unimodal_features)
        out = self.decode(multimodal_features.unsqueeze(2).unsqueeze(3), encoder_features)
        
        return {
            'encoder_features': encoder_features,
            'unimodal_features': unimodal_features,
            'global_unimodal_features': global_unimodal_features,
            'multimodal_features': multimodal_features,
            'out': out, 
        }
    
    