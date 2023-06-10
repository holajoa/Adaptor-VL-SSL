import torch.nn as nn
import torch.nn.functional as F
import torch
import torchxrayvision as xrv

from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch as smp


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetAE101UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, n_blocks):
        super().__init__()
        
        out_channels = encoder_channels[:-1]
        
        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        
        head_channels = encoder_channels[0] 
        in_channels = encoder_channels[1:]
        skip_channels = encoder_channels[1:] + [0]
        
        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
    

class ResNetAE101UNetEncoder(nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels = [1, 64, 256, 512, 1024]
        self._depth = 5
        self._in_channels = 1
        self.encoder = xrv.autoencoders.ResNetAE(weights="101-elastic")
    
    def forward(self, x):
        feat0 = x.clone()
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        feat1 = self.encoder.relu(x)
        x = self.encoder.maxpool(feat1)
        
        feat2 = self.encoder.layer1(x)
        feat3 = self.encoder.layer2(feat2)
        feat4 = self.encoder.layer3(feat3)
        feat5 = self.encoder.layer4(feat4)
        return [feat0, feat1, feat2, feat3, feat4, feat5]


smp.encoders.encoders["resnet-ae-101"] = {
    "encoder": ResNetAE101UNetEncoder,
    "params":{}, 
    "pretrained_settings":{}, 
}

# # Usage
# seg_model = smp.Unet(encoder_name="resnet-ae-101", encoder_weights=None, activation=None)
