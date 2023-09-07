"""ResNet-UNet Modified from https://github.com/zhoudaxia233/PyTorch-Unet/blob/master/resnet_unet.py"""
"""ViT Modified from https://github.com/HKU-MedAI/MGCA/blob/main/mgca/models/backbones/transformer_seg.py"""

import torch.nn as nn
import torch
import torchxrayvision as xrv
from math import log2
from typing import Tuple


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class Resize(nn.Module):
    def __init__(self, size: Tuple[int]):
        super(Resize, self).__init__()
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(
            x, size=self.size, mode="bicubic", align_corners=False
        )


class ResNetAEUNet(nn.Module):
    def __init__(
        self,
        adaptor,
        pretrained=False,
        out_channels=1,
        freeze_adaptor=True,
        input_size=224,
    ):
        super().__init__()

        resnet_ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
        self.encoder_layers = list(resnet_ae.children())[:8]
        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = nn.Sequential(*self.encoder_layers[5])
        self.block4 = nn.Sequential(*self.encoder_layers[6])
        self.block5 = nn.Sequential(*self.encoder_layers[7])

        self.adaptor = adaptor

        if input_size == 224:
            self.neck = nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1, bias=False),
                nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, bias=False),
            )
        elif input_size % 224 == 0 and (log2(input_size // 224)).is_integer():
            self.neck = nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1, bias=False),
                nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, bias=False),
                *[nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, bias=False)]
                * int(log2(input_size // 224)),
            )
        else:
            raise ValueError(
                f"Unsupported input size {input_size}, only 224, 448 and 896 are supported."
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
        self._freeze_encoder()  # Freeze encoder weights
        self.freeze_adaptor = freeze_adaptor
        if self.freeze_adaptor:
            self._freeze_adaptor()

    def _freeze_adaptor(self):
        self.freeze_adaptor = True
        for param in self.adaptor.parameters():
            param.requires_grad = False

    def _freeze_encoder(self):
        for name, block in self.named_children():
            if name.startswith("block"):
                for param in block.parameters():
                    param.requires_grad = False

    def _weights_init(self):
        for name, block in self.named_children():
            if name.startswith("block") or name == "adaptor":
                continue
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
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

    def forward(self, x, return_dict=False):
        encoder_features = self.encode(x)
        unimodal_features = encoder_features[-1]
        global_unimodal_features = self.get_global_features(unimodal_features)
        multimodal_features = self.fusion(global_unimodal_features)
        out = self.decode(
            multimodal_features.unsqueeze(2).unsqueeze(3),
            encoder_features,
        )

        if return_dict:
            return {
                "encoder_features": encoder_features,
                "unimodal_features": unimodal_features,
                "global_unimodal_features": global_unimodal_features,
                "multimodal_features": multimodal_features,
                "out": out,
            }
        return out


class ViTDecoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, features=[512, 256, 128, 64], input_size=224
    ):
        super().__init__()
        if input_size == 224:
            self.decoder_0 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=0),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
        else:
            assert input_size in [
                224,
                896,
            ], f"Unsupported image size {input_size}, only 224 and 896 are supported."
            factor = input_size // 224
            self.decoder_0 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=14 / 16),
            )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_0(x)
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x


class DINOv2Segmenter(nn.Module):
    def __init__(
        self,
        backbone,
        adaptor,
        hidden_dim=384,
        out_channels=1,
        features=[512, 256, 128, 64],
        freeze_adaptor=True,
        input_size=224,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.features = features
        self.input_size = input_size

        self.encoder = backbone
        self.adaptor = adaptor
        self.decoder = ViTDecoder(
            in_channels=hidden_dim,
            out_channels=out_channels,
            features=features,
            input_size=input_size,
        )

        self._weights_init()
        self._freeze_encoder()  # Freeze encoder weights
        self.freeze_adaptor = freeze_adaptor
        if self.freeze_adaptor:
            self._freeze_adaptor()

    def _freeze_adaptor(self):
        self.freeze_adaptor = True
        for param in self.adaptor.parameters():
            param.requires_grad = False

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _weights_init(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_dict=False):
        features = self.encoder(x, is_training=True)["x_norm_patchtokens"]
        multimodal_features = self.adaptor(features)
        batch_size, num_patches, _ = multimodal_features.shape
        patch_size = int(num_patches**0.5)
        decoder_input = multimodal_features.reshape(
            batch_size, patch_size, patch_size, self.hidden_dim
        ).permute(0, 3, 1, 2)
        output = self.decoder(decoder_input)
        if return_dict:
            return {
                "encoder_features": features,
                "multimodal_features": multimodal_features,
                "out": output,
            }
        return output
