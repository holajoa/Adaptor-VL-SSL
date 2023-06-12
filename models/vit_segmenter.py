import torch
import torch.nn as nn


class ViTDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_0  = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
    

class DINOSegmenter(nn.Module):
    def __init__(self, backbone, adaptor, hidden_dim=384, 
                 out_channels=1, features=[512, 256, 128, 64]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.features = features
        
        self.encoder = backbone 
        self.adaptor = adaptor
        self.decoder = ViTDecoder(in_channels=hidden_dim, 
                                 out_channels=out_channels, 
                                 features=features)

    def forward(self, x):
        features = self.encoder(x, is_training=True)['x_norm_patchtokens']
        features = self.adaptor(features)
        output = self.decoder(features.reshape(1, 16, 16, self.hidden_dim).permute(0, 3, 1, 2))
        return output
