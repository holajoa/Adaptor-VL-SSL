import timm
import torch.nn as nn

def load_timm_model(model_name='swin_base_patch4_window7_224', retain_head=False, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    if not retain_head:
        return nn.Sequential(*list(model.children())[:-2])
    return model

def freeze_encoder(model):
    for encoder in [model.text_model, model.vision_model]:
        for param in encoder.parameters():
            param.requires_grad = False
            