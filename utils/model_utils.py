from transformers import AutoModel
from typing import Optional 

import timm

import torch.nn as nn
from torch.nn import Module

import os


def load_timm_model(model_name='swin_base_patch4_window7_224', retain_head=False, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    if not retain_head:
        return nn.Sequential(*list(model.children())[:-2])
    return model


def load_vision_model(vision_model_type:str, 
                      vision_pretrained:Optional[str]=None,
                      retain_head:bool=False) -> Module:
    if vision_model_type == 'ae':
        import torchxrayvision as xrv
        if not vision_pretrained:
            vision_pretrained = "101-elastic"
        return xrv.autoencoders.ResNetAE(weights=vision_pretrained)
    
    if vision_model_type == 'timm':
        from utils.model_utils import load_timm_model
        if not vision_pretrained:
            vision_pretrained = "swin_base_patch4_window7_224"
        return load_timm_model(vision_pretrained, pretrained=True, retain_head=retain_head)
    
    if vision_model_type == 'hub':
        import torch.hub
        if not vision_pretrained:
            vision_pretrained = 'facebookresearch/dinov2/dinov2_vits14'
        vision_pretrained_repo, vision_pretrained = vision_pretrained.rsplit('/', 1)
        return torch.hub.load(vision_pretrained_repo, vision_pretrained)
    
    if vision_model_type == 'transformers':
        if retain_head:
            return AutoModel.from_pretrained(vision_pretrained)
        return AutoModel.from_pretrained(vision_pretrained).base_model

def get_newest_ckpt(vision_model, text_model):
    base_dir = f'/vol/bitbucket/jq619/individual-project/trained_models/pretrain/{vision_model}_{text_model}/lightning_logs/'
    base_dir = os.path.join([os.path.abspath(os.path.join(base_dir, p)) for p in os.listdir(base_dir)][-1], 'checkpoints')
    ckpt = [os.path.abspath(os.path.join(base_dir, p)) for p in os.listdir(base_dir)][-1]
    return ckpt
