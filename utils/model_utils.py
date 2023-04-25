from models.configurations import (
    TEXT_PRETRAINED_AVAILABLE,
    VISION_PRETRAINED_AVAILABLE,
    VISION_MODEL_TYPE_2_DATA_TRANSFORM,
)
from transformers import AutoModel
from typing import Optional 

from torch.nn import Module


def load_vision_model(vision_model_type:str, 
                      vision_pretrained:Optional[str]=None, 
                      retain_head:bool=False) -> Module:
    if vision_model_type == 'ae':
        import torchxrayvision as xrv
        if not vision_pretrained:
            vision_pretrained = "101-elastic"
        return xrv.autoencoders.ResNetAE(weights=vision_pretrained)
    
    if vision_model_type == 'timm':
        from utils.utils import load_timm_model
        if not vision_pretrained:
            vision_pretrained = "swin_base_patch4_window7_224"
        return load_timm_model(vision_pretrained, pretrained=True, retain_head=retain_head)
    
    if vision_model_type == 'transformers':
        if retain_head:
            return AutoModel.from_pretrained(vision_pretrained)
        return AutoModel.from_pretrained(vision_pretrained).base_model
