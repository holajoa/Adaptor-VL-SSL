from transformers import AutoModel
from typing import Optional

import torch.nn as nn
from pytorch_lightning.callbacks import TQDMProgressBar, Callback
from pytorch_lightning import LightningModule

from tqdm import tqdm
import sys
import os


def freeze_adaptor(model: LightningModule):
    for param in model.adaptor.parameters():
        param.requires_grad = False


def load_timm_model(
    model_name="swin_base_patch4_window7_224", retain_head=False, pretrained=True
):
    import timm

    model = timm.create_model(model_name, pretrained=pretrained)
    if not retain_head:
        return nn.Sequential(*list(model.children())[:-2])
    return model


def load_vision_model(
    vision_model_type: str,
    vision_pretrained: Optional[str] = None,
    retain_head: bool = False,
) -> nn.Module:
    if vision_model_type == "ae":
        import torchxrayvision as xrv

        if not vision_pretrained:
            vision_pretrained = "101-elastic"
        return xrv.autoencoders.ResNetAE(weights=vision_pretrained)

    if vision_model_type == "timm":
        from utils.model_utils import load_timm_model

        if not vision_pretrained:
            vision_pretrained = "swin_base_patch4_window7_224"
        return load_timm_model(
            vision_pretrained, pretrained=True, retain_head=retain_head
        )

    if vision_model_type == "hub":
        import torch.hub

        if not vision_pretrained:
            vision_pretrained = "facebookresearch/dinov2/dinov2_vits14"
        vision_pretrained_repo, vision_pretrained = vision_pretrained.rsplit("/", 1)
        return torch.hub.load(vision_pretrained_repo, vision_pretrained)

    if vision_model_type == "transformers":
        if retain_head:
            return AutoModel.from_pretrained(vision_pretrained)
        return AutoModel.from_pretrained(vision_pretrained).base_model


def get_newest_ckpt(vision_model, text_model, wandb=False, project_name="adaptor pretrain", postfix=""):
    if postfix:
        postfix = f"_{postfix}"
    base_dir = f"/root/adaptor-thesis/trained_models/pretrain/{vision_model}_{text_model}{postfix}/{project_name}/"
    base_dir = os.path.join(
        [os.path.abspath(os.path.join(base_dir, p)) for p in os.listdir(base_dir)][-1],
        "checkpoints",
    )
    ckpt = [os.path.abspath(os.path.join(base_dir, p)) for p in os.listdir(base_dir)][
        -1
    ]
    return ckpt


class StreamingProgressBar(TQDMProgressBar):
    def __init__(self, total: int, val_total: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total = total
        self._val_total = val_total

    def init_train_tqdm(self):
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            total=self._total,
        )
        return bar

    def init_validation_tqdm(self):
        bar = tqdm(
            desc="running validation...",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            total=self._val_total,
        )
        return bar


class TestEveryEpochCallback(Callback):
    def __init__(self, datamodule):
        self.datamodule = datamodule

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        trainer.test(pl_module, self.datamodule)
