from typing import List, Union, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPooling

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.clip.modeling_clip import CLIPOutput, clip_loss

from transformers.optimization import get_linear_schedule_with_warmup

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import tqdm
import sys


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayerWithCrossAttention(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayerWithCrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src,
        query=None,
        src_mask=None,
        src_key_padding_mask=None,
        run_cross_attn=False,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if run_cross_attn:
            assert query is not None, "Must provide a query vector"
            src2 = self.attn(
                query,
                src,
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]
        else:  # run self attention
            src2 = self.attn(
                src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class AdaptorModule(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 1,
    ):
        super(AdaptorModule, self).__init__()
        if num_layers > 1:
            self.encoder = nn.ModuleList(
                [
                    TransformerEncoderLayerWithCrossAttention(embed_dim, num_heads)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.encoder = TransformerEncoderLayerWithCrossAttention(
                embed_dim, num_heads
            )

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        encoder_layers = (
            self.encoder if isinstance(self.encoder, nn.ModuleList) else [self.encoder]
        )
        # Optional text embedding inputs
        if text_embeds is not None:
            assert (
                text_embeds.device == image_embeds.device
            ), "text and image embeddings must be on the same device"
            ## Run through cross-attention
            for layer in encoder_layers:
                image_embeds = layer(
                    src=image_embeds, query=text_embeds, run_cross_attn=True
                )
                text_embeds = layer(
                    src=text_embeds, query=image_embeds, run_cross_attn=True
                )
            return image_embeds, text_embeds
        else:
            ## Run through self-attention if no text embedding is inputed - downstream task.
            for layer in encoder_layers:
                image_embeds = layer(src=image_embeds, run_cross_attn=False)
            return image_embeds


class Project(nn.Module):
    def __init__(
        self,
        text_embed_dim: int,
        vision_embed_dim: int,
        projection_dim: int = 512,
    ):
        super(Project, self).__init__()
        self.text_embed_dim = text_embed_dim
        self.vision_embed_dim = vision_embed_dim
        self.projection_dim = projection_dim
        self.visual_projection = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_embed_dim, projection_dim, bias=False)

    def forward(
        self,
        image_embeds_: torch.Tensor,
        text_embeds_: Optional[torch.Tensor] = None,
    ):
        image_embeds = self.visual_projection(image_embeds_)
        image_embeds = image_embeds / image_embeds.norm(
            dim=-1, keepdim=True
        )  # normalized features

        if text_embeds_ is not None:
            text_embeds = self.text_projection(text_embeds_)
            text_embeds = text_embeds / text_embeds.norm(
                dim=-1, keepdim=True
            )  # normalized features
            return image_embeds, text_embeds

        return image_embeds


class Adaptor(LightningModule):
    def __init__(
        self,
        adaptor_config: Optional[BertConfig] = None,
        text_output_dim: int = 768,
        vision_output_dim: Optional[
            int
        ] = None,  # ignored if vision_model_type is huggingface
        logit_scale_init_value: float = 2.6592,  # logit_scale = 1 / temperature
        projection_dim: int = 512,
        num_layers: int = 1,
        lr: float = 1e-4,
    ):
        super(Adaptor, self).__init__()

        self.projection = Project(
            text_embed_dim=text_output_dim,
            vision_embed_dim=vision_output_dim,
            projection_dim=projection_dim,
        )
        self.adaptor_module = AdaptorModule(
            embed_dim=projection_dim, num_layers=num_layers
        )
        self.vision_output_dim = vision_output_dim
        self.text_output_dim = text_output_dim
        self.projection_dim = projection_dim

        self.logit_scale_init_value = logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.lr = lr
        self.save_hyperparameters()

    def forward(
        self,
        image_embeds_raw: torch.FloatTensor,
        text_embeds_raw: Optional[torch.FloatTensor] = None,
        return_loss: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor, CLIPOutput]:
        if text_embeds_raw is not None:
            image_embeds_full, text_embeds_full = self.projection(
                image_embeds_raw, text_embeds_raw
            )
            image_embeds_full, text_embeds_full = self.adaptor_module(
                image_embeds_full, text_embeds_full
            )  # bz x seq_len x 768 during pretraining

            if len(image_embeds_full.shape) == 2:
                image_embeds_full = image_embeds_full.unsqueeze(1)
            if len(text_embeds_full.shape) == 2:
                text_embeds_full = text_embeds_full.unsqueeze(1)
            image_embeds, text_embeds = (
                image_embeds_full[:, 0, :],
                text_embeds_full[:, 0, :],
            )

            # normalized features - is this necessary for full embeddings?
            image_embeds_full = image_embeds_full / image_embeds_full.norm(
                dim=-1, keepdim=True
            )
            text_embeds_full = text_embeds_full / text_embeds_full.norm(
                dim=-1, keepdim=True
            )

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_text = (
                torch.matmul(text_embeds, image_embeds.t()) * logit_scale
            )  # [batch_size, batch_size]
            logits_per_image = logits_per_text.T

            loss = None
            if return_loss:
                loss = clip_loss(logits_per_text)

            if not return_dict:
                output = (logits_per_image, logits_per_text, text_embeds, image_embeds)
                return ((loss,) + output) if loss is not None else output

            return CLIPOutput(
                loss=loss,
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_text,
                text_embeds=text_embeds,
                image_embeds=image_embeds,
                text_model_output=BaseModelOutputWithPooling(
                    last_hidden_state=text_embeds_full,
                    pooler_output=text_embeds,
                ),
                vision_model_output=BaseModelOutputWithPooling(
                    last_hidden_state=image_embeds_full,
                    pooler_output=image_embeds,
                ),
            )

        else:  # text_embeds is none
            image_embeds_full = self.projection(image_embeds_raw)
            image_embeds_full = self.adaptor_module(image_embeds_full)
            image_embeds_full = image_embeds_full / image_embeds_full.norm(
                dim=-1, keepdim=True
            )

            return image_embeds_full  # ignore return_dict and return_loss

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.lr_schedulers().step()
        return loss

    def _shared_eval(self, batch, batch_idx, prefix):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(f"{prefix}_loss", loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        # lr_schedule = CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=int(self.training_steps * 0.4),
        #     T_mult=0.5,
        #     eta_min=1e-8,
        # )
        lr_schedule = get_linear_schedule_with_warmup(
            optimizer=optimizer,    
            num_warmup_steps=int(self.training_steps * .02),
            num_training_steps=self.training_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_schedule}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
