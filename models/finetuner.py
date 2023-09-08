from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, Accuracy

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os


class AdaptorFinetuner(LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        model_name: str,
        text_model_name: Optional[str],
        adaptor: Union[nn.Module, LightningModule],
        in_features: int = 2048,
        num_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        binary: bool = True,
        multilabel: bool = True,
        freeze_adaptor: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        if binary:
            assert num_classes == 1
            self.train_auc = AUROC(task="binary")
            self.val_auc = AUROC(task="binary", compute_on_step=False)
            self.test_auc = AUROC(task="binary", compute_on_step=False)
            self.metric_name = "auroc"
        else:
            self.train_acc = Accuracy(
                task="multiclass", num_classes=num_classes, topk=1
            )
            self.val_acc = Accuracy(
                task="multiclass",
                num_classes=num_classes,
                topk=1,
                compute_on_step=False,
            )
            self.test_acc = Accuracy(
                task="multiclass",
                num_classes=num_classes,
                topk=1,
                compute_on_step=False,
            )
            self.metric_name = "accuracy"

        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.adaptor = adaptor
        if freeze_adaptor:
            for param in self.adaptor.parameters():
                param.requires_grad = False
        self.model_name = model_name
        self.text_model_name = text_model_name
        if self.text_model_name is not None:
            store_path = (
                "/vol/bitbucket/jq619/adaptor-thesis/saved_embeddings/dummy_text_embeds"
            )
            self.dummy_text = torch.from_numpy(
                torch.load(os.path.join(store_path, self.text_model_name + ".pt"))
            )
        self.binary = binary
        self.multilabel = multilabel

        self.linear_layer = SSLEvaluator(
            n_input=in_features,
            n_classes=num_classes,
            p=dropout,
            n_hidden=hidden_dim,
        )

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.backbone.eval()
        self.adaptor.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            auc = self.train_auc(torch.sigmoid(logits).float(), y.long())
            self.log("train_auroc_step", auc, prog_bar=True, sync_dist=True)
            self.log("train_auroc_epoch", self.train_auc, prog_bar=True, sync_dist=True)
        else:
            acc = self.train_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("train_accuracy_step", acc, prog_bar=True, sync_dist=True)
            self.log(
                "train_accuracy_epoch", self.train_acc, prog_bar=True, sync_dist=True
            )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            self.val_auc(torch.sigmoid(logits).float(), y.long())
            self.log(
                f"val_{self.metric_name}",
                self.val_auc,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        else:
            self.val_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log(
                f"val_{self.metric_name}",
                self.val_acc,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        if self.multilabel:
            self.test_auc(torch.sigmoid(logits).float(), y.long())
            # TODO: save probs and labels
            self.log(f"test_{self.metric_name}", self.test_auc, on_epoch=True)
        else:
            self.test_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log(f"test_{self.metric_name}", self.test_acc, on_epoch=True)

        return loss

    def shared_step(self, batch):
        x, y = batch["pixel_values"], batch["labels"]
        # For multi-class
        with torch.no_grad():
            if "ae" in self.model_name:
                feats = (
                    torch.flatten(self.backbone.encode(x), start_dim=2)
                    .permute(0, 2, 1)
                    .mean(dim=1)
                )
            else:
                feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        if self.text_model_name is not None:
            batch_size = feats.size(0)
            batch_text_dummy = self.dummy_text.repeat(batch_size, 1).to(feats.device)
            _, _, feats, _ = self.adaptor(
                feats, batch_text_dummy, return_loss=False, return_dict=False
            )
        else:
            feats = self.adaptor(feats)
        logits = self.linear_layer(feats)
        if self.multilabel:
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())
        else:
            # y = F.one_hot(y, num_classes=self.num_classes)
            y = y.squeeze()
            loss = F.cross_entropy(logits.float(), y.long())

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = AdamW(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_schedule = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=int(self.training_steps * 0.4),
            T_mult=1,
            eta_min=1e-8,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_schedule}

        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        effective_batch_size = trainer.accumulate_grad_batches * trainer.num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes)
            )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
