import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.dataset_utils import torch2huggingface_dataset, get_dataloader

from math import ceil


class AdaptorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        collate_fn,
        transforms,
        data_pct,
        batch_size,
        num_workers,
        crop_size=224,
        seed=42,
        **kwargs
    ):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.data_transform = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.dataset_kwargs = kwargs
        self.seed = seed
        self.datasets = dict()

    def setup(self, stage: str):
        if stage == "fit":
            train_dataset = self._get_dataset(split="train")
            # val_dataset = self._get_dataset(split="valid")
            val_dataset = self._get_dataset(split="test")

            self.train_steps = self._get_num_steps(train_dataset)
            self.val_steps = self._get_num_steps(val_dataset)

            self.datasets["train"] = torch2huggingface_dataset(
                train_dataset, streaming=False, shuffle=True, seed=self.seed
            )
            self.datasets["valid"] = torch2huggingface_dataset(
                val_dataset, streaming=False
            )
            self.datasets["train"].with_format("torch")
            self.datasets["valid"].with_format("torch")

        if stage == "test":
            test_dataset = self._get_dataset(split="test")
            self.datasets["test"] = torch2huggingface_dataset(
                test_dataset, streaming=False
            )
            self.datasets["test"].with_format("torch")

    def _get_dataset(self, split="train"):
        is_train = split == "train"
        return self.dataset(
            split=split,
            transform=self.data_transform(is_train, self.crop_size),
            data_pct=self.data_pct,
            imsize=self.crop_size,
            **self.dataset_kwargs,
        )

    def _get_num_steps(self, dataset):
        return ceil(len(dataset) / self.batch_size)

    def _get_dataloader(self, split="train", shuffle=False):
        return DataLoader(
            self.datasets[split],
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._get_dataloader(split="train")

    def val_dataloader(self):
        return self._get_dataloader(split="valid")

    def test_dataloader(self):
        return self._get_dataloader(split="test")
