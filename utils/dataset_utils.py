import torchvision.transforms as transforms
import torchxrayvision as xrv
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import ViTImageProcessor

from typing import Union, Dict

from pathlib import Path

from dataset.dataset import MultimodalPretrainingDatasetForAdaptor
from mgca.datasets.transforms import DataTransforms

from datasets import Dataset

import pickle


def get_dataloader(
    dataset,
    batch_size,
    num_workers=16,
    collate_fn=None,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
):
    return DataLoader(
        dataset,
        pin_memory=pin_memory,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
    )


class DataTransforms(object):
    """Copied from MGCA/mgca/datasets/transforms.py"""

    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        else:
            data_transforms = [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class AutoEncoderDataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        self.crop_size = crop_size
        self.data_transforms = transforms.Compose(
            [
                BatchedXRayCenterCrop(),
                xrv.datasets.XRayResizer(self.crop_size, engine="cv2"),
            ]
        )

    def __call__(self, image):
        image = xrv.datasets.normalize(np.array(image).astype(float), 255)

        # Check that images are 2D arrays
        if len(image.shape) >= 3:
            image = image[..., 0]
        elif len(image.shape) < 3:
            raise ValueError("error, dimension lower than 2 for image")

        # Add color channel
        image = image[None, :, :]
        image = self.data_transforms(image)
        image = torch.from_numpy(image)
        return image


class BatchedXRayCenterCrop(object):
    def crop_center(self, img):
        *_, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[..., starty : starty + crop_size, startx : startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


def ae_image_processor(
    imgs: np.ndarray, return_dict=True
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    imgs = xrv.datasets.normalize(imgs, 255)

    # Check that images are 2D arrays
    if len(imgs.shape) > 3:
        imgs = imgs[:, :, :, 0]
    if len(imgs.shape) < 3:
        print("error, dimension lower than 2 for image")

    # Add color channel
    imgs = imgs[:, None, :, :]

    transform = transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224, engine="cv2")]
    )
    imgs = np.array([transform(img) for img in imgs])
    imgs = torch.from_numpy(imgs)
    if return_dict:
        return {"pixel_values": imgs}
    return imgs


def timm_image_processor(imgs: np.ndarray) -> torch.Tensor:
    return ViTImageProcessor()(imgs, return_tensors="pt", return_dict=True)


def pickle_dataset(
    dataset_pkl,
    split,
    transform=None,
    data_pct=1.0,
    dataset_class: Dataset = MultimodalPretrainingDatasetForAdaptor,
    force_rebuild=False,
    **dataset_kwargs,
):
    if not Path(dataset_pkl).is_file() or force_rebuild:
        ds = dataset_class(
            split=split,
            transform=transform,
            data_pct=data_pct,
            **dataset_kwargs,
        )
        with open(dataset_pkl, "wb") as f:
            pickle.dump(ds, f, protocol=2)
            print(f"Saved dataset to: {dataset_pkl}")
    else:
        print(f"Loading dataset from: {dataset_pkl}")
        with open(dataset_pkl, "rb") as f:
            ds = pickle.load(f)
    return ds


##################################
def split_indices(n, num_shards, shuffle=False, seed=42):
    """Split the indices into num_shards parts"""
    np.random.seed(seed)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    return np.array_split(indices, num_shards)


def torch2huggingface_dataset(torch_dataset, num_shards=1, shuffle=False, seed=42):
    
    shards = split_indices(len(torch_dataset), num_shards, shuffle=shuffle, seed=seed)
    
    def gen(shard_indices):
        for idx in shard_indices:
            pixel_values, labels = torch_dataset[idx]
            yield {"pixel_values": pixel_values, "labels": labels}

    return Dataset.from_generator(
        gen, gen_kwargs={"shard_indices": shards}, 
        cache_dir="/vol/bitbucket/jq619/.cache/huggingface/datasets",
    )

# def torch2huggingface_dataset(torch_dataset, shuffle=False, seed=42):
#     shuffled_indices = np.arange(len(torch_dataset))
#     if shuffle:
#         np.random.seed(seed)
#         np.random.shuffle(shuffled_indices)

#     def gen():
#         for idx in range(len(torch_dataset)):
#             pixel_values, labels = torch_dataset[shuffled_indices[idx]]
#             yield {"pixel_values":pixel_values, "labels":labels}

#     return Dataset.from_generator(
#         gen,
#         # streaming=True,
#         cache_dir="/vol/bitbucket/jq619/.cache/huggingface/datasets",
#     )
