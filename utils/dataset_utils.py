import torchvision.transforms as transforms
import torchxrayvision as xrv
import torch
import numpy as np
from transformers import ViTImageProcessor

from typing import Union

from pathlib import Path

from datasets.dataset import MultimodalPretrainingDatasetForAdaptor
from mgca.datasets.transforms import DataTransforms

import pickle


class AutoEncoderDataTransforms(object):
    def __init__(self, is_train:bool=True, crop_size:int=224):
        self.crop_size = crop_size
        self.data_transforms = transforms.Compose([
            BatchedXRayCenterCrop(),
            xrv.datasets.XRayResizer(self.crop_size), 
        ])
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
        return img[..., starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)
  
    
def ae_image_processor(imgs:np.ndarray, return_dict=True) \
        -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    imgs = xrv.datasets.normalize(imgs, 255) 

    # Check that images are 2D arrays
    if len(imgs.shape) > 3:
        imgs = imgs[:, :, :, 0]
    if len(imgs.shape) < 3:
        print("error, dimension lower than 2 for image")

    # Add color channel
    imgs = imgs[:, None, :, :]

    transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                    xrv.datasets.XRayResizer(224)])
    imgs = np.array([transform(img) for img in imgs])
    imgs = torch.from_numpy(imgs)
    if return_dict:
        return {"pixel_values": imgs}
    return imgs 


def timm_image_processor(imgs:np.ndarray) -> torch.Tensor:
    return ViTImageProcessor()(imgs, return_tensors="pt", return_dict=True)

def pickle_dataset(dataset_pkl, split, transform, data_pct, force_rebuild=False):
    if not Path(dataset_pkl).is_file() or force_rebuild:
        dataset = MultimodalPretrainingDatasetForAdaptor(
            split=split, 
            transform=transform, 
            data_pct=data_pct, 
        )
        with open(dataset_pkl, "wb") as f:
            pickle.dump(dataset, f, protocol=2)
            print(f"Saved dataset to: {dataset_pkl}")
    else:
        print(f'Loading dataset from: {dataset_pkl}')
        with open(dataset_pkl, "rb") as f:
            dataset = pickle.load(f)
    
    return dataset
     