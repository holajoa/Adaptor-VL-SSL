
import torchxrayvision as xrv
import skimage, torch, torchvision
import numpy as np
from typing import Union


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

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    imgs = np.array([transform(img) for img in imgs])
    imgs = torch.from_numpy(imgs)
    if return_dict:
        return {"pixel_values": imgs}
    return imgs 
