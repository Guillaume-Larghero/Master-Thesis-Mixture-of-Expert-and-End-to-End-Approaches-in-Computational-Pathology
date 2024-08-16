from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def get_transforms(train=False):
    """
    Takes a list of images and applies the same augmentations to all of them.
    This is completely overengineered but it makes it easier to use in our pipeline
    as drop-in replacement for torchvision transforms.

    ## Example

    ```python
    imgs = [Image.open(f"image{i}.png") for i in range(1, 4)]
    t = get_transforms(train=True)
    t_imgs = t(imgs) # List[torch.Tensor]
    ```

    For the single image case:

    ```python
    img = Image.open(f"image{0}.png")
    # or img = np.load(some_bytes)
    t = get_transforms(train=True)
    t_img = t(img) # torch.Tensor
    ```
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    _data_transform = None

    def _get_transform(n: int = 3):
        if train:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.OneOf(
                        [
                            A.Rotate(limit=0, p=1),
                            A.Rotate(limit=90, p=1),
                            A.Rotate(limit=180, p=1),
                            A.Rotate(limit=270, p=1),
                        ],
                        p=0.5,
                    ),
                    A.Compose(
                        [
                            A.OneOf(
                                [
                                    A.ColorJitter(
                                        brightness=(0.9, 1),
                                        contrast=(0.9, 1),
                                        saturation=(0.9, 1),
                                        hue=(0, 0.1),
                                        p=1.0,
                                    ),
                                    A.Affine(
                                        scale=(0.5, 1.5),
                                        translate_percent=(0.0, 0.0),
                                        shear=(0.5, 1.5),
                                        p=1.0,
                                    ),
                                ],
                                p=0.5,
                            ),
                            A.GaussianBlur(
                                blur_limit=(1, 3), sigma_limit=(0.1, 3), p=1.0
                            ),
                        ]
                    ),
                    A.OneOf(
                        [
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                        ],
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},
            )
        else:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},
            )
        return data_transforms

    def transform_images(images: any):
        nonlocal _data_transform

        if not isinstance(images, list):
            n = 1
            images = [images]
        else:
            n = len(images)
        if _data_transform is None:
            # instantiate once
            _data_transform = _get_transform(n)

        # accepts both lists of np.Array and PIL.Image
        if isinstance(images[0], Image.Image):
            images = [np.array(img) for img in images]

        image_dict = {"image": images[0]}
        for i in range(1, n):
            image_dict[f"image{i}"] = images[i]

        transformed = _data_transform(**image_dict)
        transformed_images = [
            transformed[key] for key in transformed.keys() if "image" in key
        ]

        if len(transformed_images) == 1:
            return transformed_images[0]
        return transformed_images

    return transform_images


def pad_to_square(image, desired_size):
    old_size = image.size
    delta_w = desired_size - old_size[0]
    delta_h = desired_size - old_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return F.pad(image, padding, 0, 'constant')

def get_transforms_SC(train=False):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train:
        data_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda image: pad_to_square(image, 224)),
                transforms.Resize(224),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(0),
                        transforms.RandomRotation(90),
                        transforms.RandomRotation(180),
                        transforms.RandomRotation(270),
                    ]
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        data_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda image: pad_to_square(image, 224)),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return data_transforms
