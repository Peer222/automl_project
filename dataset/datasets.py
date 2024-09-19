"""
Data needed for the project. The data is not included in the repository due to its size.
The original data can be downloaded from the following link:
https://www.kaggle.com/datasets/coreylammie/deepweedsx

We split it for easier use of dataloaders.
We additionally provide a simplified version that does not have an imbalanced class for
ease of testing your code.
Please download the processed data from:
ml.informatik.uni-freiburg.de/~biedenka/dwx_compressed.tar.gz
"""

from __future__ import annotations

import os
from typing import Callable, Iterable
from typing_extensions import TypeAlias

import logging
from pathlib import Path

import medmnist

from torchvision import transforms
from torchvision.datasets import ImageFolder


logger = logging.getLogger(__name__)

HERE = Path(__file__).absolute().parent / "data"

Dimensions: TypeAlias = tuple[int, int, int]


def load_dataset(
    datadir: Path,
    dataset_name: str,
    resize: tuple[int, int] = (28, 28),
    transform: Iterable[Callable] = (),
):
    """Load the desired dataset.

    :param datadir:
        the path to the dataset
    :param dataset_name:
        dataset nanme
    :param resize:
        What to resize the image to
    :param transform:
        The transformation to apply to images.

    :return: The train and test datasets.
    """
    # Original dimensions are 3x256x256

    img_width, img_height = resize
    pre_processing = transforms.Compose(
        [
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(),
            *transform,
        ]
    )
    print(f"{pre_processing=}")

    if dataset_name.endswith("mnist"):
        info = medmnist.INFO[dataset_name]
        dataset_name = info["python_class"]
        dataset_type = getattr(medmnist, dataset_name)
        dataset_path = datadir / dataset_name
        if not dataset_path.exists():
            os.makedirs(dataset_path)
        target_transform = transforms.Lambda(lambda x: x.item())
        train_dataset = dataset_type(
            split="train",
            download=True,
            transform=pre_processing,
            target_transform=target_transform,
            root=dataset_path,
        )
        val_dataset = dataset_type(
            split="val",
            download=True,
            transform=pre_processing,
            target_transform=target_transform,
            root=dataset_path,
        )
        test_dataset = dataset_type(
            split="test",
            download=True,
            transform=pre_processing,
            target_transform=target_transform,
            root=dataset_path,
        )
        dimensions = (info["n_channels"], img_width, img_height)
        num_classes = len(info["label"])
        return dimensions, num_classes, train_dataset, val_dataset, test_dataset

    dimensions = (3, img_width, img_height)

    dataset_dir = datadir / dataset_name

    train_path = dataset_dir / "train"
    val_path = dataset_dir / "val"
    test_path = dataset_dir / "test"

    train_dataset = ImageFolder(root=str(train_path), transform=pre_processing)
    val_dataset = ImageFolder(root=str(val_path), transform=pre_processing)
    test_dataset = ImageFolder(root=str(test_path), transform=pre_processing)
    num_classes = len(train_dataset.classes)

    return dimensions, num_classes, train_dataset, val_dataset, test_dataset
