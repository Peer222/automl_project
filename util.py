from typing import Union, Tuple, Any, Mapping
from pathlib import Path
from argparse import Namespace

import torch.nn as nn
import config_handler

from dataclasses import dataclass
import numpy as np
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from focal_loss.focal_loss import FocalLoss
from yacs.config import CfgNode
from ConfigSpace import ConfigurationSpace, Configuration
from dataset.datasets import load_dataset


dataset_meta_type = Tuple[Tuple[int, int, int], int, Any, Any, Any]


def save_setup(
    args: Namespace, config: CfgNode, config_space: ConfigurationSpace
) -> None:
    """Saves complete setup of experiment

    Args:
        args (Namespace): Args passed from command line
        config (CfgNode): Configuration for NAS
        config_space (ConfigurationSpace): Configuration space for HPO/BOHB
    """
    with open(Path(config.save) / "args.yaml", "w") as f:
        f.write(CfgNode(vars(args)).dump())

    with open(Path(config.save) / "config.yaml", "w") as f:
        f.write(config.dump())

    config_space.to_json(Path(config.save) / "config_space.json", indent=4)


def getLoaders(
    cfg: Union[Configuration, CfgNode], resample_datasets: bool
) -> tuple[DataLoader, DataLoader, dataset_meta_type]:
    """Get DataLoader for dataset in config

    Args:
        cfg: config which contains dataset-location etc.
        resample_datasets: If true train and val are concatenated and splitted again

    Returns:
        train-loader, val-loader, metadata and datasets
    """
    dataset = cfg["dataset"]
    ds_path = cfg["datasetpath"]
    batch_size = (
        cfg.arch_search["batch_size"] if isinstance(cfg, CfgNode) else cfg["batch_size"]
    )

    augmentations = (v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(degrees=180))  # type: ignore
    dataset_meta = load_dataset(
        datadir=Path(ds_path), dataset_name=dataset, transform=augmentations
    )

    input_shape, num_classes, train_set, val_set, test_set = dataset_meta

    if resample_datasets:
        complete_set = ConcatDataset([train_set, val_set])
        train_set, val_set = random_split(complete_set, lengths=[len(train_set), len(val_set)])

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, dataset_meta


def get_class_weights(labels) -> torch.Tensor:
    """Get weights to compensate for underrepresented classes in data

    Args:
        labels (): targets which say which label the datapoint has. Used to compute class distribution

    Returns:
        weights to put into Loss function
    """
    class_counts: np.ndarray
    _, class_counts = np.unique(labels, return_counts=True)
    _class_weights: np.ndarray = 1 / (class_counts / class_counts.sum())
    return torch.tensor(_class_weights / _class_weights.mean(), dtype=torch.float)


class LogitsFocalLoss(FocalLoss):
    """Focal loss with extra wrapper to handle logits"""

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.nn.functional.softmax(x, dim=1), target)


def get_criterion(cfg_optimizer: dict[str, Any], labels) -> LogitsFocalLoss:
    """Get focal loss with smoothing and gamma set

    Args:
        labels (): labels/targets in dataset to calculate smoothing
        cfg_optimizer: config for the optimizer

    Returns:
        LogitsFocalLoss
    """
    class_weights = get_class_weights(labels)
    # smoothing is actually allowed to do nothing (cfg_optimizer["loss_class_weight_smoothing"] == 1)
    # -> given search space is contained
    smoothed_class_weights = class_weights ** cfg_optimizer[
        "loss_class_weight_smoothing"
    ] / torch.sum(class_weights ** cfg_optimizer["loss_class_weight_smoothing"])
    # gamma == 0, means normal ce. may be picked by BOHB
    # -> given search space is contained
    return LogitsFocalLoss(
        gamma=cfg_optimizer["focal_loss_gamma"], weights=smoothed_class_weights
    )


def alphas_to_probs(alphas) -> np.ndarray:
    """Convert alpha values to probabilities.
    Usually you would just use softmax,
    but we want to 'normalize' to get higher peaks/make the distribution less uniform

    Args:
        alphas (): alpha values out of darts

    Returns:
        Probability distribution
    """

    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    # alpha values are too small to be thrown directly into softmax. Resulting distribution is almost uniform
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.detach()
    normalized_alphas = np.array(alphas) / np.sum(np.abs(np.array(alphas)))
    return softmax(normalized_alphas)


def get_optimizer(
    optim_cfg: Mapping[str, Any],
    params: list[nn.Parameter],
    epochs: int,
    steps_per_epoch: int,
):
    """Create optimizer given configuration.

    Args:
        optim_cfg: optimizer configuration
        params: parameters to optimize
        epochs: how many epochs
        steps_per_epoch: how many steps per epoch

    Raises:
        ValueError: Unknown optimizer
        ValueError: Unknown scheduler

    Returns:
        optimizer and scheduler
    """
    opt_type = optim_cfg["__choice__"]
    lr = (
        optim_cfg["lr_wo_schedule"]
        if optim_cfg["lr_scheduler"] == "None"
        else optim_cfg["lr_w_schedule"]
    )
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    opt_kwargs = config_handler.get_config_for_module(optim_cfg, opt_type)
    if opt_type == "sgd":
        optimzier = torch.optim.SGD(
            lr=lr, params=params, weight_decay=weight_decay, **opt_kwargs
        )
    elif opt_type == "adam":
        optimzier = torch.optim.Adam(
            lr=lr,
            params=params,
            weight_decay=weight_decay,
            betas=(opt_kwargs["beta1"], opt_kwargs["beta2"]),
            amsgrad=opt_kwargs["amsgrad"],
        )
    elif opt_type == "adamw":
        optimzier = torch.optim.AdamW(
            lr=lr,
            params=params,
            weight_decay=weight_decay,
            betas=(opt_kwargs["beta1"], opt_kwargs["beta2"]),
            amsgrad=opt_kwargs["amsgrad"],
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}!")
    if optim_cfg["lr_scheduler"] == "None":
        lr_scheduler = None
    elif optim_cfg["lr_scheduler"] == "OneCycleLR":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimzier,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        raise ValueError(f"Unknown LRScheduler type: {optim_cfg['lr_scheduler']}")
    return optimzier, lr_scheduler


@dataclass
class StatTracker:
    avg: float = 0
    sum: float = 0
    cnt: float = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Calculate accuracy from probability distributions. Uses argmax
    Given function from template, didnt completely understand and need it

    Args:
        topk ():
        output:
        target:

    Returns:

    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1.0 / batch_size))
    return res
