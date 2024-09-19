"""
===========================
Optimization using BOHB and NAS
===========================
"""

from __future__ import annotations

import logging
from argparse import Namespace
import random

import torch
import torch.nn as nn
from functools import reduce, partial
import time

from pathlib import Path
import json

from yacs.config import CfgNode
from ConfigSpace import (
    Configuration,
    Categorical,
    GreaterThanCondition,
)
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity
from smac.scenario import Scenario
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary
from dask.distributed import get_worker
import numpy as np

from util import alphas_to_probs, get_optimizer

from naslib.optimizers import DARTSOptimizer
from naslib.defaults.trainer import Trainer

from nas.search_space import NASNetwork, save_alphas
import config_handler
import args_handler
import util


logger = logging.getLogger(__name__)

BUDGET_TYPE = "epoch"
SAMPLED_ALPHAS_PER_LEVEL = 5

RESAMPLE_TRAIN_VAL_SET = True

best_model = None
best_performance = np.inf


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
# This is specific to SMAC
def cnn_from_cfg(
    cfg: Configuration,
    seed: int,
    budget: float,
    eval_test: bool = False,
    final_training: bool = False,
) -> tuple[float, dict]:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    :param eval_test: bool
        if we would like to evaluate the test sets
    :param final_training: bool
        on final run we additionally train on validation set
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    device = cfg["device"]
    batch_size = cfg["batch_size"]

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    train_loader, val_loader, dataset_meta = util.getLoaders(cfg, resample_datasets=RESAMPLE_TRAIN_VAL_SET)
    input_shape, num_classes, train_set, val_set, test_set = dataset_meta
    if final_training:  # use val set additionally for training if it is the final model
        train_loader = DataLoader(
            ConcatDataset((train_set, val_set)),
            batch_size=batch_size,
            shuffle=True,
        )

    # Create model from sampled block alphas
    relevant_alphas = [
        cfg[f"model:cell:alphas_{i}"] for i in range(1, cfg["model:n_cells"] + 1)
    ]
    all_alphas: dict[str, list[float]] = reduce(lambda a, b: a | b, relevant_alphas)

    model_config = config_handler.get_config_for_module(cfg, "model")
    model = NASNetwork.create_from_alphas(all_alphas, model_config, cfg["dataset"])
    model.discretize_graph()
    model = model.to(model_device)

    model_summary = summary(model, input_size=(1, *input_shape), device=model_device)

    # train
    cfg_optimizer = config_handler.get_config_for_module(cfg, "optimizer")
    n_epoch = int(budget) if BUDGET_TYPE == "epoch" else 50
    optimizer, lr_scheduler = get_optimizer(cfg_optimizer, model.parameters(), epochs=n_epoch, steps_per_epoch=len(train_loader))  # type: ignore
    train_criterion = util.get_criterion(cfg_optimizer, train_set.labels)

    results_dir = Path(cfg["save"])

    train_accs = []
    train_losses = []

    n_epoch = int(budget) if BUDGET_TYPE == "epoch" else 50
    for epoch in range(n_epoch):
        if model_config["use_surrogate_in_bohb"]:
            train_frac = (epoch + 1) / n_epoch
            model.set_member_rec("surrogate_frac", (1 - train_frac) / 2)
        logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{budget}]")
        train_score, train_loss = model.train_fn(
            optimizer=optimizer,
            criterion=train_criterion,
            lr_scheduler=lr_scheduler,
            loader=train_loader,
            device=model_device,
        )

        train_accs.append(train_score)
        train_losses.append(train_loss)

        logging.info(
            f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}"
        )
    # "delete" surrogate frac after using it for training
    if model_config["use_surrogate_in_bohb"]:
        model.set_member_rec("surrogate_frac", 0)

    # Evaluate
    eval_info = model.eval_fn(val_loader, train_criterion, device)
    logging.info(f"Worker:{worker_id} => Val infos: {eval_info}")

    results = 1 - eval_info[cfg["performance_measure"]]

    # save models with potential TODO maybe use val_acc threshold instead?
    # This seems to be not possible to load right now. Due to time constraints there is no time to fix.
    if n_epoch == cfg["max_budget"]:
        models_dir = results_dir / "models"
        models_dir.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), models_dir / f"{hash(cfg)}.pth")

    data = {
        "n_params": model_summary.total_params / 1e6,
        "flops": model_summary.total_mult_adds / 1e9,
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_loss": eval_info["loss"],
        "val_acc": eval_info["acc"],
        "val_cw_acc": eval_info["cw_accuracies"],
        "val_f1_score": eval_info["f1"],
        # "val_roc_auc_score": eval_info["roc_auc"]  # is broken right now, no time to fix
    }

    if eval_test:  # Evaluate on test set (not only val)
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
        )
        test_info = model.eval_fn(test_loader, train_criterion, device)
        logging.info(f"Worker:{worker_id} => Test accuracy {test_info['acc']:.3f}")
        data["test_loss"] = test_info["loss"]
        data["test_acc"] = test_info["acc"]
        data["test_cw_acc"] = test_info["cw_accuracies"]
        data["test_f1_score"] = test_info["f1"]
        # data["test_roc_auc_score"] = test_info["roc_auc"]

    # save best model to save on retraining costs later (directly evaluate on test)
    global best_performance
    global best_model
    if best_performance > results:
        best_performance = results
        best_model = model

    # the additional information is used for MO optimization
    # TODO in our case it shouldn't be a problem to use it for logging because no multi-objective optimization is used
    return results, data


def bohbOptimize(args, base_graph: NASNetwork, time_limit: int):
    config = config_handler.load_config(args)
    configspace = config_handler.get_full_search_space(args)

    # Sample alphas per block level
    base_alphas = {k: v for k, v in base_graph.get_op_data("alpha").items() if len(v) > 0}  # type: ignore
    for block_idx in range(1, configspace["model:n_cells"].upper + 1):  # type: ignore

        def sample_one_hot_alpha() -> dict[str, list[int]]:
            """Samples one hot vector from given alphas
            The alphas are use to create a probability distribution

            Returns:
                dict which maps graph nodes to alphas, which are one hot encoded (sampled from distribution based on alphas)
            """

            def one_hot_from_alphas(alphas: torch.Tensor) -> list[int]:
                one_hot_vector: list[int] = [0] * len(alphas)
                one_hot_vector[
                    np.random.choice(len(alphas), p=alphas_to_probs(alphas))
                ] = 1
                return one_hot_vector

            return {f"block_{block_idx}_{k}": one_hot_from_alphas(alphas) for k, alphas in base_alphas.items()}  # type: ignore

        layer_categorical = Categorical(f"model:cell:alphas_{block_idx}", [sample_one_hot_alpha() for _ in range(SAMPLED_ALPHAS_PER_LEVEL)])  # type: ignore
        configspace.add(layer_categorical)

        # only use condition when enough cells have been selected
        if (
            block_idx - 1
            > 0  # work around questionable ConfigSpace design decisions (doesn't support > 0)
        ):
            use_conv_layer_n = GreaterThanCondition(
                configspace[f"model:cell:alphas_{block_idx}"],
                configspace["model:n_cells"],
                block_idx - 1,
            )
            configspace.add(use_conv_layer_n)

    # Setting up SMAC to run BOHB
    scenario = Scenario(
        name="bohb",  # "ExampleMFRunWithBOHB",
        configspace=configspace,
        deterministic=True,
        output_directory=config.save,
        seed=args.seed,
        n_trials=get_n_trials_for_hyperband_multifidelity(total_budget=args.total_budget, min_budget=args.min_budget, max_budget=args.max_budget, eta=args.eta, print_summary=True),
        max_budget=args.max_budget,
        min_budget=args.min_budget,
        n_workers=args.workers,
        walltime_limit=time_limit,
        use_default_config=True,
    )

    # You can mess with SMACs own hyperparameters here (checkout the documentation at https://automl.github.io/SMAC3)
    smac = SMAC4MF(
        target_function=cnn_from_cfg,
        scenario=scenario,
        initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=5),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_observable_budget",
            eta=args.eta,
        ),
        overwrite=True,
        logging_level=args.log_level,  # https://automl.github.io/SMAC3/main/advanced_usage/8_logging.html
    )

    # Start optimization
    return smac.optimize()


def dartsOptimize(args: Namespace):
    print(f"args: {args}")

    config = config_handler.load_config(args)
    cfg_optimizer = config_handler.get_config_for_module(
        config_handler.get_full_search_space(args).get_default_configuration(),
        "optimizer",
    )

    print("config", config.dump())

    train_queue, val_queue, dataset_meta = util.getLoaders(config, resample_datasets=RESAMPLE_TRAIN_VAL_SET)

    search_space = NASNetwork("network", config_handler.get_model_config(args))
    optimizer = DARTSOptimizer(**config.arch_search)

    _, _, train_set, _, _ = dataset_meta
    train_criterion = util.get_criterion(cfg_optimizer, train_set.labels)
    optimizer.adapt_search_space(
        search_space,
        config.dataset,
        loss=train_criterion,
    )
    optimizer.graph.print()

    trainer_cfg = config.clone()
    trainer_cfg.merge_from_other_cfg(CfgNode({"search": config.arch_search}))
    trainer = Trainer(optimizer, trainer_cfg)

    def build_search_dataloaders(_: dict) -> tuple[DataLoader, DataLoader, None]:
        return train_queue, val_queue, None

    trainer.build_search_dataloaders = build_search_dataloaders  # type: ignore
    trainer.search(
        after_epoch=partial(save_alphas, graph=optimizer.graph, config=config)
    )  # Search for an architecture

    return optimizer.graph


if __name__ == "__main__":
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the forbidden clauses.
    """
    args = args_handler.parseArgs()
    logging.basicConfig(level="INFO")
    config = config_handler.load_config(args)

    util.save_setup(
        args,
        config_handler.load_config(args),
        config_handler.get_full_search_space(args),
    )

    # Seeding. Some seeds are reset later to args.seed, but this doesn't change that everything is seeded
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Start NAS
    nas_start_time = time.time()
    final_graph_w_alphas = dartsOptimize(args)
    nas_total_time = int(time.time() - nas_start_time)
    logger.info(f"Total time taken by NAS in minutes: {nas_total_time / 60}")

    # Start BOHB
    bohb_time_limit = args.runtime - nas_total_time
    logger.info(f"Time remaining for bohb in minutes: {bohb_time_limit / 60}")
    incumbents = bohbOptimize(args, final_graph_w_alphas, bohb_time_limit)
    if not isinstance(incumbents, list):
        incumbents = [incumbents]

    # save incumbents as json
    with open(Path(config.save) / "final_config.json", "w") as f:
        incumbent_dicts = [dict(incumbent) for incumbent in incumbents]
        json.dump(incumbent_dicts, f)

    # Evaluate on test set
    print("------------------ TESTING")
    print(f"{incumbents=}")
    cfg_optimizer = config_handler.get_config_for_module(incumbents[-1], "optimizer")
    train_loader, val_loader, dataset_meta = util.getLoaders(incumbents[-1], resample_datasets=RESAMPLE_TRAIN_VAL_SET)
    input_shape, num_classes, train_set, val_set, test_set = dataset_meta
    train_criterion = util.get_criterion(cfg_optimizer, train_set.labels)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=incumbents[-1]["batch_size"],
        shuffle=False,
    )
    test_info = best_model.eval_fn(
        test_loader, train_criterion, incumbents[-1]["device"]
    )
    logging.info(f"Test accuracy {test_info['acc']:.3f}")

    print(f"{test_info=}")
