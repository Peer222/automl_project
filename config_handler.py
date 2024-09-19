from typing import Mapping, Any, Optional
from argparse import Namespace
from pathlib import Path

from ConfigSpace import (
    ConfigurationSpace,
    Integer,
    Constant,
    Categorical,
    Float,
    InCondition,
    EqualsCondition,
    NotEqualsCondition,
)
from yacs.config import CfgNode


def load_config(args: Namespace) -> CfgNode:
    """Loads configuration for darts and model

    Args:
        args (Namespace): Command line arguments

    Returns:
        CfgNode: Config
    """
    with open(args.config_file) as f:
        config = CfgNode.load_cfg(f)
        config.set_new_allowed(True)

    config.arch_search.seed = args.seed
    config.seed = args.seed  # somehow needed for plot
    config.datasetpath = str(Path(args.datasetpath).absolute())
    config.data = str(Path(args.datasetpath).resolve())
    config.device = args.device

    results_dir = Path(f"results/{config.name}/{config.seed}")
    results_dir.mkdir(exist_ok=True, parents=True)
    (results_dir / "search").mkdir(exist_ok=True)

    config.save = str(results_dir)

    return config


def get_model_config(args: Namespace) -> dict[str, Any]:
    """Get cnn model configuration for NAS/ BOHB

    Args:
        args (Namespace): Command line arguments

    Returns:
        dict[str, Any]: Model configuration
    """
    config_space = get_full_search_space(args)
    config = config_space.get_default_configuration()
    model_config = get_config_for_module(config, "model")
    # input and output shapes of model have to be added
    model_config["num_classes"] = config["num_classes"]
    model_config["input_channels"] = config["input_channels"]

    return model_config


def get_full_search_space(
    args: Namespace,
) -> ConfigurationSpace:
    """Getter for full configuration space

    Args:
        args: (Namespace): Command line arguments

    Returns:
        ConfigurationSpace: Full configuration space
    """
    config = load_config(args)

    cs = ConfigurationSpace(
        {
            "performance_measure": Constant(
                "performance_measure", config.performance_measure
            ),
            "max_budget": Constant("max_budget", args.max_budget),
            "save": Constant(
                "save", str(Path(config.save) / "bohb" / str(config.seed))
            ),
            "batch_size": Integer(
                "batch_size", (16, 256), default=config.arch_search.batch_size, log=True
            ),
            "device": Constant("device", config.device),
            "dataset": Constant("dataset", config.dataset),
            "datasetpath": Constant("datasetpath", config.datasetpath),
            "input_channels": Constant(
                "input_channels", config.dataset_meta.num_img_channels
            ),
            "num_classes": Constant("num_classes", config.dataset_meta.num_classes),
        }
    )

    cs_model = get_model_search_space(config)
    cs.add_configuration_space("model", cs_model)
    cs_optimizer = get_optimizer_search_space(args.seed)
    cs.add_configuration_space("optimizer", cs_optimizer)

    return cs


def get_model_search_space(config: CfgNode) -> ConfigurationSpace:
    """Get search space mainly used for model architecture

    Args:
        config: full config to read from

    Returns:
        ConfigurationSpace for model
    """
    cs_model = ConfigurationSpace(
        {
            "n_cells": Integer("n_cells", (1, 6), default=config.model.n_cells),
            "use_BN": Categorical("use_BN", [True, False], default=True),
            "global_avg_pooling": Categorical(
                "global_avg_pooling", [True], default=True
            ),
            "n_fc_layers": Integer(
                "n_fc_layers", (1, 3), default=config.model.n_fc_layers
            ),
            "dropout_rate": Float(
                "dropout_rate", (0.0, 0.9), default=config.model.dropout_rate, log=False
            ),
            "input_channels": Constant(
                "input_channels", config.dataset_meta.num_img_channels
            ),
            "num_classes": Constant("num_classes", config.dataset_meta.num_classes),
            "use_surrogate_in_bohb": Categorical(
                "use_surrogate_in_bohb", [True, False], default=False
            ),
        }
        | {
            f"n_out_features_fc_{i + 1}": Integer(
                f"n_out_features_fc_{i + 1}",
                (8, 256),
                default=config.model.fc_out_features[i],
                log=True,
            )
            for i in range(3)
        },
        seed=config.seed,
    )
    # Add conditions to restrict the hyperparameter space

    use_fc_layer_2 = InCondition(
        cs_model["n_out_features_fc_3"], cs_model["n_fc_layers"], [3]
    )
    use_fc_layer_1 = InCondition(
        cs_model["n_out_features_fc_2"], cs_model["n_fc_layers"], [2, 3]
    )

    # Add multiple conditions on hyperparameters at once:
    cs_model.add([use_fc_layer_2, use_fc_layer_1])

    # add HPs for each individual blocks
    for i in range(1, cs_model["n_cells"].upper + 1):  # type: ignore
        cs_cell = get_cell_search_space(config, i)
        cs_model.add_configuration_space(
            prefix=f"block{i}", configuration_space=cs_cell
        )

    return cs_model


def get_cell_search_space(config: CfgNode, cell_index: int) -> ConfigurationSpace:
    """Getter for config space of cell

    Args:
        config (CfgNode): Configuration
        cell_index (int): Cell index [1, ...]

    Returns:
        ConfigurationSpace: Cell configuration space
    """
    cs = ConfigurationSpace(seed=config.seed)
    n_channels = Integer(
        "out_channels",
        (8, 256),
        default=config.model.cell_out_channels[cell_index - 1],
        log=True,
    )
    use_bn = Categorical(
        "use_BN", (True, False), default=config.model.use_BN[cell_index - 1]
    )
    # max_pooling can be applied 4 times at most (28 -> 14 -> 7 -> 3 -> 1)
    if cell_index <= 4:
        use_max_pooling = Categorical(
            "use_max_pooling",
            (True, False),
            default=config.model.use_max_pooling[cell_index - 1],
        )
    else:
        use_max_pooling = Constant("use_max_pooling", False)

    cs.add([n_channels, use_bn, use_max_pooling])
    return cs


def get_optimizer_search_space(seed: int = 0) -> ConfigurationSpace:
    """
    A function to define a NN optimizer. Here we provide 3 optimizers: sgd, adam, adamw.
    Since all the optimizers have their own hyperparameters, we define a hierarchical search space. This is only an
    example showing how to construct such search space. Feel free to extend the search space
    :param seed:
    :return: a configuration space for optimizers
    """
    cs_optimizer = ConfigurationSpace(seed=seed)
    # first, we provide three types of optimizers
    optimizer_choice = Categorical(
        "__choice__", ["sgd", "adam", "adamw"], default="adamw"
    )
    # the hp learning rate and weight deacay are shared across all the models.
    # You could also ask each optimizer to have their own lr
    lr_wo_schedule = Float(
        "lr_wo_schedule",
        (1e-5, 1.0),
        default=1e-5,
        log=True,
    )
    lr_scheduler = Categorical(
        "lr_scheduler", ("None", "OneCycleLR"), default="OneCycleLR"
    )
    lr_w_schedule = Float("lr_w_schedule", (1e-5, 1e-1), default=1e-5, log=True)
    loss_class_weight_smoothing = Float(
        "loss_class_weight_smoothing", (0.0, 1.0), default=0.8
    )
    focal_loss_gamma = Float("focal_loss_gamma", (0.0, 5.0), default=3.0)
    use_wd = Categorical("use_wd", (True, False), default=False)
    weight_decay = Float("weight_decay", (1e-8, 0.1), default=1e-8, log=True)
    cs_optimizer.add(
        [
            optimizer_choice,
            loss_class_weight_smoothing,
            focal_loss_gamma,
            lr_scheduler,
            lr_wo_schedule,
            lr_w_schedule,
            use_wd,
            weight_decay,
        ]
    )
    wd_condition = EqualsCondition(weight_decay, use_wd, True)
    cs_optimizer.add(wd_condition)
    lr_w_schedule_condition = NotEqualsCondition(lr_w_schedule, lr_scheduler, "None")
    lr_wo_schedule_condition = EqualsCondition(lr_wo_schedule, lr_scheduler, "None")
    cs_optimizer.add(lr_w_schedule_condition)
    cs_optimizer.add(lr_wo_schedule_condition)

    # Now we show how to add the search space for each individual module
    # feel free to update these search spaces
    cs_opts = {
        "sgd": get_sgd_search_space(seed),
        "adam": get_adam_search_space(seed),
        "adamw": get_adamw_search_space(seed),
    }

    for opt_name, cs_opt in cs_opts.items():
        # adding the sub configuration space to the current search space
        # This is equivalent to add all the hps from cs_opt to cs where the hps are conditioned on the parent hp's value
        cs_optimizer.add_configuration_space(
            opt_name,
            cs_opt,
            parent_hyperparameter={"parent": optimizer_choice, "value": opt_name},
        )
    return cs_optimizer


def get_sgd_search_space(seed: int = 0):
    # sgd: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    cs_sgd = ConfigurationSpace(seed=seed)
    cs_sgd.add(
        [
            Float("momentum", (0.0, 0.9999), default=0.0),
            Categorical("nesterov", (True, False), default=False),
        ]
    )
    return cs_sgd


def get_adam_search_space(seed: int = 0):
    # adam: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    cs_adam = ConfigurationSpace(seed=seed)
    cs_adam.add(
        [
            Float("beta1", (0.5, 0.9999), default=0.5),
            Float("beta2", (0.9, 0.9999), default=0.9),
            Categorical("amsgrad", (True, False), default=False),
        ]
    )
    return cs_adam


def get_adamw_search_space(seed: int = 0):
    # adamw: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    cs_adamw = ConfigurationSpace(seed=seed)
    cs_adamw.add(
        [
            Float("beta1", (0.5, 0.9999), default=0.5),
            Float("beta2", (0.9, 0.9999), default=0.9),
            Categorical("amsgrad", (True, False), default=False),
        ]
    )
    return cs_adamw


def get_config_for_module(cfg: Mapping[str, Any], module_name: str) -> dict[str, Any]:
    """
    This function is used to extract a sub configuration that belongs to a certain module
    Note that this function needs to call for each level
    :param cfg: a configuration
    :param module_name: the module name
    :return: cfg_module: a new dict that contains all the hp values belonging to the configuration
    """
    cfg_module = {}
    for key, value in cfg.items():
        if key.startswith(module_name):
            new_key = key.replace(f"{module_name}:", "")
            cfg_module[new_key] = value
    return cfg_module
