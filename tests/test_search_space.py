from argparse import Namespace

import args_handler
import config_handler
from NASLib.naslib.optimizers import DARTSOptimizer
from nas.search_space import NASNetwork, OpsWithSkip, NASCell
from typing import Union

from util import getLoaders, get_criterion

import pytest


def get_demo_args() -> Namespace:
    return Namespace(config_file='configs/acc_config.yaml', seed=1, datasetpath='data', dataset='retinamnist',
                     device='cpu', max_budget=20)


def get_demo_model_config():
    cfg = config_handler.get_full_search_space(args).get_default_configuration()

    return config_handler.get_config_for_module(cfg, "model")


args = get_demo_args()
model_config = get_demo_model_config()


def test_node_structure():
    config = config_handler.load_config(args)

    search_space = NASNetwork("network", config_handler.get_model_config(args))
    optimizer = DARTSOptimizer(**config.arch_search)

    cfg_optimizer = config_handler.get_config_for_module(
        config_handler.get_full_search_space(args).get_default_configuration(),
        "optimizer",
    )

    train_queue, val_queue, dataset_meta = getLoaders(config)
    _, _, train_set, _, _ = dataset_meta

    train_criterion = get_criterion(cfg_optimizer, train_set.labels)

    optimizer.adapt_search_space(
        search_space,
        config.dataset,
        loss=train_criterion,
    )

    for i in range(2, 4+1):
        node = optimizer.graph.nodes[i]
        assert node['subgraph'].scope == f"block_{i-1}"
        assert type(node['subgraph']) is NASCell
        for j in range(2, 7+1):
            opnode = node['subgraph'].nodes[j]
            assert opnode['subgraph'].scope == f"block_{i-1}"
            assert type(opnode['subgraph']) is OpsWithSkip


def test_create_from_alphas():
    alphas: dict[str, list[float]] = {
        f"block_{i}_ops_{j}": [0.3, 0.2, 0.2, 0.2, 0.1]
        for i in range(1, 3 + 1) for j in range(2, 7 + 1)
    }

    newgraph = NASNetwork.create_from_alphas(alphas, model_config, "retinamnist")

    vals_per_node = {}
    for g in filter(
            lambda g: g.scope.startswith("block_") and isinstance(g, OpsWithSkip),
            newgraph._get_child_graphs(),
    ):
        g: OpsWithSkip
        vals_per_node[f"{g.scope}_{g.name}"] = g.get_all_edge_data(key='alpha')[0].tolist()

    assert set(vals_per_node.keys()) == set(alphas.keys())
    for key in vals_per_node.keys():
        assert vals_per_node[key] == pytest.approx(alphas[key])
