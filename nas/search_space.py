from typing import Union, Mapping, Any, Optional
from argparse import Namespace

from naslib.optimizers.oneshot.darts.optimizer import DARTSMixedOp
from naslib.search_spaces.core.graph import Graph, EdgeData
import numpy as np
import torch.nn as nn
import torch
from naslib.optimizers import DARTSOptimizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import time
from util import StatTracker, accuracy, alphas_to_probs
from tqdm import tqdm
from pathlib import Path
from yacs.config import CfgNode
import csv
from dask.distributed import get_worker

from .modules import (
    IdentityBlock,
    ZeroBlock,
    MaxPoolingBlock,
    ConvBlock,
    ResNetConvBlock,
    ResNetBottleneckConvBlock,
    FinalBlock,
)
import logging

logger = logging.getLogger(__name__)


class OpsWithSkip(Graph):
    OPTIMIZER_SCOPE = ["block_1", "block_2", "block_3", "all"]

    def __init__(
        self,
    ):
        """Builds structure of cell node.
        1. Add name attribute to object separately
        2. Call self.insert_ops
        3. Set input node
        """
        super().__init__()

        # input node
        self.add_node(1)

        # helper node for surrogate skip
        self.add_node(2)

        # output node
        self.add_node(3)

        # Edge for actual operations
        self.add_edge(1, 3)

        # Edges for surrogate skip
        self.add_edges_from(
            [
                (1, 2, EdgeData().finalize()),
                (2, 3),
            ]
        )

        # Combination in output node
        self.nodes[3]["comb_op"] = self.scheduled_skip

    def insert_ops(
        self, in_channels: int, out_channels: int, kernel_size: int
    ) -> "OpsWithSkip":
        """Adds operations to edges with respect to position in graph and cell

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for conv layers

        Returns:
            OpsWithSkip: OpsWithSkip object (self) with inserted operations
        """
        self.operations = [
            IdentityBlock(in_channels, out_channels),
            ZeroBlock(1, C_in=in_channels, C_out=out_channels),
            ConvBlock(in_channels, out_channels, kernel_size=kernel_size),
            ResNetConvBlock(in_channels, out_channels, kernel_size=kernel_size),
            ResNetBottleneckConvBlock(
                in_channels, out_channels, kernel_size=kernel_size
            ),
        ]

        op_edge_data: EdgeData = self.get_edge_data(1, 3)  # type: ignore
        op_edge_data.set("op", self.operations)

        skip_edge_data: EdgeData = self.get_edge_data(2, 3)  # type: ignore
        skip_edge_data.set("op", IdentityBlock(in_channels, out_channels))
        skip_edge_data.finalize()
        self.nodes[3]["comb_op"] = self.scheduled_skip

        return self

    def scheduled_skip(self, tensors):
        """Surrogate skip for every operation based on Darts- Paper

        Args:
            tensors (): two entries: first -> normal operation, second -> skip connection

        Returns:
            weighted average of skip and operation output
        """
        assert len(tensors) == 2
        return (1 - self.surrogate_frac) * tensors[0] + self.surrogate_frac * tensors[1]

    def print(self, prefix=""):
        graph_print(self, prefix=prefix)

    def set_alpha(self, alphas: nn.parameter.Parameter) -> None:
        # Set alphas for operation used by darts
        op_edge_data: EdgeData = self.get_edge_data(1, 3)  # type: ignore
        op_edge_data.set("alpha", alphas, overwrite=True)

class NASCell(Graph):
    OPTIMIZER_SCOPE = ["block_1", "block_2", "block_3", "all"]

    QUERYABLE = False

    def __init__(self) -> None:
        """Builds cell structure without operations
        1. Add name attribute to object separately
        2. Copy object
        3. Call insert_ops
        4. Set scope
        5. Set input node
        """
        super().__init__()

        # input node (nas uses two input nodes from previous and previous-previous nodes)
        self.add_node(1)

        # nas cells
        # first node stage
        for index in range(2, 6):
            ops = OpsWithSkip()
            ops.name = f"ops_{index}"
            self.add_node(index, subgraph=ops)

        # second node stage
        for index in [6, 7]:
            ops = OpsWithSkip()
            ops.name = f"ops_{index}"
            self.add_node(index, subgraph=ops)

        # output node
        # TODO by default comb_op is sum but nasbench301 uses a concat operation?
        self.add_node(8)

        self.add_node(9)

        # Actual operations are in the nodes, only use identity
        self.add_edges_from([(1, i, EdgeData().finalize()) for i in [2, 3, 4, 5]])
        self.add_edges_from([(4, 6, EdgeData().finalize())])
        self.add_edges_from([(5, 7, EdgeData().finalize())])
        self.add_edges_from([(i, 8, EdgeData().finalize()) for i in [2, 3, 6, 7]])
        # max pooling edge
        self.add_edge(8, 9)

    def insert_ops(self, config: Mapping[str, Any], stage: int) -> "NASCell":
        """Adds operations to edges with respect to model config and position in graph

        Args:
            config (Mapping[str, Any]): Model config
            stage (int): Position in graph (e.g. 2 for block_2)

        Returns:
            NASCell: NASCell object (self) with inserted operations
        """
        self.config = config
        self.stage = stage

        if stage == 1:
            in_channels: int = self.config.get("input_channels", 3)
        else:
            in_channels = self.config[f"block{stage - 1}:out_channels"]
        out_channels: int = self.config[f"block{stage}:out_channels"]

        # nas cells
        # first node stage
        node_indices = [2, 3, 4, 5]
        kernel_sizes = [3, 5, 7, 9]
        for i, k in zip(node_indices, kernel_sizes):
            self.nodes[i]["subgraph"].insert_ops(
                in_channels, out_channels, k
            ).set_input([1])

        # second node stage
        node_indices = [6, 7]
        kernel_sizes = [3, 3]
        for i, k, input in zip(node_indices, kernel_sizes, [4, 5]):
            self.nodes[i]["subgraph"].insert_ops(
                out_channels, out_channels, k
            ).set_input([input])

        # Pooling
        pool_edge_data: EdgeData = self.get_edge_data(8, 9)  # type: ignore
        pool_edge_data.set(
            "op",
            (
                MaxPoolingBlock(kernel_size=2, stride=2)
                if config[f"block{stage}:use_max_pooling"]
                else IdentityBlock(out_channels, out_channels)
            ),
        )
        pool_edge_data.finalize()

        return self

    def print(self, prefix=""):
        graph_print(self, prefix=prefix)


class NASNetwork(Graph):
    OPTIMIZER_SCOPE = "all"

    QUERYABLE = False

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        """Creates cell search space / network / graph for darts optimization

        Args:
            name (str): Name of graph
            config (Mapping[str, Any]): Model config for graph construction
        """
        super().__init__(name)
        self.scope = self.OPTIMIZER_SCOPE
        self.config = config

        self.time_train = 0
        try:
            self.my_worker_id = get_worker().name
        except ValueError:
            self.my_worker_id = 0

        # input node
        self.add_node(1)

        # intermediate nodes
        cell = NASCell()
        cell.name = "cell"

        index = 1
        for _ in range(1, config["n_cells"] + 1):
            self.add_node(
                index + 1,
                subgraph=cell.copy()
                .insert_ops(config, index)
                .set_scope(f"block_{index}")
                .set_input([index]),
            )
            index += 1

        # postprocessing and output node
        self.add_node(index + 1)

        self.add_edges_from(
            [(i, i + 1, EdgeData().finalize()) for i in range(1, index)]
        )

        fc_intermediate_features: list[int] = []
        for i in range(1, config["n_fc_layers"] + 1):
            fc_intermediate_features.append(config[f"n_out_features_fc_{i}"])

        # Final Block
        self.add_edges_from(
            [
                (
                    index,
                    index + 1,
                    EdgeData(
                        data={
                            "op": FinalBlock(
                                num_layers=config["n_fc_layers"],
                                in_channels=config[f"block{index - 1}:out_channels"],
                                intermediate_features=fc_intermediate_features,
                                out_features=config["num_classes"],
                                dropout_rate=config["dropout_rate"],
                            )
                        }
                    ).finalize(),
                )
            ]
        )

    def print_props(self):
        alpha_dict = self.get_op_data("alpha")
        props = {
            op: list(alphas_to_probs(alpha_dict[op].detach()).round(3))
            for op in alpha_dict.keys()
        }
        for op in alpha_dict.keys():
            print(f"{op}: {props[op]}")

    def print(self, prefix=""):
        self.print_props()

        graph_print(self, prefix=prefix)

    def get_op_data(
        self, key: str
    ) -> dict[str, Union[nn.parameter.Parameter, nn.Module]]:
        """Get all operations or alphas from the graph

        Args:
            key (str): Key to get data from ("op" or "alpha" are type hinted)

        Returns:
            dict[str, Union[nn.parameter.Parameter, nn.Module]]: Dictionary with node names and data
        """
        vals_per_node = {}
        for g in filter(
            lambda g: g.scope == "block_1" and isinstance(g, OpsWithSkip),
            self._get_child_graphs(),
        ):
            g: OpsWithSkip
            vals_per_node[g.name] = g.get_all_edge_data(key=key, scope="block_1")[0]
        return vals_per_node

    def count_skips(self, threshold: float) -> int:
        """Counts the number of selected skip connections in the NASCell

        Args:
            threshold (float): Probability threshold that has to be reached at least by Identity to be counted

        Returns:
            int: Number of skip connections
        """
        ops_list: dict[str, DARTSMixedOp] = self.get_op_data("op")  # type: ignore
        alphas: dict[str, nn.parameter.Parameter] = self.get_op_data("alpha")  # type: ignore

        probs_ops = [
            (alphas_to_probs(alphas[k].detach()), ops_list[k].primitives)
            for k in ops_list.keys() & alphas.keys()
        ]
        get_identity_index = lambda ops: [
            i for i, op in enumerate(ops) if isinstance(op, IdentityBlock)
        ][0]
        chosen_identity_probs = [
            (
                probs[get_identity_index(ops)]
                if get_identity_index(ops) == np.argmax(probs)
                else 0
            )
            for probs, ops in probs_ops
        ]
        logger.info(f"Chosen identity probs: {chosen_identity_probs}")

        return (np.array(chosen_identity_probs) > threshold).sum()

    @staticmethod
    def create_from_alphas(
        alphas: dict[str, list[float]], config: Mapping[str, Any], dataset: str
    ) -> "NASNetwork":
        """Factory method to create a network with given config and alphas.

        Args:
            alphas: alphas as created by darts
            config: model config
            dataset: dataset to train on (used by NASLib, otherwise not used)

        Returns:
            new NASNetwork
        """
        # use to adapt search space for recreation
        optimizer = DARTSOptimizer()

        new_graph = NASNetwork("network", config)
        # no clue why NASLib embeds the dataset into the graph, but we need it to add the alphas
        optimizer.adapt_search_space(new_graph, dataset)

        for g in filter(
            lambda g: isinstance(g, OpsWithSkip), optimizer.graph._get_child_graphs()
        ):
            g: OpsWithSkip
            g.set_alpha(
                nn.parameter.Parameter(torch.Tensor(alphas[f"{g.scope}_{g.name}"]))
            )

        optimizer.graph.parse()
        return optimizer.graph

    def discretize_graph(self) -> None:
        # Discretize graph by using argmax on the alphas -> choose only one operation per edge
        self.prepare_discretization()

        # Discretize the operations by using argmax
        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                alphas = edge.data.alpha.detach().cpu()
                edge.data.set("op", primitives[np.argmax(alphas)])

        self.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)

        # Delete all surrogate skips
        for g in filter(lambda g: isinstance(g, OpsWithSkip), self._get_child_graphs()):
            g.surrogate_frac = 0

        self.prepare_evaluation()
        self.parse()
        return

    def train_fn(
        self,
        optimizer: torch.optim.Optimizer,  # type: ignore
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion: torch.nn.Module,
        loader: DataLoader,
        device: Union[str, torch.device],
    ) -> tuple[float, float]:
        """Training method

        Args:
            optimizer (torch.optim.Optimizer): Torch optimizer
            loader (DataLoader): DataLoader of training set
            device (Union[str, torch.device]): Torch device

        Returns:
            tuple[float, float]: (Accuracy, Loss) of training epoch
        """
        time_begin = time.time()
        score_tracker = StatTracker()
        loss_tracker = StatTracker()

        self.train()

        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels.squeeze(-1))

            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            acc = accuracy(logits, labels, topk=(1,))[0]  # accuracy given by top 1
            n = images.size(0)
            loss_tracker.update(loss.item(), n)
            score_tracker.update(acc.item(), n)

        self.time_train += time.time() - time_begin
        logger.info(f"Worker:{self.my_worker_id} training time: {self.time_train}")
        return score_tracker.avg, loss_tracker.avg

    def eval_fn(
        self,
        loader: DataLoader,
        criterion: torch.nn.Module,
        device: Union[str, torch.device],
    ) -> dict[str, Any]:
        """Evaluation method

        Args:
            loader (DataLoader): DataLoader for either training, validation or test set
            device (Union[str, torch.device]): Torch device

        Returns:
            dict[str, Any]: evaluated model data as dict
        """
        acc_score_tracker = StatTracker()
        loss_score_tracker = StatTracker()
        cw_acc_tracker = {c: StatTracker() for c in range(self.config["num_classes"])}
        # roc_auc_tracker = StatTracker()
        self.eval()

        labels_l = []
        preds_l = []
        with torch.no_grad():  # no gradient needed
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                val_loss = criterion(outputs, labels)
                acc = accuracy(outputs, labels, topk=(1,))[0]
                acc_score_tracker.update(acc.item(), images.size(0))
                loss_score_tracker.update(val_loss.item(), images.size(0))

                preds = torch.argmax(outputs, dim=1)
                labels_l.append(labels)
                preds_l.append(preds)

                # pred_probs = torch.nn.functional.softmax(outputs, dim=1)
                # roc_auc_tracker.update(float(roc_auc_score(labels, pred_probs, average="macro", multi_class='ovo')), images.size(0))

                # compute class-wise accuracies
                for c in cw_acc_tracker.keys():
                    mask = labels == c
                    masked_outputs = outputs[mask]
                    masked_labels = labels[mask]
                    if len(masked_labels):
                        cw_acc = accuracy(masked_outputs, masked_labels, topk=(1,))[0]
                        cw_acc_tracker[c].update(cw_acc.item(), n=len(masked_labels))

            if self.my_worker_id:
                logger.debug(
                    f"(=> Worker:{self.my_worker_id}) Accuracy: {acc_score_tracker.avg:.4f}, Val Loss: {loss_score_tracker.avg:.4f}"
                )
            else:
                logger.debug(
                    f"Accuracy: {acc_score_tracker.avg:.4f}, Val Loss: {loss_score_tracker.avg:.4f}"
                )

        return {
            "acc": acc_score_tracker.avg,
            "loss": loss_score_tracker.avg,
            "cw_accuracies": {c: tracker.avg for c, tracker in cw_acc_tracker.items()},
            "f1": f1_score(torch.cat(labels_l), torch.cat(preds_l), average="macro"),
            # "roc_auc": roc_auc_tracker.avg,
        }


def graph_print(graph, prefix=""):
    for n in graph.nodes:
        node = graph.nodes[n]
        if "subgraph" in node:
            print(
                f"{prefix}node[{n}] = (name={node['subgraph'].name}, scope={node['subgraph'].scope}, nodes={node['subgraph'].nodes})"
            )
            node["subgraph"].print(prefix + "  ")


def save_alphas(epoch: int, graph: NASNetwork, config: CfgNode) -> None:
    """After epoch callback for NAS.
    Should be passed with `partial(save_alphas, graph=optimizewr.graph, config=config)`

    Args:
        epoch (int): Current epoch
        graph (NASNetwork): Graph that is optimized
        config (CfgNode): Configuration of NAS run
    """
    fp = Path(config.save) / "alphas.csv"
    alphas = graph.get_op_data("alpha")
    exists = fp.exists()
    with open(fp, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            keys = [node for node in alphas.keys()]
            keys += ["epoch"]
            writer.writerow(keys)
        # epochs should start with 1
        values = [value.tolist() for value in alphas.values()]
        values += [epoch + 1]
        writer.writerow(values)
