import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from util import alphas_to_probs
from .utils import style_plot


def parse_data(data: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
    """Parses saved data into format for plotting

    Args:
        data (pd.DataFrame): Raw data

    Returns:
        tuple[list[pd.DataFrame], list[str]]: Parsed weights data for each node in cell, node names
    """
    cols = list(data.columns)
    operations = ["Identity", "Zero", "Conv", "ResNet", "Bottleneck"]
    epochs = data[cols[-2]]
    seeds = data[cols[-1]]

    nodes = cols[:-2]

    splitted_data = []
    for node in nodes:
        alphas = data[node].apply(yaml.safe_load).to_list()
        parsed_data = {"operation": [], "alpha": [], "epoch": [], "seed": []}
        for j, row in enumerate(alphas):
            probs = alphas_to_probs(row).tolist()
            for i in range(len(probs)):
                parsed_data["operation"].append(f"{operations[i]}")
                parsed_data["alpha"].append(probs[i])
                parsed_data["epoch"].append(epochs.iloc[j])
                parsed_data["seed"].append(seeds.iloc[j])

        assert len(alphas[0]) == len(
            operations
        ), f"{len(alphas[0])=}, {len(operations)=}"
        parsed_df = pd.DataFrame(parsed_data)
        splitted_data += [parsed_df]

    node_names = [f"Node {i + 2}" for i in range(len(nodes))]
    return splitted_data, node_names


def plot(data: pd.DataFrame, result_dir: Path) -> None:
    """Plots alphas/ weights for operations

    Args:
        data (pd.DataFrame): Raw data
        result_dir (Path): Directory to save plots
    """
    splitted_data, cols = parse_data(data)

    sns.set_theme("poster", "whitegrid", font_scale=1.5)
    for d, prefix in zip(splitted_data, cols):
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.tight_layout(rect=(0.03, 0.03, 1.02, 0.99))

        plt.title(f"{prefix}: Architecture Weights", pad=20)
        plt.xlabel("epochs")
        plt.ylabel("share", labelpad=20)
        plt.ylim(0)

        style_plot(ax)

        sns.lineplot(data=d, x="epoch", y="alpha", hue="operation")
        plt.legend(ncol=2)
        plt.savefig(result_dir / f"{prefix}_architecture_weights", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting progress of darts weights")

    parser.add_argument(
        "dirpaths",
        nargs="*",
        type=Path,
        help="Paths to result directories for different seeds (e.g. results/config_1/1/)",
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=None,
        help="Plotting directory to save plots into (defaults to dirpath/plots/alphas or dirpath[0].parent/plots/alphas for multiple runs)",
    )

    args = parser.parse_args()
    for dirpath in args.dirpaths:
        result_dir: Path = dirpath / "plots" / "alphas"
        if args.result_dir:
            result_dir = args.result_dir
        result_dir.mkdir(exist_ok=True, parents=True)

        arch_weights = pd.read_csv(dirpath / "alphas.csv")
        arch_weights["seed"] = dirpath.name

        plot(arch_weights, result_dir)

    if len(args.dirpaths) > 1:
        result_dir = args.dirpaths[0].parent / "plots" / "alphas"
        if args.result_dir:
            result_dir = args.result_dir / "all"
        result_dir.mkdir(exist_ok=True, parents=True)

        dfs = []
        for dirpath in args.dirpaths:
            df = pd.read_csv(dirpath / "alphas.csv")
            df["seed"] = dirpath.name
            dfs.append(df)

        arch_weights = pd.concat(dfs)
        plot(arch_weights, result_dir)
