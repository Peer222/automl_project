import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .utils import style_plot

MIN_YSPACE_TO_PLOT_BORDER = 0.05
MIN_XSPACE_TO_PLOT_BORDER = 0.5


def parse_data(data: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    """Parses saved data into format for plotting

    Args:
        data (pd.DataFrame): Raw data
        metric (str): Metric to extract

    Returns:
        tuple[pd.DataFrame, str]: Parsed metric values dataframe, type [list, instance]
    """
    parsed_data = {
        "epoch": [],
        metric: [],
        "config_id": [],
    }
    data_type = "instance"
    for entry in data["data"]:
        config_id = entry[0]
        run_data = entry[-1]

        if not metric in run_data.keys():
            continue
        if isinstance(run_data[metric], list):
            data_type = "list"
            for i, loss in enumerate(run_data[metric]):
                parsed_data["config_id"].append(config_id)
                parsed_data["epoch"].append(i + 1)
                parsed_data[metric].append(loss)
        else:
            epochs = len(run_data["train_losses"])
            parsed_data["config_id"].append(config_id)
            parsed_data["epoch"].append(epochs)
            parsed_data[metric].append(run_data[metric])

    return pd.DataFrame(parsed_data), data_type


def plot(data: pd.DataFrame, result_dir: Path, metric: str, ylabel: str) -> None:
    """Plots given metric in appropiate format

    Args:
        data (pd.DataFrame): Raw data
        result_dir (Path): Directory save plots
        metric (str): Metric from data that should be plotted
        ylabel (str): Label of y-axis
    """
    parsed_data, data_type = parse_data(data, metric)

    if data_type == "list":
        plot_curves(parsed_data, result_dir, metric, ylabel)
    elif data_type == "instance":
        plot_dots(parsed_data, result_dir, metric, ylabel)


def plot_curves(
    parsed_data: pd.DataFrame, result_dir: Path, metric: str, ylabel: str
) -> None:
    """Plots metric with respect to epochs

    Args:
        parsed_data (pd.DataFrame): Parsed data
        result_dir (Path): Directory to save plots
        metric (str): Metric from data that should be plotted
        ylabel (str): Label of y-axis
    """
    sns.set_theme("poster", "whitegrid", font_scale=1.5)

    _, ax = plt.subplots(1, 1, figsize=(11, 8))

    plt.tight_layout(rect=(0.03, 0.03, 1.02, 0.99))

    plt.title(f"BOHB progress", pad=20)
    plt.xlabel("epochs")
    plt.ylabel(ylabel, labelpad=20)

    plt.ylim(
        max(0, parsed_data[metric].min() - MIN_YSPACE_TO_PLOT_BORDER),
        parsed_data[metric].max() + MIN_YSPACE_TO_PLOT_BORDER,
    )

    style_plot(ax)

    sns.lineplot(data=parsed_data, x="epoch", y=metric, hue="config_id", legend=False)

    plt.savefig(result_dir / f"bohb_{metric}", dpi=300)


def plot_dots(
    parsed_data: pd.DataFrame, result_dir: Path, metric: str, ylabel: str, hue: str = "config_id"
) -> None:
    sns.set_theme("poster", "whitegrid", font_scale=1.5)

    _, ax = plt.subplots(1, 1, figsize=(11, 8))

    plt.tight_layout(rect=(0.03, 0.03, 1.02, 0.99))

    plt.title(f"BOHB Progress", pad=20)
    plt.xlabel("epochs")
    plt.ylabel(ylabel, labelpad=20)

    plt.ylim(
        max(0, parsed_data[metric].min() - MIN_YSPACE_TO_PLOT_BORDER),
        parsed_data[metric].max() + MIN_YSPACE_TO_PLOT_BORDER,
    )
    plt.xlim(0, parsed_data["epoch"].max() + MIN_XSPACE_TO_PLOT_BORDER)

    style_plot(ax)

    sns.scatterplot(
        data=parsed_data, x="epoch", y=metric, hue=hue, legend=False, ax=ax
    )
    sns.lineplot(
        data=parsed_data, x="epoch", y=metric, hue=hue, legend=False, ax=ax
    )


    plt.savefig(result_dir / f"bohb_{metric}", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting progress (validation accuracies) of bohb")

    parser.add_argument(
        "dirpaths", nargs="*", type=Path, help="Paths to result directories (e.g. results/config_1/1/)"
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=None,
        help="Plotting directory to save plots into (arch_weights subdirectory created automatically)",
    )

    args = parser.parse_args()

    result_dir: Path = args.dirpaths[0].parent / "plots" / "bohb"
    if args.result_dir:
        result_dir = args.result_dir
    result_dir.mkdir(exist_ok=True, parents=True)
    dfs = []
    for dirpath in args.dirpaths:
        with open(dirpath / "bohb" / dirpath.name / "runhistory.json", "r") as f:
            run_history = json.load(f)
            df, _ = parse_data(run_history, "val_acc")
            df["seed"] = int(dirpath.name)
            df["config_id"] = df["config_id"] + 1000 * df["seed"]
            dfs.append(df)

    parsed_data = pd.concat(dfs)
    plot_dots(parsed_data, result_dir, "val_acc", ylabel="validation accuracy", hue="config_id")
    
    #plot(run_history, result_dir, "train_accs", "train accuracy")
    #plot(run_history, result_dir, "val_acc", "validation accuracy")
