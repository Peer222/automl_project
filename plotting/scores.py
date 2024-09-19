import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Iterable
import dataframe_image as dfi
import re
import statistics
import matplotlib as mpl


def parse_data(data: dict, metrics: Iterable[str]) -> pd.DataFrame:
    """Parses saved data into format for calculating values

    Args:
        data (pd.DataFrame): Raw data
        metric (str): Metric to extract

    Returns:
        tuple[pd.DataFrame, str]: Parsed metric values dataframe, type [list, instance]
    """
    parsed_data = {metric: [] for metric in metrics}
    for entry in data["data"]:
        run_data = entry[-1]

        for metric in metrics:
            if not metric in run_data.keys():
                continue
            parsed_data[metric].append(run_data[metric])

    return pd.DataFrame(parsed_data)


def plot(dfs: list[pd.DataFrame], test_info: list[dict], result_dir: Path) -> None:
    """Create a table listing basic statistics over multiple seeds/runs

    Args:
        dfs: dataframe with metrics per run
        test_info: parsed test_info (evaluation on test set) printed by the run
    """

    def multicolumn_from_str(col: str) -> tuple[str, str]:
        m = re.search(r"([\w|-]+)\s(.*)$", col)
        try:
            return (m.group(1), m.group(2))
        except:
            return ("", col)

    best_dfs = []
    for df, test_info in zip(dfs, test_infos):
        # keep only column with highest acc
        df_max = df[df["val_acc"] == df["val_acc"].max()].iloc[[0]].copy()

        df = pd.DataFrame()
        df["F1 Score"] = (test_info["f1"], float(df_max["val_f1_score"].item()))
        df["Accuracy"] = (test_info["acc"], float(df_max["val_acc"].item()))
        df["Split"] = ("Test", "Val")
        df["Mode"] = ("Ours (2h)", "Ours (2h)")
        best_dfs.append(df)


    # Hardcode baseline df. This is not pretty, but time was short
    baseline_val_accuracies = [0.617, 0.608, 0.608]
    baseline_val_f1_scores = [0.385, 0.438, 0.376]

    baseline_val_df = pd.DataFrame()
    baseline_val_df["Accuracy"] = baseline_val_accuracies
    baseline_val_df["F1 Score"] = baseline_val_f1_scores
    baseline_val_df["Split"] = "Val"
    baseline_val_df["Mode"] = "Baseline (6h)"
    best_dfs.append(baseline_val_df)

    baseline_test_accuracies = [0.478, 0.537, 0.525]
    baseline_test_f1_scores = [0.218, 0.321, 0.316]

    baseline_test_df = pd.DataFrame()
    baseline_test_df["Accuracy"] = baseline_test_accuracies
    baseline_test_df["F1 Score"] = baseline_test_f1_scores
    baseline_test_df["Split"] = "Test"
    baseline_test_df["Mode"] = "Baseline (6h)"
    best_dfs.append(baseline_test_df)

    best_df = pd.concat(best_dfs)
    sns.set_theme(context="poster", style='whitegrid', palette="tab10", font_scale=1.5)
    mpl.rcParams['legend.loc'] = "upper left"
    mpl.rc('legend',fontsize='smaller')

    fig, ax = plt.subplots(figsize=(11,8), dpi=300)
    sns.boxplot(data=best_df, x="Accuracy", y="Split", hue="Mode", whis=999)
    plt.legend(bbox_to_anchor=(0.0, 1.3), loc='upper left', ncol=2)
    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.tight_layout()
    plt.ylabel(" ")
    plt.savefig(result_dir / "accuracy.png")
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    sns.boxplot(data=best_df, x="F1 Score", y="Split", hue="Mode", whis=999)
    plt.legend(bbox_to_anchor=(0, 1.3), loc='upper left', ncol=2)
    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.ylabel(" ")
    plt.tight_layout()
    plt.savefig(result_dir / "f1.png")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting progress of bohb")

    parser.add_argument(
        "--dirpaths",
        type=Path,
        nargs="*",
        required=True,
        help="Paths to result directory (e.g. results/config_1/1/), delimited by ,",
    )
    parser.add_argument(
        "--logpaths",
        type=Path,
        nargs="*",
        required=True,
        help="Paths to log where we can find test evaluation, delimited by ,",
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

    def get_run_history(dirpath: Path):
        with open(dirpath / "bohb" / dirpath.name / "runhistory.json", "r") as f:
            return json.load(f)

    def get_test_info(logpath: Path):
        with open(logpath) as f:
            # we need some ugly parsing because we forgot logging properly instead of only printing
            test_info_re = re.search(r"^test_info=(.*)$", f.read(), re.MULTILINE).group(
                1
            )
            return json.loads(
                re.sub(r"(\d):", r'"\1":', test_info_re.replace("'", '"'))
            )

    dirpaths = args.dirpaths
    logpaths = args.logpaths
    run_histories = list(map(get_run_history, dirpaths))
    parsed_histories = list(
        map(
            lambda history: parse_data(history, ("val_f1_score", "val_acc")),
            run_histories,
        )
    )
    test_infos = list(map(get_test_info, logpaths))

    plot(parsed_histories, test_infos, result_dir)
