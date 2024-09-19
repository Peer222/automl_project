from pathlib import Path
import argparse


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MF example using BOHB.")
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        help="Configuration file for NAS",
        required=True,
    )

    parser.add_argument(
        "--runtime",
        default=21600,
        type=int,
        help="Running time (seconds) allocated to run the algorithm",
    )
    parser.add_argument(
        "--max_budget",
        type=float,
        default=20,
        help="maximal budget to use with BOHB",
    )
    parser.add_argument(
        "--min_budget",
        type=float,
        default=5,
        help="Minimum budget for BOHB",
    )
    parser.add_argument("--eta", type=int, default=2, help="eta for BOHB")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="num of workers to use with BOHB"
    )
    parser.add_argument(
        "--total_budget", type=int, default=1000, help="Budget to run SMAC for"
    )
    parser.add_argument(
        "--log_level",
        choices=[
            "NOTSET",
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARN",
            "WARNING",
            "INFO",
            "DEBUG",
        ],
        default="NOTSET",
        help="Logging level",
    )
    parser.add_argument(
        "--datasetpath",
        type=str,
        default="./data/",
        help="Path to directory containing the dataset",
    )
    return parser.parse_args()
