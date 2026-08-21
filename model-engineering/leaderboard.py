"""Print a formatted experiment leaderboard from runs/summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Print experiment leaderboard")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--sort-by",
        default="roc_auc",
        help="Column to rank runs by (desc). Default: roc_auc",
    )
    args = parser.parse_args()

    summary_path = Path(args.runs_dir) / "summary.csv"
    if not summary_path.exists():
        print(f"No summary file found at {summary_path}. Run training first.")
        return

    df = pd.read_csv(summary_path)

    sort_col = args.sort_by if args.sort_by in df.columns else "roc_auc"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)

    print("\n=== Model Leaderboard ===\n")
    print(df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
