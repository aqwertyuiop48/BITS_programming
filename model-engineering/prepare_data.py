"""Prepare a raw CSV into the processed training format used by the repo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.common import ensure_dir


def _parse_csv_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw data for training")
    parser.add_argument("--input-csv", required=True, help="Path to raw input CSV")
    parser.add_argument(
        "--output-csv",
        default="data/processed/train.csv",
        help="Path to processed output CSV",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Name of target column expected by training",
    )
    parser.add_argument(
        "--rename-target-from",
        default="",
        help="Rename this source column to the target column name",
    )
    parser.add_argument(
        "--drop-columns",
        default="",
        help="Comma-separated columns to drop before saving",
    )
    parser.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Allow processing unlabeled data when the target column is absent",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    df.columns = [str(column).strip() for column in df.columns]

    if args.rename_target_from:
        if args.rename_target_from not in df.columns:
            raise ValueError(
                f"Column '{args.rename_target_from}' not found in raw data for target rename"
            )
        df = df.rename(columns={args.rename_target_from: args.target_col})

    if args.target_col not in df.columns and not args.allow_missing_target:
        raise ValueError(
            f"Target column '{args.target_col}' not found. "
            "Use --rename-target-from if the raw file uses a different name."
        )

    drop_columns = _parse_csv_list(args.drop_columns)
    if drop_columns:
        missing_drop_columns = [column for column in drop_columns if column not in df.columns]
        if missing_drop_columns:
            raise ValueError(
                f"Cannot drop missing columns: {missing_drop_columns}"
            )
        df = df.drop(columns=drop_columns)

    df = df.drop_duplicates().reset_index(drop=True)

    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)

    print(f"Prepared data written to: {output_csv}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    if args.target_col in df.columns:
        print(f"Target column: {args.target_col}")
    else:
        print("Target column: not present (allowed for inference prep)")


if __name__ == "__main__":
    main()