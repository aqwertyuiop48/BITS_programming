"""Run batch inference with a saved model and export predictions to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.inference.predictor import load_model, predict_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--run-dir", default="runs/run_001")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_csv = Path(args.input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model = load_model(run_dir)
    X = pd.read_csv(input_csv)
    preds = predict_dataframe(model, X)

    output_csv = Path(args.output_csv) if args.output_csv else run_dir / "predictions_inference.csv"
    preds.to_csv(output_csv, index=False)

    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
