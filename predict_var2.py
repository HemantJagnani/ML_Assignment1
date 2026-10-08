"""predict_var2.py - Generates predictions for Phase 2: Subterranean Thermal Reservoir Mapping.

Inputs:
  CSV file containing 3D spatial features: x1, x2, x3
Output:
  CSV file containing single column 'y' (Thermal Anomaly Score), matching sample_submission.csv format.

Usage:
  # Predict on assigned assignment test set (default):
  python predict_var2.py

  # Predict on any custom test CSV file:
  python predict_var2.py --input path/to/custom_test_var2.csv --output my_preds_var2.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from train_var2 import (
    VAR2_FEATURES,
    DEFAULT_MODEL_FILE,
    train_var2,
)

DEFAULT_TEST_FILE = "BT2024226_test_var2.csv"
DEFAULT_PRED_FILE = "BT2024226_pred_var2.csv"


def predict_var2(
    input_csv: str = DEFAULT_TEST_FILE,
    output_csv: str = DEFAULT_PRED_FILE,
    model_file: str = DEFAULT_MODEL_FILE,
) -> np.ndarray:
    """Load Phase 2 model, predict Thermal Anomaly Score, validate format, and save to CSV."""
    model_path = Path(model_file)
    if not model_path.exists():
        print(f"Model '{model_file}' not found. Training model now...")
        model = train_var2(model_out=model_file)
    else:
        model = joblib.load(model_path)

    print("=" * 65)
    print("PHASE 2 INFERENCE: SUBTERRANEAN THERMAL RESERVOIR PREDICTION")
    print("=" * 65)
    print(f"Reading query features from: {input_csv}")
    df = pd.read_csv(input_csv)

    missing = [c for c in VAR2_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input file '{input_csv}' is missing required features: {missing}")

    X = df[VAR2_FEATURES].to_numpy()
    preds = model.predict(X)

    if np.isnan(preds).any() or np.isinf(preds).any():
        raise ValueError("Predictions contain NaN or Infinite values.")

    # Write output CSV strictly adhering to sample_submission.csv format
    out_df = pd.DataFrame({"y": preds})
    out_df.to_csv(output_csv, index=False)

    print(f"Predictions saved to: {output_csv} ({len(out_df)} rows)")
    print(f"Prediction statistics:")
    print(f"  Mean:   {preds.mean():.4f}")
    print(f"  Std:    {preds.std():.4f}")
    print(f"  Min:    {preds.min():.4f}")
    print(f"  Max:    {preds.max():.4f}")
    return preds


def main():
    parser = argparse.ArgumentParser(description="Predict Thermal Anomaly Score for Reservoir Mapping (Phase 2).")
    parser.add_argument("--input", default=DEFAULT_TEST_FILE, help="Path to input test CSV")
    parser.add_argument("--output", default=DEFAULT_PRED_FILE, help="Path for output predictions CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL_FILE, help="Path to trained model file")
    args = parser.parse_args()
    predict_var2(input_csv=args.input, output_csv=args.output, model_file=args.model)


if __name__ == "__main__":
    main()
