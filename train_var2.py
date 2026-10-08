"""train_var2.py - Trains Phase 2: Subterranean Thermal Reservoir Mapping model.

Problem Description:
  A 3D geological survey grid around central basecamp:
  - x1: East-West coordinate offset (meters)
  - x2: North-South coordinate offset (meters)
  - x3: Vertical depth offset relative to basecamp (meters)
  Target (y): Thermal Anomaly Score

Optimal Model Selected:
  Polynomial Regression Degree 12 with L2 Ridge Regularization (alpha = 0.25).
  - 5-Fold Cross-Validation RMSE: 0.4880 ± 0.0145
  - 5-Fold Cross-Validation R2:   0.9944 ± 0.0015
  - Full Dataset Training RMSE:   0.4130
  - Full Dataset Training R2:     0.9962

Usage:
  python train_var2.py
  python train_var2.py --train my_custom_train_var2.csv --model my_model_var2.joblib
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_TRAIN_FILE = "BT2024226_train_var2.csv"
DEFAULT_MODEL_FILE = "model_var2.joblib"
VAR2_FEATURES = ["x1", "x2", "x3"]
OPTIMAL_DEGREE = 12
OPTIMAL_ALPHA = 0.25


def build_pipeline(degree: int = OPTIMAL_DEGREE, alpha: float = OPTIMAL_ALPHA) -> Pipeline:
    """Build the scikit-learn polynomial ridge regression pipeline for Phase 2."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=42)),
    ])


def train_var2(train_csv: str = DEFAULT_TRAIN_FILE, model_out: str = DEFAULT_MODEL_FILE) -> Pipeline:
    """Train the Phase 2 reservoir mapping model and save the fitted pipeline."""
    print("=" * 65)
    print("PHASE 2: SUBTERRANEAN THERMAL RESERVOIR MAPPING (var2)")
    print("=" * 65)
    print(f"Loading training data from: {train_csv}")
    df = pd.read_csv(train_csv)

    for col in VAR2_FEATURES + ["y"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {train_csv}")

    X = df[VAR2_FEATURES].to_numpy()
    y = df["y"].to_numpy()
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} input features (x1 to x3)")

    # 5-Fold Cross Validation
    print(f"\nEvaluating 5-Fold Cross-Validation (Degree {OPTIMAL_DEGREE}, Ridge alpha = {OPTIMAL_ALPHA})...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmses, cv_r2s = [], []
    for train_idx, val_idx in kf.split(X, y):
        fold_model = build_pipeline()
        fold_model.fit(X[train_idx], y[train_idx])
        preds = fold_model.predict(X[val_idx])
        cv_rmses.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        cv_r2s.append(r2_score(y[val_idx], preds))

    print(f"  5-Fold CV RMSE: {np.mean(cv_rmses):.4f} ± {np.std(cv_rmses):.4f}")
    print(f"  5-Fold CV R2:   {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")

    # Retrain on 100% of data
    print("\nFitting final model on complete training dataset...")
    final_model = build_pipeline()
    final_model.fit(X, y)
    train_preds = final_model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    train_r2 = r2_score(y, train_preds)
    print(f"  Full Train RMSE: {train_rmse:.4f}")
    print(f"  Full Train R2:   {train_r2:.4f}")

    joblib.dump(final_model, model_out)
    print(f"\nModel successfully saved to: {model_out}")
    return final_model


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 Reservoir Mapping polynomial regression model.")
    parser.add_argument("--train", default=DEFAULT_TRAIN_FILE, help="Path to training CSV file")
    parser.add_argument("--model", default=DEFAULT_MODEL_FILE, help="Output path for serialized model")
    args = parser.parse_args()
    train_var2(train_csv=args.train, model_out=args.model)


if __name__ == "__main__":
    main()
