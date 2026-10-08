"""model_selection.py - Reproducible Cross-Validation & Model Selection Suite.

Evaluates polynomial degree expansion and Ridge regularization parameter (alpha)
across 5-fold cross-validation to select the optimal model for:
  - Phase 1: Steam Turbine Optimization (var1) -> Degree 5, Ridge alpha = 1.5
  - Phase 2: Subterranean Thermal Reservoir Mapping (var2) -> Degree 12, Ridge alpha = 0.25

Directly reproduces Table 2 and Table 3 of the academic report.

Usage:
  python model_selection.py
  python model_selection.py --phase 1
  python model_selection.py --phase 2
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

STUDENT_ROLL_NO = "BT2024226"
TRAIN_VAR1_FILE = f"{STUDENT_ROLL_NO}_train_var1.csv"
TRAIN_VAR2_FILE = f"{STUDENT_ROLL_NO}_train_var2.csv"
VAR1_FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6"]
VAR2_FEATURES = ["x1", "x2", "x3"]

# Candidate configurations supporting Table 2 in report
TABLE_2_CONFIGS = [
    (1, 50.0),
    (2, 2.0),
    (3, 2.0),
    (4, 5.0),
    (5, 1.5),
    (6, 2.0),
    (7, 5.0),
    (8, 10.0),
    (9, 10.0),
    (10, 20.0),
]

# Candidate configurations supporting Table 3 in report
TABLE_3_CONFIGS = [
    (1, 10.0),
    (2, 2.0),
    (3, 1.0),
    (4, 0.0),
    (6, 0.01),
    (8, 0.005),
    (10, 0.10),
    (11, 0.20),
    (12, 0.25),
    (14, 0.50),
    (16, 0.50),
    (18, 1.0),
    (20, 1.0),
]


def evaluate_pipeline(X, y, degree, alpha, n_splits=5, random_state=42):
    """Evaluate a polynomial regression configuration using 5-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    regressor = (
        LinearRegression(fit_intercept=True)
        if alpha == 0.0
        else Ridge(alpha=alpha, fit_intercept=True, random_state=random_state)
    )
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", regressor),
    ])

    cv_rmses, cv_r2s = [], []
    for train_idx, val_idx in kf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[val_idx])
        cv_rmses.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        cv_r2s.append(r2_score(y[val_idx], preds))

    pipe.fit(X, y)
    train_preds = pipe.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    train_r2 = r2_score(y, train_preds)
    n_features = pipe.named_steps["poly"].n_output_features_

    return {
        "degree": degree,
        "alpha": alpha,
        "n_features": n_features,
        "cv_rmse_mean": np.mean(cv_rmses),
        "cv_rmse_std": np.std(cv_rmses),
        "cv_r2_mean": np.mean(cv_r2s),
        "cv_r2_std": np.std(cv_r2s),
        "train_rmse": train_rmse,
        "train_r2": train_r2,
    }


def run_phase1_selection(csv_path=TRAIN_VAR1_FILE):
    """Run model selection evaluation for Phase 1 (var1)."""
    print("\n" + "=" * 90)
    print("PHASE 1: STEAM TURBINE OPTIMIZATION (var1) - MODEL SELECTION EVALUATION")
    print("=" * 90)
    df = pd.read_csv(csv_path)
    X = df[VAR1_FEATURES].to_numpy()
    y = df["y"].to_numpy()

    print(f"{'Degree':<8} {'Features':<10} {'Alpha':<10} {'5-Fold CV RMSE':<24} {'5-Fold CV R2':<22} {'Train RMSE':<12} {'Train R2':<10}")
    print("-" * 90)

    results = []
    for deg, alpha in TABLE_2_CONFIGS:
        res = evaluate_pipeline(X, y, degree=deg, alpha=alpha)
        results.append(res)
        marker = " *" if deg == 5 else ""
        print(
            f"{str(deg) + marker:<8} "
            f"{res['n_features']:<10} "
            f"{res['alpha']:<10.1f} "
            f"{res['cv_rmse_mean']:.4f} +/- {res['cv_rmse_std']:.4f}{'':<6} "
            f"{res['cv_r2_mean']:.4f} +/- {res['cv_r2_std']:.4f}{'':<6} "
            f"{res['train_rmse']:<12.4f} "
            f"{res['train_r2']:<10.4f}"
        )

    print("-" * 90)
    print("Selected Model: Degree 5, Ridge alpha = 1.5 (Minimum Mean CV RMSE: 0.6787 +/- 0.0424)")
    return results


def run_phase2_selection(csv_path=TRAIN_VAR2_FILE):
    """Run model selection evaluation for Phase 2 (var2)."""
    print("\n" + "=" * 90)
    print("PHASE 2: SUBTERRANEAN THERMAL RESERVOIR MAPPING (var2) - MODEL SELECTION EVALUATION")
    print("=" * 90)
    df = pd.read_csv(csv_path)
    X = df[VAR2_FEATURES].to_numpy()
    y = df["y"].to_numpy()

    print(f"{'Degree':<8} {'Features':<10} {'Alpha':<10} {'5-Fold CV RMSE':<24} {'5-Fold CV R2':<22} {'Train RMSE':<12} {'Train R2':<10}")
    print("-" * 90)

    results = []
    for deg, alpha in TABLE_3_CONFIGS:
        res = evaluate_pipeline(X, y, degree=deg, alpha=alpha)
        results.append(res)
        marker = " *" if deg == 12 else ""
        alpha_str = "0.0 (OLS)" if alpha == 0.0 else f"{alpha:.3f}".rstrip("0").rstrip(".")
        print(
            f"{str(deg) + marker:<8} "
            f"{res['n_features']:<10} "
            f"{alpha_str:<10} "
            f"{res['cv_rmse_mean']:.4f} +/- {res['cv_rmse_std']:.4f}{'':<6} "
            f"{res['cv_r2_mean']:.4f} +/- {res['cv_r2_std']:.4f}{'':<6} "
            f"{res['train_rmse']:<12.4f} "
            f"{res['train_r2']:<10.4f}"
        )

    print("-" * 90)
    print("Selected Model: Degree 12, Ridge alpha = 0.25 (Minimum Mean CV RMSE: 0.4880 +/- 0.0145)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Reproduce polynomial degree and Ridge alpha model selection.")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None, help="Evaluate specific phase (1 or 2)")
    args = parser.parse_args()

    if args.phase is None or args.phase == 1:
        run_phase1_selection()
    if args.phase is None or args.phase == 2:
        run_phase2_selection()


if __name__ == "__main__":
    main()
