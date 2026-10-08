"""predict.py - Master inference script to generate predictions for both Phase 1 and Phase 2.

Student Roll Number: BT2024226

This script runs inference for both independent problems:
  - Phase 1: Steam Turbine Optimization (var1) -> produces BT2024226_pred_var1.csv
  - Phase 2: Subterranean Thermal Reservoir Mapping (var2) -> produces BT2024226_pred_var2.csv

For individual phase execution or custom files:
  python predict_var1.py --input <file> --output <file>
  python predict_var2.py --input <file> --output <file>
"""

import argparse
from pathlib import Path
from predict_var1 import predict_var1, DEFAULT_TEST_FILE as TEST_VAR1_DEFAULT, DEFAULT_PRED_FILE as PRED_VAR1_DEFAULT
from predict_var2 import predict_var2, DEFAULT_TEST_FILE as TEST_VAR2_DEFAULT, DEFAULT_PRED_FILE as PRED_VAR2_DEFAULT


def run_all_predictions(
    var1_in=TEST_VAR1_DEFAULT,
    var1_out=PRED_VAR1_DEFAULT,
    var2_in=TEST_VAR2_DEFAULT,
    var2_out=PRED_VAR2_DEFAULT,
):
    """Run predictions for both Phase 1 and Phase 2."""
    p1 = None
    if Path(var1_in).exists():
        p1 = predict_var1(input_csv=var1_in, output_csv=var1_out)
        print()

    p2 = None
    if Path(var2_in).exists():
        p2 = predict_var2(input_csv=var2_in, output_csv=var2_out)

    return p1, p2


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for both Phase 1 and Phase 2.")
    parser.add_argument("--var1", default=TEST_VAR1_DEFAULT, help="Path to var1 test CSV")
    parser.add_argument("--out1", default=PRED_VAR1_DEFAULT, help="Output path for var1 predictions")
    parser.add_argument("--var2", default=TEST_VAR2_DEFAULT, help="Path to var2 test CSV")
    parser.add_argument("--out2", default=PRED_VAR2_DEFAULT, help="Output path for var2 predictions")
    args = parser.parse_args()

    run_all_predictions(
        var1_in=args.var1,
        var1_out=args.out1,
        var2_in=args.var2,
        var2_out=args.out2,
    )


if __name__ == "__main__":
    main()
