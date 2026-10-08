"""train.py - Master training script to train models for both Phase 1 and Phase 2.

Student Roll Number: BT2024226

This script runs training for both independent problems:
  - Phase 1: Steam Turbine Optimization (var1) -> Degree 5 Ridge (alpha=1.5)
  - Phase 2: Subterranean Thermal Reservoir Mapping (var2) -> Degree 12 Ridge (alpha=0.25)

For individual phase execution:
  python train_var1.py
  python train_var2.py
"""

from train_var1 import train_var1, MODEL_VAR1_FILE, VAR1_FEATURES
from train_var2 import train_var2, MODEL_VAR2_FILE, VAR2_FEATURES


def main():
    print("=================================================================")
    print("RUNNING MASTER TRAINING PIPELINE (PHASE 1 AND PHASE 2)")
    print("=================================================================\n")
    train_var1()
    print()
    train_var2()
    print("\nAll models trained and serialized successfully!")


if __name__ == "__main__":
    main()
