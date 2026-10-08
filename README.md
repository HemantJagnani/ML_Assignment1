# Polynomial Regression Assignment (Machine Learning)

**Student Name:** Hemant Jagnani  
**Student Roll Number:** `BT2024226`  
**Course:** Machine Learning  
**GitHub Repository:** [https://github.com/HemantJagnani/ML_Assignment1](https://github.com/HemantJagnani/ML_Assignment1)  

---

## 1. Assignment Overview & Two Distinct Problems

This assignment contains two completely distinct physical regression problems:

### Phase 1: Power Plant Steam Turbine Optimization (`var1`)
- **Domain:** Geothermal multi-stage steam turbine efficiency.
- **Input Features (6):** `x1` (high-pressure steam valve), `x2` (coolant flow), `x3` (hydraulic pump pressure), `x4` (turbine blade pitch), `x5` (gas exhaust rate), `x6` (steam inlet pressure).
- **Target ($y$):** Net Power Score.
- **Selected Model:** Polynomial Degree 5 with L2 Ridge Regularization ($\alpha = 1.50$).
- **Performance:** 5-Fold CV RMSE = **0.6787 ± 0.0424**, CV $R^2$ = **0.9537 ± 0.0067**.

### Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)
- **Domain:** 3D seismic/thermal sensor network mapping around basecamp landmark.
- **Input Features (3):** `x1` (East-West offset in meters), `x2` (North-South offset in meters), `x3` (vertical depth offset in meters).
- **Target ($y$):** Thermal Anomaly Score.
- **Selected Model:** Polynomial Degree 12 with L2 Ridge Regularization ($\alpha = 0.25$).
- **Performance:** 5-Fold CV RMSE = **0.4880 ± 0.0145**, CV $R^2$ = **0.9944 ± 0.0015**.

---

## 2. File Organization

All files are arranged cleanly in the root directory:

```
.
├── BT2024226_train_var1.csv    # Phase 1 training data (Steam turbine, 6 features + y, 1000 rows)
├── BT2024226_test_var1.csv     # Phase 1 test queries (6 features, 1000 rows)
├── BT2024226_pred_var1.csv     # [DELIVERABLE] Completed test predictions for Phase 1
├── train_var1.py               # Dedicated training script for Phase 1 (Degree 5 Ridge)
├── predict_var1.py             # Dedicated inference script for Phase 1
├── model_var1.joblib           # Pre-trained pipeline for Phase 1
│
├── BT2024226_train_var2.csv    # Phase 2 training data (Reservoir mapping, 3 features + y, 1000 rows)
├── BT2024226_test_var2.csv     # Phase 2 test queries (3 features, 1000 rows)
├── BT2024226_pred_var2.csv     # [DELIVERABLE] Completed test predictions for Phase 2
├── train_var2.py               # Dedicated training script for Phase 2 (Degree 12 Ridge)
├── predict_var2.py             # Dedicated inference script for Phase 2
├── model_var2.joblib           # Pre-trained pipeline for Phase 2
│
├── train.py                    # Master training script (runs both Phase 1 and Phase 2)
├── predict.py                  # Master inference script (runs both Phase 1 and Phase 2)
├── model_selection.py          # Reproducible 5-fold CV degree/alpha evaluation (reproduces Tables 2 & 3)
├── test_pipeline.py            # Automated test suite (run with pytest or python)
│
├── BT2024226_report.pdf        # [DELIVERABLE] 4-page technical academic report
├── sample_submission.csv       # Sample submission format benchmark
├── requirements.txt            # Minimal Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This documentation
```

---

## 3. How to Use the Code

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Working with Phase 1 (Steam Turbine)
```bash
# Train Phase 1 model:
python train_var1.py

# Predict Phase 1 on assigned test set (outputs BT2024226_pred_var1.csv):
python predict_var1.py

# Predict on new custom turbine data:
python predict_var1.py --input path/to/custom_data.csv --output custom_predictions.csv
```

### 3.3 Working with Phase 2 (Reservoir Mapping)
```bash
# Train Phase 2 model:
python train_var2.py

# Predict Phase 2 on assigned test set (outputs BT2024226_pred_var2.csv):
python predict_var2.py

# Predict on new custom reservoir data:
python predict_var2.py --input path/to/custom_data.csv --output custom_predictions.csv
```

### 3.4 Reproducible Model Selection & Cross-Validation Evaluation
Reproduce the cross-validation and training metrics across polynomial degrees and Ridge $\alpha$ values supporting Table 2 and Table 3 of the report:
```bash
# Evaluate both Phase 1 (var1) and Phase 2 (var2):
python model_selection.py

# Evaluate a specific phase:
python model_selection.py --phase 1
python model_selection.py --phase 2
```

### 3.5 Running Both Phases at Once (Master Scripts)
```bash
# Retrain both models:
python train.py

# Generate both prediction files:
python predict.py
```

### 3.6 Run Test Suite
```bash
pytest test_pipeline.py -v
```
*(or `python test_pipeline.py`)*
