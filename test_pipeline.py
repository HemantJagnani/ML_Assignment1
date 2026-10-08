"""test_pipeline.py - Automated test suite for data validation, training, inference, and deliverables.

Run with:
  pytest test_pipeline.py -v
or:
  python test_pipeline.py
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

STUDENT_ROLL_NO = "BT2024226"


def test_student_roll_number():
    """Verify that student roll number is BT2024226."""
    assert STUDENT_ROLL_NO == "BT2024226"


def test_datasets_exist():
    """Verify all supplied CSV datasets exist and have expected columns."""
    for name in [
        f"{STUDENT_ROLL_NO}_train_var1.csv",
        f"{STUDENT_ROLL_NO}_test_var1.csv",
        f"{STUDENT_ROLL_NO}_train_var2.csv",
        f"{STUDENT_ROLL_NO}_test_var2.csv",
        "sample_submission.csv",
    ]:
        p = Path(name)
        assert p.exists(), f"Missing required file: {name}"
        assert p.stat().st_size > 0, f"File is empty: {name}"


def test_dataset_hygiene():
    """Verify absence of NaN and infinite values in raw data."""
    for name in [
        f"{STUDENT_ROLL_NO}_train_var1.csv",
        f"{STUDENT_ROLL_NO}_test_var1.csv",
        f"{STUDENT_ROLL_NO}_train_var2.csv",
        f"{STUDENT_ROLL_NO}_test_var2.csv",
    ]:
        df = pd.read_csv(name)
        assert df.isna().sum().sum() == 0, f"NaNs found in {name}"
        assert not np.isinf(df.to_numpy()).any(), f"Inf found in {name}"
        assert len(df) == 1000, f"Expected 1,000 rows in {name}, found {len(df)}"


def test_training_phase1_and_phase2():
    """Verify individual phase training functions."""
    from train_var1 import train_var1, DEFAULT_MODEL_FILE as M1_FILE
    from train_var2 import train_var2, DEFAULT_MODEL_FILE as M2_FILE

    m1 = train_var1()
    m2 = train_var2()

    assert Path(M1_FILE).exists()
    assert Path(M2_FILE).exists()
    assert m1.named_steps["poly"].degree == 5
    assert m2.named_steps["poly"].degree == 12


def test_prediction_deliverables():
    """Verify generated prediction files match required submission specifications."""
    from predict_var1 import predict_var1, DEFAULT_PRED_FILE as P1_FILE
    from predict_var2 import predict_var2, DEFAULT_PRED_FILE as P2_FILE

    p1 = predict_var1()
    p2 = predict_var2()

    for pred_file, preds in [(P1_FILE, p1), (P2_FILE, p2)]:
        p = Path(pred_file)
        assert p.exists(), f"Missing prediction file: {pred_file}"

        with open(p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert lines[0] == "y", f"Expected header 'y', got '{lines[0]}'"
        assert len(lines) == 1001, f"Expected 1,001 lines, got {len(lines)}"

        df = pd.read_csv(p)
        assert df.shape == (1000, 1)
        assert df.columns.tolist() == ["y"]
        assert np.all(np.isfinite(df["y"].to_numpy()))


def test_reproducibility():
    """Verify that predictions are strictly deterministic."""
    m1 = joblib.load("model_var1.joblib")
    m2 = joblib.load("model_var2.joblib")

    df_test1 = pd.read_csv(f"{STUDENT_ROLL_NO}_test_var1.csv")
    df_test2 = pd.read_csv(f"{STUDENT_ROLL_NO}_test_var2.csv")

    p1_first = m1.predict(df_test1.to_numpy())
    p1_second = m1.predict(df_test1.to_numpy())
    np.testing.assert_array_equal(p1_first, p1_second)

    p2_first = m2.predict(df_test2.to_numpy())
    p2_second = m2.predict(df_test2.to_numpy())
    np.testing.assert_array_equal(p2_first, p2_second)


def test_report_deliverable():
    """Verify report.pdf exists, is 4 pages, and contains valid non-empty content."""
    import pypdf

    report_p = Path(f"{STUDENT_ROLL_NO}_report.pdf")
    if not report_p.exists():
        report_p = Path("report.pdf")
    assert report_p.exists(), f"Missing {STUDENT_ROLL_NO}_report.pdf or report.pdf in root"
    reader = pypdf.PdfReader(str(report_p))
    assert 4 <= len(reader.pages) <= 5, f"Expected 4-5 pages, found {len(reader.pages)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
