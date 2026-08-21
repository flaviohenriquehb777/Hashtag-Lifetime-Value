import numpy as np
import pandas as pd
from src.evaluation import compute_regression_metrics, compute_residuals, format_metrics_summary


def test_compute_regression_metrics_perfect_prediction():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["R2"] == 1.0
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE"] == 0.0
    assert metrics["MedAE"] == 0.0


def test_compute_residuals():
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([90.0, 210.0])
    df_res = compute_residuals(y_true, y_pred)
    assert len(df_res) == 2
    assert "residual" in df_res.columns
    assert df_res["residual"].iloc[0] == 10.0
    assert df_res["residual"].iloc[1] == -10.0


def test_format_metrics_summary():
    models_metrics = {
        "Linear Regression": {"R2": 0.85, "RMSE": 10.0, "MAE": 8.0, "MAPE": 0.05, "MedAE": 7.0},
        "Random Forest": {"R2": 0.90, "RMSE": 8.0, "MAE": 6.0, "MAPE": 0.04, "MedAE": 5.0},
    }
    df_summary = format_metrics_summary(models_metrics)
    assert len(df_summary) == 2
    assert df_summary.iloc[0]["modelo"] == "Random Forest"
