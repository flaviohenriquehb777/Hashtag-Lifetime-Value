"""
evaluation.py
==============
Avaliação de modelos, cálculo de métricas de regressão e geração de diagnósticos de performance.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Calcula o conjunto consolidado de métricas de regressão:
    - R² (Coeficiente de Determinação)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - MedAE (Median Absolute Error)
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    r2 = float(r2_score(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    medae = float(median_absolute_error(y_true_arr, y_pred_arr))

    # Proteção para MAPE caso haja zeros no target
    mask_nonzero = y_true_arr != 0
    if np.any(mask_nonzero):
        mape = float(mean_absolute_percentage_error(y_true_arr[mask_nonzero], y_pred_arr[mask_nonzero]))
    else:
        mape = float("nan")

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "MedAE": medae,
    }


def compute_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Gera um DataFrame estruturado contendo valores reais, preditos, resíduos e erro percentual.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr

    df_res = pd.DataFrame({
        "actual": y_true_arr,
        "predicted": y_pred_arr,
        "residual": residuals,
        "abs_error": np.abs(residuals),
        "pct_error": np.where(y_true_arr != 0, np.abs(residuals) / np.abs(y_true_arr), np.nan),
    })
    return df_res


def format_metrics_summary(models_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Consolida múltiplos dicionários de métricas de modelos em um DataFrame ordenado por R² decrescente.
    """
    df = pd.DataFrame.from_dict(models_metrics, orient="index")
    df.index.name = "modelo"
    if "R2" in df.columns:
        df = df.sort_values(by="R2", ascending=False)
    return df.reset_index()
