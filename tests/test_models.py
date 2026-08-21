from __future__ import annotations

import pandas as pd

from src.data_loader import clean_raw_data, split_data
from src.features import prepare_datasets_for_modeling
from src.models import (
    evaluate_models_kfold,
    evaluate_models_on_final_base,
    evaluate_models_on_final_base_test,
)


def test_evaluate_models_kfold_returns_expected_table():
    df_raw = pd.read_csv("data/raw/ltv_base.csv", nrows=1500)
    df, _ = clean_raw_data(df_raw)
    train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1)

    prepared = prepare_datasets_for_modeling(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaling="robust",
        log_target=True,
        categorical_min_frequency=100,
    )

    results, _ = evaluate_models_kfold(prepared.X_train, prepared.y_train, n_splits=5, random_state=42)

    assert results.shape[0] == 4
    assert "modelo" in results.columns
    assert "r2_medio" in results.columns
    for i in range(1, 6):
        assert f"fold_{i}" in results.columns


def test_evaluate_models_on_final_base_returns_fold_and_summary_tables():
    df = pd.read_csv("data/final/ltv_base_final.csv", nrows=2000)
    folds, summary, split_info = evaluate_models_on_final_base(df, n_splits=5, random_state=42)

    assert folds.shape == (5, 4)
    assert summary.shape == (4, 2)
    assert split_info["linhas"].sum() == len(df)


def test_evaluate_models_on_final_base_test_returns_metrics_table():
    df = pd.read_csv("data/final/ltv_base_final.csv", nrows=2000)
    metrics, split_info = evaluate_models_on_final_base_test(df, random_state=42)

    assert metrics.shape == (4, 4)
    assert list(metrics.columns) == ["modelo", "R2", "RMSE", "MAE"]
    assert split_info["linhas"].sum() == len(df)
