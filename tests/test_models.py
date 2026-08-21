from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.data_loader import clean_raw_data, split_data
from src.features import prepare_datasets_for_modeling
from src.models import (
    evaluate_models_kfold,
    evaluate_models_on_final_base,
    evaluate_models_on_final_base_test,
)

@pytest.fixture
def dummy_clean_df():
    """Gera um DataFrame limpo para testes de modelos."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(n)],
        "LTV": np.random.uniform(100, 1000, n),
        "data_compra": dates,
        "valor_1_compra": np.random.uniform(50, 500, n),
        "recorrente_1_compra": np.random.choice([0, 1], n),
        "Produto Fonte": np.random.choice(["Excel", "Power BI", "Python"], n),
        "Fonte Campanha": np.random.choice(["Google", "Facebook", "Instagram"], n),
        "Sexo": np.random.choice(["Masculino", "Feminino", "Outros"], n),
        "Formacao": np.random.choice(["Superior", "Médio", "Técnico"], n),
        "mes_compra": np.random.randint(1, 13, n),
        "dia_semana_compra": np.random.randint(0, 7, n)
    })
    for col in ["Produto Fonte", "Fonte Campanha", "Sexo", "Formacao"]:
        df[col] = df[col].astype("string")
    df["recorrente_1_compra"] = df["recorrente_1_compra"].astype("Int64")
    return df

def test_evaluate_models_kfold_returns_expected_table(dummy_clean_df):
    train_df, val_df, test_df = split_data(dummy_clean_df, test_size=0.2, validation_size=0.1)

    prepared = prepare_datasets_for_modeling(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaling="robust",
        log_target=True,
        categorical_min_frequency=None, # Desativar agrupamento raro para teste pequeno
    )

    results, _ = evaluate_models_kfold(prepared.X_train, prepared.y_train, n_splits=5, random_state=42)

    assert results.shape[0] == 4
    assert "modelo" in results.columns
    assert "r2_medio" in results.columns
    for i in range(1, 6):
        assert f"fold_{i}" in results.columns

def test_evaluate_models_on_final_base_returns_fold_and_summary_tables(dummy_clean_df):
    folds, summary, split_info = evaluate_models_on_final_base(dummy_clean_df, n_splits=5, random_state=42)

    assert folds.shape == (5, 4)
    assert summary.shape == (4, 2)
    assert split_info["linhas"].sum() == len(dummy_clean_df)

def test_evaluate_models_on_final_base_test_returns_metrics_table(dummy_clean_df):
    metrics, split_info = evaluate_models_on_final_base_test(dummy_clean_df, random_state=42)

    assert metrics.shape == (4, 4)
    assert list(metrics.columns) == ["modelo", "R2", "RMSE", "MAE"]
    assert split_info["linhas"].sum() == len(dummy_clean_df)
