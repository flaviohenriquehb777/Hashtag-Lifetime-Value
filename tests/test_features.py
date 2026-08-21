from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.data_loader import clean_raw_data, split_data
from src.features import (
    TEMPORAL_FEATURE_COLUMNS,
    create_temporal_features,
    inverse_transform_target,
    prepare_datasets_for_modeling,
    transform_target,
)

@pytest.fixture
def dummy_raw_df():
    """Gera um DataFrame fictício para testes de features."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(n)],
        "LTV": np.random.uniform(100, 1000, n),
        "data_compra": np.arange(44927, 44927 + n),
        "valor_1_compra": np.random.uniform(50, 500, n),
        "recorrente_1_compra": np.random.choice([0, 1], n),
        "Produto Fonte": np.random.choice(["Excel", "Power BI", "Python"], n),
        "Fonte Campanha": np.random.choice(["Google", "Facebook", "Instagram"], n),
        "Sexo": np.random.choice(["Masculino", "Feminino", "Outros"], n),
        "Formacao": np.random.choice(["Superior", "Médio", "Técnico"], n),
        "Renda": np.random.randint(2000, 10000, n)
    })

def test_create_temporal_features_adds_expected_columns(dummy_raw_df):
    df, _ = clean_raw_data(dummy_raw_df, rare_category_min_count=0)
    enriched = create_temporal_features(df, date_col="data_compra")

    for col in TEMPORAL_FEATURE_COLUMNS[:-1]:
        assert col in enriched.columns

def test_prepare_datasets_for_modeling_fits_only_on_train(dummy_raw_df):
    df, _ = clean_raw_data(dummy_raw_df, rare_category_min_count=0)
    train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1)

    val_df = val_df.copy()
    val_df.loc[val_df.index[0], "Produto Fonte"] = "Categoria Nunca Vista"

    prepared = prepare_datasets_for_modeling(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaling="robust",
        log_target=True,
        categorical_min_frequency=None,
    )

    assert list(prepared.X_train.columns) == list(prepared.X_val.columns)
    assert list(prepared.X_train.columns) == list(prepared.X_test.columns)
    assert not any("Categoria Nunca Vista" in col for col in prepared.X_train.columns)
    assert "Renda" not in prepared.X_train.columns
    assert "recorrente_1_compra" not in prepared.X_train.columns
    assert prepared.metadata["fit_only_on_train"] is True

def test_target_log_transform_is_reversible():
    y = pd.Series([10.0, 100.0, 1000.0], dtype="Float64")
    logged = transform_target(y, use_log=True)
    restored = inverse_transform_target(logged, use_log=True)

    assert np.allclose(restored, y.astype(float).to_numpy())
