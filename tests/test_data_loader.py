from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import (
    DEFAULT_EXPECTED_COLUMNS,
    EXPECTED_FINAL_DTYPES,
    clean_raw_data,
    load_config,
    load_raw_data,
    prepare_modeling_frame,
    split_data,
    validate_dtypes,
    validate_schema,
    validate_values,
)

@pytest.fixture
def dummy_raw_df():
    """Gera um DataFrame fictício para testes sem dependência de arquivos externos."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(n)],
        "LTV": np.random.uniform(100, 1000, n),
        "data_compra": np.arange(44927, 44927 + n), # Serial Excel para 2023-01-01 em diante
        "valor_1_compra": np.random.uniform(50, 500, n),
        "recorrente_1_compra": np.random.choice([0, 1], n),
        "Produto Fonte": np.random.choice(["Excel", "Power BI", "Python"], n),
        "Fonte Campanha": np.random.choice(["Google", "Facebook", "Instagram"], n),
        "Sexo": np.random.choice(["Masculino", "Feminino", "Outros"], n),
        "Formacao": np.random.choice(["Superior", "Médio", "Técnico"], n),
        "Renda": np.random.randint(2000, 10000, n)
    })

def test_load_config_returns_dict():
    # Criar um config temporário se não existir para o teste
    cfg_path = Path("config/config.yaml")
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            f.write("paths:\n  raw_data: 'data/raw/ltv_base.csv'")
    
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, dict)
    assert "paths" in cfg

def test_validate_schema_success(dummy_raw_df):
    assert validate_schema(dummy_raw_df, DEFAULT_EXPECTED_COLUMNS) is True

def test_clean_raw_data_converts_types(dummy_raw_df):
    df, report = clean_raw_data(dummy_raw_df, rare_category_min_count=0)
    
    assert isinstance(report, dict)
    assert df["LTV"].dtype.kind in {"f", "i"}
    assert df["valor_1_compra"].dtype.kind in {"f", "i"}
    assert pd.api.types.is_datetime64_any_dtype(df["data_compra"])
    assert str(df["ID"].dtype) == "string"
    assert str(df["recorrente_1_compra"].dtype) == "Int64"
    assert str(df["Renda"].dtype) == "Int64"

def test_split_data_temporal(dummy_raw_df):
    df, _ = clean_raw_data(dummy_raw_df, rare_category_min_count=0)
    train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1, date_column="data_compra")

    assert len(train_df) > 0
    assert val_df is not None and len(val_df) > 0
    assert len(test_df) > 0
    assert train_df["data_compra"].max() <= val_df["data_compra"].min()

def test_prepare_modeling_frame(dummy_raw_df):
    df, _ = clean_raw_data(dummy_raw_df, rare_category_min_count=0)
    df_model, rep = prepare_modeling_frame(df, drop_columns=["Renda"])

    assert "Renda" not in df_model.columns
    assert "Renda" in rep["dropped_columns"]
