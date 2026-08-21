from __future__ import annotations

import pandas as pd

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


def test_load_config_returns_dict():
    cfg = load_config("config/config.yaml")
    assert isinstance(cfg, dict)
    assert "paths" in cfg


def test_load_raw_data_has_expected_columns():
    df = load_raw_data("data/raw/ltv_base.csv")
    validate_schema(df, DEFAULT_EXPECTED_COLUMNS)


def test_clean_raw_data_converts_types_and_missing_sentinels():
    df_raw = pd.read_csv("data/raw/ltv_base.csv", nrows=500)
    df, report = clean_raw_data(df_raw)

    assert isinstance(report, dict)
    assert df["LTV"].dtype.kind in {"f", "i"}
    assert df["valor_1_compra"].dtype.kind in {"f", "i"}
    assert pd.api.types.is_datetime64_any_dtype(df["data_compra"])
    assert str(df["ID"].dtype) == "string"
    assert str(df["recorrente_1_compra"].dtype) == "Int64"
    assert str(df["Renda"].dtype) == "Int64"

    assert not (df["Sexo"] == "0").any()
    assert not (df["Formacao"] == "0").any()

    validate_dtypes(df, EXPECTED_FINAL_DTYPES)
    validate_values(df)


def test_split_data_temporal_returns_three_splits():
    df_raw = pd.read_csv("data/raw/ltv_base.csv", nrows=2000)
    df, _ = clean_raw_data(df_raw)
    train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1, date_column="data_compra")

    assert len(train_df) > 0
    assert val_df is not None and len(val_df) > 0
    assert len(test_df) > 0
    assert train_df["data_compra"].max() <= val_df["data_compra"].min()
    assert val_df["data_compra"].max() <= test_df["data_compra"].min()


def test_prepare_modeling_frame_drops_renda_and_multicollinearity():
    df_raw = pd.read_csv("data/raw/ltv_base.csv", nrows=5000)
    df, _ = clean_raw_data(df_raw)
    df_model, rep = prepare_modeling_frame(df)

    assert "Renda" not in df_model.columns
    assert rep["dropped_columns"] == ["Renda"]
    assert "recorrente_1_compra" not in df_model.columns
