"""
data_loader.py
==============
Responsável pela ingestão, validação e split de dados brutos.

Funções planejadas:
    - load_raw_data: Carrega CSV de data/raw/.
    - validate_schema: Valida colunas e tipos esperados.
    - split_data: Divide em treino, validação e teste com estratificação temporal.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .cleaning import (
    CleaningReport,
    coerce_decimal_comma,
    coerce_excel_serial_date,
    group_rare_categories,
    iqr_upper_bound,
    normalize_with_mapping,
    replace_sentinels_with_nan,
    set_above_to_nan,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carrega configurações do projeto."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


DEFAULT_EXPECTED_COLUMNS = [
    "ID",
    "LTV",
    "data_compra",
    "valor_1_compra",
    "recorrente_1_compra",
    "Produto Fonte",
    "Fonte Campanha",
    "Sexo",
    "Formacao",
    "Renda",
]


DEFAULT_TEXT_NORMALIZATION = {
    "Lanþamento": "Lançamento",
    "Trßfego Direto": "Tráfego Direto",
    "Anßlise de Dados": "Análise de Dados",
    "CiÛncia de Dados": "Ciência de Dados",
    "VitalÝcio": "Vitalício",
    "TÚcnico": "Técnico",
    "MÚdio": "Médio",
    "NÒo": "Não",
    "nÒo": "não",
    "NÒo-binßrio": "Não-binário",
    "binßrio": "binário",
}

DEFAULT_MODEL_DROP_COLUMNS = ["Renda"]
DEFAULT_COLLINEAR_PAIRS = [("valor_1_compra", "recorrente_1_compra")]
EXPECTED_FINAL_DTYPES = {
    "ID": "string",
    "LTV": "Float64",
    "data_compra": "datetime64[ns]",
    "valor_1_compra": "Float64",
    "recorrente_1_compra": "Int64",
    "Produto Fonte": "string",
    "Fonte Campanha": "string",
    "Sexo": "string",
    "Formacao": "string",
    "Renda": "Int64",
}


def load_raw_data(filepath: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Carrega dados brutos a partir de um arquivo CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, encoding=encoding)


def validate_schema(df: pd.DataFrame, expected_columns: list) -> bool:
    """Valida se o DataFrame contém as colunas esperadas."""
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    return True


def clean_raw_data(
    df_raw: pd.DataFrame,
    *,
    text_normalization: dict[str, str] | None = None,
    renda_upper_cap_strategy: str = "iqr",
    renda_absolute_max: float = 1_000_000,
    rare_category_min_count: int = 100,
) -> tuple[pd.DataFrame, dict]:
    df = df_raw.copy()

    nulls_before = df.isna().sum().to_dict()
    conversions: dict[str, str] = {}
    notes: list[str] = []

    validate_schema(df, DEFAULT_EXPECTED_COLUMNS)

    df["ID"] = df["ID"].astype("string").str.strip()
    conversions["ID"] = "object->string"

    df["LTV"] = coerce_decimal_comma(df["LTV"])
    conversions["LTV"] = "decimal_comma->float"

    df["valor_1_compra"] = coerce_decimal_comma(df["valor_1_compra"])
    conversions["valor_1_compra"] = "decimal_comma->float"

    df["data_compra"] = coerce_excel_serial_date(df["data_compra"])
    conversions["data_compra"] = "excel_serial->datetime"

    df["Sexo"] = replace_sentinels_with_nan(df["Sexo"], {"0"})
    df["Formacao"] = replace_sentinels_with_nan(df["Formacao"], {"0"})

    df["Renda"] = pd.to_numeric(df["Renda"], errors="coerce")
    df["Renda"] = set_above_to_nan(df["Renda"], renda_absolute_max)

    if renda_upper_cap_strategy == "iqr":
        renda_upper = iqr_upper_bound(df["Renda"].dropna())
        df["Renda"] = set_above_to_nan(df["Renda"], renda_upper)
        notes.append(f"renda_upper_iqr={renda_upper:.2f}")

    mapping = text_normalization or DEFAULT_TEXT_NORMALIZATION
    for col in ["Produto Fonte", "Fonte Campanha", "Sexo", "Formacao"]:
        df[col] = normalize_with_mapping(df[col], mapping)

    if rare_category_min_count and rare_category_min_count > 0:
        df["Produto Fonte"] = group_rare_categories(df["Produto Fonte"], min_count=rare_category_min_count, other_label="Outros")
        df["Formacao"] = group_rare_categories(df["Formacao"], min_count=rare_category_min_count, other_label="Outros")
        df["Sexo"] = df["Sexo"].replace({"Não quero declarar": "Não declarado"}).astype("object")
        df["Sexo"] = group_rare_categories(df["Sexo"], min_count=rare_category_min_count, other_label="Outros")

    df["recorrente_1_compra"] = pd.to_numeric(df["recorrente_1_compra"], errors="coerce").astype("Int64")
    df["Renda"] = df["Renda"].astype("Int64")

    for col in ["Produto Fonte", "Fonte Campanha", "Sexo", "Formacao"]:
        df[col] = df[col].astype("string")

    conversions["Renda"] = "numeric->Int64"
    conversions["recorrente_1_compra"] = "numeric->Int64"
    for col in ["Produto Fonte", "Fonte Campanha", "Sexo", "Formacao"]:
        conversions[col] = "normalized->string"

    nulls_after = df.isna().sum().to_dict()
    report = CleaningReport(
        rows_in=int(df_raw.shape[0]),
        rows_out=int(df.shape[0]),
        nulls_before={k: int(v) for k, v in nulls_before.items()},
        nulls_after={k: int(v) for k, v in nulls_after.items()},
        conversions=conversions,
        notes=notes,
    )

    return df, asdict(report)


def validate_values(df: pd.DataFrame) -> bool:
    if (df["LTV"] < 0).any():
        raise ValueError("LTV contém valores negativos.")
    if (df["valor_1_compra"] < 0).any():
        raise ValueError("valor_1_compra contém valores negativos.")
    if not set(df["recorrente_1_compra"].dropna().unique().tolist()).issubset({0, 1}):
        raise ValueError("recorrente_1_compra deve ser binária (0/1).")
    if df["data_compra"].isna().any():
        raise ValueError("data_compra contém valores inválidos após conversão.")
    if df["ID"].duplicated().any():
        raise ValueError("ID contém duplicados.")
    return True


def validate_dtypes(
    df: pd.DataFrame,
    expected_dtypes: dict[str, str] | None = None,
) -> bool:
    expected = expected_dtypes or EXPECTED_FINAL_DTYPES

    for col, expected_dtype in expected.items():
        if col not in df.columns:
            raise ValueError(f"Coluna ausente para validação de tipo: {col}")

        if expected_dtype == "datetime64[ns]":
            if not pd.api.types.is_datetime64_ns_dtype(df[col]):
                raise TypeError(f"{col} deveria ser datetime64[ns], mas está como {df[col].dtype}.")
            continue

        actual = str(df[col].dtype)
        if actual != expected_dtype:
            raise TypeError(f"{col} deveria ser {expected_dtype}, mas está como {actual}.")

    return True


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
    date_column: str | None = "data_compra",
    stratify_column: str | None = None,
) -> tuple:
    """Divide os dados em treino/validação/teste."""
    if date_column and date_column in df.columns:
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        n = len(df_sorted)
        n_test = int(round(n * test_size))
        n_val = int(round(n * validation_size))
        if n_test <= 0 or n - n_test <= 0:
            raise ValueError("test_size inválido para o tamanho do dataset.")
        if n_val < 0 or n - n_test - n_val <= 0:
            raise ValueError("validation_size inválido para o tamanho do dataset.")

        test_df = df_sorted.iloc[n - n_test :].copy()
        val_df = df_sorted.iloc[n - n_test - n_val : n - n_test].copy() if n_val > 0 else None
        train_df = df_sorted.iloc[: n - n_test - n_val].copy()
        return train_df, val_df, test_df

    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
    train_val, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if validation_size and validation_size > 0:
        val_ratio = validation_size / (1 - test_size)
        stratify_tv = (
            train_val[stratify_column]
            if stratify_column and stratify_column in train_val.columns
            else None
        )
        train_df, val_df = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify_tv,
        )
        return train_df, val_df, test_df

    return train_val, None, test_df


def load_clean_validate_split(
    config_path: str = "config/config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict]:
    config = load_config(config_path)
    raw_path = config["paths"]["raw_data"]
    df_raw = load_raw_data(raw_path)
    df, report = clean_raw_data(df_raw)
    validate_dtypes(df)
    validate_values(df)

    split_cfg = config.get("data_split", {})
    train_df, val_df, test_df = split_data(
        df,
        test_size=float(split_cfg.get("test_size", 0.2)),
        validation_size=float(split_cfg.get("validation_size", 0.1)),
        random_state=int(split_cfg.get("random_state", 42)),
        date_column="data_compra",
        stratify_column=split_cfg.get("stratify_column"),
    )

    return train_df, val_df, test_df, report


def prepare_modeling_frame(
    df: pd.DataFrame,
    *,
    drop_columns: list[str] | None = None,
    drop_collinear_pairs: list[tuple[str, str]] | None = None,
    corr_threshold: float = 0.75,
) -> tuple[pd.DataFrame, dict]:
    drops = drop_columns or DEFAULT_MODEL_DROP_COLUMNS
    pairs = drop_collinear_pairs or DEFAULT_COLLINEAR_PAIRS

    df_model = df.drop(columns=drops, errors="ignore")

    num = df_model.select_dtypes(include=[np.number]).corr().abs()
    dropped_due_to_corr: list[str] = []
    for a, b in pairs:
        if a in num.columns and b in num.columns:
            if float(num.loc[a, b]) >= corr_threshold and b in df_model.columns:
                df_model = df_model.drop(columns=[b])
                dropped_due_to_corr.append(b)

    report = {
        "dropped_columns": [c for c in drops if c in df.columns],
        "dropped_due_to_multicollinearity": dropped_due_to_corr,
        "corr_threshold": float(corr_threshold),
    }

    return df_model, report
