"""
features.py
===========
Pipeline de engenharia de atributos e pré-processamento para modelagem de LTV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler


DEFAULT_TARGET_COLUMN = "LTV"
DEFAULT_ID_COLUMN = "ID"
DEFAULT_DATE_COLUMN = "data_compra"
DEFAULT_DROP_COLUMNS = ["Renda", "recorrente_1_compra"]
DEFAULT_CATEGORICAL_COLUMNS = [
    "Produto Fonte",
    "Fonte Campanha",
    "Sexo",
    "Formacao",
]
DEFAULT_NUMERICAL_COLUMNS = ["valor_1_compra"]
TEMPORAL_FEATURE_COLUMNS = [
    "purchase_year",
    "purchase_month",
    "purchase_quarter",
    "purchase_dayofweek",
    "purchase_is_weekend",
    "days_since_train_start",
]


@dataclass
class PreparedFeatureSet:
    X_train: pd.DataFrame
    X_val: pd.DataFrame | None
    X_test: pd.DataFrame | None
    y_train: pd.Series
    y_val: pd.Series | None
    y_test: pd.Series | None
    pipeline: Pipeline
    metadata: dict


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Remove colunas que não devem seguir para a modelagem."""

    def __init__(self, drop_columns: list[str] | None = None):
        self.drop_columns = drop_columns or []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = [col for col in X.columns if col not in self.drop_columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns_].copy()


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extrai atributos temporais sem usar informação futura do conjunto de validação/teste."""

    def __init__(self, date_col: str = DEFAULT_DATE_COLUMN, drop_original: bool = True):
        self.date_col = date_col
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        dates = pd.to_datetime(X[self.date_col], errors="coerce")
        self.reference_date_ = dates.min()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        dates = pd.to_datetime(X_out[self.date_col], errors="coerce")

        X_out["purchase_year"] = dates.dt.year.astype("Float64")
        X_out["purchase_month"] = dates.dt.month.astype("Float64")
        X_out["purchase_quarter"] = dates.dt.quarter.astype("Float64")
        X_out["purchase_dayofweek"] = dates.dt.dayofweek.astype("Float64")
        X_out["purchase_is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype("Float64")
        X_out["days_since_train_start"] = (dates - self.reference_date_).dt.days.astype("Float64")

        if self.drop_original:
            X_out = X_out.drop(columns=[self.date_col])

        return X_out


def create_rfm_features(
    df: pd.DataFrame,
    customer_col: str = DEFAULT_ID_COLUMN,
    date_col: str = DEFAULT_DATE_COLUMN,
    monetary_col: str = "valor_1_compra",
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cria agregações RFM para bases transacionais."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    ref = reference_date or (work[date_col].max() + pd.Timedelta(days=1))

    rfm = (
        work.groupby(customer_col)
        .agg(
            recency_days=(date_col, lambda x: (ref - x.max()).days),
            frequency=(customer_col, "size"),
            monetary=(monetary_col, "sum"),
        )
        .reset_index()
    )
    return rfm


def create_temporal_features(df: pd.DataFrame, date_col: str = DEFAULT_DATE_COLUMN) -> pd.DataFrame:
    """Cria features temporais determinísticas a partir da data de compra."""
    transformer = TemporalFeatureExtractor(date_col=date_col, drop_original=False)
    return transformer.fit_transform(df)


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str = "onehot",
    min_frequency: int = 100,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Aplica OneHotEncoder com proteção adicional para categorias infrequentes."""
    if strategy != "onehot":
        raise ValueError("A estratégia suportada nesta fase é apenas 'onehot'.")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        min_frequency=min_frequency,
        sparse_output=False,
    )
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(columns),
        index=df.index,
    )
    return encoded_df, encoder


def build_feature_pipeline(
    *,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
    scaling: str = "robust",
    date_col: str = DEFAULT_DATE_COLUMN,
    drop_columns: list[str] | None = None,
    categorical_min_frequency: int = 100,
) -> Pipeline:
    """Monta o pipeline de engenharia + pré-processamento."""
    categorical_columns = categorical_columns or DEFAULT_CATEGORICAL_COLUMNS
    numerical_columns = numerical_columns or DEFAULT_NUMERICAL_COLUMNS
    drop_columns = [DEFAULT_ID_COLUMN, *(drop_columns or DEFAULT_DROP_COLUMNS)]

    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("scaling deve ser 'standard' ou 'robust'.")

    numeric_features = [*numerical_columns, *TEMPORAL_FEATURE_COLUMNS]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "to_object",
                FunctionTransformer(
                    lambda X: X.astype("object").where(pd.notna(X), np.nan),
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    min_frequency=categorical_min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("selector", FeatureSelector(drop_columns=drop_columns)),
            ("temporal", TemporalFeatureExtractor(date_col=date_col, drop_original=True)),
            ("preprocessor", preprocessor),
        ]
    )


def transform_target(y: pd.Series, use_log: bool = False) -> pd.Series:
    """Transforma o alvo opcionalmente com log1p."""
    y_series = pd.Series(y).astype("Float64")
    if use_log:
        return np.log1p(y_series).astype("Float64")
    return y_series


def inverse_transform_target(y: pd.Series | np.ndarray, use_log: bool = False) -> np.ndarray:
    """Reverte a transformação do alvo."""
    arr = np.asarray(y, dtype=float)
    if use_log:
        return np.expm1(arr)
    return arr


def _to_feature_frame(pipeline: Pipeline, transformed: np.ndarray, index: pd.Index) -> pd.DataFrame:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    return pd.DataFrame(transformed, columns=clean_names, index=index)


def prepare_datasets_for_modeling(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    *,
    target_col: str = DEFAULT_TARGET_COLUMN,
    scaling: str = "robust",
    log_target: bool = False,
    categorical_min_frequency: int = 100,
) -> PreparedFeatureSet:
    """
    Ajusta o pipeline somente no treino e transforma treino/validação/teste.
    """
    X_train_raw = train_df.drop(columns=[target_col]).copy()
    y_train = transform_target(train_df[target_col], use_log=log_target)

    pipeline = build_feature_pipeline(
        scaling=scaling,
        categorical_min_frequency=categorical_min_frequency,
    )
    pipeline.fit(X_train_raw, y_train)

    X_train = _to_feature_frame(pipeline, pipeline.transform(X_train_raw), train_df.index)

    X_val = None
    y_val = None
    if val_df is not None:
        X_val_raw = val_df.drop(columns=[target_col]).copy()
        X_val = _to_feature_frame(pipeline, pipeline.transform(X_val_raw), val_df.index)
        y_val = transform_target(val_df[target_col], use_log=log_target)

    X_test = None
    y_test = None
    if test_df is not None:
        X_test_raw = test_df.drop(columns=[target_col]).copy()
        X_test = _to_feature_frame(pipeline, pipeline.transform(X_test_raw), test_df.index)
        y_test = transform_target(test_df[target_col], use_log=log_target)

    metadata = {
        "target_col": target_col,
        "log_target": log_target,
        "scaling": scaling,
        "categorical_min_frequency": categorical_min_frequency,
        "dropped_columns_before_pipeline": [DEFAULT_ID_COLUMN, *DEFAULT_DROP_COLUMNS],
        "temporal_features": TEMPORAL_FEATURE_COLUMNS,
        "fit_only_on_train": True,
    }

    return PreparedFeatureSet(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        pipeline=pipeline,
        metadata=metadata,
    )
