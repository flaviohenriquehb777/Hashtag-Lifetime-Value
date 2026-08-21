"""
models.py
=========
Treinamento e validação cruzada para modelos de LTV.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def get_requested_models(random_state: int = 42) -> dict[str, object]:
    """Retorna os 4 modelos solicitados para comparação."""
    return {
        "DummyRegressor (média)": DummyRegressor(strategy="mean"),
        "Regressão Linear": LinearRegression(),
        "Regressão Polinomial (grau 2) + Linear": Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linear", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_baseline(X_train, y_train, model_type: str = "linear"):
    """Treina um modelo baseline entre Dummy e Regressão Linear."""
    models = get_requested_models()
    if model_type == "dummy":
        model = models["DummyRegressor (média)"]
    elif model_type == "linear":
        model = models["Regressão Linear"]
    else:
        raise ValueError("model_type deve ser 'dummy' ou 'linear'.")

    model.fit(X_train, y_train)
    return model


def train_advanced(X_train, y_train, model_name: str = "random_forest", params: dict | None = None):
    """Treina um modelo avançado entre Polinomial+Linear e Random Forest."""
    params = params or {}
    if model_name == "polynomial_linear":
        model = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=params.get("degree", 2), include_bias=False)),
                ("linear", LinearRegression()),
            ]
        )
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
    else:
        raise ValueError("model_name deve ser 'polynomial_linear' ou 'random_forest'.")

    model.fit(X_train, y_train)
    return model


def evaluate_models_kfold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Executa validação cruzada KFold=5 e retorna tabela com R² por dobra e média.
    """
    models = get_requested_models(random_state=random_state)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows: list[dict] = []
    for model_name, model in models.items():
        fold_scores: list[float] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train), start=1):
            X_tr = X_train.iloc[train_idx]
            X_va = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_va = y_train.iloc[valid_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            score = r2_score(y_va, y_pred)
            fold_scores.append(float(score))

        row = {"modelo": model_name}
        for fold_idx, score in enumerate(fold_scores, start=1):
            row[f"fold_{fold_idx}"] = score
        row["r2_medio"] = float(np.mean(fold_scores))
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("r2_medio", ascending=False).reset_index(drop=True)
    return results, models


def build_gemini_style_pipelines() -> dict[str, Pipeline]:
    """Replica o desenho de modelagem usado no experimento do Gemini."""
    num_cols = ["valor_1_compra"]
    cat_cols = [
        "Produto Fonte",
        "Fonte Campanha",
        "Sexo",
        "Formacao",
        "dia_semana_compra",
        "mes_compra",
    ]
    pass_cols = ["recorrente_1_compra"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                cat_cols,
            ),
            ("pass", "passthrough", pass_cols),
        ]
    )

    return {
        "1. DummyRegressor (Média)": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "2. Regressão Linear": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "3. Regressão Polinomial (g=2) + Ridge": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        ),
        "4. Random Forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ]
        ),
    }


def evaluate_models_on_final_base(
    df: pd.DataFrame,
    *,
    target_col: str = "LTV",
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Replica o experimento com a ltv_base_final.csv:
    - train_test_split aleatório 80/20
    - preprocessor dentro do pipeline
    - KFold=5 com scoring R²
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipelines = build_gemini_style_pipelines()

    results: dict[str, np.ndarray] = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        results[name] = scores

    df_results = pd.DataFrame(results, index=[f"Fold {i}" for i in range(1, n_splits + 1)])
    summary = pd.DataFrame(
        {
            "Média R²": df_results.mean(),
            "Desvio Padrão": df_results.std(),
        }
    )

    split_info = pd.DataFrame(
        {
            "split": ["train", "test"],
            "linhas": [len(X_train), len(X_test)],
        }
    )

    return df_results, summary, split_info


def evaluate_models_on_final_base_test(
    df: pd.DataFrame,
    *,
    target_col: str = "LTV",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Treina os 4 modelos na base final e compara desempenho no conjunto de teste
    usando R2, RMSE e MAE.
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipelines = build_gemini_style_pipelines()
    rows: list[dict[str, float | str]] = []

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append(
            {
                "modelo": name,
                "R2": float(r2_score(y_test, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
            }
        )

    metrics = (
        pd.DataFrame(rows)
        .sort_values(by=["R2", "RMSE", "MAE"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    split_info = pd.DataFrame(
        {
            "split": ["train", "test"],
            "linhas": [len(X_train), len(X_test)],
        }
    )

    return metrics, split_info


def hyperparameter_tuning(X_train, y_train, model_name: str, n_trials: int = 50):
    """Placeholder mantido para fases futuras."""
    raise NotImplementedError("Tuning não faz parte desta etapa.")


def save_model(model, filepath: str) -> None:
    """Serializa e salva o modelo treinado."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filepath: str):
    """Carrega um modelo serializado."""
    return joblib.load(filepath)
