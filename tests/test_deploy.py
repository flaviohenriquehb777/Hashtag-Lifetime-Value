from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from src.deploy import (
    create_ltv_calculator_workbook,
    default_test_clients,
    fit_linear_calculator_artifacts,
    predict_linear_formula_python,
)


def test_predict_linear_formula_python_returns_float():
    df = pd.read_csv("data/final/ltv_base_final.csv", nrows=5000)
    artifacts = fit_linear_calculator_artifacts(df, random_state=42)
    client = default_test_clients()[0]

    prediction = predict_linear_formula_python(artifacts, client)

    assert isinstance(prediction, float)


def test_create_ltv_calculator_workbook_creates_expected_structure(tmp_path: Path):
    df = pd.read_csv("data/final/ltv_base_final.csv", nrows=5000)
    artifacts = fit_linear_calculator_artifacts(df, random_state=42)
    output_path = tmp_path / "ltv_calculator.xlsx"

    created_path = create_ltv_calculator_workbook(artifacts, output_path)

    assert created_path.exists()

    workbook = load_workbook(created_path, data_only=False)
    assert workbook.sheetnames == ["Calculadora", "Apoio", "Documentacao"]
    assert "LTV_Previsto" in workbook.defined_names
    assert "CAC_Maximo_Recomendado" in workbook.defined_names
