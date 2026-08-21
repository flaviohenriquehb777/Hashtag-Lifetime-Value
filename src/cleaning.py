from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningReport:
    rows_in: int
    rows_out: int
    nulls_before: dict[str, int]
    nulls_after: dict[str, int]
    conversions: dict[str, str]
    notes: list[str]


def coerce_decimal_comma(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def coerce_excel_serial_date(series: pd.Series, origin: str = "1899-12-30") -> pd.Series:
    s = series.astype("string").str.replace(",", ".", regex=False)
    num = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(num, unit="D", origin=origin, errors="coerce")


def replace_sentinels_with_nan(series: pd.Series, sentinels: set[Any]) -> pd.Series:
    return series.replace(list(sentinels), np.nan)


def normalize_with_mapping(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    s = series.astype("string")
    for src, dst in mapping.items():
        s = s.str.replace(src, dst, regex=False)
    s = s.str.strip()
    return s.astype("object")


def iqr_upper_bound(series: pd.Series, multiplier: float = 1.5) -> float:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return float(q3 + multiplier * iqr)


def set_above_to_nan(series: pd.Series, max_value: float) -> pd.Series:
    s = series.copy()
    s = pd.to_numeric(s, errors="coerce")
    s[s > max_value] = np.nan
    return s


def group_rare_categories(
    series: pd.Series,
    *,
    min_count: int = 100,
    other_label: str = "Outros",
) -> pd.Series:
    s = series.astype("object")
    vc = s.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return s.where(~s.isin(rare), other_label)

