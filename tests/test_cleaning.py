import numpy as np
import pandas as pd
from src.cleaning import (
    coerce_decimal_comma,
    coerce_excel_serial_date,
    replace_sentinels_with_nan,
    normalize_with_mapping,
    iqr_upper_bound,
    set_above_to_nan,
    group_rare_categories,
)


def test_coerce_decimal_comma():
    s = pd.Series(["10,50", "20,00", "invalid", "100.25"])
    res = coerce_decimal_comma(s)
    assert res.iloc[0] == 10.50
    assert res.iloc[1] == 20.00
    assert pd.isna(res.iloc[2])
    assert res.iloc[3] == 100.25


def test_coerce_excel_serial_date():
    s = pd.Series(["44562", "invalid"])
    res = coerce_excel_serial_date(s)
    assert pd.notna(res.iloc[0])
    assert pd.isna(res.iloc[1])


def test_replace_sentinels_with_nan():
    s = pd.Series(["99", "N/A", "Valid"])
    res = replace_sentinels_with_nan(s, {"99", "N/A"})
    assert pd.isna(res.iloc[0])
    assert pd.isna(res.iloc[1])
    assert res.iloc[2] == "Valid"


def test_normalize_with_mapping():
    s = pd.Series(["  Sao Paulo  ", "Rio  "])
    res = normalize_with_mapping(s, {"Sao Paulo": "SP", "Rio": "RJ"})
    assert res.iloc[0] == "SP"
    assert res.iloc[1] == "RJ"


def test_iqr_upper_bound_and_set_above():
    s = pd.Series([10, 12, 11, 13, 12, 14, 100])
    ub = iqr_upper_bound(s, multiplier=1.5)
    assert ub < 100
    res = set_above_to_nan(s, ub)
    assert pd.isna(res.iloc[-1])
    assert not pd.isna(res.iloc[0])


def test_group_rare_categories():
    s = pd.Series(["A"] * 105 + ["B"] * 10 + ["C"] * 5)
    res = group_rare_categories(s, min_count=50, other_label="Outros")
    assert (res == "A").sum() == 105
    assert (res == "Outros").sum() == 15
