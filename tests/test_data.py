"""Tests for src/data.py: cleaning rules on a tiny synthetic CSV, plus one check on the real data."""
from urllib.error import URLError

import pandas as pd
import pytest

from src.data import download, load_clean

COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]


def write_csv(tmp_path, rows):
    path = tmp_path / "telco.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def test_cleaning_rules(tmp_path):
    # Row 2 is a brand-new customer: tenure 0 and a blank TotalCharges (as in the real file).
    path = write_csv(tmp_path, [
        ["A-1", 12, 50.0, "600.5", "No"],
        ["B-2", 0, 30.0, " ", "Yes"],
    ])
    X, y = load_clean(path)

    assert "customerID" not in X.columns and "Churn" not in X.columns
    assert list(y) == [0, 1]
    assert X["TotalCharges"].dtype == float
    assert list(X["TotalCharges"]) == [600.5, 0.0]


def test_blank_total_charges_with_positive_tenure_is_rejected(tmp_path):
    # Same blank, but the customer HAS been billed: filling with 0 would hide a data error.
    path = write_csv(tmp_path, [["A-1", 12, 50.0, " ", "No"]])
    with pytest.raises(ValueError, match="TotalCharges"):
        load_clean(path)


@pytest.fixture(scope="module")
def real_data():
    try:
        return load_clean(download())
    except URLError:
        pytest.skip("no network and no cached copy of the dataset")


def test_real_dataset_shape_and_churn_rate(real_data):
    X, y = real_data
    assert X.shape == (7043, 19)
    assert set(y.unique()) == {0, 1}
    assert y.mean() == pytest.approx(0.2654, abs=1e-3)
    assert not X.isna().any().any()
