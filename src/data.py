"""Download and clean the IBM Telco Customer Churn dataset.

Cleaning here is row-wise and rule-based (parse a column, use a fixed value for customers not
billed yet). Nothing is learned from the data, so doing it before the train/test split cannot
leak. Anything that is fitted (imputers, encoders, scalers) lives in the Pipeline in modeling.py.

Run `python -m src.data` to download the file and print a one-line summary.
"""
import urllib.request
from pathlib import Path

import pandas as pd

# IBM sample data set, public GitHub mirror (no login needed).
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv"
EXPECTED_SHAPE = (7043, 21)


def download(path: Path = RAW_PATH) -> Path:
    """Fetch the raw CSV once and check it has the expected shape."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    shape = pd.read_csv(path).shape
    if shape != EXPECTED_SHAPE:
        raise ValueError(f"{path} has shape {shape}, expected {EXPECTED_SHAPE}: delete it and retry")
    return path


def load_clean(path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return the features X (19 columns) and the target y (1 = the customer churned)."""
    df = pd.read_csv(path)

    # TotalCharges is text because 11 rows hold a blank. In the real file all of them are
    # customers with tenure 0 (signed up, not billed yet), so 0 is their true value. If a blank
    # ever shows up for a billed customer it is a data error, and we stop instead of hiding it.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if (df.loc[total.isna(), "tenure"] != 0).any():
        raise ValueError("TotalCharges is blank for a customer with tenure > 0")
    df["TotalCharges"] = total.fillna(0.0)

    y = (df.pop("Churn") == "Yes").astype(int)
    X = df.drop(columns="customerID")
    return X, y


if __name__ == "__main__":
    X, y = load_clean(download())
    print(f"{len(X)} rows, churn {y.mean():.3f}")
