# Explaining customer churn: an end-to-end XAI case study

Final project for *Explainable AI* (Comillas ICAI, 2026/27), Option A.

**Task.** Binary classification: given a telecom customer, will they leave the service (churn)?
**Data.** IBM Telco Customer Churn: 7,043 customers x 21 columns, 26.5 % churn
([source](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)).
**Stakeholder.** The retention team of an operator, who decides with a limited budget whom to contact and with which offer, and whoever audits the model. Knowing *why* a customer is at risk is what makes the right offer possible.

## Setup

Python 3.12 is required (`dice-ml` does not support 3.13 yet).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

On macOS, XGBoost needs a recent `libomp`; if `import xgboost` fails, run `brew upgrade libomp`.

## Run

```bash
.venv/bin/python -m src.data     # download the dataset to data/raw/ and print a summary
.venv/bin/python -m pytest       # run the tests (~10 s)

# Data, models and evaluation (~2 min). Writes reports/metrics.json.
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_model.ipynb
```

## Layout

```
src/         data loading and cleaning (data.py), split, pipeline and evaluation protocol (modeling.py)
notebooks/   01_model.ipynb: task, data, protocol, baselines and results
tests/       pytest tests
reports/     metrics.json written by the notebooks
data/        raw data, downloaded by `python -m src.data` (not versioned)
```

## Credits and external resources

- Dataset: IBM sample data sets (link above).
