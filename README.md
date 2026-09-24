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
.venv/bin/python -m pytest       # run the tests (warnings are errors; a few minutes)

# Notebooks, in this order (02 reads the thresholds that 01 writes to reports/metrics.json).
# Each writes its figures to reports/figures/ and can also be opened in JupyterLab.
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_model.ipynb    # data, models, threshold, calibration
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_xai.ipynb      # explanations, sanity checks, local cases
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_actions.ipynb  # model improvement, levers, counterfactuals

reports/build_report.sh          # report.md -> report.pdf (needs pandoc and Google Chrome)
```

## Layout

```
src/         data.py (load and clean) · modeling.py (split, pipeline, protocol, threshold and cost) ·
             xai.py (SHAP and linear contributions per feature, sanity checks, counterfactual search) · plots.py (figure style)
notebooks/   01_model.ipynb · 02_xai.ipynb · 03_actions.ipynb
tests/       pytest tests (each rule was broken on purpose to check that a test fails)
reports/     report.md and build_report.sh (source of the PDF), figures/, metrics.json written by the notebooks
data/        raw data, downloaded by `python -m src.data` (not versioned)
```

## What the project does

Binary classification of telecom churn. A logistic regression and an XGBoost model are trained with a protocol that
avoids leakage (stratified split, nested cross-validation, threshold chosen on out-of-fold predictions, no resampling).
The XGBoost model is explained with TreeSHAP, permutation importance and the coefficients of the linear model, the
explanations are tested with two sanity checks (label randomisation, shuffling the top-ranked features), and they are
used to simplify the model and to identify levers for the retention team, with counterfactuals from DiCE. Every number
in the report comes from the notebooks (figures in `reports/figures/`, headline metrics in `reports/metrics.json`).

## Credits and external resources

- **Data:** IBM sample data sets, "Telco Customer Churn" (link above).
- **Libraries:** scikit-learn, XGBoost, SHAP (TreeSHAP), DiCE (`dice-ml`), pandas, NumPy, SciPy, matplotlib, pytest, JupyterLab.
  Their public documentation was used for the APIs; no code was copied from tutorials or repositories.
- **Methods, cited in the report:** SHAP (Lundberg and Lee, 2017; Lundberg et al., 2020), permutation importance (Fisher,
  Rudin and Dominici, 2019), sanity checks for explanations (Adebayo et al., 2018, whose saliency-map tests for images are
  adapted here to tabular SHAP), diverse counterfactual explanations (Mothilal, Sharma and Tan, 2020).
- **Generative AI:** the project was developed with the help of a generative AI assistant (Claude Code), as the
  assignment allows (section 8). The analysis, code and text are to be understood and defended by the author.
