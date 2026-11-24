# Explaining customer churn: an end-to-end XAI case study

Final project for **Explainable AI** (Option A), Comillas ICAI, academic year 2026/27.
Author: **Nicolás del Val**.

📄 **Report:** [`reports/report.pdf`](reports/report.pdf) · source: [`reports/report.tex`](reports/report.tex)

A telecom retention team must decide whom to contact with a limited budget, and an auditor
must decide whether to trust the model and its explanations. This project trains a churn
classifier, then *evaluates the explanations* rather than only producing them: TreeSHAP,
permutation importance and DiCE counterfactuals are checked for stability and sanity, and
are then used to simplify the model and examine candidate retention levers.

## At a glance

| Aspect | Detail |
|---|---|
| **Task** | Binary `Churn` classification from customer demographics, services, contract and billing |
| **Data** | [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — 7,043 rows × 21 columns, 26.5 % churn |
| **Models** | Majority baseline, logistic regression (interpretable reference), XGBoost (adopted) |
| **Protocol** | Stratified split, nested cross-validation, out-of-fold predictions, training-only preprocessing, no resampling; cost-based threshold selection (expected cost for 2:1, 5:1 and 10:1 contact/miss ratios); calibration checks |
| **Explanations** | TreeSHAP, permutation importance, DiCE counterfactuals, label randomisation and feature-shuffling sanity checks, deletion-based reference for global rankings |
| **Extension** | Feature-removal study (exploratory: test explanations were inspected before the hypothesis was fixed) and retention-lever analysis |

Key numbers: XGBoost test AUC 0.848 (logistic 0.841) with a small train–CV gap; at the 5:1
cost ratio the adopted model uses a 0.17 threshold and strictly improves on both "contact
nobody" and "contact everybody". The report is the reference for all figures and caveats.

## Repository layout

```
src/         data.py     load and clean the raw data
             modeling.py split, pipelines, nested CV protocol, threshold and cost
             xai.py      SHAP / linear contributions, sanity checks, counterfactual search
             plots.py    shared figure style
notebooks/   01_model.ipynb    data, models, threshold, calibration
             02_xai.ipynb      global and local explanations, sanity checks
             03_actions.ipynb  model simplification, levers, counterfactuals
tests/       pytest suite (warnings are errors)
reports/     report.tex + report.pdf, build_report.sh, figures/, result JSONs
data/        downloaded by `python -m src.data` (not versioned)
```

## Setup

Python **3.12** is required (`dice-ml` does not support 3.13 yet).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

On macOS, XGBoost needs a recent `libomp`; if `import xgboost` fails, run `brew upgrade libomp`.

## Reproduce

```bash
.venv/bin/python -m src.data     # download the dataset to data/raw/ and print a summary
.venv/bin/python -m pytest       # tests (a few minutes)

# Run in this order: 02 reads the thresholds that 01 writes to reports/metrics.json.
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_model.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_xai.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_actions.ipynb

reports/build_report.sh          # report.tex -> report.pdf (pdfLaTeX, two passes)
```

The notebooks write their figures to `reports/figures/` and their numbers to
`reports/metrics.json`, `reports/xai_checks_initial.json` and `reports/actions_metrics.json`;
every number and figure in the report comes from these executed notebooks (seed 42).
`build_report.sh` needs a TeX distribution with the packages listed in the report preamble.

## Credits

- **Data:** IBM sample data sets, "Telco Customer Churn".
- **Libraries:** scikit-learn, XGBoost, SHAP (TreeSHAP), DiCE (`dice-ml`), pandas, NumPy, SciPy,
  matplotlib, pytest, JupyterLab. Public documentation only; no code copied from tutorials.
- **Methods:** SHAP (Lundberg & Lee, 2017; Lundberg et al., 2020), permutation importance
  (Fisher, Rudin & Dominici, 2019), sanity checks for explanations (Adebayo et al., 2018),
  diverse counterfactuals (Mothilal, Sharma & Tan, 2020). Full references in the report.
- **Generative AI:** developed with the help of generative AI assistants (Claude Code and Codex),
  as the assignment allows (section 8); the code and text are understood and defended by the author.
