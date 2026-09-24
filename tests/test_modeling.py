"""Tests for src/modeling.py. Synthetic data with a known signal, so every check can be reasoned about."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.modeling import (SEED, build_pipeline, churn_metrics, customer_costs, cv_and_gap, cv_fold_scores,
                          expected_cost, make_models, oof_proba, pick_threshold, split)

CONTRACTS = ("monthly", "yearly", "biennial")


def make_frame(rng, n, contracts=CONTRACTS):
    """Two numeric columns (only `tenure` carries signal) and one categorical column."""
    return pd.DataFrame({
        "tenure": rng.integers(0, 72, n).astype(float),
        "noise": rng.normal(size=n),
        "contract": rng.choice(contracts, n),
    })


def label_from_tenure(X, rng):
    """Short tenure -> likely churn, with some noise so the task is not trivially separable."""
    return (X["tenure"] + rng.normal(scale=15, size=len(X)) < 30).astype(int)


def make_data(seed=0, n_train=600, n_test=300):
    rng = np.random.default_rng(seed)
    X_train = make_frame(rng, n_train)
    X_test = make_frame(rng, n_test)
    return X_train, label_from_tenure(X_train, rng), X_test, label_from_tenure(X_test, rng)


ESTIMATORS = {
    "tuned logistic (GridSearchCV)": lambda: make_models()["logistic"],
    "plain xgboost pipeline": lambda: build_pipeline(XGBClassifier(n_estimators=50, max_depth=4, random_state=0)),
}


def test_split_is_stratified_disjoint_and_seeded():
    rng = np.random.default_rng(0)
    X = make_frame(rng, 1000)
    y = pd.Series((rng.random(1000) < 0.2).astype(int))

    X_tr, X_te, y_tr, y_te = split(X, y, seed=1)

    assert (len(X_tr), len(X_te)) == (800, 200)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y.mean()) < 0.005 and abs(y_te.mean() - y.mean()) < 0.005
    same_seed = split(X, y, seed=1)[1].index
    other_seed = split(X, y, seed=2)[1].index
    assert list(same_seed) == list(X_te.index)
    assert list(other_seed) != list(X_te.index)


def test_category_unseen_in_training_is_encoded_as_zeros_with_a_warning():
    X_train, y_train, _, _ = make_data()
    rng = np.random.default_rng(1)
    X_new = make_frame(rng, 5, contracts=("two_year",))  # a level the encoder has never seen

    model = build_pipeline(LogisticRegression(), scale=True).fit(X_train, y_train)

    with pytest.warns(UserWarning, match="unknown categories"):
        proba = model.predict_proba(X_new)[:, 1]
    assert proba.shape == (5,) and np.isfinite(proba).all()


@pytest.mark.parametrize("make_estimator", ESTIMATORS.values(), ids=ESTIMATORS.keys())
def test_everything_computed_from_train_ignores_the_test_set(make_estimator):
    # Two different test sets (other features AND other labels). Anything fitted, selected or
    # scored on train must come out identical; only the test AUC may move. A scaler, encoder or
    # grid search that looked at the test rows would change the train-side numbers.
    X_train, y_train, X_a, y_a = make_data(seed=0)
    _, _, X_b, y_b = make_data(seed=5)

    a = cv_and_gap(make_estimator(), X_train, y_train, X_a, y_a)
    b = cv_and_gap(make_estimator(), X_train, y_train, X_b, y_b)

    train_side = ["train_auc", "cv_auc", "cv_std", "gap", "cv_pr_auc"]
    assert {k: a[k] for k in train_side} == {k: b[k] for k in train_side}
    assert a["test_auc"] != b["test_auc"]  # the two test sets really were different


@pytest.mark.parametrize("make_estimator", ESTIMATORS.values(), ids=ESTIMATORS.keys())
def test_test_auc_is_that_of_a_model_fitted_on_train_only(make_estimator):
    X_train, y_train, X_test, y_test = make_data()

    row = cv_and_gap(make_estimator(), X_train, y_train, X_test, y_test)

    train_only = clone(make_estimator()).fit(X_train, y_train)
    expected = roc_auc_score(y_test, train_only.predict_proba(X_test)[:, 1])
    assert row["test_auc"] == expected


def test_cv_and_gap_depends_on_the_seed_and_only_on_it():
    X_train, y_train, X_test, y_test = make_data()
    run = lambda seed: cv_and_gap(make_models()["logistic"], X_train, y_train, X_test, y_test, seed=seed)

    assert run(1) == run(1)
    assert run(1)["cv_auc"] != run(2)["cv_auc"]


# --- Decision threshold and churn-class metrics -------------------------------------------------
# Six customers, two of them churners. At threshold 0.5 the model gets one churner right (TP), misses
# the other (FN) and wrongly flags two loyal customers (FP); two loyal customers are cleared (TN).
# The counts are all different on purpose, so precision, recall and F1 take three different values
# and swapping any two of them would be noticed.
Y = np.array([0, 0, 0, 0, 1, 1])
P = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])


def test_churn_metrics_are_for_the_churn_class_not_weighted():
    m = churn_metrics(Y, P, threshold=0.5)

    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["f1"] == pytest.approx(0.4)  # a weighted F1 would be 0.514 and hide the churners
    assert m["accuracy"] == pytest.approx(3 / 6)


def test_a_model_that_never_flags_churn_scores_zero_recall_despite_high_accuracy():
    m = churn_metrics(Y, np.zeros(6), threshold=0.5)

    assert m["recall"] == 0 and m["precision"] == 0 and m["f1"] == 0
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_expected_cost_counts_missed_churners_and_wasted_offers():
    # 2 churners missed and 1 loyal customer flagged. The counts differ on purpose: with equal
    # counts, swapping which error carries the ratio would give the same number.
    pred = np.array([0, 0, 0, 1, 0, 0])
    assert list(customer_costs(Y, pred, cost_ratio=5)) == [0, 0, 0, 1, 5, 5]
    assert expected_cost(Y, pred, cost_ratio=5) == pytest.approx((5 * 2 + 1 * 1) / 6)
    assert expected_cost(Y, pred, cost_ratio=1) == pytest.approx(3 / 6)


def test_threshold_falls_as_a_missed_churner_gets_more_expensive():
    # Perfectly calibrated probabilities: the cost-optimal threshold is 1 / (1 + cost_ratio).
    rng = np.random.default_rng(0)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(int)

    thresholds = {r: pick_threshold(y, p, cost_ratio=r) for r in (1, 2, 5, 10)}

    assert thresholds[10] < thresholds[5] < thresholds[2] < thresholds[1]
    for ratio, t in thresholds.items():
        assert t == pytest.approx(1 / (1 + ratio), abs=0.03)


def test_threshold_ties_go_to_the_lowest_value():
    # Every customer churns: any threshold up to 0.3 flags all of them at zero cost. Take the first.
    assert pick_threshold(np.array([1, 1, 1]), np.array([0.3, 0.5, 0.7]), cost_ratio=5) == pytest.approx(0.01)


def test_oof_predictions_come_from_models_that_never_saw_the_row():
    # Labels are pure noise but every row has a unique numeric fingerprint, so a flexible model
    # memorises (most of) the labels of the rows it was fitted on. Out-of-fold predictions cannot.
    # Memorisation is partial because XGBoost bins each feature into 256 values.
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"fingerprint": rng.permutation(400).astype(float), "contract": "monthly"})
    y = pd.Series(rng.integers(0, 2, 400))
    estimator = build_pipeline(XGBClassifier(n_estimators=300, max_depth=6, random_state=0))

    oof = oof_proba(estimator, X, y)
    in_sample = clone(estimator).fit(X, y).predict_proba(X)[:, 1]

    assert oof.shape == (400,) and ((0 <= oof) & (oof <= 1)).all()
    assert roc_auc_score(y, in_sample) > 0.85          # the model can memorise...
    assert abs(roc_auc_score(y, oof) - 0.5) < 0.1      # ...but not out of fold


def test_oof_predictions_are_in_row_order_and_work_with_a_grid_search():
    # Real signal this time: if the rows came back shuffled, the AUC against y would collapse.
    X_train, y_train, _, _ = make_data()

    oof = oof_proba(make_models()["logistic"], X_train, y_train)

    assert oof.shape == (len(y_train),)
    assert roc_auc_score(y_train, oof) > 0.75


# --- Per-fold scores for comparing variants of the model ---------------------------------------

def test_fold_scores_use_the_given_threshold_and_the_same_folds_every_time():
    X_train, y_train, X_test, y_test = make_data()
    estimator = build_pipeline(LogisticRegression(), scale=True)

    everybody = cv_fold_scores(estimator, X_train, y_train, threshold=0.0, cost_ratio=5)
    nobody = cv_fold_scores(estimator, X_train, y_train, threshold=1.01, cost_ratio=5)

    assert len(everybody) == 5                                      # one row per fold
    assert (everybody["recall"] == 1).all() and (nobody["recall"] == 0).all()
    churn_rate = y_train.mean()
    assert everybody["precision"].mean() == pytest.approx(churn_rate, abs=0.02)   # everybody flagged: precision = churn rate
    assert (nobody["precision"] == 0).all()                         # nobody flagged: defined as 0, not an error
    # Contacting everybody wastes an offer on every loyal customer; contacting nobody misses every churner.
    assert everybody["cost"].mean() == pytest.approx(1 - churn_rate, abs=0.02)
    assert nobody["cost"].mean() == pytest.approx(5 * churn_rate, abs=0.05)
    again = cv_fold_scores(estimator, X_train, y_train, threshold=0.0, cost_ratio=5)
    assert everybody.equals(again)


def test_fold_scores_that_do_not_depend_on_the_threshold_match_the_other_protocol_functions():
    # The decision rule of the model-improvement section rests on `auc`, so it must be the same AUC, on the same
    # folds, as cv_and_gap reports; `pr_auc` likewise, and `brier` (lower is better) is checked against sklearn.
    X_train, y_train, X_test, y_test = make_data()
    estimator = build_pipeline(LogisticRegression(), scale=True)

    scores = cv_fold_scores(estimator, X_train, y_train, threshold=0.3, cost_ratio=5)
    reference = cv_and_gap(estimator, X_train, y_train, X_test, y_test)
    brier = -cross_val_score(estimator, X_train, y_train, scoring="neg_brier_score",
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))

    assert scores["auc"].mean() == pytest.approx(reference["cv_auc"])
    assert scores["pr_auc"].mean() == pytest.approx(reference["cv_pr_auc"])
    assert scores["brier"].to_numpy() == pytest.approx(brier)
