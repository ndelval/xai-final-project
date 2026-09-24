"""Tests for src/modeling.py. Synthetic data with a known signal, so every check can be reasoned about."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.modeling import build_pipeline, cv_and_gap, make_models, split

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
