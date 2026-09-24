"""Tests for src/xai.py. The risk is in the bookkeeping: contributions must stay additive and end up on the right feature."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.modeling import build_pipeline
from src.xai import (MAX_RANDOMIZED_SPEARMAN, RANDOM_QUANTILE, auc_drop, bootstrap_rows, by_original_feature,
                     importance_under_refits, linear_contributions, median_case, shap_by_feature,
                     shuffle_labels, top_k_vs_random_k)


def make_data(n=600, seed=0):
    """Columns in an order where the text columns come BEFORE the numeric ones, and a shuffled index."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "contract": rng.choice(["monthly", "yearly", "biennial"], n),
        "tenure": rng.integers(0, 72, n).astype(float),
        "partner": rng.choice(["Yes", "No"], n),
        "noise": rng.normal(size=n),
    }, index=rng.permutation(n) + 1000)
    y = (X["tenure"] + 10 * (X["contract"] == "monthly") + rng.normal(scale=15, size=n) < 35).astype(int)
    return X, y


def test_levels_are_added_to_the_feature_they_belong_to():
    # "Streaming" is a prefix of "StreamingTV": a careless match would give the TV levels to the wrong feature.
    columns = ["tenure", "Streaming", "StreamingTV", "Contract"]
    names = ["num__tenure", "cat__Streaming_Yes", "cat__StreamingTV_Yes", "cat__StreamingTV_No", "cat__Contract_One year"]
    values = np.array([[1.0, 2.0, 4.0, 8.0, 16.0]])

    grouped = by_original_feature(values, names, columns)

    assert list(grouped.columns) == columns
    assert list(grouped.iloc[0]) == [1.0, 2.0, 12.0, 16.0]


def test_a_column_that_belongs_to_no_feature_is_an_error():
    with pytest.raises(ValueError, match="cat__Mystery_Yes"):
        by_original_feature(np.zeros((1, 1)), ["cat__Mystery_Yes"], ["tenure"])


def test_shap_values_are_additive_and_one_column_per_original_feature():
    X, y = make_data()
    model = build_pipeline(XGBClassifier(n_estimators=40, max_depth=2, random_state=0)).fit(X, y)

    values, base = shap_by_feature(model, X, X)

    assert list(values.columns) == list(X.columns)   # 4 features, not 1 + 1 + 3 + 1 one-hot columns
    assert values.index.equals(X.index)
    margin = base + values.sum(axis=1)
    np.testing.assert_allclose(expit(margin), model.predict_proba(X)[:, 1], atol=1e-5)


def test_shap_is_relative_to_the_average_reference_customer():
    # Additivity holds for ANY base value, so it cannot catch a wrong reference. What defines the reference is
    # that each feature's contribution averages to 0 over it and that the base is its average log-odds.
    X, y = make_data()
    model = build_pipeline(XGBClassifier(n_estimators=40, max_depth=2, random_state=0)).fit(X, y)
    reference, others = X.iloc[:300], X.iloc[300:]

    on_reference, base = shap_by_feature(model, reference, reference)
    on_others, base_others = shap_by_feature(model, others, reference)

    assert (on_reference.mean().abs() < 1e-5).all()
    assert base == pytest.approx(logit(model.predict_proba(reference)[:, 1]).mean(), abs=1e-4)
    assert base_others == pytest.approx(base)     # same reference, same base, whoever is explained
    np.testing.assert_allclose(expit(base_others + on_others.sum(axis=1)), model.predict_proba(others)[:, 1], atol=1e-5)


def test_shap_puts_the_weight_on_the_features_that_carry_the_signal():
    X, y = make_data()
    model = build_pipeline(XGBClassifier(n_estimators=40, max_depth=2, random_state=0)).fit(X, y)

    values, _ = shap_by_feature(model, X, X)

    importance = values.abs().mean()
    assert importance["tenure"] > 5 * importance["noise"]
    assert importance["contract"] > importance["noise"]   # the monthly contract also raises the risk


def test_linear_contributions_are_additive_and_relative_to_the_reference_set():
    X, y = make_data()
    model = build_pipeline(LogisticRegression(), scale=True).fit(X, y)
    older = X.assign(tenure=X["tenure"] + 30)   # same customers, 30 months older

    contrib, base = linear_contributions(model, older, X)

    assert list(contrib.columns) == list(X.columns) and contrib.index.equals(X.index)
    np.testing.assert_allclose(base + contrib.sum(axis=1), model.decision_function(older), atol=1e-9)
    # Relative to the reference customers (X), older customers push tenure's contribution clearly away
    # from zero. Centring on the explained set itself would make its average contribution exactly 0.
    assert abs(contrib["tenure"].mean()) > 0.5
    reference_contrib, _ = linear_contributions(model, X, X)
    assert abs(reference_contrib["tenure"].mean()) < 1e-9


# --- Sanity checks of the explanations ---------------------------------------------------------
# Synthetic data with a known truth: `signal_a` and `signal_b` drive the label, the rest is noise.

def make_signal_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({name: rng.normal(size=n) for name in ["signal_a", "signal_b", "noise_1", "noise_2", "noise_3", "noise_4"]})
    y = pd.Series((X["signal_a"] + 0.8 * X["signal_b"] + rng.normal(scale=0.7, size=n) > 0).astype(int))
    return X, y


def small_xgb():
    return build_pipeline(XGBClassifier(n_estimators=60, max_depth=3, random_state=0))


def test_shuffling_labels_makes_the_explanation_collapse():
    X, y = make_signal_data()
    X_train, X_test, y_train = X.iloc[:800], X.iloc[800:], y.iloc[:800]
    real, _ = shap_by_feature(small_xgb().fit(X_train, y_train), X_test, X_train)
    real = real.abs().mean()

    shuffled = importance_under_refits(small_xgb, X_train, y_train, X_test, shuffle_labels, n_refits=8)

    spearman = shuffled.apply(lambda row: row.corr(real, method="spearman"), axis=1)
    assert real.nlargest(2).index.isin(["signal_a", "signal_b"]).all()         # the real model finds the truth
    assert spearman.mean() < MAX_RANDOMIZED_SPEARMAN                            # ...and the shuffled ones do not
    # Control: refits that keep the labels (only the rows are resampled) must stay much closer to the real
    # ranking, otherwise a low Spearman could just mean "the ranking is noisy from one fit to the next".
    # (One refit alone is noisy with 6 features, so the comparison is between means.)
    resampled = importance_under_refits(small_xgb, X_train, y_train, X_test, bootstrap_rows, n_refits=8)
    control = resampled.apply(lambda row: row.corr(real, method="spearman"), axis=1)
    assert control.mean() > spearman.mean() + 0.4


def test_the_ranking_of_the_true_features_beats_random_features_when_they_are_shuffled():
    X, y = make_signal_data()
    X_train, X_test, y_train, y_test = X.iloc[:800], X.iloc[800:], y.iloc[:800], y.iloc[800:]
    model = small_xgb().fit(X_train, y_train)
    good_ranking = ["signal_a", "signal_b", "noise_1", "noise_2", "noise_3", "noise_4"]

    drop, random_drops = top_k_vs_random_k(model, X_test, y_test, good_ranking, k=2)

    assert drop > np.quantile(random_drops, RANDOM_QUANTILE)


def test_a_ranking_that_puts_noise_first_does_not_pass_the_same_check():
    # Same model, same data, same k: only the ranking changes. The check must be able to say "no".
    X, y = make_signal_data()
    X_train, X_test, y_train, y_test = X.iloc[:800], X.iloc[800:], y.iloc[:800], y.iloc[800:]
    model = small_xgb().fit(X_train, y_train)
    bad_ranking = ["noise_1", "noise_2", "noise_3", "noise_4", "signal_a", "signal_b"]

    drop, random_drops = top_k_vs_random_k(model, X_test, y_test, bad_ranking, k=2)

    assert drop <= np.quantile(random_drops, RANDOM_QUANTILE)


def test_shuffling_noise_costs_nothing_and_shuffling_signal_costs_a_lot_on_unseen_data():
    # On the data it was fitted on, a flexible model has memorised some noise and shuffling it hurts (0.04 here);
    # what matters is unseen data, which is also where the check is run.
    X, y = make_signal_data()
    X_train, X_test, y_train, y_test = X.iloc[:800], X.iloc[800:], y.iloc[:800], y.iloc[800:]
    model = small_xgb().fit(X_train, y_train)
    rng = np.random.default_rng(0)

    assert auc_drop(model, X_test, y_test, ["noise_1", "noise_2"], n_repeats=5, rng=rng) < 0.02
    assert auc_drop(model, X_test, y_test, ["signal_a"], n_repeats=5, rng=rng) > 0.1


def test_median_case_is_the_middle_customer_of_the_group_and_only_of_that_group():
    proba = np.array([0.9, 0.1, 0.5, 0.7, 0.3, 0.8])
    group = np.array([True, False, True, True, False, True])   # probabilities 0.9, 0.5, 0.7, 0.8 -> median 0.8

    assert median_case(proba, group) == 5
    assert median_case(proba, ~group) == 4                       # the other group: 0.1, 0.3 -> upper median 0.3
