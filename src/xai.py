"""Explanations of the fitted models. Everything is reported per ORIGINAL feature and in log-odds.

Both explainers work on the encoded matrix the model actually sees (one-hot columns), where one
feature such as Contract is spread over several columns. Contributions are additive, so we add the
columns of each feature back together: that is what a person reads ("Contract", not "Contract_One year").
"""
import itertools

import numpy as np
import pandas as pd
import shap
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from src.modeling import SEED

# Pass/fail rules of the sanity checks, fixed BEFORE running them on the real data.
MAX_RANDOMIZED_SPEARMAN = 0.5  # explanations of models fitted on shuffled labels may correlate at most this much with the real ones
RANDOM_QUANTILE = 0.95         # shuffling the k top-ranked features must hurt more than this quantile of shuffling k random ones


def by_original_feature(values, encoded_names, columns):
    """Add up the contributions of the encoded columns that come from the same original feature."""
    owners = []
    for name in encoded_names:
        encoded = name.split("__", 1)[1]  # drop the "num__" / "cat__" prefix added by ColumnTransformer
        found = [c for c in columns if encoded == c or encoded.startswith(c + "_")]
        if len(found) != 1:
            raise ValueError(f"cannot tell which feature the encoded column {name} belongs to: {found}")
        owners.append(found[0])
    grouped = pd.DataFrame(np.asarray(values), columns=owners).T.groupby(level=0, sort=False).sum().T
    return grouped.reindex(columns=columns)


def shap_by_feature(model, X, X_reference):
    """TreeSHAP values of a fitted (prep -> XGBoost) pipeline, relative to the average `X_reference` customer.

    TreeSHAP's own base value is a cover-weighted expectation of the trees, not the average over any
    data, so its contributions do not average to 0 anywhere. We subtract each feature's average
    contribution over the reference customers (and add it to the base), which keeps the sum intact:
    base + sum of a row's values = the model's log-odds for that customer. Now the base is the average
    log-odds of the reference customers, exactly as in `linear_contributions`. For an additive model
    (depth-1 trees) this equals interventional SHAP with that reference as background.
    Returns (values per original feature, base value).
    """
    prep, booster = model.named_steps["prep"], model.named_steps["model"]
    explainer, names = shap.TreeExplainer(booster), prep.get_feature_names_out()

    def contributions(data):
        explanation = explainer(prep.transform(data))
        values = by_original_feature(np.asarray(explanation.values, dtype=float), names, data.columns)
        return values, float(np.mean(explanation.base_values))

    values, base = contributions(X)
    reference_mean = contributions(X_reference)[0].mean()
    values = values - reference_mean
    values.index = X.index
    return values, base + float(reference_mean.sum())


def linear_contributions(model, X, X_reference):
    """Contribution of each feature to the log-odds of a fitted (prep -> logistic regression) pipeline.

    Same idea and units as SHAP for a linear model: coefficient times the distance to the average
    reference customer. Returns (contributions per original feature, base value).
    """
    prep, logistic = model.named_steps["prep"], model.named_steps["model"]
    encoded, mean_reference = prep.transform(X), prep.transform(X_reference).mean(axis=0)
    coef = logistic.coef_[0]
    contributions = by_original_feature((encoded - mean_reference) * coef, prep.get_feature_names_out(), X.columns)
    contributions.index = X.index
    return contributions, float(logistic.intercept_[0] + mean_reference @ coef)


def shuffle_labels(rng, X, y):
    """Break the link between features and target: the same rows with the labels in random order."""
    return X, rng.permutation(np.asarray(y))


def bootstrap_rows(rng, X, y):
    """Control: the same data resampled with replacement, with the link between features and target intact."""
    rows = rng.integers(0, len(X), len(X))
    return X.iloc[rows], np.asarray(y)[rows]


def importance_under_refits(make_model, X_train, y_train, X_explain, perturb, n_refits=10, seed=SEED):
    """Mean |SHAP| per feature of models refitted on perturbed training data: one row per refit.

    `make_model()` returns a fresh unfitted pipeline; `perturb(rng, X, y)` returns the (X, y) to fit on.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_refits):
        X_fit, y_fit = perturb(rng, X_train, y_train)
        values, _ = shap_by_feature(make_model().fit(X_fit, y_fit), X_explain, X_train)
        rows.append(values.abs().mean())
    return pd.DataFrame(rows)


def auc_drop(model, X, y, features, n_repeats, rng):
    """Mean fall in ROC-AUC when each of the given columns is shuffled independently of the others."""
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    drops = []
    for _ in range(n_repeats):
        shuffled = X.copy()
        for feature in features:
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        drops.append(baseline - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
    return float(np.mean(drops))


def top_k_vs_random_k(model, X, y, ranking, k, n_random=30, n_repeats=5, seed=SEED):
    """Fall in AUC after shuffling the k top-ranked features, and after shuffling many random sets of k.

    Columns are shuffled in the original data, before the pipeline. Returns (drop of the top k, drops of the random sets).
    """
    rng = np.random.default_rng(seed)
    top = auc_drop(model, X, y, list(ranking[:k]), n_repeats, rng)
    pool = sorted(ranking)  # the random sets do not depend on the order of the ranking, so rankings are compared on the same ones
    random = [auc_drop(model, X, y, list(rng.choice(pool, k, replace=False)), n_repeats, rng)
              for _ in range(n_random)]
    return top, np.array(random)


def median_case(proba, group):
    """Position of the customer whose probability is the median of the customers in `group` (upper median if even).

    A reproducible way to pick a *typical* customer of a group, instead of the most extreme one.
    """
    positions = np.flatnonzero(group)
    return positions[np.argsort(np.asarray(proba)[positions])][len(positions) // 2]


class ShiftedThreshold:
    """Wraps a fitted model so that its decision boundary at `threshold` looks like 0.5, which is what DiCE assumes."""

    def __init__(self, model, threshold):
        self.model, self.shift = model, logit(threshold)

    def predict_proba(self, X):
        probability = np.clip(self.model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        shifted = expit(logit(probability) - self.shift)
        return np.column_stack([1 - shifted, shifted])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def fewest_changes(model, row, options, threshold):
    """The fewest changes to a customer that bring the predicted risk below `threshold`, trying every combination.

    `row` is a one-row DataFrame (it keeps the column types); `options` maps each feature that may change to
    the values it may take, which must include the current one. It counts changes, not their cost. Returns ({feature: (old, new)}, new probability), or None
    if no combination gets below the threshold. Among combinations with as few changes, the lowest risk wins.
    """
    features, original = list(options), row.iloc[0]
    for feature in features:
        if original[feature] not in options[feature]:
            raise ValueError(f"the current value of {feature} ({original[feature]!r}) must be one of its options")
    candidates = pd.concat([row] * int(np.prod([len(v) for v in options.values()])), ignore_index=True)
    candidates[features] = pd.DataFrame(list(itertools.product(*options.values())), columns=features)
    probability = model.predict_proba(candidates)[:, 1]
    n_changes = sum((candidates[f] != original[f]).to_numpy().astype(int) for f in features)
    below = np.flatnonzero(probability < threshold)
    if len(below) == 0:
        return None
    best = below[np.lexsort((probability[below], n_changes[below]))[0]]
    changes = {f: (original[f], candidates.loc[best, f]) for f in features if candidates.loc[best, f] != original[f]}
    return changes, float(probability[best])
