"""Split, pipeline and evaluation protocol shared by every experiment."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             make_scorer, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
MAX_AUC_GAP = 0.03  # "not overfitted" means train AUC minus CV AUC below this. Fixed before tuning.
SUSPICIOUS_AUC = 0.90  # a CV AUC above this on Telco (papers report ~0.84-0.86) is leakage until proven otherwise


def split(X, y, seed=SEED):
    """Stratified 80/20 split. Telco has no time axis, so a random split is appropriate."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def build_pipeline(estimator, scale=False):
    """One-hot encode the text columns and, for linear models, standardise the numeric ones.

    Yes/No columns become a single 0/1 column (no mirror pair that would split an attribution or
    a coefficient in two). Columns are picked by dtype, so features added later flow through.
    Everything is fitted inside the pipeline: cross-validation refits it on each training fold.
    """
    numeric = StandardScaler() if scale else "passthrough"
    prep = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
         make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])


def make_models(seed=SEED):
    """The three models compared in the project. The tuned ones search their grid by CV on train.

    No class re-weighting or resampling: class imbalance is handled with a decision threshold
    and churn-class metrics, which avoids distorting the predicted probabilities.
    """
    inner_cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    logistic = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=1000), scale=True),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        scoring="roc_auc", cv=inner_cv)
    xgboost = GridSearchCV(
        build_pipeline(XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                     random_state=seed)),
        {"model__max_depth": [1, 2, 3], "model__n_estimators": [150, 300, 600]},
        scoring="roc_auc", cv=inner_cv)
    return {
        "majority": build_pipeline(DummyClassifier(strategy="prior")),
        "logistic": logistic,
        "xgboost": xgboost,
    }


def cv_and_gap(estimator, X_train, y_train, X_test, y_test, seed=SEED):
    """ROC-AUC and PR-AUC in 5-fold CV over train and on test, plus the train-minus-CV AUC gap.

    Train and CV scores come from the same models (each fold's model scored on the rows it was
    fitted on and on the rows it never saw), so the gap compares like with like. For the tuned
    models the CV is nested: the grid search is redone inside every outer fold, so the CV score
    is not inflated by picking the best configuration on the same folds.
    The test set is only scored, by one model fitted on all of train. It never takes part in
    fitting or tuning. PR-AUC of a model with no skill equals the churn rate (0.265).
    """
    outer_cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv = cross_validate(estimator, X_train, y_train, cv=outer_cv, return_train_score=True,
                        scoring={"auc": "roc_auc", "pr_auc": "average_precision"})
    train_auc, cv_auc = cv["train_auc"].mean(), cv["test_auc"].mean()

    proba = clone(estimator).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    return {"train_auc": train_auc, "cv_auc": cv_auc, "cv_std": cv["test_auc"].std(),
            "test_auc": roc_auc_score(y_test, proba), "gap": train_auc - cv_auc,
            "cv_pr_auc": cv["test_pr_auc"].mean(), "test_pr_auc": average_precision_score(y_test, proba)}


def oof_proba(estimator, X_train, y_train, seed=SEED):
    """Churn probability of every training row, from models that never saw that row (5-fold CV).

    This is what the decision threshold is chosen on: the test set is not touched, and the
    probabilities are not the over-confident ones a model gives to rows it was fitted on.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba")[:, 1]


def churn_metrics(y_true, proba, threshold):
    """Metrics of the churn class at a threshold, plus PR-AUC and Brier score (threshold-free).

    Always the churn class, never a weighted average: with 26.5 % churn a weighted score is
    dominated by the loyal customers and hides how well the churners are found.
    """
    pred = np.asarray(proba) >= threshold
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred, zero_division=0),
            "accuracy": accuracy_score(y_true, pred),
            "pr_auc": average_precision_score(y_true, proba),
            "brier": brier_score_loss(y_true, proba)}


def customer_costs(y_true, pred, cost_ratio):
    """Cost of each customer's decision: a missed churner costs `cost_ratio`, a wasted offer costs 1.

    `y_true` and `pred` (True = contact the customer) are matched by position, not by index.
    """
    y_true, pred = np.asarray(y_true), np.asarray(pred).astype(bool)
    return np.where(y_true == 1, cost_ratio * ~pred, pred)


def expected_cost(y_true, pred, cost_ratio):
    """Average cost per customer."""
    return customer_costs(y_true, pred, cost_ratio).mean()


def pick_threshold(y_true, proba, cost_ratio):
    """The threshold in [0.01, 0.99] with the lowest expected cost on the given probabilities."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y_true, proba >= t, cost_ratio) for t in grid]
    return grid[int(np.argmin(costs))]


def cv_fold_scores(estimator, X, y, threshold, cost_ratio, seed=SEED):
    """Score of each of the 5 CV folds: ROC-AUC, PR-AUC, Brier, and recall, precision and expected cost at `threshold`.

    The folds are the same for every estimator and the threshold is chosen in advance, not fitted on
    these folds, so that variants of a model can be compared fold by fold. Nested for the tuned models.
    """
    def at_threshold(score):
        return make_scorer(lambda y_true, proba: score(y_true, proba >= threshold), response_method="predict_proba")

    scoring = {"auc": "roc_auc", "pr_auc": "average_precision", "neg_brier": "neg_brier_score",
               "recall": at_threshold(recall_score),
               "precision": at_threshold(lambda y_true, flagged: precision_score(y_true, flagged, zero_division=0)),
               "neg_cost": at_threshold(lambda y_true, flagged: -expected_cost(y_true, flagged, cost_ratio))}
    scores = cross_validate(estimator, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring=scoring)
    return pd.DataFrame({"auc": scores["test_auc"], "pr_auc": scores["test_pr_auc"], "brier": -scores["test_neg_brier"],
                         "recall": scores["test_recall"], "precision": scores["test_precision"],
                         "cost": -scores["test_neg_cost"]})
