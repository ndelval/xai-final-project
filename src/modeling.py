"""Split, pipeline and evaluation protocol shared by every experiment."""
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
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
