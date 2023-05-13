"""Define training methods"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    k_fold_splits: int = 5,
    model_dict: dict = {
        "Logistic_Regression": LogisticRegression(),
        "Random_Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(tree_method="hist", enable_categorical=True),
    },
) -> pd.DataFrame:
    # pylint: disable=dangerous-default-value
    """
    This method is designed to simplify the training of multiple models
    assuming they all come from a sklearn model or it has the required methods
    to be trained using the current sklearn structure.
    We return a pandas DataFrame containing the metrics for each split in a
    k-fold cross-validation scheme. We also print the training results for
    each model.
    """
    results = {}
    scoring = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
    ]
    for name, model in model_dict.items():
        kfold = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)
        cross_validation_results = cross_validate(
            model, x_train, y_train, cv=kfold, scoring=scoring
        )
        model = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(
            y_test, y_pred, target_names=["no", "yes"]
        )
        print(f"{name}\n{report}")
        results[name] = cross_validation_results
    return pd.DataFrame(results)
