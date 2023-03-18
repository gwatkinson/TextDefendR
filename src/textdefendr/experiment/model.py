import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from textdefendr.utils.file_io import mkdir_if_dne


class ExperimentModel:
    def __init__(
        self,
        classifier: BaseEstimator,
        feature_names: list,
        serialization_dir: Path,
    ):
        self.classifier = classifier
        self.feature_names = feature_names
        self.label_encoder = LabelEncoder()
        self.serialization_dir = serialization_dir
        self.metrics = {}
        self.n_features = None

    def fit(self, X, y):
        y_encode = self.label_encoder.fit_transform(y)
        self.n_features = len(self.label_encoder.classes_)

        self.classifier.fit(X, y_encode)

    def evaluate(self, X, y, split_name):
        y_true = self.label_encoder.transform(y)
        y_pred = self.classifier.predict(X)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        if self.n_features == 2:
            pos_label = self.label_encoder.transform(["perturbed"])[0]
            metrics["recall"] = recall_score(y_true, y_pred, pos_label=pos_label)
            metrics["precision"] = precision_score(y_true, y_pred, pos_label=pos_label)
            metrics["f1_score"] = f1_score(y_true, y_pred)
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)

        new_metrics = {split_name + "_" + key: value for key, value in metrics.items()}

        self.metrics.update(new_metrics)

        print(new_metrics)

    def extend_metrics(self) -> None:
        # Sort keys
        for key in [
            "best_params",
            "classes_labels",
            "important_features",
            "feature_names",
            "coef",
            "intercept",
        ]:
            self.metrics[key] = None

        self.metrics["feature_names"] = list(
            self.feature_names
        )  # make sure these are lists since numpy ND arrays are not JSON serializable
        self.metrics["classes_labels"] = list(self.label_encoder.classes_)

        if isinstance(self.classifier, GridSearchCV):
            self.metrics["best_params"] = self.classifier.best_params_
            classifier = self.classifier.best_estimator_
        else:
            classifier = self.classifier

        if isinstance(classifier, Pipeline):
            classifier = classifier[-1]

        if isinstance(classifier, LogisticRegression):
            self.metrics["coef"] = extract_coef_logistic(
                classifier, self.feature_names, self.label_encoder.classes_
            )

    def save(self) -> None:
        """
        Save model and a dictionary of metrics to self.serialization_dir
        """
        print("--- saving metrics to", self.serialization_dir)
        mkdir_if_dne(self.serialization_dir)
        with open(Path(self.serialization_dir, "metrics.json"), "w") as ofp:
            try:
                json.dump(self.metrics, ofp, indent=4)
            except Exception as e:
                print("Saving metrics failed:")
                print(e)
        try:
            joblib.dump(
                self.classifier,
                Path(self.serialization_dir, "model.joblib"),
            )
        except Exception as e:
            print("Saving model failed:")
            print(e)


def extract_coef_logistic(
    classifier: LogisticRegression, feature_names: list, classes_labels: list
) -> dict:
    """Get a dataframe of the logistic regression coefficients.

    Classes are in columns and features in rows.

    Parameters
    ----------
    classifier : LogisticRegression
        model
    feature_names : list
        list of feature names.
    classes_labels : list
        list of classes labels.

    Returns
    -------
    dict
        dataframe as dict
    """
    if len(classes_labels) == 2:
        columns = [classes_labels[1]]
    else:
        columns = classes_labels

    df_coef = pd.DataFrame(
        classifier.coef_.T,
        feature_names,
        columns=columns,
    )
    return df_coef.to_dict()
