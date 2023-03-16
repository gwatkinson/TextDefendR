import json
from pathlib import Path

import joblib
import numpy
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from nlp_adversarial_attacks.utils.file_io import mkdir_if_dne


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

    def fit(self, X_train, y_train):
        y_train_encode = self.label_encoder.fit_transform(y_train)

        self.classifier.fit(X_train, y_train_encode)

    def evaluate(self, X_train, y_train, split_name):
        y_train_encode = self.label_encoder.transform(y_train)
        y_pred_train = self.classifier.predict(X_train)
        acc_train = accuracy_score(y_pred_train, y_train_encode)
        bacc_train = balanced_accuracy_score(y_pred_train, y_train_encode)

        conf_matrix_train = confusion_matrix(y_train_encode, y_pred_train).tolist()

        metrics = {
            "accuracy": acc_train,
            "balanced_accuracy": bacc_train,
            "confusion_matrix": conf_matrix_train,
        }

        class_names = self.label_encoder.classes_
        if len(class_names) == 2:
            train_roc_auc = roc_auc_score(y_train, y_pred_train)
            metrics["roc_auc"] = train_roc_auc

        new_metrics = {split_name + "_" + key: value for key, value in metrics.items()}

        self.metrics.update(new_metrics)

        print(new_metrics)

    def compute_metrics(self) -> None:
        # Sort keys
        for key in [
            "best_params",
            "label_classes",
            "important_features",
            "feature_names",
            "coef",
            "intercept",
        ]:
            self.metrics[key] = None

        self.metrics["feature_names"] = list(
            self.feature_names
        )  # make sure these are lists since numpy ND arrays are not JSON serializable
        self.metrics["label_classes"] = list(self.label_encoder.classes_)

        if isinstance(self.classifier, GridSearchCV):
            self.metrics["best_params"] = self.classifier.best_params_
            classifier = self.classifier.best_estimator_
        else:
            classifier = self.classifier

        if isinstance(classifier, Pipeline):
            classifier = classifier[-1]

        if isinstance(classifier, LogisticRegression):
            self.metrics["coef"] = classifier.coef_.tolist()
            self.metrics["intercept"] = classifier.intercept_.tolist()
            self.metrics["important_features"] = self.extract_important_features(
                coefs=numpy.array(self.metrics["coef"])
            )

    def extract_important_features(self, coefs, k=10) -> dict:
        """
        Extract the most important features of the model.
        Inputs:
            lr_coef: coef_ from a lr model
            label_map: mapping from int label to str label (maybe I missremembered and its the inverse)
            feature_names: feature names corresponding to each dim
        Returns: Feature importances of shape=(no. classes, no. features).
        """
        out = {}
        # coefs = self.classifier.coef_
        label_map = self.label_encoder.classes_

        # extract feature coefficients for each class
        for i, coef in enumerate(coefs):
            best_coef_indices = numpy.argsort(coef)[::-1]
            # display the top features for this class
            label_name = label_map[i] if coefs.shape[0] > 1 else label_map[1]
            # print('{} (Top {:,} features)'.format(label_name, k))
            out[label_name] = []
            for ndx in best_coef_indices[:k]:
                # print('{}: {:.3f}'.format(feature_names[ndx], coef[ndx]))
                out[label_name].append((self.feature_names[ndx], coef[ndx]))
            # print('')
        # print(out)
        return out

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
