import argparse
import multiprocessing
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle as shuffle

from textdefendr.experiment import ExperimentModel
from textdefendr.feature_extraction import FeatureExtractor, get_feature_dim_names
from textdefendr.utils.file_io import mkdir_if_dne


def check_data_experiments(experiment_dir):
    print("--- searching exp data in", experiment_dir)
    assert Path(experiment_dir, "train.csv").exists(), (
        "cannot find train.csv in "
        + experiment_dir
        + ", did you distribute and make the experiment correctly?"
    )
    assert Path(experiment_dir, "val.csv").exists(), (
        "cannot find val.csv in "
        + experiment_dir
        + ", did you distribute and make the experiment correctly?"
    )
    assert Path(experiment_dir, "test_release.csv").exists(), (
        "cannot find test_release.csv in "
        + experiment_dir
        + ", did you distribute and make the experiment correctly?"
    )


def get_feature_names_by_setting(feature_setting):
    feature_setting = feature_setting.lower()

    if feature_setting == "bert":
        feature_names = ["tp_bert"]
    elif feature_setting == "bert+tp":
        feature_names = sorted(
            [fcn.__name__ for fcn in FeatureExtractor(add_tags=["tp"]).extractors]
        )
    elif feature_setting == "bert+tp+lm":
        feature_names = sorted(
            [fcn.__name__ for fcn in FeatureExtractor(add_tags=["tp", "lm"]).extractors]
        )
    elif feature_setting == "all":
        feature_names = sorted(
            [
                fcn.__name__
                for fcn in FeatureExtractor(add_tags=["tp", "lm", "tm"]).extractors
            ]
        )
    else:
        raise ValueError(
            "Invalid feature setting. Must be bert / bert+tp / bert+tp+lm or all."
        )
    return feature_names


def get_feature_dim_names_by_feature_names(feature_names):
    out = []
    for _fname in feature_names:
        out += get_feature_dim_names(_fname)
    return out


def joblib_to_x_and_y(joblib_dir, feature_names):
    print("--- loading from", joblib_dir)
    valid_features = feature_names
    instance_dict = joblib.load(joblib_dir)

    print("num of instances in the file: ", len(instance_dict))
    out_x = []
    out_y = []
    for k in instance_dict:
        instance = instance_dict[k]

        # Features
        feats = []
        for f in valid_features:
            feat = instance["deliverable"][f]
            feats.append(feat)

        feats = np.concatenate([i[0] for i in feats])
        out_x.append(feats)

        unused_feats = [
            f for f in instance["deliverable"].keys() if f not in valid_features
        ]
        for uf in unused_feats:
            del instance["deliverable"][uf]  # save memory

        # Label
        out_y.append(instance["target_label"])

    del instance_dict
    print("loaded, labels: ", Counter(out_y))
    assert len(out_x) == len(out_y)
    return out_x, out_y


def run_experiment(experiment_dir, feature_setting, model_type, args):
    names_of_feats_to_use = get_feature_names_by_setting(feature_setting)

    print("--- using these features:")
    print(names_of_feats_to_use)

    output_dir = Path(experiment_dir, model_type, feature_setting)
    print("--- output to", output_dir)

    feature_dim_names = get_feature_dim_names_by_feature_names(names_of_feats_to_use)

    train_joblib_path = Path(experiment_dir, "train.joblib")
    val_joblib_path = Path(experiment_dir, "val.joblib")
    test_joblib_path = Path(experiment_dir, "test_release.joblib")

    X_train, y_train = joblib_to_x_and_y(
        train_joblib_path,
        names_of_feats_to_use,
    )
    X_val, y_val = joblib_to_x_and_y(
        val_joblib_path,
        names_of_feats_to_use,
    )
    X_test, y_test = joblib_to_x_and_y(
        test_joblib_path,
        names_of_feats_to_use,
    )
    training_main(
        model_type=model_type,
        output_dir=output_dir,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_dim_names,
        args=args,
    )


def training_main(
    model_type,
    output_dir: Path,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    feature_names,
    args,
):
    print("--- training main starting")

    print("skip if done:", args.skip_if_done)
    if Path(output_dir, "model.joblib").exists() and args.skip_if_done:
        print("looks you already have a model in there, skipping")
        exit(0)

    assert set(y_train) == set(y_val), "inconsistant label classes in ytrain and yval"

    print(f"\nno. train: {len(y_train):,}")
    print(f"no. val: {len(y_val):,}")
    print(f"no. test: {len(y_test):,}")
    print(f"no. features: {len(X_train[0]):,}\n")

    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = shuffle(X_val, y_val)

    # not shuffling test data

    if args.train_frac < 1.0:
        assert args.train_frac > 0.0
        n_train = int(len(X_train) * args.train_frac)
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    mkdir_if_dne(output_dir)

    ss = StandardScaler()
    if model_type == "LR":
        clf_temp = LogisticRegression(
            solver=args.solver,
            penalty=args.penalty,
            random_state=0,
            n_jobs=args.model_n_jobs,
            class_weight="balanced",
        )
        param_grid = {"logisticregression__C": [1e-1, 1e0]}
    elif model_type == "LGB":
        clf_temp = LGBMClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            random_state=0,
            n_jobs=args.model_n_jobs,
            class_weight="balanced",
        )
        param_grid = {
            "lgbmclassifier__n_estimators": [50, 100],
            "lgbmclassifier__max_depth": [3, 5],
            "lgbmclassifier__num_leaves": [2, 15],
            "lgbmclassifier__boosting_type": ["gbdt"],
        }
    elif model_type == "RF":
        clf_temp = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=0,
            n_jobs=args.model_n_jobs,
            class_weight="balanced",
        )
        param_grid = {
            "randomforestclassifier__n_estimators": [50, 100],
            "randomforestclassifier__max_depth": [3, 5],
            "randomforestclassifier__min_samples_leaf": [2, 4],
        }
    elif model_type == "DT":
        clf_temp = DecisionTreeClassifier(random_state=0, class_weight="balanced")
        param_grid = {
            "decisiontreeclassifier__max_depth": [3, 5, None],
            "decisiontreeclassifier__min_samples_leaf": [1, 2, 4, 10],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    pipeline = make_pipeline(ss, clf_temp)
    if not args.disable_tune:
        clf = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            verbose=3,
            n_jobs=args.cv_n_jobs,
        )
    else:
        clf = pipeline

    model = ExperimentModel(
        classifier=clf,
        feature_names=feature_names,
        serialization_dir=output_dir,
    )

    print("--- starting training")
    start_time = time.time()
    model.fit(X_train, y_train)
    print("--- finished training")
    print(
        f"--- training time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}"
    )

    model.evaluate(X_train, y_train, "train")
    model.evaluate(X_val, y_val, "validation")
    model.evaluate(X_test, y_test, "test")
    model.extend_metrics()

    model.save()


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default="data_tcab/detection-experiments/allocine/distilcamembert/default/clean_vs_all/",
        help="Directory of the distributed experiment.",
    )
    parser.add_argument(
        "--feature_setting",
        type=str,
        choices=["bert", "bert+tp", "bert+tp+lm", "all"],
        default="all",
        help="Set of features to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["LR", "DT", "RF", "LGB"],
        default="LR",
        help="Classification model.",
    )
    parser.add_argument(
        "--skip_if_done",
        action="store_true",
        help="Skip if an experiment is already runned.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test model.",
    )
    parser.add_argument(
        "--model_n_jobs",
        type=int,
        default=1,
        help="No. jobs to run in parallel for the model.",
    )
    parser.add_argument(
        "--cv_n_jobs",
        type=int,
        default=1,
        help="No. jobs to run in parallel for gridsearch.",
    )
    parser.add_argument("--solver", type=str, default="lbfgs", help="LR solver.")
    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="LR penalty.",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=1,
        help="Fraction of train data to train with.",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100, help="No. boosting rounds for lgb."
    )
    parser.add_argument(
        "--max_depth", type=int, default=5, help="Max. depth for each tree."
    )
    parser.add_argument(
        "--num_leaves", type=int, default=32, help="No. leaves per tree."
    )
    parser.add_argument(
        "--disable_tune",
        action="store_true",
        help="Disable hyperparameters tuning with gridsearch.",
    )
    args = parser.parse_args(raw_args)

    print("starting experiment:", args.experiment_dir)

    ncpu = multiprocessing.cpu_count()
    print(f"CPU available : {ncpu}")

    if args.test:
        print("RUNNING IN TEST MODE")
        args.train_frac = 0.01

    check_data_experiments(args.experiment_dir)

    run_experiment(
        experiment_dir=args.experiment_dir,
        feature_setting=args.feature_setting,
        model_type=args.model,
        args=args,
    )


if __name__ == "__main__":
    main()
