import argparse
from pathlib import Path

import pandas as pd

from textdefendr.experiment import Experiment
from textdefendr.utils.pandas_ops import (
    create_ideal_train_test_split,
    downsample_clean_to_max_nonclean_class,
    downsample_clean_to_sum_nonclean_class,
)


def get_splitted_exp(in_dir):
    print("Reading train.csv from ", Path(in_dir, "train.csv"))
    train_df = pd.read_csv(Path(in_dir, "train.csv"))
    print("Reading val.csv from ", Path(in_dir, "val.csv"))
    val_df = pd.read_csv(Path(in_dir, "val.csv"))
    print("Reading test.csv from ", Path(in_dir, "test.csv"))
    test_df = pd.read_csv(Path(in_dir, "test.csv"))
    return train_df, val_df, test_df


def drop_attacks_by_count(train_df, val_df, test_df, min_instance_per_class=10):
    """
    https://stackoverflow.com/questions/29403192/convert-series-returned-by-pandas-series-value-counts-to-a-dictionary
    """
    dropped_attacks = []
    counts = train_df["attack_name"].value_counts()
    drop_warning = ""
    for k in counts.to_dict().keys():
        if counts[k] < min_instance_per_class:
            drop_warning = drop_warning + k + ":" + str(counts[k]) + ", "
            dropped_attacks.append(k)
    if drop_warning != "":
        print(
            drop_warning,
            " have been dropped due to having class count smaller than ",
            min_instance_per_class,
        )
    train_df = train_df[
        ~train_df["attack_name"].isin(counts[counts < min_instance_per_class].index)
    ]
    val_df = val_df[~val_df["attack_name"].isin(dropped_attacks)]
    test_df = test_df[~test_df["attack_name"].isin(dropped_attacks)]
    return train_df, val_df, test_df


def add_target_label_column(df_row, experiment_setting):
    assert experiment_setting in (
        "clean_vs_all",
        "multiclass_with_clean",
    ), "incorrect experiment setting"
    if experiment_setting == "clean_vs_all":
        if df_row["attack_name"] == "clean":
            return "clean"
        else:
            return "perturbed"
    elif experiment_setting == "multiclass_with_clean":
        return df_row["attack_name"]


def filter_by_exp_setting(
    train_df, val_df, test_df_release, test_df_hidden, experiment_setting
) -> Experiment:
    # not touching test_df.csv as it's going to remain "hidden till the end"

    train_df["target_label"] = train_df.apply(
        lambda row: add_target_label_column(row, experiment_setting), axis=1
    )
    val_df["target_label"] = val_df.apply(
        lambda row: add_target_label_column(row, experiment_setting), axis=1
    )
    test_df_release["target_label"] = test_df_release.apply(
        lambda row: add_target_label_column(row, experiment_setting), axis=1
    )
    test_df_hidden["target_label"] = test_df_hidden.apply(
        lambda row: add_target_label_column(row, experiment_setting), axis=1
    )
    if experiment_setting == "clean_vs_all":
        train_df = downsample_clean_to_sum_nonclean_class(train_df)
        val_df = downsample_clean_to_sum_nonclean_class(val_df)
        # test_df = downsample_clean_to_sum_nonclean_class(test_df)

        return Experiment(
            train_df=train_df,
            val_df=val_df,
            test_df_release=test_df_release,
            test_df_hidden=test_df_hidden,
            name="clean_vs_all",
        )
    elif experiment_setting == "multiclass_with_clean":
        train_df = downsample_clean_to_max_nonclean_class(train_df)
        val_df = downsample_clean_to_max_nonclean_class(val_df)
        # test_df = downsample_clean_to_max_nonclean_class(test_df)

        return Experiment(
            train_df=train_df,
            val_df=val_df,
            test_df_release=test_df_release,
            test_df_hidden=test_df_hidden,
            name="multiclass_with_clean",
        )
    else:
        raise ValueError(f"invalid experiment_setting {experiment_setting}")


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="allocine",
        help="Dataset attacked.",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="distilcamembert",
        help="Target model type.",
    )
    parser.add_argument(
        "--embeddings_name",
        type=str,
        default="default",
        help="Embeddings name (prefix).",
    )
    parser.add_argument(
        "--experiment_setting",
        type=str,
        choices=["clean_vs_all", "multiclass_with_clean"],
        default="clean_vs_all",
        help="Binary or multiclass detection.",
    )
    args = parser.parse_args(raw_args)

    try:
        in_dir = Path(
            "data_tcab",
            "official_TCAB_splits",
            "splits_by_dataset_and_tm",
            args.target_dataset,
            args.target_model,
        )
        assert Path(in_dir, "train.csv").exists(), f"no train.csv in {in_dir}"
        assert Path(in_dir, "val.csv").exists(), f"no val.csv in {in_dir}"
        assert Path(in_dir, "test.csv").exists(), f"no test.csv in {in_dir}"
    except AssertionError as e:
        print(e)

    out_dir = Path(
        "data_tcab",
        "detection-experiments",
        args.target_dataset,
        args.target_model,
        args.embeddings_name,
        args.experiment_setting,
    )
    train_df, val_df, test_df = get_splitted_exp(in_dir)

    assert (
        set(train_df["attack_name"])
        == set(val_df["attack_name"])
        == set(test_df["attack_name"])
    ), "attacks missing in train, val or test split"

    print("Dropping attacks by count, if less than < 10")
    train_df, val_df, test_df = drop_attacks_by_count(train_df, val_df, test_df)
    test_df_release, test_df_hidden = create_ideal_train_test_split(
        test_df, split_ratio=0.5
    )

    exp = filter_by_exp_setting(
        train_df, val_df, test_df_release, test_df_hidden, args.experiment_setting
    )
    exp.aux = vars(args)
    exp.dump(exp_root_dir=out_dir)


if __name__ == "__main__":
    main()
