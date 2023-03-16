from pathlib import Path

import pandas as pd

from nlp_adversarial_attacks.utils.file_io import mkdir_if_dne
from nlp_adversarial_attacks.utils.hashing import get_pk_tuple_from_pandas_row
from nlp_adversarial_attacks.utils.magic_vars import (
    PRIMARY_KEY_FIELDS,
    SUPPORTED_TARGET_MODEL_DATASETS,
    SUPPORTED_TARGET_MODELS,
)
from nlp_adversarial_attacks.utils.pandas_ops import (
    convert_nested_list_to_df,
    create_ideal_train_test_split,
    drop_for_column_outside_of_values,
    no_duplicate_entry,
    no_duplicate_index,
    show_df_stats,
)

RANDOM_SEED = 22


def get_src_instance_identifier_from_pandas_row(pandas_row):
    """
    returns an tuple idenfier unique to src instance

    e.g. #777 sentence in SST is attacked by 7 attacks, those will share this identifier
    """
    return (
        pandas_row["target_model_dataset"],
        pandas_row["test_index"],
    )


def get_dataset_df(whole_catted_dataset_path):
    """
    Read in the whole_catted_dataset.csv. Do some sanity check on it as well
    Pad useful column called pk, and another src instance identifier
    """
    odf = pd.read_csv(whole_catted_dataset_path)
    print("--- dropping unsupported datasets")
    odf = drop_for_column_outside_of_values(
        odf, "target_model_dataset", SUPPORTED_TARGET_MODEL_DATASETS
    )
    odf = odf.sample(frac=1, random_state=RANDOM_SEED)
    odf["pk"] = odf.apply(lambda row: get_pk_tuple_from_pandas_row(row), axis=1)
    odf["unique_src_instance_identifier"] = odf.apply(
        lambda row: get_src_instance_identifier_from_pandas_row(row), axis=1
    )
    assert no_duplicate_index(odf)
    assert no_duplicate_entry(odf, "pk")
    return odf


def get_splits_for_dataset(dataset, df):
    print("--- filtering for dataset ", df.shape)
    df = drop_for_column_outside_of_values(df, "target_model_dataset", [dataset])
    train_df, test_val_df = create_ideal_train_test_split(df, split_ratio=0.6)
    val_df, test_df = create_ideal_train_test_split(test_val_df, split_ratio=0.5)
    print("--- Train DF stats ---")
    print(show_df_stats(train_df))
    print("--- Test DF stats ---")
    print(show_df_stats(test_df))
    print("--- Val DF stats ---")
    print(show_df_stats(val_df))

    if dataset == "wikipedia_personal":
        assert len(set(train_df["attack_name"])) >= len(
            set(val_df["attack_name"])
        ), "train_df has more attacks than val_df"
        assert set(train_df["attack_name"]).union(set(val_df["attack_name"])) == set(
            train_df["attack_name"]
        ), "attacks missing in either of the splits"
        assert len(set(train_df["attack_name"])) >= len(
            set(test_df["attack_name"])
        ), "train_df has more attacks than test_df"
        assert set(train_df["attack_name"]).union(set(test_df["attack_name"])) == set(
            train_df["attack_name"]
        ), "attacks missing in either of the splits"
    else:
        assert (
            set(train_df["attack_name"])
            == set(test_df["attack_name"])
            == set(val_df["attack_name"])
        ), "attacks missing in either of the splits"

    assert (
        set(train_df["target_model_dataset"])
        == set(test_df["target_model_dataset"])
        == set(val_df["target_model_dataset"])
    ), "discrepancy in target_model_dataset column across splits"
    assert (
        set(train_df["target_model"])
        == set(test_df["target_model"])
        == set(val_df["target_model"])
    ), "discrepancy in target_model distribution across splits"
    return train_df, val_df, test_df


def get_splits_for_tm(tm, train_df, val_df, test_df):
    print("--- filtering for target model ")
    train_df = drop_for_column_outside_of_values(train_df, "target_model", [tm])
    val_df = drop_for_column_outside_of_values(val_df, "target_model", [tm])
    test_df = drop_for_column_outside_of_values(test_df, "target_model", [tm])
    print("--- Train DF stats ---")
    print(show_df_stats(train_df))
    print("--- Test DF stats ---")
    print(show_df_stats(test_df))
    print("--- Val DF stats ---")
    print(show_df_stats(val_df))

    assert (
        set(train_df["target_model_dataset"])
        == set(test_df["target_model_dataset"])
        == set(val_df["target_model_dataset"])
    ), "discrepancy in target_model_dataset column across splits"

    dataset = list(set(train_df["target_model_dataset"]))[0]
    if dataset == "wikipedia_personal":
        assert len(set(train_df["attack_name"])) >= len(
            set(val_df["attack_name"])
        ), "train_df has less attacks than val_df"
        assert set(train_df["attack_name"]).union(set(val_df["attack_name"])) == set(
            train_df["attack_name"]
        ), "attacks missing in either of the splits"
        assert len(set(train_df["attack_name"])) >= len(
            set(test_df["attack_name"])
        ), "train_df has more attacks than test_df"
        assert set(train_df["attack_name"]).union(set(test_df["attack_name"])) == set(
            train_df["attack_name"]
        ), "attacks missing in either of the splits"
    else:
        assert (
            set(train_df["attack_name"])
            == set(test_df["attack_name"])
            == set(val_df["attack_name"])
        ), "attacks missing in either of the splits"

    assert (
        set(train_df["target_model"])
        == set(test_df["target_model"])
        == set(val_df["target_model"])
    ), "discrepancy in target_model distribution across splits"
    return train_df, val_df, test_df


def main():
    print("--- reading data")
    df = get_dataset_df("data_tcab/whole_catted_dataset.csv")

    print("--- dropping duplicates")
    df = df.drop_duplicates(subset=PRIMARY_KEY_FIELDS)

    train_list = []
    val_list = []
    test_list = []

    print("--- making splits across all datasets ---")

    combined_dump_path = Path("data_tcab", "official_TCAB_splits", "combined")
    mkdir_if_dne(combined_dump_path)

    for dataset in SUPPORTED_TARGET_MODEL_DATASETS:
        if dataset in df["target_model_dataset"].unique():
            print(dataset)
            train_df_temp, val_df_temp, test_df_temp = get_splits_for_dataset(
                dataset, df
            )
            dataset_dump_path = Path(
                "data_tcab", "official_TCAB_splits", "splits_by_dataset_and_tm", dataset
            )
            mkdir_if_dne(dataset_dump_path)
            for tm in SUPPORTED_TARGET_MODELS:
                if tm in df["target_model"].unique():
                    train_df_temp_, val_df_temp_, test_df_temp_ = get_splits_for_tm(
                        tm, train_df_temp, val_df_temp, test_df_temp
                    )
                    dataset_and_tm_dump_path = Path(
                        "data_tcab",
                        "official_TCAB_splits",
                        "splits_by_dataset_and_tm",
                        dataset,
                        tm,
                    )
                    mkdir_if_dne(dataset_and_tm_dump_path)
                    s1 = set(train_df_temp_["unique_src_instance_identifier"])
                    s2 = set(val_df_temp_["unique_src_instance_identifier"])
                    s3 = set(test_df_temp_["unique_src_instance_identifier"])
                    assert len(s1.intersection(s2)) == 0
                    assert len(s1.intersection(s3)) == 0
                    assert len(s2.intersection(s3)) == 0
                    train_df_temp_.to_csv(
                        Path(dataset_and_tm_dump_path, "train.csv"), index=False
                    )
                    val_df_temp_.to_csv(
                        Path(dataset_and_tm_dump_path, "val.csv"), index=False
                    )
                    test_df_temp_.to_csv(
                        Path(dataset_and_tm_dump_path, "test.csv"), index=False
                    )

                    print("---", dataset, tm, "--- DONE.")

            s1 = set(train_df_temp["unique_src_instance_identifier"])
            s2 = set(val_df_temp["unique_src_instance_identifier"])
            s3 = set(test_df_temp["unique_src_instance_identifier"])
            assert len(s1.intersection(s2)) == 0
            assert len(s1.intersection(s3)) == 0
            assert len(s2.intersection(s3)) == 0

            train_df_temp.to_csv(Path(dataset_dump_path, "train.csv"), index=False)
            val_df_temp.to_csv(Path(dataset_dump_path, "val.csv"), index=False)
            test_df_temp.to_csv(Path(dataset_dump_path, "test.csv"), index=False)

            print("---", dataset, "--- DONE.")

            train_list.append(train_df_temp)
            val_list.append(val_df_temp)
            test_list.append(test_df_temp)

    print("--- combining splits across all datasets ---")

    train_df = convert_nested_list_to_df(train_list)
    val_df = convert_nested_list_to_df(val_list)
    test_df = convert_nested_list_to_df(test_list)

    s1 = set(train_df["unique_src_instance_identifier"])
    s2 = set(val_df["unique_src_instance_identifier"])
    s3 = set(test_df["unique_src_instance_identifier"])
    assert len(s1.intersection(s2)) == 0
    assert len(s1.intersection(s3)) == 0
    assert len(s2.intersection(s3)) == 0

    train_df.to_csv(Path(combined_dump_path, "train.csv"), index=False)
    test_df.to_csv(Path(combined_dump_path, "test.csv"), index=False)
    val_df.to_csv(Path(combined_dump_path, "val.csv"), index=False)

    print("--- DONE.")


if __name__ == "__main__":
    main()
