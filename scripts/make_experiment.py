import argparse
from pathlib import Path

import joblib
import pandas as pd

from textdefendr.utils.file_io import grab_joblibs, load_json
from textdefendr.utils.hashing import get_pk_tuple
from textdefendr.utils.magic_vars import (
    PRIMARY_KEY_FIELDS,
    SUPPORTED_TARGET_DATASETS,
    SUPPORTED_TARGET_MODELS,
)
from textdefendr.utils.pandas_ops import no_duplicate_index


def check_joblib_dict(samples_dict):
    """
    Checks to make sure each example has the correct
        number of fields in the primary key; if not,
        this method attempts to fill in the missing values.

    Input
        samples_dict: dict of extracted features
            format: key - sample index, value - dict.

    Return
        samples_dict, with updated primary keys and unique IDs.
    """
    for _key, value in samples_dict.items():
        assert len(value["primary_key"]) == len(PRIMARY_KEY_FIELDS)
    return samples_dict


def load_known_instances(
    rootdir_with_joblib_file,
    target_model,
    target_dataset,
    embeddings_name,
):
    assert target_model in SUPPORTED_TARGET_MODELS
    assert target_dataset in SUPPORTED_TARGET_DATASETS

    print("\n--- loading known instances from ", rootdir_with_joblib_file)
    print(f"--- target_model: {target_model}, target_dataset: {target_dataset}")

    all_jblb = grab_joblibs(rootdir_with_joblib_file)

    filter_string = "_".join([embeddings_name, target_model, target_dataset])
    all_jblb = [jblb for jblb in all_jblb if jblb.name.startswith(filter_string)]

    print(f"--- No. joblib files of varied size to read: {len(all_jblb):,}")

    known_samples = {}

    for i, jblb in enumerate(all_jblb):
        print(f"[{i}] {jblb}")  # to avoid name collision

        holder = check_joblib_dict(joblib.load(jblb))

        for idx in holder.keys():
            instance = holder[idx]
            pk = instance["primary_key"]
            pk = tuple(pk)
            assert isinstance(pk, tuple)
            if pk in known_samples:
                raise ValueError("Duplicate instance in joblib files.")
            known_samples[pk] = instance

    print(
        f"\ndone, no. unique instances with extracted features: {len(known_samples):,}"
    )
    return known_samples


def df_to_instance_subset(df, known_samples):
    assert no_duplicate_index(df)

    out = {}
    no_repr_count = 0

    for idx in df.index:
        pk = get_pk_tuple(df, idx)
        pk = tuple(pk)

        if pk in known_samples:
            out[pk] = known_samples[pk]
            del known_samples[pk]

            out[pk]["attack_name"] = df.at[idx, "attack_name"]
            out[pk]["target_label"] = df.at[idx, "target_label"]
            out[pk]["original_text"] = df.at[idx, "original_text"]
            out[pk]["perturbed_text"] = df.at[idx, "perturbed_text"]
        else:
            no_repr_count += 1

    print(f"    cannot find embeddings for {no_repr_count:,} / {len(df):,} instances")

    return out


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="data_tcab/detection-experiments/allocine/distilcamembert/default/clean_vs_all/",
        help="Directory of the distributed experiment to be made.",
    )
    main_args = parser.parse_args(raw_args)
    out_dir = main_args.experiment_dir

    assert Path(out_dir, "settings.json").exists(), (
        "cannot find settings.json in " + out_dir
    )
    assert Path(out_dir, "train.csv").exists(), "cannot find train.csv in " + out_dir
    assert Path(out_dir, "val.csv").exists(), "cannot find train.csv in " + out_dir
    assert Path(out_dir, "test_release.csv").exists(), (
        "cannot find test.csv in " + out_dir
    )

    exp_args = load_json(Path(out_dir, "settings.json"))
    exp_args = argparse.Namespace(**exp_args)

    print("making exp into... ", out_dir)

    known_instances = load_known_instances(
        "data_tcab/embeddings/",
        target_model=exp_args.target_model,
        target_dataset=exp_args.target_dataset,
        embeddings_name=exp_args.embeddings_name,
    )

    print("\n--- creating train.joblib")
    train_df = pd.read_csv(Path(out_dir, "train.csv"))
    train_data = df_to_instance_subset(train_df, known_instances)
    joblib.dump(train_data, Path(out_dir, "train.joblib"))

    print("\n--- creating val.joblib")
    val_df = pd.read_csv(Path(out_dir, "val.csv"))
    val_data = df_to_instance_subset(val_df, known_instances)
    joblib.dump(val_data, Path(out_dir, "val.joblib"))

    print("\n--- creating test_release.joblib")
    test_df_release = pd.read_csv(Path(out_dir, "test_release.csv"))
    test_release_data = df_to_instance_subset(test_df_release, known_instances)
    joblib.dump(test_release_data, Path(out_dir, "test_release.joblib"))


if __name__ == "__main__":
    main()
