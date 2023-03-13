from pathlib import Path

import pandas as pd

from nlp_adversarial_attacks.reactdetect.utils.magic_vars import SUPPORTED_ATTACKS
from nlp_adversarial_attacks.reactdetect.utils.pandas_ops import (
    concat_multiple_df,
    drop_for_column_outside_of_values,
    refresh_index,
    show_df_stats,
)

# -------------------------------- GLOBAL VARS ------------------------------- #

RANDOM_SEED = 1  # make data sampling replicable

# ---------------------------------- MACROS ---------------------------------- #


def grab_csvs(root_dir):
    root_dir = Path(root_dir)
    # files = list(root_dir.glob("**/*.csv"))
    files = [
        "attacks/allocine/distilcamembert/none/clean/0/results.csv",
        "attacks/allocine/distilcamembert/textattack/bae/2000/results.csv",
        "attacks/allocine/distilcamembert/textattack/deepwordbug/2000/results.csv",
        "attacks/allocine/distilcamembert/textattack/input_reduction/2000/results.csv",
        "attacks/allocine/distilcamembert/textattack/pwws/2000/results.csv",
        "attacks/allocine/distilcamembert/textattack/textbugger/2000/results.csv",
        "attacks/allocine/distilcamembert/textattack/textfooler/2000/results.csv",
    ]
    return files


def all_cols_nan_to_strnone(df):
    columns = df.columns
    instruction = {}
    for cn in columns:
        instruction[cn] = "None"
    return df.fillna(instruction)


def address_clean_samples(df):
    for idx in df.index:
        if df.at[idx, "status"] == "clean":
            df.at[idx, "perturbed_text"] = df.at[idx, "original_text"]
            df.at[idx, "attack_name"] = "clean"
    return df


if __name__ == "__main__":
    print("--- loading all data")
    csvs = grab_csvs("attacks/")
    datas = []
    for csv in csvs:
        print(f"reading {csv}")
        df = pd.read_csv(csv)
        datas.append(df)
    whole_catted_dataset = concat_multiple_df(datas)
    assert "test_ndx" not in whole_catted_dataset.keys()
    assert "dataset" not in whole_catted_dataset.keys()
    whole_catted_dataset["test_ndx"] = whole_catted_dataset["test_index"]
    whole_catted_dataset["original_text_identifier"] = whole_catted_dataset[
        "test_index"
    ]
    whole_catted_dataset["dataset"] = whole_catted_dataset["target_model_dataset"]
    whole_catted_dataset = refresh_index(whole_catted_dataset)
    whole_catted_dataset = all_cols_nan_to_strnone(whole_catted_dataset)
    whole_catted_dataset = address_clean_samples(whole_catted_dataset)
    print("done, all data statistics: ")
    print(show_df_stats(whole_catted_dataset))

    # whole_catted_dataset.to_csv('whole_catted_dataset_unfiltered.csv')

    print("--- doing global filtering over all data")
    print("dropping unsuccessful attacks...")
    whole_catted_dataset = drop_for_column_outside_of_values(
        whole_catted_dataset, "status", ["success", "clean"]
    )

    print("dropping invalid attacks...")
    VALID_ATTACKS = SUPPORTED_ATTACKS
    whole_catted_dataset = drop_for_column_outside_of_values(
        whole_catted_dataset, "attack_name", VALID_ATTACKS
    )

    whole_catted_dataset = refresh_index(whole_catted_dataset)
    print("done, all data statistics: ")
    print(show_df_stats(whole_catted_dataset))

    print("-- saving to disk")
    whole_catted_dataset.to_csv("data_tcab/whole_catted_dataset.csv", index=False)
