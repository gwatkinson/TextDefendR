from pathlib import Path

import pandas as pd

from nlp_adversarial_attacks.utils.magic_vars import SUPPORTED_ATTACKS
from nlp_adversarial_attacks.utils.pandas_ops import (
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


def main():
    print("--- loading all data")
    csvs = grab_csvs("attacks/")
    datas = []
    for csv in csvs:
        print(f"reading {csv}")
        df = pd.read_csv(csv)
        datas.append(df)
    attack_dataset = concat_multiple_df(datas)
    attack_dataset = refresh_index(attack_dataset)
    attack_dataset = all_cols_nan_to_strnone(attack_dataset)
    attack_dataset = address_clean_samples(attack_dataset)
    print("done, all data statistics: ")
    print(show_df_stats(attack_dataset))

    print("--- doing global filtering over all data")
    print("dropping unsuccessful attacks...")
    attack_dataset = drop_for_column_outside_of_values(
        attack_dataset, "status", ["success", "clean"]
    )

    print("dropping invalid attacks...")
    attack_dataset = drop_for_column_outside_of_values(
        attack_dataset, "attack_name", SUPPORTED_ATTACKS
    )

    attack_dataset = refresh_index(attack_dataset)
    print("done, all data statistics: ")
    print(show_df_stats(attack_dataset))

    print("-- saving to disk")
    attack_dataset.to_csv("data_tcab/attack_dataset.csv", index=False)


if __name__ == "__main__":
    main()
