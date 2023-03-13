"""
Common pandas operations
"""
import math
from collections import Counter

import pandas as pd


def concat_multiple_df(list_of_dfs):
    return refresh_index(pd.concat(list_of_dfs, axis=0))


def refresh_index(df):
    return df.reset_index(drop=True)


def no_duplicate_index(df):
    return df.index.is_unique


def drop_for_column_outside_of_values(df, column_name, values):
    return df[df[column_name].isin(values)]


def show_df_stats(df):
    out = ""
    out += "total_instances: " + str(len(df)) + ", \n"
    out += "attack_name: " + str(dict(Counter(df["attack_name"]))) + ", \n"
    out += (
        "target_model_dataset: "
        + str(dict(Counter(df["target_model_dataset"])))
        + ", \n"
    )
    out += "target_model: " + str(dict(Counter(df["target_model"]))) + ", \n"
    out += "status: " + str(dict(Counter(df["status"]))) + ", \n"
    out += "attack_toolchain: " + str(dict(Counter(df["attack_toolchain"]))) + ", \n"
    out += "scenario: " + str(dict(Counter(df["scenario"]))) + ", \n"
    return out


def restrict_max_instance_for_class(
    in_df,
    attack_name_to_clip,
    max_instance_per_class=math.inf,
    min_instance_per_class=0,
    make_copy=False,
):
    """
    restrict the instances per class, class meaning attack_name
    """
    if make_copy:
        # make a copy of the dataset
        df = in_df.copy()
    else:
        df = in_df

    attack_names = sorted(set(df["attack_name"].unique()))  # duplicate class count ptsd

    # get data for each attack using the assigned samples
    dfs = []
    n_samples_per_class = []  # holds the number of samples for each attack
    for attack in attack_names:
        gf = df[(df["attack_name"] == attack)].copy()

        # downsample each class's dataframe untill add of the classes are <= maximum samples per class
        if len(gf) > max_instance_per_class and attack == attack_name_to_clip:
            gf = gf.head(max_instance_per_class)
        n_samples_per_class.append(len(gf))
        dfs.append(gf)

    df = pd.concat(dfs)
    df = refresh_index(df)
    assert no_duplicate_index(df)

    return df
