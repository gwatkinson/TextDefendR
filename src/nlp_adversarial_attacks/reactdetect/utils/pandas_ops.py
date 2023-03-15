"""
Common pandas operations
"""
import math
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm


def concat_multiple_df(list_of_dfs):
    return refresh_index(pd.concat(list_of_dfs, axis=0))


def refresh_index(df):
    return df.reset_index(drop=True)


def no_duplicate_index(df):
    return df.index.is_unique


def no_duplicate_entry(df, column_name):
    return df[column_name].is_unique


def drop_for_column_outside_of_values(df, column_name, values):
    return df[df[column_name].isin(values)]


def drop_for_column_inside_values(df, column_name, values):
    return df[~df[column_name].isin(values)]


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


def downsample_clean_to_max_nonclean_class(df):
    class_distribution = Counter(df["attack_name"])
    max_num_other_than_clean_per_class = sorted(
        [i[1] for i in class_distribution.items() if i[0] != "clean"]
    )[-1]
    return restrict_max_instance_for_class(
        df, "clean", max_num_other_than_clean_per_class
    )


def downsample_clean_to_sum_nonclean_class(df):
    class_distribution = Counter(df["attack_name"])
    max_num_other_than_clean_per_class = sum(
        [i[1] for i in class_distribution.items() if i[0] != "clean"]
    )
    return restrict_max_instance_for_class(
        df, "clean", max_num_other_than_clean_per_class
    )


def split_df_by_column(idf, column):
    """
    https://stackoverflow.com/questions/40498463/python-splitting-dataframe-into-multiple-dataframes-based-on-column-values-and/40498517
    split a dataframe into sub dataframes, each grouped by unique values of that col
    return list of sub-dataframes
    """
    out = []
    for _region, df_region in idf.groupby(column):
        out.append(df_region)
    return out


def convert_nested_list_to_df(df_list):
    """
    Converts a list of pd.DataFrame objects into one pd.DataFrame object.
    """
    return pd.concat(df_list)


def create_ideal_train_test_split(df, split_ratio=0.6):
    idf_groups = split_df_by_column(df, "unique_src_instance_identifier")
    train_groups = []
    test_groups = []
    len_train = 0
    len_test = 0

    for small_df in tqdm(idf_groups):
        if len_train == 0 or len_train / (len_train + len_test) < split_ratio:
            train_groups.append(small_df)
            len_train += len(small_df)
        else:
            test_groups.append(small_df)
            len_test += len(small_df)

    train_df = convert_nested_list_to_df(train_groups)
    test_df = convert_nested_list_to_df(test_groups)
    s1 = set(train_df["unique_src_instance_identifier"])
    s2 = set(test_df["unique_src_instance_identifier"])
    assert len(s1.intersection(s2)) == 0, "there duplicate entries across splits!"
    return train_df, test_df
