import json
import re
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
from IPython.core.display import HTML, display
from pandas.io.formats.style import Styler


def display_side_by_side(dfs: list[Union[pd.DataFrame, Styler]]):
    """Display tables side by side.

    Parameters
    ----------
    dfs : list[Union[pd.DataFrame, Styler]]
        list of DataFrames
    """
    output = ""
    for df in dfs:
        if isinstance(df, pd.DataFrame):
            df_styler = df.style
        else:
            df_styler = df
        output += df_styler.set_table_attributes("style='display:inline'")._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))


def display_freq_categorical(serie: pd.Series):
    """Display frequency distribution of a categorical variable

    Parameters
    ----------
    serie : pd.Series
    """
    name = serie.name
    df = serie.value_counts().to_frame().rename(columns={name: "Frequency"})
    df.index.name = name
    df["Percent"] = df["Frequency"] / df["Frequency"].sum()
    display(
        df.head(25).style.format(
            {
                "Percent": "{:.2%}".format,
            }
        )
    )


def load_attack(folder_path: str) -> dict:
    """Load attack data from a folder.

    Parameters
    ----------
    folder_path : str

    Returns
    -------
    dict
    """
    folder_path = Path(folder_path)

    with open(Path(folder_path, "config.yml")) as f:
        config = yaml.safe_load(f)

    df_results = pd.read_csv(Path(folder_path, "results.csv"))
    df_results["label"] = df_results["ground_truth"].replace(
        {0: "Negative", 1: "Positive"}
    )

    with open(Path(folder_path, "summary.json")) as f:
        summary = json.load(f)["Attack Results"]

    attack_time = None
    with open(Path(folder_path, "log.txt")) as f:
        for line in f.readlines():
            results = re.search(r"Attack time: (.*)", line)
            if results:
                attack_time = results.group(1)
                break

    summary["Attack time"] = pd.Timedelta(attack_time).seconds

    return {"config": config, "df_results": df_results, "summary": summary}
