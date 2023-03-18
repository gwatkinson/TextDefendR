import json
import re
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from IPython.core.display import HTML, display
from pandas.io.formats.style import Styler
from sklearn.metrics import ConfusionMatrixDisplay

from nlp_adversarial_attacks.utils.file_io import load_json


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


def extract_infos(metrics_path):
    parts = metrics_path.parts
    infos = {
        "target_dataset": parts[-7],
        "target_model": parts[-6],
        "embeddings": parts[-5],
        "setting": parts[-4],
        "classification_model": parts[-3],
        "feature_setting": parts[-2],
    }
    expe_json = load_json(Path(metrics_path.parents[2], "settings.json"))
    assert infos["target_dataset"] == expe_json["target_dataset"]
    assert infos["target_model"] == expe_json["target_model"]
    assert infos["embeddings"] == expe_json["embeddings_name"]
    assert infos["setting"] == expe_json["setting"]
    infos["is_binary"] = expe_json["is_binary"]

    return infos


def select_metrics_columns(columns):
    new_columns = []
    tags = ["accuracy", "recall", "precision", "f1_score", "roc_auc"]
    for column in columns:
        if any(tag in column for tag in tags):
            new_columns.append(column)
    return new_columns


def select_experiment(
    df, columns_order, metrics=None, split=None, verbose=False, **kwargs
):
    df_temp = df.copy()
    description = []
    for column, modality in kwargs.items():
        description.append(f"{column} = {modality}")
        if isinstance(modality, list):
            df_temp = df_temp[df_temp[column].isin(modality)]
        else:
            df_temp = df_temp[df_temp[column] == modality]

    description = ", ".join(description)
    if verbose:
        display(HTML(f"<h3>- {description}<h3/>"))

    if len(df_temp) == 1:
        return df_temp.iloc[0]

    columns = df.columns
    if metrics:
        columns = select_metrics_columns(columns)
        if split:
            columns = [col for col in columns if col.startswith(split)]
        columns = list(columns_order.keys()) + columns
    return df_temp[columns]


def remove_columns(df, tag_list):
    columns = []
    for col in df.columns:
        if not any(tag in col for tag in tag_list):
            columns.append(col)
    return df[columns]


def sort_function(order_list):
    return lambda serie: serie.map(lambda x: order_list.index(x))


def columns_order_to_sort(columns_order):
    columns_key = {}
    for column in columns_order:
        columns_key[column] = sort_function(columns_order[column])
    return columns_key


def display_tables(df, columns_order, columns_key, transpose=False, sort_metric=None):
    for column in columns_order.keys():
        display(HTML(f"<h3>{column}<h3/>"))
        df_temp = df.groupby(column).mean(numeric_only=True).drop(columns=["is_binary"])
        if sort_metric:
            df_temp = df_temp.sort_values(sort_metric, ascending=False)
        else:
            df_temp = df_temp.sort_index(key=columns_key[column])
        if transpose:
            df_temp = df_temp.transpose()
            df_temp.index.name = "feature"
            gradient_axis = 1
        else:
            gradient_axis = 0
        df_temp_style = df_temp.style.background_gradient(cmap=CMAP, axis=gradient_axis)
        display(df_temp_style)


def scatter_plot(
    df, columns_order, ax_column, x_column, color_column, metric, ymin=0.2
):
    fig, axes = plt.subplots(
        1, len(columns_order[ax_column]), figsize=(4 * len(columns_order[ax_column]), 3)
    )
    for i, ax_modality in enumerate(columns_order[ax_column]):
        sns.stripplot(
            data=df[df[ax_column] == ax_modality],
            x=x_column,
            y=metric,
            hue=color_column,
            order=columns_order[x_column],
            ax=axes[i],
        )
        axes[i].set_title(ax_modality)

    for ax in axes:
        ax.set_ylim(ymin, 1)
        ax.grid(axis="y")

    plt.show()


def plot_confusion_matrix_serie(
    serie, binary_classes=None, multi_classes=None, title=""
):
    if serie[0].shape[0] == 2:
        classes = binary_classes
        figsize = (4 * len(serie), 4)
        xticks_rotation = False
    else:
        classes = multi_classes
        figsize = (6 * len(serie), 6.5)
        xticks_rotation = True
    fig, axes = plt.subplots(1, len(serie), figsize=figsize)
    for i, (index, cm) in enumerate(serie.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(index)
        xlabels = axes[i].get_xticklabels()
        if xticks_rotation:
            axes[i].set_xticklabels(
                xlabels, rotation=40, ha="right", rotation_mode="anchor"
            )
    plt.tight_layout()
    fig.suptitle(title)
    plt.show()


CMAP = sns.diverging_palette(5, 250, as_cmap=True)
CMAP_R = sns.diverging_palette(250, 5, as_cmap=True)
