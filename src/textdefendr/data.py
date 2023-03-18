import pandas as pd


def load_dataset():
    # TODO : download from Internet.
    df = pd.read_csv("data_tcab/attack_dataset.csv")

    df.rename(columns={"perturbed_text": "text"}, inplace=True)

    df["perturbed"] = df["attack_name"] != "clean"

    return df[["text", "perturbed", "attack_name"]]
