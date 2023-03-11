import os

import pandas as pd
from datasets import load_dataset


def main():
    # load dataset from the HuggingFace Hub
    dataset = load_dataset("allocine")

    # convert to dataframes, and prepare the data for training
    train_df = pd.DataFrame.from_dict(dataset["train"])
    val_df = pd.DataFrame.from_dict(dataset["validation"])
    test_df = pd.DataFrame.from_dict(dataset["test"])

    train_df = train_df.rename(columns={"review": "text"})
    val_df = val_df.rename(columns={"review": "text"})
    test_df = test_df.rename(columns={"review": "text"})

    print("Saving datasets...")
    columns = ["text", "label"]
    out_dir = "data/allocine/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    train_df.to_csv(out_dir + "train.csv", index=None, columns=columns)
    val_df.to_csv(out_dir + "val.csv", index=None, columns=columns)
    test_df.to_csv(out_dir + "test.csv", index=None, columns=columns)


if __name__ == "__main__":
    main()
