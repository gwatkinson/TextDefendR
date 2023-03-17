import argparse
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


def download_allocine():
    # load dataset from the HuggingFace Hub
    dataset = load_dataset("allocine")

    dataset = dataset.rename_column("review", "text")

    export_dir = Path("data/allocine/")
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving allocine dataset in {export_dir}")
    columns = ["text", "label"]

    dataset["train"].to_csv(Path(export_dir, "train.csv"), index=False, columns=columns)
    dataset["validation"].to_csv(
        Path(export_dir, "val.csv"), index=False, columns=columns
    )
    dataset["test"].to_csv(Path(export_dir, "test.csv"), index=False, columns=columns)


def download_attack_dataset():
    # load dataset from the HuggingFace Hub
    dataset = load_dataset(
        "baptiste-pasquier/attack-dataset", use_auth_token=True, split="all"
    )

    export_dir = Path("data_tcab/")
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = Path(export_dir, "attack_dataset.csv")
    print(f"Saving attack dataset in {export_path}")
    dataset.to_csv(export_path, index=False)


def download_attack_embeddings():
    cache_dir = snapshot_download(
        "baptiste-pasquier/attack-dataset",
        allow_patterns=["*.joblib"],
        repo_type="dataset",
    )

    export_dir = Path("data_tcab/embeddings")
    print(f"Saving embeddings in {export_dir}")
    for file in Path(cache_dir).iterdir():
        print(f"Saving {file.name}")
        shutil.copy(file, export_dir)


def main(raw_args=None):
    # must have cms
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["allocine", "attack_dataset", "attack_embeddings"],
        help="Dataset to download.",
    )
    # get args, sanity check
    args = parser.parse_args(raw_args)

    if args.dataset == "allocine":
        download_allocine()
    elif args.dataset == "attack_dataset":
        download_attack_dataset()
    elif args.dataset == "attack_embeddings":
        download_attack_embeddings()
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")


if __name__ == "__main__":
    main()
