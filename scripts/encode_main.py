import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from nlp_adversarial_attacks.batch_encoding.encode_samplewise_features import (
    encode_all_properties,
)
from nlp_adversarial_attacks.reactdetect.utils.file_io import mkfile_if_dne
from nlp_adversarial_attacks.reactdetect.utils.magic_vars import (
    SUPPORTED_ATTACKS,
    SUPPORTED_TARGET_MODEL_DATASETS,
    SUPPORTED_TARGET_MODELS,
)
from nlp_adversarial_attacks.reactdetect.utils.pandas_ops import (
    restrict_max_instance_for_class,
    show_df_stats,
)


def holder_to_disk(holder, fname):
    """
    Holder is a nested dict, see `encode_samplewise_features.py`
    """
    joblib.dump(holder, fname)


if __name__ == "__main__":
    start = time.time()
    # arg sanity checks

    # must have cms
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="distilcamembert",
        help="Target model type.",
    )
    parser.add_argument(
        "--target_model_dataset",
        type=str,
        default="allocine",
        help="Dataset attacked.",
    )
    parser.add_argument(
        "--target_model_train_dataset",
        type=str,
        default="allocine",
        help="Dataset used to train the target model.",
    )
    parser.add_argument(
        "--attack_name",
        type=str,
        default="ALL",
        help="Name of the attack or ALL or ALLBUTCLEAN.",
    )
    parser.add_argument(
        "--max_clean_instance",
        type=int,
        default=0,
        help="Only consider certain number of clean instances; 0 = consider all.",
    )

    # encode models
    parser.add_argument(
        "--tp_model",
        type=str,
        default="sentence-transformers/bert-base-nli-mean-tokens",
        help="Sentence embeddings model for text properties features.",
    )
    parser.add_argument(
        "--lm_perplexity_model",
        type=str,
        default="gpt2",
        help="GPT2 model for lm perplexity features. (e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2)",
    )
    parser.add_argument(
        "--lm_proba_model",
        type=str,
        default="roberta-base",
        help="Roberta model for lm proba features. (e.g. roberta-base, roberta-large, distilroberta-base)",
    )
    parser.add_argument(
        "--target_model_name_or_path",
        type=str,
        default="baptiste-pasquier/distilcamembert-allocine",
        help="Fine-tuned target model to load from cache or download (HuggingFace).",
    )

    # other might be useful cmds
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="If True only computes first 10 instance.",
    )
    parser.add_argument(
        "--disable_tqdm",
        type=bool,
        default=False,
        help="If True silent tqdm progress bar.",
    )
    parser.add_argument(
        "--prefix_file_name",
        type=str,
        default="",
        help="Prefix for resulting file name.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="ALL",
        help="Tasks to perform in string format (e.g 'TP,LM_PROBA,LM_PERPLEXITY,TM').",
    )

    # get args, sanity check
    args = parser.parse_args()
    assert args.target_model in SUPPORTED_TARGET_MODELS
    assert args.target_model_dataset in SUPPORTED_TARGET_MODEL_DATASETS
    assert args.target_model_train_dataset in SUPPORTED_TARGET_MODEL_DATASETS
    assert (
        args.attack_name in SUPPORTED_ATTACKS
        or args.attack_name == "ALL"
        or args.attack_name == "ALLBUTCLEAN"
    )
    assert type(args.max_clean_instance) == int

    # io
    print("--- reading csv")
    DF = pd.read_csv("data_tcab/whole_catted_dataset.csv")
    print()
    print("--- stats before filtering")
    print(show_df_stats(DF))

    # compute
    print("--- filtering dataframe")
    if args.attack_name == "ALL":
        print("--- attack name is ALL, using all attacks")
        DF = DF[
            (DF["target_model_dataset"] == args.target_model_dataset)
            & (DF["target_model_train_dataset"] == args.target_model_train_dataset)
            & (DF["target_model"] == args.target_model)
        ]
    elif args.attack_name == "ALLBUTCLEAN":
        print("--- attack name is ALLBUTCLEAN, using all attacks but clean")
        DF = DF[
            (DF["target_model_dataset"] == args.target_model_dataset)
            & (DF["target_model_train_dataset"] == args.target_model_train_dataset)
            & (DF["target_model"] == args.target_model)
            & (DF["attack_name"] != "clean")
        ]
    else:
        DF = DF[
            (DF["target_model_dataset"] == args.target_model_dataset)
            & (DF["target_model_dataset"] == args.target_model_dataset)
            & (DF["target_model"] == args.target_model)
            & (DF["attack_name"] == args.attack_name)
        ]

    print(" done , instance distribution: ")
    print(show_df_stats(DF))

    if args.max_clean_instance > 0:
        print("--- dropping clean instance to ", args.max_clean_instance)
        print(" done , instance distribution: ")
        DF = restrict_max_instance_for_class(
            in_df=DF,
            attack_name_to_clip="clean",
            max_instance_per_class=args.max_clean_instance,
        )
        print(show_df_stats(DF))

    print()
    print("--- starting the encoding process")

    # if test use only 10 sample
    if args.test:
        print("*** WARNING, TEST MODE, only encode 10 samples")
        DF = DF.head(10)

    # encode everything. DF in, dict out
    HOLDER = encode_all_properties(
        DF,
        tp_model=args.tp_model,
        lm_perplexity_model=args.lm_perplexity_model,
        lm_proba_model=args.lm_proba_model,
        tm_model=args.target_model,
        tm_model_name_or_path=args.target_model_name_or_path,
        disable_tqdm=args.disable_tqdm,
        tasks=args.tasks,
    )
    file_name = (
        "_".join(
            [args.target_model, args.target_model_dataset, args.attack_name, args.tasks]
        )
        + ".joblib"
    )

    print("--- saving to disk")
    if args.prefix_file_name:
        file_name = args.prefix_file_name + "_" + file_name
    if args.test:
        file_name = "test_" + file_name
    file_path = Path("data_tcab/reprs/samplewise", file_name)
    mkfile_if_dne(file_path)
    holder_to_disk(HOLDER, file_path)
    print(f"saved in {file_path}")

    print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")
