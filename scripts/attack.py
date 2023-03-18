"""
Attacks a dataset given a trained model.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from textattack import AttackArgs, Attacker
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.datasets import Dataset

import textdefendr.attack.utils as utils
from textdefendr.attack import ModelWrapper, get_attack_recipe, predict, save_results
from textdefendr.models.model_loading import load_target_model

# set environments
os.environ["TA_CACHE_DIR"] = ".cache"  # textattack cache
os.environ["TRANSFORMERS_CACHE"] = ".cache"  # transformers cache


here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + "/../../")  # REACT root directory


def main(args):
    start = time.time()

    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)

    # make sure the dataset matches the scenario
    if args.dataset_name in [
        "nuclear_energy",
        "climate-change_waterloo",
        "imdb",
        "sst",
        "allocine",
    ]:
        assert args.task_name == "sentiment"

    elif args.dataset_name in [
        "wikipedia",
        "hatebase",
        "civil_comments",
        "wikipedia_personal",
        "wikipedia_aggression",
        "reddit_dataset",
        "gab_dataset",
    ]:
        assert args.task_name == "abuse"

    elif args.dataset_name == "fnc1":
        assert args.task_name == "fake_news"

    else:
        raise ValueError(f"unknown dataset_name {args.dataset_name}")

    # make sure there is no attack toolchain if generating 'clean' samples
    if args.attack_toolchain == "none":
        assert args.attack_name == "clean"
    if args.attack_name == "clean":
        assert args.attack_toolchain == "none"
        assert args.attack_n_samples == 0

    # setup output directory
    out_dir = Path(
        args.dir_out,
        args.dataset_name,
        args.model_name,
        args.attack_toolchain,
        args.attack_name,
        str(args.attack_n_samples),
    )
    os.makedirs(out_dir, exist_ok=True)
    logger = utils.get_logger(Path(out_dir, "log.txt"))
    logger.info(args)
    logger.info(f"Timestamp: {datetime.now()}")

    # save config file
    utils.cmd_args_to_yaml(args, Path(out_dir, "config.yml"))

    # set no. labels
    num_labels = 2
    if args.dataset_name in ["nuclear_energy", "climate-change_waterloo"]:
        num_labels = 3

    # load trained model
    logger.info("\nLoading trained model...")
    model = load_target_model(
        model_name=args.model_name,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        num_labels=num_labels,
        max_seq_len=args.model_max_seq_len,
        device=device,
    )

    # read in the test set
    logger.info("\nLoading test data set...")
    dir_dataset = Path(args.dir_dataset, args.dataset_name)

    if os.path.exists(Path(dir_dataset, "test.csv")):
        test_df = pd.read_csv(Path(dir_dataset, "test.csv"))

    elif Path(dir_dataset, "data.csv"):
        test_df = pd.read_csv(Path(dir_dataset, "data.csv"))

    if "index" in test_df.columns:
        del test_df["index"]

    assert test_df.columns.tolist() == ["text", "label"]
    test_df = test_df.reset_index().rename(columns={"index": "test_index"})

    # generate predictions on the test set
    t = time.time()
    logger.info("Making prediction on the test set...")
    pred_test, proba_test = predict(args, model, test_df, device, logger=logger)
    test_df["pred"] = pred_test
    logger.info(f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-t))}")

    # save predictions on the test set
    if args.attack_name == "clean":
        test_text_list = test_df["text"].tolist()
        y_test = test_df["label"].values
        test_indices = test_df["test_index"].tolist()

        # record metadata for each prediction
        results = []
        for i, proba in enumerate(proba_test):
            text = test_text_list[i]
            proba = proba_test[i]

            result = {
                "scenario": args.task_name,
                "target_model": args.model_name,
                "target_model_train_dataset": args.target_model_train_dataset,
                "attack_toolchain": args.attack_toolchain,
                "attack_name": args.attack_name,
                "target_dataset": args.dataset_name,
                "test_index": test_indices[i],
                "ground_truth": y_test[i],
                "original_text": text,
                "perturbed_text": text,
                "original_output": proba,
                "perturbed_output": proba,
                "status": "clean",
                "num_queries": 0,
                "frac_words_changed": 0,
            }

            results.append(result)

        save_results(results, out_dir, logger)

        # cleanup and exit
        utils.remove_logger(logger)
        exit(0)

    # get label names and encode labels if necessary
    if args.dataset_name in [
        "wikipedia",
        "civil_comments",
        "hatebase",
        "wikipedia_personal",
        "wikipedia_aggression",
        "reddit_dataset",
        "gab_dataset",
    ]:
        label_names = ["non-toxic", "toxic"]

    elif args.dataset_name == "fnc1":
        label_map = {"agree": 0, "disagree": 1, "discuss": 2, "unrelated": 3}
        inverse_label_map = {v: k for k, v in label_map.items()}
        label_names = [inverse_label_map[i] for i in range(len(inverse_label_map))]

    elif args.dataset_name in ["imdb", "sst", "allocine"]:
        label_names = ["negative", "positive"]

    elif args.dataset_name in ["nuclear_energy", "climate-change_waterloo"]:
        label_names = ["negative", "neutral", "positive"]

    else:
        raise ValueError(f"unknown dataset {args.dataset_name}")

    # focus on correctly predicted TOXIC instances for abuse
    if args.task_name == "abuse":
        temp_df = test_df[(test_df["label"] == 1) & (test_df["pred"] == 1)].copy()

    # focus on correctly predicted instances
    else:
        temp_df = test_df[test_df["label"] == test_df["pred"]].copy()

    # attack text prioritizing longer text instances
    # temp_df["length"] = temp_df["text"].apply(lambda x: len(x.split()))
    # temp_df = temp_df.sort_values("length", ascending=False)
    if args.attack_n_samples:
        temp_df = temp_df[: args.attack_n_samples]
    temp_df = temp_df.reset_index(drop=True)
    logger.info("\nBeginning attacks...")
    logger.info(f"No. test: {len(test_df):,}, no. candidates: {len(temp_df):,}")

    # result containers
    results = []
    t = time.time()  # total time

    # selected indices information
    y_test = temp_df["label"].values
    test_indices = temp_df["test_index"].values

    # TextAttack
    if args.attack_toolchain in ["textattack"]:
        # prepare data
        dataset = Dataset(
            list(zip(temp_df["text"], temp_df["label"])), label_names=label_names
        )

        # prepare attacker
        model_wrapper = ModelWrapper(model)
        attack_recipe = get_attack_recipe(args.attack_toolchain, args.attack_name)
        attack = attack_recipe.build(model_wrapper)
        attack_args = AttackArgs(
            num_examples=-1,
            query_budget=args.attack_query_budget,
            log_summary_to_json=Path(out_dir, "summary.json").as_posix(),
        )
        attacker = Attacker(attack, dataset, attack_args)

        attack_results = attacker.attack_dataset()

        # attack test set
        for i, attack_result in enumerate(attack_results):
            # get attack status
            status = "success"
            if isinstance(attack_result, FailedAttackResult):
                status = "failed"
            elif isinstance(attack_result, SkippedAttackResult):
                status = "skipped"

            # get original and peturbed objects
            og = attack_result.original_result
            pp = attack_result.perturbed_result

            num_words_changed = len(og.attacked_text.all_words_diff(pp.attacked_text))

            result = {
                "scenario": args.task_name,
                "target_model": args.model_name,
                "target_model_train_dataset": args.target_model_train_dataset,
                "attack_toolchain": args.attack_toolchain,
                "attack_name": args.attack_name,
                "target_dataset": args.dataset_name,
                "test_index": test_indices[i],
                "ground_truth": y_test[i],
                "original_text": og.attacked_text.text,
                "perturbed_text": pp.attacked_text.text,
                "original_output": og.raw_output.tolist(),
                "perturbed_output": pp.raw_output.tolist(),
                "status": status,
                "num_queries": attack_result.num_queries,
            }
            try:
                result["frac_words_changed"] = num_words_changed / len(
                    og.attacked_text.words
                )
            except ZeroDivisionError:
                result["frac_words_changed"] = -1
            results.append(result)

            s = "Result {} (dataset={}, model={}, attack={}, no. queries={:,}):"
            logger.info(
                s.format(
                    i + 1,
                    args.dataset_name,
                    args.model_name,
                    args.attack_name,
                    result["num_queries"],
                )
            )
            logger.info(attack_result.goal_function_result_str() + "\n")
            # logger.info(attack_result.__str__(color_method="ansi"))

        # save leftover results
        logger.info(f"\nSaving results to {out_dir}/...")
        save_results(results, out_dir, logger)

    logger.info(
        f"\nAttack time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-t))}"
    )
    logger.info(
        f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # I/O settings
    parser.add_argument(
        "--dir_dataset",
        type=str,
        default="data/",
        help="Central directory for storing datasets.",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="attacks/",
        help="Central directory for storing attacks.",
    )

    # Experiment settings
    parser.add_argument(
        "--task_name",
        type=str,
        default="sentiment",
        help="e.g., abuse, sentiment or fake_news.",
    )

    # Model parameters
    parser.add_argument(
        "--model_name", type=str, default="distilcamembert", help="Model type."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="baptiste-pasquier/distilcamembert-allocine",
        help="Fine-tuned model configuration to load from cache or download (HuggingFace).",
    )
    parser.add_argument(
        "--model_max_seq_len", type=int, default=512, help="Max. no. tokens per string."
    )
    parser.add_argument(
        "--model_batch_size", type=int, default=32, help="No. instances per mini-batch."
    )

    # Data parameters
    parser.add_argument(
        "--dataset_name", type=str, default="allocine", help="Dataset to attack."
    )
    parser.add_argument(
        "--target_model_train_dataset",
        type=str,
        default="allocine",
        help="Dataset used to train the target model.",
    )

    # Attack parameters
    parser.add_argument(
        "--attack_toolchain",
        type=str,
        default="textattack",
        help="e.g., textattack or none.",
    )
    parser.add_argument(
        "--attack_name",
        type=str,
        default="deepwordbug",
        help="Name of the attack; clean = no attack.",
    )
    parser.add_argument(
        "--attack_query_budget",
        type=int,
        default=0,
        help="Max. no. of model queries per attack; 0 = infinite budget.",
    )
    parser.add_argument(
        "--attack_n_samples",
        type=int,
        default=10,
        help="No. samples to attack; 0 = attack all samples.",
    )

    # Other parameters
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed value to use for reproducibility.",
    )

    args = parser.parse_args()
    main(args)
