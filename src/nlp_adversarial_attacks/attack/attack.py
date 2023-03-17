from pathlib import Path

import numpy as np
import pandas as pd
import textattack
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm


def get_attack_recipe(attack_toolchain, attack_name, skip_words=None):
    """
    Load attack recipe.
    """

    if attack_toolchain in ["textattack"]:
        if attack_name == "bae":
            recipe = textattack.attack_recipes.BAEGarg2019

        elif attack_name == "bert":
            recipe = textattack.attack_recipes.BERTAttackLi2020

        elif attack_name == "checklist":
            recipe = textattack.attack_recipes.CheckList2020

        elif attack_name == "clare":
            recipe = textattack.attack_recipes.CLARE2020

        elif attack_name == "deepwordbug":
            recipe = textattack.attack_recipes.DeepWordBugGao2018

        elif attack_name == "faster_genetic":
            recipe = textattack.attack_recipes.FasterGeneticAlgorithmJia2019

        elif attack_name == "genetic":
            recipe = textattack.attack_recipes.GeneticAlgorithmAlzantot2018

        elif attack_name == "hotflip":
            recipe = textattack.attack_recipes.HotFlipEbrahimi2017

        elif attack_name == "iga_wang":
            recipe = textattack.attack_recipes.IGAWang2019

        elif attack_name == "input_reduction":
            recipe = textattack.attack_recipes.InputReductionFeng2018

        elif attack_name == "kuleshov":
            recipe = textattack.attack_recipes.Kuleshov2017

        elif attack_name == "pruthi":
            recipe = textattack.attack_recipes.Pruthi2019

        elif attack_name == "pso":
            recipe = textattack.attack_recipes.PSOZang2020

        elif attack_name == "pwws":
            recipe = textattack.attack_recipes.PWWSRen2019

        elif attack_name == "textbugger":
            recipe = textattack.attack_recipes.TextBuggerLi2018

        elif attack_name == "textfooler":
            recipe = textattack.attack_recipes.TextFoolerJin2019

        else:
            raise ValueError(f"Unknown recipe {attack_name}")
    else:
        raise ValueError(f"Unknown toolchain {attack_toolchain}")

    return recipe


def predict(args, model, test_df, device, logger=None):
    """
    Generate predictions on a given test set.
    """

    if logger:
        logger.info(f"No. test samples: {len(test_df):,}")

    # create dataloader
    test_data = test_df["text"].tolist()
    test_dataloader = DataLoader(
        test_data,
        sampler=SequentialSampler(test_data),
        batch_size=args.model_batch_size,
    )

    # generate predictions
    all_preds = []
    for text_list in tqdm(test_dataloader):
        # make predictions for this batch
        with torch.no_grad():
            preds = model(text_list)
            all_preds.append(preds.cpu().numpy().tolist())

    # concat all predictions
    all_preds = np.vstack(all_preds)
    y_pred = np.argmax(all_preds, axis=1)
    y_proba = softmax(all_preds, axis=1)

    # evaluate performance
    if logger:
        y_test = test_df["label"].values

        acc = accuracy_score(y_test, y_pred)
        s = f"Accuracy: {acc:.3f}"

        # multiclass classification
        if y_proba.shape[1] > 2:
            con_mat = confusion_matrix(y_test, y_pred)
            s += f"Confusion matrix:\n{con_mat}"

        logger.info(s)

    return y_pred, y_proba


def save_results(results, out_dir, logger):
    """
    Save adversarial samples and summary.
    """

    # compile results
    df = pd.DataFrame(results)

    # rearrange columns
    all_cols = df.columns.tolist()
    required_cols = [
        "scenario",
        "target_model",
        "target_model_train_dataset",
        "attack_toolchain",
        "attack_name",
        "target_dataset",
        "test_index",
        "original_text",
        "perturbed_text",
        "ground_truth",
        "original_output",
        "perturbed_output",
        "status",
        "num_queries",
        "frac_words_changed",
    ]
    extra_cols = [x for x in all_cols if x not in required_cols]
    if len(extra_cols):
        raise ValueError(f"Extra columns: {extra_cols}")
    df = df[required_cols]

    # save all samples
    pd.set_option("display.max_columns", 100)
    logger.info(f"\nResults:\n{df.head()}")
    df.to_csv(Path(out_dir, "results.csv"), index=False)
