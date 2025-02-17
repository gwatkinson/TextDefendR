# TextDefendR: Detecting adversarial attacks on French text classifiers <!-- omit from toc -->

[![Code quality](https://github.com/baptiste-pasquier/textdefendr/actions/workflows/quality.yml/badge.svg)](https://github.com/baptiste-pasquier/textdefendr/actions/workflows/quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- [:mag\_right: Overview](#mag_right-overview)
- [:hammer\_and\_wrench: Installation](#hammer_and_wrench-installation)
- [:arrow\_forward: Quickstart](#arrow_forward-quickstart)
- [:memo: Usage](#memo-usage)
  - [1. DistilCamemBERT fine-tuning on Allociné](#1-distilcamembert-fine-tuning-on-allociné)
    - [1.1. Model fine-tuning (optional)](#11-model-fine-tuning-optional)
    - [1.2. Model evaluation](#12-model-evaluation)
  - [2. TCAB Dataset Generation](#2-tcab-dataset-generation)
    - [2.1. Download the Allociné dataset](#21-download-the-allociné-dataset)
    - [2.2. Run some attacks with TextAttack (optional)](#22-run-some-attacks-with-textattack-optional)
    - [2.3. Run attacks for TCAB dataset](#23-run-attacks-for-tcab-dataset)
    - [2.4. Generate the attack dataset](#24-generate-the-attack-dataset)
  - [3. TCAB Benchmark](#3-tcab-benchmark)
    - [3.1. Encode the dataset with feature extraction](#31-encode-the-dataset-with-feature-extraction)
    - [3.2. Split data by model and trained dataset](#32-split-data-by-model-and-trained-dataset)
    - [3.3. Distribute data for detection experiments](#33-distribute-data-for-detection-experiments)
    - [3.4. Merge experiment data with feature extraction](#34-merge-experiment-data-with-feature-extraction)
    - [3.5. Run experiment](#35-run-experiment)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## :mag_right: Overview

TextDefendR is a library for detecting adversarial attacks on NLP classification models.
It provides:
- a script to generate attacks on a Transformers model and create a dataset of several attacks;
- several tools to extract embeddings on generated samples;
- experiments to train classifiers for attack detection.

The project reproduces the results from the following paper: "*Identifying Adversarial Attacks on Text Classifiers*" [[1]](#1) and uses the associated code (see [Acknowledgments](#6-acknowledgments)).

## :hammer_and_wrench: Installation

1. Clone the repository
```bash
git clone https://github.com/baptiste-pasquier/textdefendr
```

2. Install the project
- With `poetry` ([installation](https://python-poetry.org/docs/#installation)):
```bash
poetry install
```
- With `pip` :
```bash
pip install -e .
```

3. (Optional) Install Pytorch CUDA
```bash
poe torch_cuda
```

## :arrow_forward: Quickstart

:bulb: Notebook: [quickstart.ipynb](notebooks/quickstart.ipynb)

```python
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from textdefendr.encoder import TextEncoder
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

The dataset contains 9000 samples of attacks on Allociné + 20000 original reviews.
The `attack_name` column shows the name of the attack used, or "clean" for original texts.
The `perturbed_text` column contains the text modified by an attack, or the original text for unattacked samples.

```python
df = load_dataset("baptiste-pasquier/attack-dataset", split="all").to_pandas()
df = df.sample(1000, random_state=42)
```

To train a binary classification model, you can consider the binary variable that indicates whether a text comes from an attack.

```python
X = df["perturbed_text"]
y = df["perturbed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Let's encode text samples with several language model embeddings.

```python
encoder = TextEncoder(
    enable_tp=True,
    enable_lm_perplexity=True,
    enable_lm_proba=True,
    device=device)
X_train_encoded = encoder.fit_transform(X_train)
```

Now it is possible to use any usual classifier.

```python
clf = LogisticRegression(random_state=42)
clf.fit(X_train_encoded, y_train)
```

```python
X_test_encoded = encoder.transform(X_test)
clf.score(X_test_encoded, y_test)
```

## :memo: Usage

### 1. DistilCamemBERT fine-tuning on Allociné

#### 1.1. Model fine-tuning (optional)



- Fine-tune with TextAttack:
```{bash}
textattack train --model cmarkea/distilcamembert-base --dataset allocine --num-epochs 3 --learning_rate 5e-5 --num_warmum_steps 500 --weight_decay 0.01 --per-device-train-batch-size 16 --gradient_accumulation_steps 4 --load_best_model_at_end true --log-to-tb
```
- Fine-tune with Transformers: [model_finetuning.ipynb](notebooks/model_finetuning.ipynb)

The fine-tuned model is available on HuggingFace: https://huggingface.co/baptiste-pasquier/distilcamembert-allocine

#### 1.2. Model evaluation
- Evaluate the fine-tuned model with TextAttack:
```{bash}
textattack eval --model-from-huggingface baptiste-pasquier/distilcamembert-allocine --dataset-from-huggingface allocine --num-examples 1000 --dataset-split test
```
- Evaluate with Transformers: [model_evaluation.ipynb](notebooks/model_evaluation.ipynb)

The model offers an accuracy score of 97%.


### 2. TCAB Dataset Generation

This section provides
a database of attacks with a fine-tuned DistilCamemBERT model on the task of Allociné reviews classification.

:globe_with_meridians: Reference: https://github.com/react-nlp/tcab_generation

#### 2.1. Download the Allociné dataset

Run
```{bash}
python scripts/download_data.py allocine
```
This generates a `train.csv`, `val.csv`, and `test.csv` in `data/allocine/` directory.

#### 2.2. Run some attacks with TextAttack (optional)

Run
```{bash}
textattack attack --model-from-huggingface baptiste-pasquier/distilcamembert-allocine --dataset-from-huggingface allocine --recipe deepwordbug --num-examples 50
```

#### 2.3. Run attacks for TCAB dataset

Run
```{bash}
python scripts/attack.py
```
:memo: Usage
```
usage: attack.py [-h] [--dir_dataset DIR_DATASET] [--dir_out DIR_OUT]
                 [--task_name TASK_NAME] [--model_name MODEL_NAME]
                 [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]  
                 [--model_max_seq_len MODEL_MAX_SEQ_LEN]
                 [--model_batch_size MODEL_BATCH_SIZE] [--dataset_name DATASET_NAME]  
                 [--target_model_train_dataset TARGET_MODEL_TRAIN_DATASET]
                 [--attack_toolchain ATTACK_TOOLCHAIN] [--attack_name ATTACK_NAME]  
                 [--attack_query_budget ATTACK_QUERY_BUDGET]
                 [--attack_n_samples ATTACK_N_SAMPLES] [--random_seed RANDOM_SEED]  

options:
  -h, --help            show this help message and exit
  --dir_dataset DIR_DATASET
                        Central directory for storing datasets. (default: data/)  
  --dir_out DIR_OUT     Central directory for storing attacks. (default: attacks/)  
  --task_name TASK_NAME
                        e.g., abuse, sentiment or fake_news. (default: sentiment)  
  --model_name MODEL_NAME
                        Model type. (default: distilcamembert)
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Fine-tuned model configuration to load from cache or download  
                        (HuggingFace). (default: baptiste-pasquier/distilcamembert-  
                        allocine)
  --model_max_seq_len MODEL_MAX_SEQ_LEN
                        Max. no. tokens per string. (default: 512)
  --model_batch_size MODEL_BATCH_SIZE
                        No. instances per mini-batch. (default: 32)
  --dataset_name DATASET_NAME
                        Dataset to attack. (default: allocine)
  --target_model_train_dataset TARGET_MODEL_TRAIN_DATASET
                        Dataset used to train the target model. (default: allocine)  
  --attack_toolchain ATTACK_TOOLCHAIN
                        e.g., textattack or none. (default: textattack)
  --attack_name ATTACK_NAME
                        Name of the attack; clean = no attack. (default: deepwordbug)  
  --attack_query_budget ATTACK_QUERY_BUDGET
                        Max. no. of model queries per attack; 0 = infinite budget.  
                        (default: 0)
  --attack_n_samples ATTACK_N_SAMPLES
                        No. samples to attack; 0 = attack all samples. (default: 10)  
  --random_seed RANDOM_SEED
                        Random seed value to use for reproducibility. (default: 0)
```

:bulb: Notebook step-by-step: [run_attack.ipynb](/notebooks/run_attack.ipynb)

:bulb: Notebook attack statistics: [attack_statistics.ipynb](/notebooks/attack_statistics.ipynb)

#### 2.4. Generate the attack dataset

Combines all attacks into a single csv file: `data_tcab/attack_dataset.csv`

Run
```{bash}
python scripts/generate_attack_dataset.py
```

Our dataset is available on HuggingFace: https://huggingface.co/datasets/baptiste-pasquier/attack-dataset

### 3. TCAB Benchmark

:globe_with_meridians: Reference: https://github.com/react-nlp/tcab_benchmark

:inbox_tray: If you do not have a `attack_dataset.csv` dataset, you can download it from HuggingFace:
```{bash}
python scripts/download_data.py attack_dataset
```

:bulb: **You can run all this section in the following notebook:** [run_all_benchmark.ipynb](/notebooks/run_all_benchmark.ipynb)

#### 3.1. Encode the dataset with feature extraction

Extract embeddings for the detection model :
- TP: text properties
- TM: target model properties
- LM: language model properties

The result is stored ase a `.joblib` file under `data_tcab/embeddings/` directory.

Run
```{bash}
python scripts/encode_main.py
```
:memo: Usage
```
usage: encode_main.py [-h] [--target_model TARGET_MODEL]
                      [--target_dataset TARGET_DATASET]
                      [--target_model_train_dataset TARGET_MODEL_TRAIN_DATASET]
                      [--attack_name ATTACK_NAME]
                      [--max_clean_instance MAX_CLEAN_INSTANCE] [--tp_model TP_MODEL]  
                      [--lm_perplexity_model LM_PERPLEXITY_MODEL]
                      [--lm_proba_model LM_PROBA_MODEL]
                      [--target_model_name_or_path TARGET_MODEL_NAME_OR_PATH] [--test]  
                      [--disable_tqdm] [--embeddings_name EMBEDDINGS_NAME]
                      [--tasks TASKS]

options:
  -h, --help            show this help message and exit
  --target_model TARGET_MODEL
                        Target model type. (default: distilcamembert)
  --target_dataset TARGET_DATASET
                        Dataset attacked. (default: allocine)
  --target_model_train_dataset TARGET_MODEL_TRAIN_DATASET
                        Dataset used to train the target model. (default: allocine)  
  --attack_name ATTACK_NAME
                        Name of the attack or ALL or ALLBUTCLEAN. (default: ALL)  
  --max_clean_instance MAX_CLEAN_INSTANCE
                        Only consider certain number of clean instances; 0 = consider  
                        all. (default: 0)
  --tp_model TP_MODEL   Sentence embeddings model for text properties features.
                        (default: sentence-transformers/bert-base-nli-mean-tokens)  
  --lm_perplexity_model LM_PERPLEXITY_MODEL
                        GPT2 model for lm perplexity features. (e.g. gpt2,
                        gpt2-medium, gpt2-large, gpt2-xl, distilgpt2) (default: gpt2)  
  --lm_proba_model LM_PROBA_MODEL
                        Roberta model for lm proba features. (e.g. roberta-base,  
                        roberta-large, distilroberta-base) (default: roberta-base)  
  --target_model_name_or_path TARGET_MODEL_NAME_OR_PATH
                        Fine-tuned target model to load from cache or download
                        (HuggingFace). (default: baptiste-pasquier/distilcamembert-  
                        allocine)
  --test                Only computes first 10 instance. (default: False)
  --disable_tqdm        Silent tqdm progress bar. (default: False)
  --embeddings_name EMBEDDINGS_NAME
                        Prefix for resulting file name. (default: default)
  --tasks TASKS         Tasks to perform in string format (e.g.
                        'TP,LM_PROBA,LM_PERPLEXITY,TM'). (default: ALL)
```

For instance, to use the `DistilCamemBERT` model trained on the `Allociné` dataset as the target model, a version of `DistilCamemBERT` for the TP and LM probabilities, and a French implementation of `GPT2` for the perplexity, run:
```{bash}
python scripts/encode_main.py \
    --tp_model cmarkea/distilcamembert-base-nli \
    --lm_perplexity_model asi/gpt-fr-cased-small \
    --lm_proba_model cmarkea/distilcamembert-base \
    --embeddings_name fr+small \
```
This will create the file `data_tcab/embeddings/fr+small_distilcamembert_allocine_ALL_ALL.joblib`. This command is quite long, around 7 hours for the `Allociné` dataset.

To generate only the TP features with another model, run :
```{bash}
python scripts/encode_main.py \
    --tp_model google/canine-c \
    --embeddings_name fr+canine \
    --tasks TP
```
This will create the file `data_tcab/embeddings/fr+canine_distilcamembert_allocine_ALL_TP.joblib`. This is quite fast compared to the previous command (around 5 minutes).

:bulb: Notebook step-by-step: [run_encode_main.ipynb](/notebooks/run_encode_main.ipynb)

:bulb: Notebook step-by-step for encode_samplewise_features: [encode_samplewise_features.ipynb](/notebooks/encode_samplewise_features.ipynb)

:bulb: Notebook for feature extraction: [feature_extraction.ipynb](/notebooks/feature_extraction.ipynb)

:inbox_tray: You can download the embeddings from HuggingFace:
```{bash}
python scripts/download_data.py attack_embeddings
```

#### 3.2. Split data by model and trained dataset

```{bash}
python scripts/make_official_dataset_splits.py
```

#### 3.3. Distribute data for detection experiments

Create `train.csv`, `val.csv` and `test.csv` under `data_tcab/detection-experiments/` directory.
```{bash}
python scripts/distribute_experiments.py
```
:memo: Usage
```
usage: distribute_experiments.py [-h] [--target_dataset TARGET_DATASET]
                                 [--target_model TARGET_MODEL]
                                 [--embeddings_name EMBEDDINGS_NAME]
                                 [--experiment_setting {clean_vs_all,multiclass_with_clean}]

options:
  -h, --help            show this help message and exit
  --target_dataset TARGET_DATASET
                        Dataset attacked. (default: allocine)
  --target_model TARGET_MODEL
                        Target model type. (default: distilcamembert)
  --embeddings_name EMBEDDINGS_NAME
                        Embeddings name (prefix). (default: default)
  --experiment_setting {clean_vs_all,multiclass_with_clean}
                        Binary or multiclass detection. (default: clean_vs_all)
```

#### 3.4. Merge experiment data with feature extraction

Take an experiment directory that contains train and test csv files and make them into joblib files using cached features in `data_tcab/embeddings/` directory.
```{bash}
python scripts/make_experiment.py
```
:memo: Usage
```
usage: make_experiment.py [-h] [--experiment_dir EXPERIMENT_DIR]

options:
  -h, --help            show this help message and exit
  --experiment_dir EXPERIMENT_DIR
                        Directory of the distributed experiment to be made. (default:  
                        data_tcab/detection-
                        experiments/allocine/distilcamembert/default/clean_vs_all/)
```

#### 3.5. Run experiment

Take an experiment directory that contains train and test joblib files, then a classification model and log model, outputs and metrics in a unique subdirectory.
```{bash}
python scripts/run_experiment.py
```
:memo: Usage
```
usage: run_experiment.py [-h] [--experiment_dir EXPERIMENT_DIR]
                         [--feature_setting {bert,bert+tp,bert+tp+lm,all}]
                         [--model {LR,DT,RF,LGB}] [--skip_if_done] [--test]
                         [--model_n_jobs MODEL_N_JOBS] [--cv_n_jobs CV_N_JOBS]
                         [--solver SOLVER] [--penalty {l1,l2}]
                         [--train_frac TRAIN_FRAC] [--n_estimators N_ESTIMATORS]  
                         [--max_depth MAX_DEPTH] [--num_leaves NUM_LEAVES]
                         [--disable_tune]

options:
  -h, --help            show this help message and exit
  --experiment_dir EXPERIMENT_DIR
                        Directory of the distributed experiment. (default:
                        data_tcab/detection-
                        experiments/allocine/distilcamembert/default/clean_vs_all/)  
  --feature_setting {bert,bert+tp,bert+tp+lm,all}
                        Set of features to use. (default: all)
  --model {LR,DT,RF,LGB}
                        Classification model. (default: LR)
  --skip_if_done        Skip if an experiment is already runned. (default: False)  
  --test                Quick test model. (default: False)
  --model_n_jobs MODEL_N_JOBS
                        No. jobs to run in parallel for the model. (default: 1)
  --cv_n_jobs CV_N_JOBS
                        No. jobs to run in parallel for gridsearch. (default: 1)  
  --solver SOLVER       LR solver. (default: lbfgs)
  --penalty {l1,l2}     LR penalty. (default: l2)
  --train_frac TRAIN_FRAC
                        Fraction of train data to train with. (default: 1)
  --n_estimators N_ESTIMATORS
                        No. boosting rounds for lgb. (default: 100)
  --max_depth MAX_DEPTH
                        Max. depth for each tree. (default: 5)
  --num_leaves NUM_LEAVES
                        No. leaves per tree. (default: 32)
  --disable_tune        Disable hyperparameters tuning with gridsearch. (default:  
                        False)
```

## References

<a id="1">[1]</a> Xie, Zhouhang, et al. "[Identifying adversarial attacks on text classifiers](https://arxiv.org/abs/2201.08555)". arXiv:2201.08555 (2022).

## Acknowledgments

The project relies mainly on the following repositories:
- https://github.com/REACT-NLP/tcab_generation
- https://github.com/REACT-NLP/tcab_benchmark
