# NLP adversarial attacks <!-- omit from toc -->

[![Code quality](https://github.com/baptiste-pasquier/nlp-adversarial-attacks/actions/workflows/quality.yml/badge.svg)](https://github.com/baptiste-pasquier/nlp-adversarial-attacks/actions/workflows/quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [2.1. TCAB Dataset Generation](#21-tcab-dataset-generation)
    - [2.1.1. Download the Allociné dataset](#211-download-the-allociné-dataset)
    - [2.1.2. Fine-tune the model (optional)](#212-fine-tune-the-model-optional)
    - [2.1.3. Test `textattack` (optional)](#213-test-textattack-optional)
    - [2.1.4. Generate the TCAB dataset](#214-generate-the-tcab-dataset)


## 1. Installation

1. Clone the repository
```bash
git clone https://github.com/baptiste-pasquier/nlp-adversarial-attacks
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

3. Install pre-commit
```bash
pre-commit install
```

4. (Optional) Install Pytorch CUDA
```bash
poe torch_cuda
```

## 2. Usage

### 2.1. TCAB Dataset Generation

This section provides
a database of attacks with a fine-tuned DistilCamemBERT model on the task of Allociné reviews classification.

:globe_with_meridians: Reference: https://github.com/react-nlp/tcab_generation

#### 2.1.1. Download the Allociné dataset

Run
```{bash}
python scripts/download_data.py
```
This generates a `train.csv`, `val.csv`, and `test.csv`.

#### 2.1.2. Fine-tune the model (optional)

Run
```{bash}
textattack train --model cmarkea/distilcamembert-base --dataset allocine --num-epochs 3 --learning-rate 5e-5 --per-device-train-batch-size 64 --log-to-tb
```
The fine-tuned model is available on HuggingFace: https://huggingface.co/baptiste-pasquier/distilcamembert-allocine


Evaluate the fine-tuned model:
```{bash}
textattack eval --model-from-huggingface baptiste-pasquier/distilcamembert-allocine --dataset-from-huggingface allocine --num-examples 1000 --dataset-split test
```
The model offers an accuracy score of 97%.

#### 2.1.3. Test `textattack` (optional)

Run
```{bash}
textattack attack --model-from-huggingface baptiste-pasquier/distilcamembert-allocine --dataset-from-huggingface allocine --recipe deepwordbug --num-examples 50
```

#### 2.1.4. Generate the TCAB dataset

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

:bulb: Notebook attack statistics: [run_attack.ipynb](/notebooks/attack_statistics.ipynb)
