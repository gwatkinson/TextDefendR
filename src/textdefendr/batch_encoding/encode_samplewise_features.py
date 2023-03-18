import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from textdefendr.feature_extraction import FeatureExtractor
from textdefendr.models.model_loading import load_target_model
from textdefendr.utils.hashing import get_pk_tuple, hash_pk_tuple
from textdefendr.utils.magic_vars import NUM_LABELS_LOOKUP
from textdefendr.utils.pandas_ops import no_duplicate_index

assert (
    torch.cuda.is_available()
), "encoding features is quite expensive, defenitely use gpus"
CUDA_DEVICE = torch.device("cuda")


# ---------------------------------- MACROS ---------------------------------- #


def get_value_holder(df):
    """
    Given a df, return a dict keyed by index
    """
    out = {}
    for idx in df.index:
        out[idx] = {}
        out[idx]["num_successful_loop"] = 0
        out[idx]["deliverable"] = {}
        pk = get_pk_tuple(df, idx)
        out[idx]["primary_key"] = pk
        out[idx]["unique_id"] = hash_pk_tuple(pk)
    return out


def show_sample_instance(holder, index):
    """
    un-mutating printing util
    """
    out = {
        "num_successful_loop": holder[index]["num_successful_loop"],
        "primary_key": holder[index]["primary_key"],
        "unique_id": holder[index]["unique_id"],
        "deliverable": {},
    }

    for feat_name in holder[index]["deliverable"].keys():
        feat_shape = "arr/list of shape: " + str(
            np.array(holder[index]["deliverable"][feat_name]).shape
        )
        out["deliverable"][feat_name] = feat_shape
    print(out)


# ----------------- HEAVEY LIFTING PORTION FOR ENCODING STUFF ---------------- #


def encode_text_properties(df, holder, bert_model_name, disable_tqdm=False):
    """
    input is df, return a bool array of len(DF)

    Text properties should be fairly pain-free,
        aside from BERT that is a bit slow, there shouldnt be too much overhead/loading bunch of external
        models, etc
    """
    print("preparing text properties encoding")
    assert no_duplicate_index(df)

    # define feature extractor
    fe = FeatureExtractor(add_tags=["tp"])

    # then load the bert model (out-of-box)
    print("--- loading lm")
    print(f"AutoModel: {bert_model_name}")
    lm_bert_model = AutoModel.from_pretrained(bert_model_name).to(CUDA_DEVICE)
    lm_bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    print("--- lm loaded")

    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            res = fe(
                return_dict=True,
                text_list=pd.Series([df.at[idx, "perturbed_text"]]),
                lm_bert_model=lm_bert_model,
                lm_bert_tokenizer=lm_bert_tokenizer,
                device=CUDA_DEVICE,
            )
            for extractor_name in res.keys():
                _, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]["deliverable"][extractor_name] = values
            holder[idx]["num_successful_loop"] += 1
        except Exception as e:
            print("**" * 40)
            try:
                print(df.at[idx, "perturbed_text"], " failed")
            except:  # noqa: B001, E722
                print("cannot event print offending text somehow, prob. decoding err")
            print("reason: ", e)

    del lm_bert_model
    del lm_bert_tokenizer

    return holder


def encode_lm_perplexity(df, holder, lm_causal_model_gpt_name, disable_tqdm=False):
    """
    LM properties envolves loading lm masked model
    """
    print("preparing lm perplexity encoding")
    assert no_duplicate_index(df)

    # define the feature extractor
    fe = FeatureExtractor(add_specific=["lm_perplexity"])

    # then load the language models (out-of-box)
    print("--- loading lm")
    print(f"GPT2LMHeadModel: {lm_causal_model_gpt_name}")
    lm_causal_model_gpt = GPT2LMHeadModel.from_pretrained(lm_causal_model_gpt_name).to(
        CUDA_DEVICE
    )
    lm_causal_tokenizer_gpt = GPT2TokenizerFast.from_pretrained(
        lm_causal_model_gpt_name
    )
    print("--- lm loaded")

    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            res = fe(
                return_dict=True,
                text_list=pd.Series([df.at[idx, "perturbed_text"]]),
                lm_causal_model=lm_causal_model_gpt,
                lm_causal_tokenizer=lm_causal_tokenizer_gpt,
                device=CUDA_DEVICE,
            )
            for extractor_name in res.keys():
                _, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]["deliverable"][extractor_name] = values
            holder[idx]["num_successful_loop"] += 1
        except Exception as e:
            print("**" * 40)
            try:
                print(df.at[idx, "perturbed_text"], " failed")
            except:  # noqa: B001, E722
                print("cannot event print offending text somehow, prob. decoding err")
            print("reaason: ", e)

    del lm_causal_model_gpt

    return holder


def encode_lm_proba(df, holder, lm_masked_model_roberta_name, disable_tqdm=False):
    """
    LM properties envolves loading lm masked model
    """
    print("preparing lm proba encoding")
    assert no_duplicate_index(df)

    # define the feature extractor
    fe = FeatureExtractor(add_specific=["lm_proba_and_rank"])

    # load mlm
    print("--- loading lm")
    print(f"RobertaForMaskedLM: {lm_masked_model_roberta_name}")
    lm_masked_model_roberta = AutoModelForMaskedLM.from_pretrained(
        lm_masked_model_roberta_name, return_dict=True
    ).to(CUDA_DEVICE)
    lm_masked_tokenizer_roberta = AutoTokenizer.from_pretrained(
        lm_masked_model_roberta_name
    )
    print("--- lm loaded")

    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            res = fe(
                return_dict=True,
                text_list=pd.Series([df.at[idx, "perturbed_text"]]),
                lm_masked_model=lm_masked_model_roberta,
                lm_masked_tokenizer=lm_masked_tokenizer_roberta,
                device=CUDA_DEVICE,
            )
            for extractor_name in res.keys():
                _, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]["deliverable"][extractor_name] = values
            holder[idx]["num_successful_loop"] += 1
        except Exception as e:
            print("**" * 40)
            try:
                print(df.at[idx, "perturbed_text"], " failed")
            except:  # noqa: B001, E722
                print("cannot event print offending text somehow, prob. decoding err")
            print("reason: ", e)

    del lm_masked_model_roberta

    return holder


def encode_tm_properties(
    df,
    holder,
    target_model_name,
    pretrained_model_name_or_path,
    disable_tqdm=False,
):
    # ------------ LOAD TM, DETERMINE NUM_LABELS AUTO ------------ #
    print("preparing tm properties encoding")
    assert no_duplicate_index(df)

    assert "target_dataset" in df.columns
    assert df["target_dataset"].nunique() == 1
    target_dataset = df["target_dataset"][0]

    assert "target_model" in df.columns
    assert df["target_model"].nunique() == 1

    # lookup how many labels are there
    num_labels = NUM_LABELS_LOOKUP[target_dataset]

    print("--- loading target model")
    print(
        f"{pretrained_model_name_or_path} ({target_model_name} trained on {target_dataset})"
    )
    target_model = load_target_model(
        model_name=target_model_name,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        num_labels=num_labels,
        max_seq_len=None,
        device=CUDA_DEVICE,
    )
    regions = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0), (0.0, 1.0)]
    print("--- target model loaded")

    ######################################################################

    # define the feature extractor
    fe = FeatureExtractor(add_tags=["tm"])

    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            perturbed_text = df.at[idx, "perturbed_text"]
            perturbed_output = np.argmax(df.at[idx, "perturbed_output"])
            res = fe(
                return_dict=True,
                text_list=pd.Series([perturbed_text]),
                labels=pd.Series([perturbed_output]),
                target_model=target_model,
                device=CUDA_DEVICE,
                regions=regions,
            )
            for extractor_name in res.keys():
                _, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]["deliverable"][extractor_name] = values
            holder[idx]["num_successful_loop"] += 1
        except Exception as e:
            print("**" * 40)
            try:
                print(df.at[idx, "perturbed_text"], " failed")
            except:  # noqa: B001, E722
                print("cannot event print offending text somehow, prob. decoding err")
            print("reason: ", e)

    del target_model

    return holder


def encode_all_properties(
    df,
    tp_model,
    lm_perplexity_model,
    lm_proba_model,
    tm_model,
    tm_model_name_or_path,
    disable_tqdm=False,
    tasks="ALL",
):
    """
    Takes in a df,
    returns a nested dict called holder, as the data object
    """
    tasks = tasks.upper().split(",")
    for task in tasks:
        assert task in ["ALL", "TP", "LM_PROBA", "LM_PERPLEXITY", "TM"]
    assert no_duplicate_index(df)
    holder = get_value_holder(df)

    try:
        if "ALL" in tasks or "TP" in tasks:
            holder = encode_text_properties(
                df, holder, tp_model, disable_tqdm=disable_tqdm
            )
        if "ALL" in tasks or "LM_PERPLEXITY" in tasks:
            holder = encode_lm_perplexity(
                df, holder, lm_perplexity_model, disable_tqdm=disable_tqdm
            )
        if "ALL" in tasks or "LM_PROBA" in tasks:
            holder = encode_lm_proba(
                df, holder, lm_proba_model, disable_tqdm=disable_tqdm
            )
        if "ALL" in tasks or "TM" in tasks:
            holder = encode_tm_properties(
                df, holder, tm_model, tm_model_name_or_path, disable_tqdm=disable_tqdm
            )
    except KeyboardInterrupt as ki:
        print("Keyboard interrupt, saving partial results")
        print(ki)

    print("=" * 40)
    print("--- all done")

    loop_num = 4  # 4 extactor pipes
    keys_to_rm = []
    for h in holder.keys():
        if holder[h]["num_successful_loop"] == loop_num:
            pass
        else:
            keys_to_rm.append(h)

    len_holder = len(holder)
    _failed_extraction_count = 0
    for _failed_extraction_count, k in enumerate(keys_to_rm, start=1):
        del holder[k]

    print("total failed extraction: ", _failed_extraction_count, "out of", len_holder)
    print("a sample holder value for sanity check")
    print()
    print()
    sample_holder_item_key = list(holder.keys())[0]
    show_sample_instance(holder, sample_holder_item_key)
    print()

    print("=" * 40)

    return holder


def encode_only_tp_model_properties(
    df,
    tp_model,
    disable_tqdm=False,
):
    """
    Takes in a df,
    returns a nested dict called holder, as the data object
    """
    assert no_duplicate_index(df)
    holder = get_value_holder(df)

    holder = encode_text_properties(df, holder, tp_model, disable_tqdm=disable_tqdm)
    loop_num = 4  # 4 extactor pipes

    print("=" * 40)
    print("--- all done")

    keys_to_rm = []
    for h in holder.keys():
        if holder[h]["num_successful_loop"] == loop_num:
            pass
        else:
            keys_to_rm.append(h)

    len_holder = len(holder)
    _failed_extraction_count = 0
    for _failed_extraction_count, k in enumerate(keys_to_rm, start=1):
        del holder[k]

    print("total failed extraction: ", _failed_extraction_count, "out of", len_holder)
    print("a sample holder value for sanity check")
    print()
    print()
    sample_holder_item_key = list(holder.keys())[0]
    show_sample_instance(holder, sample_holder_item_key)
    print()

    print("=" * 40)

    return holder
