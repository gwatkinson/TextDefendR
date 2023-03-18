import inspect

import pandas as pd

from textdefendr.feature_extraction.feature_extractor import FeatureExtractor
from textdefendr.models import load_target_model


def get_feature_dim_names(feature_fcn_name):
    stub_kwargs = {
        "return_dict": True,
        "text_list": pd.Series(["hello world"]),
        "lm_bert_model": None,
        "lm_bert_tokenizer": None,
        "lm_causal_model": None,
        "lm_causal_tokenizer": None,
        "lm_masked_model": None,
        "lm_masked_tokenizer": None,
        "labels": pd.Series([0]),
        "device": "cpu",
        "get_feature_dim_names": True,
    }
    fe = FeatureExtractor(add_tags=["tm", "lm", "tp"])
    out = []
    corresponding_extractor_fcn = None
    for f in fe.extractors:
        if f.__name__ == feature_fcn_name:
            corresponding_extractor_fcn = f

    if corresponding_extractor_fcn is not None:
        fcn_params = [
            param.name
            for param in inspect.signature(
                corresponding_extractor_fcn
            ).parameters.values()
        ]

        # add target model if necessary:
        if "target_model" in fcn_params:
            target_model = load_target_model(
                model_name="distilcamembert",
                pretrained_model_name_or_path="baptiste-pasquier/distilcamembert-allocine",
                num_labels=2,
                max_seq_len=None,
                device="cpu",
            )
            stub_kwargs["target_model"] = target_model

        # get subset of kwargs specifically for this function
        kwargs_ = {k: stub_kwargs[k] for k in fcn_params if k in stub_kwargs}

        corresponding_extractor_fcn(**kwargs_, feature_list=out)
    else:
        raise ValueError(f"Not found: {feature_fcn_name}")

    return out


if __name__ == "__main__":
    print(get_feature_dim_names("tp_bert"))
