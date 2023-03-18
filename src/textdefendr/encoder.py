import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from textdefendr.feature_extraction import FeatureExtractor
from textdefendr.models.model_loading import load_target_model


class TextEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        enable_tp: bool = False,
        enable_lm_perplexity: bool = False,
        enable_lm_proba: bool = False,
        enable_target_model: bool = False,
        tp_model: str = "sentence-transformers/bert-base-nli-mean-tokens",
        lm_perplexity_model: str = "gpt2",
        lm_proba_model: str = "roberta-base",
        target_model: str | None = None,
        device: torch.device | str = "cpu",
    ):
        self.enable_tp = enable_tp
        self.enable_lm_perplexity = enable_lm_perplexity
        self.enable_lm_proba = enable_lm_proba
        self.enable_target_model = enable_target_model
        self.tp_model = tp_model
        self.lm_perplexity_model = lm_perplexity_model
        self.lm_proba_model = lm_proba_model
        self.target_model = target_model
        self.device = device

        self.total_step = (
            self.enable_tp
            + self.enable_lm_perplexity
            + self.enable_lm_proba
            + self.enable_target_model
        )
        if self.total_step == 0:
            raise ValueError("Please choose at least one encoding step.")

    def fit(self, X):
        X = column_or_1d(X, warn=True)

        return self

    def transform(self, X):
        X = column_or_1d(X, warn=True)

        with tqdm(total=self.total_step) as t:
            res = []
            if self.enable_tp:
                tqdm.write(f"Encoding text properties with {self.tp_model}...")
                res.append(self._encode_tp_properties(X))
                t.update()
            if self.enable_lm_perplexity:
                tqdm.write(f"Encoding perplexity with {self.lm_perplexity_model}...")
                res.append(self._encode_lm_perplexity(X))
                t.update()
            if self.enable_lm_proba:
                tqdm.write(f"Encoding proba and rank with {self.lm_proba_model}...")
                res.append(self._encode_lm_proba(X))
                t.update()
            if self.enable_target_model:
                tqdm.write(
                    f"Encoding target model properties with {self.target_model}..."
                )
                res.append(self._encode_tm_properties(X))
                t.update()

        return np.hstack(res)

    def _encode_tp_properties(self, X):
        fe = FeatureExtractor(add_tags=["tp"])
        lm_bert_model = AutoModel.from_pretrained(self.tp_model).to(self.device)
        lm_bert_tokenizer = AutoTokenizer.from_pretrained(self.tp_model)

        features, feature_names = fe(
            return_dict=False,
            text_list=pd.Series(X),
            lm_bert_model=lm_bert_model,
            lm_bert_tokenizer=lm_bert_tokenizer,
            device=self.device,
        )

        return features

    def _encode_lm_perplexity(self, X):
        fe = FeatureExtractor(add_specific=["lm_perplexity"])
        lm_causal_model_gpt = GPT2LMHeadModel.from_pretrained(
            self.lm_perplexity_model
        ).to(self.device)
        lm_causal_tokenizer_gpt = GPT2TokenizerFast.from_pretrained(
            self.lm_perplexity_model
        )

        features, feature_names = fe(
            return_dict=False,
            text_list=pd.Series(X),
            lm_causal_model=lm_causal_model_gpt,
            lm_causal_tokenizer=lm_causal_tokenizer_gpt,
            device=self.device,
        )

        return features

    def _encode_lm_proba(self, X):
        fe = FeatureExtractor(add_specific=["lm_proba_and_rank"])

        lm_masked_model_roberta = AutoModelForMaskedLM.from_pretrained(
            self.lm_proba_model, return_dict=False
        ).to(self.device)
        lm_masked_tokenizer_roberta = AutoTokenizer.from_pretrained(self.lm_proba_model)

        features, feature_names = fe(
            return_dict=False,
            text_list=pd.Series(X),
            lm_masked_model=lm_masked_model_roberta,
            lm_masked_tokenizer=lm_masked_tokenizer_roberta,
            device=self.device,
        )

        return features

    def _encode_tm_properties(self, X):
        raise NotImplementedError()
        fe = FeatureExtractor(add_tags=["tm"])

        predictions = ...
        num_labels = ...

        target_model = load_target_model(
            model_name="bert",
            pretrained_model_name_or_path=self.target_model,
            num_labels=num_labels,
            max_seq_len=None,
            device=self.device,
        )

        regions = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0), (0.0, 1.0)]

        features, feature_names = fe(
            return_dict=False,
            text_list=pd.Series(X),
            labels=pd.Series(predictions),
            target_model=target_model,
            device=self.device,
            regions=regions,
        )

        return features
