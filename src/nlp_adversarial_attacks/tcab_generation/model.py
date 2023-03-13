import torch
from textattack.models.wrappers import HuggingFaceModelWrapper

from nlp_adversarial_attacks.models.target_models import AutoClassifier


class ModelWrapper(HuggingFaceModelWrapper):
    """Wrapper to interface more easily with TextAttack.."""

    def __init__(self, model: AutoClassifier):
        classifier = model.classifier
        tokenizer = model.tokenizer
        max_seq_len = model.max_seq_len

        super().__init__(classifier, tokenizer)
        self.max_seq_len = max_seq_len

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits
