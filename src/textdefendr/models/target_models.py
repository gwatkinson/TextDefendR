"""
Auto classification model.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class AutoClassifier(torch.nn.Module):
    """
    Simple text classification model using a pretrained
    AutoModelSequenceClassifier model to tokenize, embed, and classify the input.
    """

    def __init__(
        self, pretrained_model_name_or_path, num_labels, device, max_seq_len=None
    ):
        super().__init__()

        self.device = device
        self.max_seq_len = max_seq_len

        # load pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, return_dict=True, num_labels=num_labels
        )

        if self.max_seq_len is None:
            self.max_seq_len = (
                512
                if self.tokenizer.model_max_length == int(1e30)
                else self.tokenizer.model_max_length
            )

    def forward(self, text_list):
        """
        Define the forward pass.
        """

        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(self.device)
        return self.classifier(**inputs).logits

    def gradient(self, text, label, loss_fn=None):
        """
        Return gradients for this sample.
        """
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        self.classifier.zero_grad()  # reset gradients
        pred = self.forward(text)  # forward pass
        label = label.to(self.device)
        loss = loss_fn(pred, label)  # compute loss
        loss.backward()  # backward pass
        gradients = [p.grad for p in self.classifier.parameters()]
        return gradients
