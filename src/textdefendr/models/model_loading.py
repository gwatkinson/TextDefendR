from textdefendr.models.target_models import AutoClassifier


def _get_model(
    model_name,
    pretrained_model_name_or_path,
    num_labels,
    max_seq_len,
    device,
):
    """
    Return a new instance of the text classification model.
    """
    if model_name in ["bert", "roberta", "distilcamembert"]:
        model = AutoClassifier(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_labels=num_labels,
            device=device,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"Unknown model {model_name}!")

    return model


def load_target_model(
    model_name,
    pretrained_model_name_or_path,
    num_labels,
    max_seq_len,
    device,
):
    """
    Load trained model weights and sets model to evaluation mode.
    """
    model = _get_model(
        model_name=model_name,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        num_labels=num_labels,
        max_seq_len=max_seq_len,
        device=device,
    )
    model = model.to(device)
    model.eval()

    return model
