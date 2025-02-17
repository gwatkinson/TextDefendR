import time

import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, SequentialSampler

from .model_wrappers import NodeActivationWrapper, SaliencyWrapper
from .utils import apply_tags


@apply_tags(["tm", "classification"])
def tm_posterior(
    text_list, target_model, device, batch_size=32, logger=None, feature_list=None
):
    """
    Input: Pandas.Series of strings;
           Torch.nn model;
           Device ('cpu' or 'gpu').
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             output posteriors (softmax applied to logits).
    """
    # create dataloader
    data = text_list.tolist()
    dataloader = DataLoader(
        data, sampler=SequentialSampler(data), batch_size=batch_size
    )

    # activate evaluation mode
    target_model.eval()

    # genrate predictions on the test set
    all_preds = []
    for step, batch_text_list in enumerate(dataloader):
        if logger:
            s = "[TM: POSTERIOR] batch {:,} no. samples: {:,}"
            logger.info(s.format(step + 1, batch_size * (step + 1)))

        # make predictions for this batch
        with torch.no_grad():
            preds = target_model(batch_text_list)
            all_preds.append(preds.cpu().numpy().tolist())

    # concat all predictions
    all_preds = np.vstack(all_preds)
    y_proba = softmax(all_preds, axis=1)

    if feature_list is not None:
        for i in range(y_proba.shape[1]):
            feature_list.append(f"tm_output_{i}")

    return y_proba


@apply_tags(["tm", "classification"])
def tm_gradient(
    text_list,
    labels,
    target_model,
    device="cpu",
    logger=None,
    regions=None,
    quantiles=None,
    feature_list=None,
):
    """
    Input: Pandas.Series of strings;
           Pandas.Series of integer labels;
           Torch.nn model;
           Device ('cpu' or 'gpu').
    Returns: 2D Numpy.Array of shape=(no. samples, (1 + 1 + no. quantiles) * no. regions * no. layers))
             gradient statistics (mean, var, quantiles) for different
             input regions for each layer of gradients.
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    if regions is None:
        regions = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0), (0.0, 1.0)]

    # timer
    start = time.time()
    # move quantiles list to device
    quantiles = torch.tensor(quantiles, dtype=torch.float32).to(device)
    # create dataloader
    data = list(zip(text_list.tolist(), labels.values))
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1)
    # activate evaluation mode
    target_model.eval()
    # compute loss for each sample
    feature_vec_list = []
    for i, (text, label) in enumerate(dataloader):
        feature_vec = []
        # get no. words in this sample
        text = text[0].split()
        num_words = len(text)
        # compute features for each input region
        for k, (start_frac, end_frac) in enumerate(regions):
            # get input region text
            start_ndx = int(num_words * start_frac)
            end_ndx = int(num_words * end_frac)
            region_text = " ".join(text[start_ndx:end_ndx])
            # compute gradients
            gradients = target_model.gradient(region_text, label)
            # process gradients layer by layer
            for j, gradient in enumerate(gradients):
                # get number of parameters for this layer
                gradient = gradient.flatten().detach()
                # handle small layers
                if gradient.shape[0] < len(quantiles):
                    feature_vec.append(torch.mean(gradient).item())
                    feature_vec.append(torch.var(gradient).item())
                    # add feature names
                    if i == 0 and feature_list is not None:
                        feature_list.append(f"tm_gradient_mean_layer{j}_region_{k}")
                        feature_list.append(f"tm_gradient_var_layer{j}_region_{k}")
                # compute quantiles of bigger layers
                else:
                    # add statistical features for this region
                    feature_vec.append(torch.mean(gradient).item())
                    feature_vec.append(torch.var(gradient).item())
                    # ERROR: input tensor to quantile() is sometimes too large
                    try:
                        no_error = True
                        feature_vec += list(torch.quantile(gradient, quantiles).cpu())
                    # no CUDA memory
                    except RuntimeError:
                        no_error = False
                    # add feature names
                    if i == 0 and feature_list is not None:
                        feature_list.append(f"tm_gradient_mean_layer{j}_region{k}")
                        feature_list.append(f"tm_gradient_var_layer{j}_region{k}")
                        if no_error:
                            feature_list += [
                                f"tm_gradient_quant{m}_layer{j}_region{k}"
                                for m in range(len(quantiles))
                            ]
        # add feature vector to the list of feature vectors
        feature_vec_list.append(feature_vec)
        # progress update
        if logger and (i + 1) % 100 == 0:
            s = "[TM: GRADIENT] no. samples: {:,}...{:.3f}s"
            logger.info(s.format(i + 1, time.time() - start))
    return np.vstack(feature_vec_list)


@apply_tags(["tm"])
def tm_activation(
    text_list,
    target_model,
    device,
    logger=None,
    regions=None,
    quantiles=None,
    feature_list=None,
):
    """
    Input: Pandas.Series of strings;
           Torch.nn model;
           Device ('cpu' or 'gpu').
    Returns: 2D Numpy.Array of shape=(no. samples, (1 + 1 + no. quantiles) * no. regions * no. layers)),
             activation statistics (mean, var, quantiles) for different
               input regions for each layer of nodes.
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    if regions is None:
        regions = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0), (0.0, 1.0)]

    # timer
    start = time.time()

    # move quantiles list to device
    # quantiles = torch.tensor(quantiles, dtype=torch.float32).to(device)

    # create dataloader
    data = text_list.tolist()
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1)

    # get max. no. tokens for the target model
    if hasattr(target_model, "max_seq_len"):
        max_length = target_model.max_seq_len
    else:
        max_length = 10000

    # activate evaluation mode
    target_model.eval()

    # wrap model for ability to add hooks to get node activations
    target_model = NodeActivationWrapper(target_model)

    # compute loss for each sample
    feature_vec_list = []
    for i, text in enumerate(dataloader):
        feature_vec = []

        # get no. words in this sample
        text = text[0].split()[:max_length]
        num_words = len(text)

        # compute features for each input region
        for j, (start_frac, end_frac) in enumerate(regions):
            # get input region text
            start_ndx = int(num_words * start_frac)
            end_ndx = int(num_words * end_frac)
            region_text = " ".join(text[start_ndx:end_ndx])

            # reset gradients and extract activations
            activations = target_model.activation(region_text)

            # process activations layer by layer
            for k, activation in enumerate(activations):
                # flatten token / node activation matrix
                activation = activation.flatten().detach().cpu().numpy()

                # handle small layers
                if activation.shape[0] < len(quantiles):
                    feature_vec.append(np.mean(activation))
                    feature_vec.append(np.var(activation))

                    # add feature names
                    if i == 0 and feature_list is not None:
                        feature_list.append(f"tm_activation_mean_layer{k}_region{j}")
                        feature_list.append(f"tm_activation_var_layer{k}")

                # compute quantiles for bigger layers
                else:
                    # add statistical features for this activation layer
                    feature_vec.append(np.mean(activation))
                    feature_vec.append(np.var(activation))
                    feature_vec += list(np.quantile(activation, quantiles))

                    # add feature names
                    if i == 0 and feature_list is not None:
                        feature_list.append(f"tm_activation_mean_layer{k}_region{j}")
                        feature_list.append(f"tm_activation_var_layer{k}_region{j}")
                        feature_list += [
                            f"tm_activation_quant{m}_layer{k}_region{j}"
                            for m in range(len(quantiles))
                        ]

        # add feature vector to the list of feature vectors
        feature_vec_list.append(feature_vec)

        # progress update
        if logger and i % 100 == 0:
            s = "[TM: ACTIVATION] no. samples: {:,}...{:.3f}s"
            logger.info(s.format(i + 1, time.time() - start))

    return np.vstack(feature_vec_list)


@apply_tags(["tm"])
def tm_saliency(
    text_list,
    labels,
    target_model,
    device,
    saliency_type="simple_gradient",
    logger=None,
    regions=None,
    quantiles=None,
    feature_list=None,
):
    """
    Input: Pandas.Series of strings;
           Pandas.Series of integer labels;
           Torch.nn model;
           Device ('cpu' or 'gpu').
    Returns: 2D Numpy.Array of shape=(no. samples, (1 + 1 + no. quantiles) * no. regions * no. layers)),
             activation statistics (mean, var, quantiles) for different
               input regions for each layer of nodes.
    Reference: https://github.com/allenai/allennlp/blob/master/allennlp/\
               interpret/saliency_interpreters/simple_gradient.py
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    if regions is None:
        regions = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0), (0.0, 1.0)]

    # timer
    start = time.time()

    # move quantiles list to device
    quantiles = torch.tensor(quantiles, dtype=torch.float32).to(device)

    # create dataloader
    data = list(zip(text_list.tolist(), labels.values))
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1)

    # activate evaluation mode
    target_model.eval()

    # wrap model with hooks to obtain embedding layer outputs and gradients
    target_model = SaliencyWrapper(target_model, saliency_type=saliency_type)

    # compute loss for each sample
    feature_vec_list = []
    for i, (text, label) in enumerate(dataloader):
        feature_vec = []

        # extract embedding layer outputs and gradients
        saliency = target_model.saliency(text, label).to(device)

        # aggregate token attributions to create a fixed-length feature vector
        feature_vec.append(torch.mean(saliency).item())
        feature_vec.append(torch.var(saliency).item())
        feature_vec += list(torch.quantile(saliency, quantiles).cpu().tolist())

        # add feature names
        if i == 0 and feature_list is not None:
            feature_list.append(f"tm_{saliency_type}_saliency_mean")
            feature_list.append(f"tm_{saliency_type}_saliency_var")
            feature_list += [
                f"tm_{saliency_type}_saliency_quant{j}" for j in range(len(quantiles))
            ]

        # add feature vector to the list of feature vectors
        feature_vec_list.append(feature_vec)

        # progress update
        if logger and i % 100 == 0:
            s = "[TM: SALIENCY] no. samples: {:,}...{:.3f}s"
            logger.info(s.format(i + 1, time.time() - start))

    return np.vstack(feature_vec_list)
