import numpy as np
import scipy
import torch


def _perplexity(logits, labels, pad_token=3):
    for i in range(len(labels)-1, -1, -1):
        if labels[i] != pad_token:
            last_not_pad_id = i
            break
    logits = logits[:last_not_pad_id + 1]
    labels = labels[:last_not_pad_id + 1]
    log_probas = scipy.special.log_softmax(logits, axis=1).astype(np.float32)
    log_probas = [log_probas[i][labels[i]] for i in range(len(labels))]
    l = np.mean(log_probas)
    return 2 ** (-l)


def perplexity(logits, labels, pad_token=3):
    pp = []
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    for cur_logits, cur_labels in zip(logits, labels):
        pp.append(_perplexity(np.array(cur_logits), np.array(cur_labels).astype(int), pad_token))
    return np.mean(pp)
