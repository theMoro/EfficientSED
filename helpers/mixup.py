import numpy as np
import torch


def get_mixup_coefficients(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    return rn_indices, lam


def apply_mixup(x, perm, alpha):
    batch_size = x.size(0)
    x = x * alpha.reshape(batch_size, 1, 1, 1) + x[perm] * (
            1. - alpha.reshape(batch_size, 1, 1, 1))
    return x