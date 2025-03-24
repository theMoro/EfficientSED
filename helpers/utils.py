import torch
import numpy as np
import random


def worker_init_fn(x):
    seed = (torch.initial_seed() + x * 1000) % 2 ** 31  # problem with nearly seeded randoms

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

def replace_state_dict_key(state_dict: dict, old: str, new: str):
    """Replaces `old` in all keys of `state_dict` with `new`."""
    keys = list(state_dict.keys())  # take snapshot of the keys
    for key in keys:
        if old in key:
            state_dict[key.replace(old, new)] = state_dict.pop(key)
    return state_dict

