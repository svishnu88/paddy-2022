import numpy as np

def get_split(items=None,split=0.2,seed=None):
    np.random.seed(seed)
    np.random.shuffle(items)
    n = len(items)
    split_idx = int(n * (1-split))
    return items[:split_idx],items[split_idx:]

