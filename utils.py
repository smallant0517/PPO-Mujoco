import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

@torch.no_grad()
def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    # 1 - Var[y - yhat] / Var[y]
    var_y = torch.var(y_true)
    if var_y.item() < 1e-12:
        return float("nan")
    return (1.0 - torch.var(y_true - y_pred) / var_y).item()