import numpy as np
import torch


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def optional_tensor_to_numpy(x: torch.Tensor | None) -> np.ndarray | None:
    if x is None:
        return None
    return x.detach().cpu().numpy()