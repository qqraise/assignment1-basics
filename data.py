import numpy as np
import numpy.typing as npt
import torch
from typing import Tuple

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    max_start = n - context_length
    starts = np.random.randint(0, max_start, size=(batch_size,))
    idx = starts[:, None] + np.arange(context_length)[None, :]
    x_np = dataset[idx]
    y_np = dataset[idx + 1]
    x = torch.from_numpy(x_np).long().to(device)
    y = torch.from_numpy(y_np).long().to(device)
    return x, y
