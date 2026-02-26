import os
from typing import BinaryIO, IO, Union
import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: Union[str, os.PathLike, BinaryIO, IO[bytes]]) -> None:
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src, map_location="cpu")
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return int(obj["iteration"])
