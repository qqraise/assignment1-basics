import torch
from jaxtyping import Bool, Float, Int
from collections.abc import Iterable, Callable

from einops import rearrange
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from math import sqrt, pi, cos

def cross_entropy(inputs: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log( torch.exp(shifted).sum(dim=-1, keepdim=True) )
    loss = -torch.gather(shifted - log_sum_exp, dim=-1, index=targets.unsqueeze(-1)).mean()
    return loss

def gradient_clipping(params: Iterable[Tensor], max_norm: float) -> None:
    """
    Clip the gradient norm of the parameters.
    """
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

def get_lr_cosine_schedule( 
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> Float:
    """
    Get the learning rate at iteration it.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + cos(progress * pi))


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)

    def step(self, closure=None):
        """
        Perform one step of optimization.
        m: first moment
        v: second moment
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state["step"] += 1
                step = state["step"]
                m = state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                v = state["v"] = beta2 * state["v"] + (1 - beta2) * (grad * grad)
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                p.data.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss
