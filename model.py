import torch
from jaxtyping import Bool, Float, Int

from einops import rearrange
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from math import sqrt

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, device=self.device, dtype=self.dtype)
            )
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None
        # 使用截断正态分布初始化参数，均值为0，标准差为1，截断范围为 [-3, 3]
        sigma = sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.weight, std=sigma, a = -3 * sigma, b = 3 * sigma)
        
    def forward(self, x: Float[Tensor, "batch in_features"]) -> Float[Tensor, "batch out_features"]:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding_matrix = nn.Parameter(
            torch.empty(self.vocab_size, self.embedding_dim, device=self.device, dtype=self.dtype)
        )
        # 使用截断正态分布初始化参数，均值为0，标准差为1，截断范围为 [-3, 3]
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=1.0, a = -3.0, b = 3.0)
        
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        return self.embedding_matrix[token_ids]

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(self.d_model, self.d_ff, bias=False, device=self.device, dtype=self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, bias=False, device=self.device, dtype=self.dtype)
        self.w3 = Linear(self.d_model, self.d_ff, bias=False, device=self.device, dtype=self.dtype)
        self.silu = lambda x: x * torch.sigmoid(x)
        
    def forward(self, x: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch d_model"]:
        return self.w2.forward( self.silu(self.w1.forward(x)) * self.w3.forward(x))

def softmax(x: Float[Tensor, "... dk"]) -> Float[Tensor, "... d_k"]:
    max_x = x.max(dim=-1, keepdim=True).values
    shift_x = x - max_x
    exp_x = torch.exp(shift_x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, "... values d_v"]:
    d_k = Q.shape[-1]
    QK = Q @ K.transpose(-2, -1) # [..., queries, keys]
    scores = QK / (d_k ** 0.5) # [..., queries, keys]
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1) # [..., queries, keys]
    y = attn_weights @ V # [..., queries, d_v]
    return y

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, seq_len: int, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        half_dim = d_k // 2
        inv_freq = theta ** (-torch.arange(0, half_dim, device=self.device, dtype=self.dtype) / half_dim)
        idx = torch.arange(seq_len, device=self.device, dtype=self.dtype)
        theta_table = torch.outer(idx, inv_freq)

        self.register_buffer("sin", theta_table.sin(), persistent=False)
        self.register_buffer("cos", theta_table.cos(), persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        cos = self.cos[token_positions].unsqueeze(1) # type: ignore
        sin = self.sin[token_positions].unsqueeze(1) # type: ignore
        x2 = rearrange(x, "... t (d two) -> ... t d two", two=2)
        rot_even = x2[..., :, 0] * cos - x2[..., :, 1] * sin
        rot_odd = x2[..., :, 1] * cos + x2[..., :, 0] * sin
        y2 = torch.stack([rot_even, rot_odd], dim=-1)
        return rearrange(y2, "... t d two -> ... t (d two)")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int, apply_rope: bool = False, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.head_dim = self.d_model // self.num_heads
        if apply_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.head_dim, seq_len=self.seq_len, device=self.device, dtype=self.dtype)
        else:
            self.rope = None
        self.q_proj = Linear(self.d_model, self.d_model, bias=False, device=self.device, dtype=self.dtype)
        self.k_proj = Linear(self.d_model, self.d_model, bias=False, device=self.device, dtype=self.dtype)
        self.v_proj = Linear(self.d_model, self.d_model, bias=False, device=self.device, dtype=self.dtype)
        self.out_proj = Linear(self.d_model, self.d_model, bias=False, device=self.device, dtype=self.dtype)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool)), persistent=False)


    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        actual_seq_len = x.shape[-2]
        Q = rearrange(self.q_proj.forward(x), 
                      "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        K = rearrange(self.k_proj.forward(x), 
                      "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        V = rearrange(self.v_proj.forward(x),
                      "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(actual_seq_len, device=self.device, dtype=torch.long)
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        mask = self.mask[:actual_seq_len, :actual_seq_len] # type: ignore
        y = scaled_dot_product_attention(Q, K, V, mask=mask)
        y = rearrange(y, "b h s d -> b s (h d)")
        y = self.out_proj.forward(y)
        return y
        
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x / rms
        return y * self.weight

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int, d_ff: int, apply_rope: bool = False, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.apply_rope = apply_rope
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.ln1 = RMSNorm(self.d_model, eps=1e-5, device=self.device, dtype=self.dtype)
        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, self.seq_len, self.apply_rope, self.theta, self.device, self.dtype)
        self.ln2 = RMSNorm(self.d_model, eps=1e-5, device=self.device, dtype=self.dtype)
        self.ffn = SwiGLU(self.d_model, d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        y = self.ln1.forward(x)
        a = self.attn.forward(y, token_positions)
        x = x + a
        y2 = self.ln2.forward(x)
        f = self.ffn.forward(y2)
        x = x + f
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, apply_rope: bool = False, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.apply_rope = apply_rope
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.token_embeddings = EmbeddingModule(self.vocab_size, self.d_model, device=self.device, dtype=self.dtype)
        self.layers = nn.ModuleList([TransformerBlock(self.d_model, self.num_heads, self.seq_len, self.d_ff, self.apply_rope, self.theta, self.device, self.dtype) for _ in range(self.num_layers)])
        self.ln_final = RMSNorm(self.d_model, eps=1e-5, device=self.device, dtype=self.dtype)
        self.lm_head = Linear(self.d_model, self.vocab_size, bias=False, device=self.device, dtype=self.dtype)

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        y = self.token_embeddings.forward(x)
        b, s = y.shape[0], y.shape[1]
        pos = torch.arange(s, device=self.device if self.device is not None else y.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        for block in self.layers:
            y = block.forward(y, pos)
        y = self.ln_final.forward(y)
        y = self.lm_head.forward(y)
        return y

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        mapped = {}
        for k, v in state_dict.items():
            nk = k
            if nk == "token_embeddings.weight":
                nk = "token_embeddings.embedding_matrix"
            elif nk.startswith("layers.") and ".attn.output_proj.weight" in nk:
                nk = nk.replace(".attn.output_proj.weight", ".attn.out_proj.weight")
            elif nk == "ln_final.weight":
                nk = "ln_final.weight"
            elif nk == "lm_head.weight":
                nk = "lm_head.weight"
            mapped[nk] = v
        return super().load_state_dict(mapped, strict=strict, assign=assign)
    
