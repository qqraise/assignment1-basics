import os
import time
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from tqdm import tqdm
import tiktoken
from model import TransformerLM
from optimizer import cross_entropy, gradient_clipping, AdamW, get_lr_cosine_schedule

BASE_DIR = "/home/qiuwenchang.1/work/cs336/assigment1/data"
TRAIN_TXT = os.path.join(BASE_DIR, "TinyStoriesV2-GPT4-train.txt")
VALID_TXT = os.path.join(BASE_DIR, "TinyStoriesV2-GPT4-valid.txt")

class Config:
    project_name = "tinystories-temp"
    batch_size = 16
    context_length = 128
    d_model = 256
    d_ff = 896
    num_layers = 4
    num_heads = 8
    rope_theta = 10000.0
    vocab_size = 50257
    learning_rate = 3e-4
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-8
    weight_decay = 0.01
    gradient_clipping = 1.0
    warmup_steps = 100
    total_steps = 120
    eval_freq = 40
    eval_batch = 3
    log_freq = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

enc = tiktoken.get_encoding("gpt2")

def load_tokens(path: str, max_chars: int) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)
    ids = enc.encode(text)
    return np.array(ids, dtype=np.int64)

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    n = len(dataset)
    if n < context_length + 1:
        raise RuntimeError("dataset too small for requested context_length")
    starts = np.random.randint(0, n - (context_length + 1), size=(batch_size,))
    idx = starts[:, None] + np.arange(context_length)[None, :]
    x = torch.from_numpy(dataset[idx]).long().to(device)
    y = torch.from_numpy(dataset[idx + 1]).long().to(device)
    return x, y

def main():
    train_ids = load_tokens(TRAIN_TXT, max_chars=600_000)
    valid_ids = load_tokens(VALID_TXT, max_chars=200_000)
    C = Config
    model = TransformerLM(
        vocab_size=C.vocab_size,
        seq_len=C.context_length,
        d_model=C.d_model,
        num_layers=C.num_layers,
        num_heads=C.num_heads,
        d_ff=C.d_ff,
        apply_rope=True,
        theta=C.rope_theta,
        device=C.device,
    )
    if C.device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = model.to(C.device)
    optimizer = AdamW(
        model.parameters(),
        lr=C.learning_rate,
        weight_decay=C.weight_decay,
        eps=C.epsilon,
        betas=(C.beta1, C.beta2),
    )
    start = time.time()
    for step in tqdm(range(1, C.total_steps + 1)):
        lr = get_lr_cosine_schedule(
            step, C.learning_rate, C.learning_rate * 0.05, C.warmup_steps, C.total_steps
        )
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ids, C.batch_size, C.context_length, C.device)
        model.train()
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), C.gradient_clipping)
        optimizer.step()
        if step % C.log_freq == 0:
            with torch.no_grad():
                grads = [
                    p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
                ]
                grad_norm = (sum((g.norm().item() ** 2 for g in grads))) ** 0.5
            print(
                f"step={step} loss={loss.item():.4f} lr={lr:.6f} grad_norm={grad_norm:.4f} time={time.time()-start:.1f}s"
            )
        if step % C.eval_freq == 0:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for _ in range(C.eval_batch):
                    vx, vy = get_batch(valid_ids, C.batch_size, C.context_length, C.device)
                    vlogits = model(vx)
                    vloss = cross_entropy(vlogits, vy)
                    eval_losses.append(vloss.item())
            print(f"val@{step}: loss={sum(eval_losses)/len(eval_losses):.4f}")
    print("Training finished. Total time:", time.time() - start)

if __name__ == "__main__":
    main()
