import os
from datetime import datetime
import time
import wandb
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import math
from typing import cast
from tqdm import tqdm
import tiktoken
from config import PretrainedConfig
from data import get_batch
from model import TransformerLM
from serialization import load_checkpoint, save_checkpoint
from optimizer import cross_entropy, gradient_clipping, AdamW, get_lr_cosine_schedule



def train(step, dataset: npt.NDArray, model: nn.Module, opt: torch.optim.Optimizer, config: PretrainedConfig):
    inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
    model.train()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)

    opt.zero_grad()
    loss.backward()

    gradient_clipping(model.parameters(), config.gradient_clipping)
    opt.step()

    return loss.item()

def evaluate(dataset: npt.NDArray, model: nn.Module, config: PretrainedConfig):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config.eval_batch):
            inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def train_model(config: PretrainedConfig):
    run = wandb.init(project=config.project_name, name=datetime.now().strftime("%Y%m%d_%H%M%S"), config=config.__dict__)
    print("wandb.init OK")
    wandb.define_metric("train/*", step_metric="trainer/step")
    wandb.define_metric("val/*", step_metric="trainer/step")
    
    ckpt_dir = config.checkpoint_dir or "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device(config.device)

    #设置PyTorch中乘法精度
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    enc = tiktoken.get_encoding("gpt2")
    def ensure_bin(path: str) -> str:
        if path.endswith(".bin"):
            return path
        base, ext = os.path.splitext(path)
        bin_path = base + ".bin"
        if not os.path.exists(bin_path):
            with open(path, "r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
                while True:
                    chunk = fin.read(1_000_000)
                    if not chunk:
                        break
                    ids = enc.encode(chunk, disallowed_special=())
                    np.array(ids, dtype=np.uint16).tofile(fout)
        return bin_path
    train_path = ensure_bin(config.train_path)
    valid_path = ensure_bin(config.valid_path)
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_path, dtype=np.uint16, mode="r")
    max_id = int(max(int(train_data.max()), int(valid_data.max())))
    if config.vocab_size <= max_id:
        config.vocab_size = max_id + 1
    # 
    # 模型
    model = TransformerLM(
        vocab_size=config.vocab_size,
        seq_len=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        apply_rope=True,
        theta=config.rope_theta,
        device=config.device,
    )

    if config.use_compile:
        print("Compiling model for better performance...")
        model = cast(nn.Module, torch.compile(model))

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.epsilon,
        betas=(config.beta1, config.beta2)
    )
    
    print("train device: ", config.device)
    print("train data size: ", train_data.shape[0], "valid data size: ", valid_data.shape[0])
    total_tokens_processed = config.batch_size * config.context_length * config.total_steps
    print("total tokens processed: ", total_tokens_processed)
    if total_tokens_processed < 327680000:
        print("warning: total_tokens_processed < 327680000, may underfit.")
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 训练循环
    start_time = time.time()
    step = 0
    last_log_time = start_time
    tokens_per_step = config.batch_size * config.context_length
    for step in tqdm(range(1, config.total_steps + 1)):
        # 更新學習率
        lr = get_lr_cosine_schedule(
            step,
            config.learning_rate,
            config.learning_rate*0.05,
            config.warmup_steps,
            config.total_steps
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train
        loss = train(step, train_data, model, optimizer, config)

        if step % config.log_freq == 0:
            grad_norm = math.sqrt(
                sum(
                    (p.grad.data.norm().item() ** 2)
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                )
            )
            now = time.time()
            dt = max(1e-6, now - last_log_time)
            tps = tokens_per_step / dt
            ppl = math.exp(loss) if loss < 50 else float("inf")
            wandb.log({
                'trainer/step': step,
                'train/loss': loss, 
                'train/grad_norm': grad_norm, 
                'train/lr': lr, 
                'train/wallclock_time': now - start_time,
                'train/perplexity': ppl,
                'train/tokens_per_sec': tps,
                'train/samples_per_sec': config.batch_size / dt,
            }, step=step)
            print(f"step = {step}, loss = {loss}, lr = {lr}, grad_norm = {grad_norm}")
            last_log_time = now

        # 验证
        if step % config.eval_freq == 0:
            eval_loss = evaluate(valid_data, model, config)
            eval_ppl = math.exp(eval_loss) if eval_loss < 50 else float("inf")
            wandb.log({
                'trainer/step': step,
                'val/loss': eval_loss,
                'val/wallclock_time': time.time() - start_time,
                'val/perplexity': eval_ppl,
            }, step=step)
            print(f"step = {step}, eval_loss = {eval_loss}")
        
        # 保存checkpoint
        if step % config.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, step, os.path.join(ckpt_dir, f"checkpoint_{step}.pt"))
            print(f"Checkpoint saved to {ckpt_dir}/checkpoint_{step}.pt")

    eval_loss = evaluate(valid_data, model, config)
    wandb.log({
        'trainer/step': step,
        'val/loss': eval_loss,
        'val/wallclock_time': time.time() - start_time,
        'val/perplexity': math.exp(eval_loss) if eval_loss < 50 else float("inf"),
    }, step=step)
    print(f"final evaluation loss: {eval_loss}")
    
    save_checkpoint(model, optimizer, step, os.path.join(ckpt_dir, f"checkpoint_{step}.pt"))
    
    wandb.finish()

if __name__ == "__main__":
    data_dir = "/home/qiuwenchang.1/work/cs336/assigment1/data/"
    config = PretrainedConfig(
        project_name="TinyStoriesV2-GPT4",
        vocab_path = data_dir + "TinyStoriesV2-GPT4-bpe-vocab.json",
        merges_path = data_dir + "TinyStoriesV2-GPT4-bpe-merges.txt",
        special_tokens = ["<|endoftext|>"],
        train_path = data_dir + "TinyStoriesV2-GPT4-train.txt",
        valid_path = data_dir + "TinyStoriesV2-GPT4-valid.txt",
        # batch_size=8,
        # context_length=128,
        # d_model=256,
        # d_ff=896,
        # num_layers=2,
        # num_heads=4,
        # vocab_size=50257,
        # use_compile=False,
        log_freq=1,
        # eval_freq=20,
        # total_steps=60,
    )
    train_model(config)
