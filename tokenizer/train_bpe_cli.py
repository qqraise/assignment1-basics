import os
import json
import sys
import time
import argparse
from tokenizer.train_bpe import train_bpe

def _print_progress(prefix, done, total):
    if total <= 0:
        p = 0.0
    else:
        p = done * 100.0 / total
    sys.stdout.write(f"\r{prefix} {done}/{total} ({p:.1f}%)")
    sys.stdout.flush()

def _extract_sample(src, dst, sample_bytes):
    size = os.path.getsize(src)
    target = min(sample_bytes, size)
    written = 0
    last = 0
    t0 = time.time()
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            b = len(line.encode("utf-8"))
            fout.write(line)
            written += b
            if written - last >= max(1, target // 100):
                last = written
                _print_progress("Sampling", min(written, target), target)
            if written >= target:
                break
    sys.stdout.write("\n")
    sys.stdout.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--sample_bytes", type=int, default=20_000_000)
    ap.add_argument("--prefix", type=str, default="bpe")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sample_path = os.path.join(args.outdir, f"{args.prefix}_sample.txt")
    if not os.path.exists(sample_path):
        _extract_sample(args.input, sample_path, args.sample_bytes)
    def cb(done, total):
        _print_progress("Training", done, total)
    vocab, merges = train_bpe(sample_path, args.vocab_size, args.special_tokens, progress=cb)
    sys.stdout.write("\n")
    sys.stdout.flush()
    vocab_out = {str(k): v.decode("latin-1") for k, v in vocab.items()}
    merges_out = [[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges]
    vocab_file = os.path.join(args.outdir, f"{args.prefix}-v{args.vocab_size}-vocab.json")
    merges_file = os.path.join(args.outdir, f"{args.prefix}-v{args.vocab_size}-merges.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=False)
    with open(merges_file, "w", encoding="utf-8") as f:
        json.dump(merges_out, f, ensure_ascii=False)
    print("Saved:", vocab_file)
    print("Saved:", merges_file)

if __name__ == "__main__":
    main()
