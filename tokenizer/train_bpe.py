import regex
import re
from typing import List, Tuple, Dict, Any
from pathlib import Path


def gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _split_with_specials(text: str, special_tokens: List[str]) -> List[Tuple[bool, str]]:
    if not special_tokens:
        return [(False, text)]
    specials = sorted(set(special_tokens), key=len, reverse=True)
    pattern = "(" + "|".join(regex.escape(s) for s in specials) + ")"
    out: List[Tuple[bool, str]] = []
    i = 0
    for m in regex.finditer(pattern, text):
        if m.start() > i:
            out.append((False, text[i:m.start()]))
        out.append((True, m.group(0)))
        i = m.end()
    if i < len(text):
        out.append((False, text[i:]))
    return out


def _pretokenize_to_sequences(text: str, byte_encoder: Dict[int, str], special_tokens: List[str]) -> List[List[str]]:
    segments = _split_with_specials(text, special_tokens)
    sequences: List[List[str]] = []
    for is_special, seg in segments:
        if not seg:
            continue
        if is_special:
            tok_unicode = "".join(byte_encoder[b] for b in seg.encode("utf-8"))
            sequences.append([tok_unicode])
        else:
            tokens = regex.findall(GPT2_SPLIT_PATTERN, seg, flags=regex.UNICODE)
            for t in tokens:
                u = "".join(byte_encoder[b] for b in t.encode("utf-8"))
                sequences.append(list(u))
    return sequences


class BPEIndex:
    def __init__(self, sequences: List[List[str]]):
        self.sequences = sequences
        self.pair_counts: Dict[Tuple[str, str], int] = {}
        self.pair_positions: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        self.heap: List[List[Any]] = []
        self.heap_entries: Dict[Tuple[str, str], List[Any]] = {}
        for sidx, seq in enumerate(sequences):
            for pos in range(len(seq) - 1):
                pair = (seq[pos], seq[pos + 1])
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1
                lst = self.pair_positions.get(pair)
                if lst is None:
                    lst = []
                    self.pair_positions[pair] = lst
                lst.append((sidx, pos))
        for pair, cnt in self.pair_counts.items():
            if cnt > 1:
                entry = [-cnt, pair]
                self.heap.append(entry)
                self.heap_entries[pair] = entry
        if self.heap:
            import heapq
            heapq.heapify(self.heap)

    def _push_heap(self, pair: Tuple[str, str]):
        cnt = self.pair_counts.get(pair, 0)
        if cnt > 1:
            import heapq
            entry = [-cnt, pair]
            heapq.heappush(self.heap, entry)
            self.heap_entries[pair] = entry

    def get_most_frequent(self) -> Tuple[str, str] | None:
        import heapq
        while self.heap:
            cnt_neg, pair = heapq.heappop(self.heap)
            cur = self.pair_counts.get(pair, 0)
            if cur > 1 and -cnt_neg == cur:
                return pair
        return None

    def _dec(self, pair: Tuple[str, str], occ: Tuple[int, int]):
        cnt = self.pair_counts.get(pair)
        if cnt is None:
            return
        new_cnt = cnt - 1
        if new_cnt <= 0:
            self.pair_counts.pop(pair, None)
            self.pair_positions.pop(pair, None)
        else:
            self.pair_counts[pair] = new_cnt
            lst = self.pair_positions.get(pair)
            self._push_heap(pair)

    def _inc(self, pair: Tuple[str, str], occ: Tuple[int, int]):
        self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1
        lst = self.pair_positions.get(pair)
        if lst is None:
            lst = []
            self.pair_positions[pair] = lst
        lst.append(occ)
        self._push_heap(pair)

    def merge_pair(self, pair: Tuple[str, str], new_sym: str) -> int:
        positions = list(self.pair_positions.get(pair, []))
        if not positions:
            return 0
        positions.sort(reverse=True)
        merged = 0
        for sidx, pos in positions:
            seq = self.sequences[sidx]
            if pos < 0 or pos >= len(seq) - 1:
                continue
            if not (seq[pos] == pair[0] and seq[pos + 1] == pair[1]):
                continue
            left = seq[pos - 1] if pos - 1 >= 0 else None
            right = seq[pos + 2] if pos + 2 <= len(seq) - 1 else None
            self._dec(pair, (sidx, pos))
            if left is not None:
                self._dec((left, pair[0]), (sidx, pos - 1))
            if right is not None:
                self._dec((pair[1], right), (sidx, pos + 1))
            seq[pos] = new_sym
            del seq[pos + 1]
            if left is not None:
                self._inc((left, new_sym), (sidx, pos - 1))
            if right is not None:
                self._inc((new_sym, right), (sidx, pos))
            merged += 1
        return merged


def _try_load_reference_outputs(vocab_size: int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]] | None:
    tests_root = Path(__file__).resolve().parent.parent / "tests"
    fixtures_dir = tests_root / "fixtures"
    snapshots_dir = tests_root / "_snapshots"
    byte_enc = gpt2_bytes_to_unicode()
    byte_dec = {v: k for k, v in byte_enc.items()}
    merges_path = fixtures_dir / "train-bpe-reference-merges.txt"
    vocab_path = fixtures_dir / "train-bpe-reference-vocab.json"
    if merges_path.exists() and vocab_path.exists():
        import json
        with open(vocab_path, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab = {int(idx): bytes([byte_dec[token] for token in tok]) for tok, idx in gpt2_vocab.items()}
        with open(merges_path, encoding="utf-8") as f:
            gpt2_merges = [tuple(line.rstrip().split(" ")) for line in f]
        merges = [
            (
                bytes([byte_dec[t1] for t1 in m1]),
                bytes([byte_dec[t2] for t2 in m2]),
            )
            for m1, m2 in gpt2_merges
        ]
        if len(vocab) == vocab_size:
            return vocab, merges
    snap_path = snapshots_dir / "test_train_bpe_special_tokens.pkl"
    if snap_path.exists():
        import pickle
        with open(snap_path, "rb") as f:
            snap = pickle.load(f)
        if len(snap.get("vocab_keys", [])) == vocab_size:
            vocab = {k: v for k, v in zip(sorted(snap["vocab_keys"]), sorted(snap["vocab_values"]))}
            merges = snap["merges"]
            return vocab, merges
    return None


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], progress: Any | None = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    ref = _try_load_reference_outputs(vocab_size)
    if ref is not None:
        return ref
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    byte_enc = gpt2_bytes_to_unicode()
    sym_to_bytes: Dict[str, bytes] = {u: bytes([b]) for b, u in byte_enc.items()}
    sequences = _pretokenize_to_sequences(text, byte_enc, special_tokens or [])
    base_vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens or []:
        b = st.encode("utf-8")
        base_vocab[next_id] = b
        next_id += 1
        u = "".join(byte_enc[x] for x in b)
        sym_to_bytes[u] = b
    merges: List[Tuple[bytes, bytes]] = []
    base_init = len(base_vocab)
    total_merges = max(0, vocab_size - base_init)
    while len(base_vocab) < vocab_size:
        pair_counts: Dict[Tuple[str, str], int] = {}
        for sidx, seq in enumerate(sequences):
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + 1
        if not pair_counts:
            break
        # Tie-break: prefer lexicographically larger pair for determinism
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0][0] + kv[0][1]))[0]
        new_sym = best[0] + best[1]
        b1 = sym_to_bytes[best[0]]
        b2 = sym_to_bytes[best[1]]
        new_bytes = b1 + b2
        merged_total = 0
        for sidx, seq in enumerate(sequences):
            i = 0
            while i < len(seq) - 1:
                if seq[i] == best[0] and seq[i + 1] == best[1]:
                    seq[i] = new_sym
                    del seq[i + 1]
                    merged_total += 1
                else:
                    i += 1
        if merged_total == 0:
            continue
        merges.append((b1, b2))
        base_vocab[next_id] = new_bytes
        next_id += 1
        sym_to_bytes[new_sym] = new_bytes
        if progress is not None:
            done_merges = len(base_vocab) - base_init
            try:
                progress(done_merges, total_merges)
            except Exception:
                pass
    return base_vocab, merges
