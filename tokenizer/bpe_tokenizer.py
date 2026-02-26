
import regex
from typing import List, Dict



def gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(i) for i in cs]
    return dict(zip(bs, cs))

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pre_tokenize(doc: str) -> List[str]:
    # 输出：List[str]，每个元素是“字节级 unicode”的单个预分词 token
    byte_encoder = gpt2_bytes_to_unicode()
    tokens = regex.findall(GPT2_SPLIT_PATTERN, doc, flags=regex.UNICODE)
    return ["".join(byte_encoder[b] for b in token.encode("utf-8")) for token in tokens]
    

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]] = [], special_tokens: list[str] | None = []):
        self.vocab_size = len(vocab)
        self.id_to_bytes = vocab
        self.bytes_to_id = {b: i for i, b in vocab.items()}
        self.byte_decoder = {u: b for b, u in gpt2_bytes_to_unicode().items()}
        self.merges = merges
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens
    
    def _split_special(self, text: str) -> List[tuple[bool, str]]:
        if not self.special_tokens:
            return [(False, text)]
        specials = sorted(set(self.special_tokens), key=len, reverse=True)
        pattern = "(" + "|".join(regex.escape(s) for s in specials) + ")"
        segments: List[tuple[bool, str]] = []
        i = 0
        for m in regex.finditer(pattern, text):
            if m.start() > i:
                segments.append((False, text[i:m.start()]))
            segments.append((True, m.group(0)))
            i = m.end()
        if i < len(text):
            segments.append((False, text[i:]))
        return segments
    
    def _token_to_bytes(self, token: str) -> List[bytes]:
        return [bytes([self.byte_decoder[ch]]) for ch in token]
    
    def _apply_merges(self, symbols: List[bytes]) -> List[bytes]:
        if not self.merges_rank or len(symbols) < 2:
            return symbols
        while True:
            best_idx = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merges_rank:
                    r = self.merges_rank[pair]
                    if best_rank is None or r < best_rank:
                        best_idx = i
                        best_rank = r
            if best_idx is None:
                break
            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]
        return symbols
    
    def _symbols_to_ids(self, symbols: List[bytes]) -> List[int]:
        out: List[int] = []
        for sym in symbols:
            if sym in self.bytes_to_id:
                out.append(self.bytes_to_id[sym])
            else:
                for b in sym:
                    out.append(self.bytes_to_id[bytes([b])])
        return out

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for is_special, seg in self._split_special(text):
            if is_special:
                b = seg.encode("utf-8")
                if b in self.bytes_to_id:
                    out.append(self.bytes_to_id[b])
                else:
                    for x in b:
                        out.append(self.bytes_to_id[bytes([x])])
            else:
                for token in pre_tokenize(seg):
                    symbols = self._token_to_bytes(token)
                    symbols = self._apply_merges(symbols)
                    out.extend(self._symbols_to_ids(symbols))
        return out
    
    def decode(self, ids: List[int]) -> str:
        buf = bytearray()
        for i in ids:
            buf += self.id_to_bytes[i]
        return buf.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable):
        for chunk in iterable:
            for _id in self.encode(chunk):
                yield _id
    
    

if __name__ == "__main__":
    import json
    base_dir = "/home/qiuwenchang.1/work/cs336/assigment1/data/"
    with open(base_dir + "TinyStoriesV2-GPT4-bpe-vocab.json", "r") as f:
        vocab_json = json.load(f)
    with open(base_dir + "TinyStoriesV2-GPT4-bpe-merges.json", "r") as f:
        merges_json = json.load(f)
    vocab = {int(k): v.encode("latin-1") for k, v in vocab_json.items()}
    merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in merges_json]
    tokenizer = BPETokenizer(vocab=vocab, merges=merges)
    test_string = "Hello, world!"
    token_ids = tokenizer.encode(test_string)
    print(token_ids)
