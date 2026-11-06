"""
Datasets: synthetic tasks (copy, adding) and text corpora (Tiny Shakespeare, WikiText-2).
Every task exposes get_batch(split, batch_size, generator) -> (inputs, targets).
"""

import os
import urllib.request

import torch

from .tokenizer import CharTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
IGNORE = -100   # targets with this value are skipped by cross_entropy


# ---------------------------------------------------------------------------
# Text corpora
# ---------------------------------------------------------------------------
def load_tiny_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_wikitext2():
    """
    Expects the raw WikiText-2 files in data/cache/wikitext-2-raw/
    (wiki.train.raw, wiki.valid.raw). Download them from the official source
    or the Hugging Face dataset 'Salesforce/wikitext' (wikitext-2-raw-v1).
    """
    d = os.path.join(DATA_DIR, "wikitext-2-raw")
    if not os.path.exists(os.path.join(d, "wiki.train.raw")):
        _download_wikitext2(d)
    try:
        with open(os.path.join(d, "wiki.train.raw"), encoding="utf-8") as f:
            train = f.read()
        with open(os.path.join(d, "wiki.valid.raw"), encoding="utf-8") as f:
            val = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"WikiText-2 not found in {d}. See load_wikitext2 docstring.") from e
    return train, val


WIKITEXT_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"


def _download_wikitext2(d):
    """Fetch the raw WikiText-2 parquet files from Hugging Face (needs pandas + pyarrow)."""
    import pandas as pd
    os.makedirs(d, exist_ok=True)
    for split, fname in (("train", "wiki.train.raw"), ("validation", "wiki.valid.raw")):
        pq = os.path.join(d, f"{split}.parquet")
        urllib.request.urlretrieve(WIKITEXT_URL.format(split=split), pq)
        with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
            f.write("".join(pd.read_parquet(pq)["text"].tolist()))


class TextTask:
    """Next-character prediction on random windows of a corpus."""

    def __init__(self, name="shakespeare", seq_len=256, val_fraction=0.1):
        if name == "shakespeare":
            text = load_tiny_shakespeare()
            self.tokenizer = CharTokenizer.from_text(text)
            ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            n = int(len(ids) * (1 - val_fraction))
            self.data = {"train": ids[:n], "val": ids[n:]}
        elif name == "wikitext2":
            train, val = load_wikitext2()
            self.tokenizer = CharTokenizer.from_text(train + val)
            self.data = {
                "train": torch.tensor(self.tokenizer.encode(train), dtype=torch.long),
                "val": torch.tensor(self.tokenizer.encode(val), dtype=torch.long),
            }
        else:
            raise ValueError(name)
        self.seq_len = seq_len
        self.vocab_size = self.tokenizer.vocab_size

    def get_batch(self, split, batch_size, generator=None):
        data = self.data[split]
        ix = torch.randint(len(data) - self.seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i: i + self.seq_len] for i in ix])
        y = torch.stack([data[i + 1: i + 1 + self.seq_len] for i in ix])
        return x, y


# ---------------------------------------------------------------------------
# Synthetic tasks
# ---------------------------------------------------------------------------
class CopyTask:
    """
    Memorize-then-recall:
        input:  a1 a2 ... aK  [SEP]  _  _ ... _
        target: (ignored)            a1 a2 ... aK
    Token 0 = blank, 1 = SEP, 2.. = data symbols.
    The model must store K tokens in its state and replay them in order.
    """

    def __init__(self, n_copy=16, n_symbols=8):
        self.n_copy = n_copy
        self.vocab_size = n_symbols + 2
        self.seq_len = 2 * n_copy + 1

    def get_batch(self, split, batch_size, generator=None):
        K = self.n_copy
        data = torch.randint(2, self.vocab_size, (batch_size, K), generator=generator)
        blank = torch.zeros(batch_size, K, dtype=torch.long)
        sep = torch.ones(batch_size, 1, dtype=torch.long)
        x = torch.cat([data, sep, blank], dim=1)                   # (b, 2K+1)
        y = torch.full_like(x, IGNORE)
        y[:, K:2 * K] = data                                        # predict a_i at SEP/blank slots
        return x, y


class AddingTask:
    """
    Classic long-range adding problem (Hochreiter & Schmidhuber).
    Each step has 2 features: a random value in [0, 1] and a 0/1 marker.
    Exactly two steps are marked (one in each half). Target: sum of the two marked values.
    Predicting 1.0 every time gives MSE ≈ 0.167; solving the task needs selective memory.
    """

    def __init__(self, seq_len=200):
        self.seq_len = seq_len
        self.in_dim = 2

    def get_batch(self, split, batch_size, generator=None):
        L = self.seq_len
        values = torch.rand(batch_size, L, generator=generator)
        marks = torch.zeros(batch_size, L)
        i1 = torch.randint(0, L // 2, (batch_size,), generator=generator)
        i2 = torch.randint(L // 2, L, (batch_size,), generator=generator)
        rows = torch.arange(batch_size)
        marks[rows, i1] = 1.0
        marks[rows, i2] = 1.0
        x = torch.stack([values, marks], dim=-1)                    # (b, L, 2)
        y = values[rows, i1] + values[rows, i2]                     # (b,)
        return x, y


class MQARTask:
    """
    Multi-query associative recall (as in the Zoology benchmark, Arora et al. 2023).

        k1 v1 k2 v2 ... kK vK   n n n k7 n n k2 n ...
                                      ^ target v7   ^ target v2

    Vocabulary: 0 = pad, keys, values and noise tokens come from separate ranges.
    Every key is queried once, at a random later position, in random order.
    Pure SSMs must squeeze all K pairs into a fixed state; attention can look them up.
    """

    def __init__(self, seq_len=256, n_pairs=16, n_keys=40, n_values=40, n_noise=47, val_seq_len=None):
        self.seq_len, self.K = seq_len, n_pairs
        self.val_seq_len = val_seq_len or seq_len
        self.k0, self.v0, self.n0 = 1, 1 + n_keys, 1 + n_keys + n_values
        self.n_keys, self.n_values, self.n_noise = n_keys, n_values, n_noise
        self.vocab_size = 1 + n_keys + n_values + n_noise
        assert 2 * n_pairs + n_pairs <= seq_len and n_pairs <= n_keys

    def get_batch(self, split, batch_size, generator=None):
        L = self.val_seq_len if split == "val" else self.seq_len
        K, g = self.K, generator
        keys = torch.stack([torch.randperm(self.n_keys, generator=g)[:K] for _ in range(batch_size)]) + self.k0
        vals = torch.randint(0, self.n_values, (batch_size, K), generator=g) + self.v0
        x = torch.randint(0, self.n_noise, (batch_size, L), generator=g) + self.n0
        x[:, 0:2 * K:2], x[:, 1:2 * K:2] = keys, vals
        y = torch.full_like(x, IGNORE)
        for b in range(batch_size):
            pos = torch.randperm(L - 2 * K, generator=g)[:K] + 2 * K
            order = torch.randperm(K, generator=g)
            x[b, pos] = keys[b, order]
            y[b, pos] = vals[b, order]
        return x, y


class NeedleTask:
    """
    One fact hidden in a long stretch of noise, asked for at the very end:

        n n n [MARK] key value n n n ... n n [MARK] key  -> target: value

    Train at seq_len, evaluate at val_seq_len (longer) to test length generalization.
    """

    def __init__(self, seq_len=512, val_seq_len=None, n_keys=16, n_values=16, n_noise=32):
        self.seq_len, self.val_seq_len = seq_len, val_seq_len or seq_len
        self.MARK, self.k0, self.v0, self.n0 = 1, 2, 2 + n_keys, 2 + n_keys + n_values
        self.n_keys, self.n_values, self.n_noise = n_keys, n_values, n_noise
        self.vocab_size = 2 + n_keys + n_values + n_noise

    def get_batch(self, split, batch_size, generator=None):
        L, g = (self.val_seq_len if split == "val" else self.seq_len), generator
        x = torch.randint(0, self.n_noise, (batch_size, L), generator=g) + self.n0
        key = torch.randint(0, self.n_keys, (batch_size,), generator=g) + self.k0
        val = torch.randint(0, self.n_values, (batch_size,), generator=g) + self.v0
        pos = torch.randint(0, L - 6, (batch_size,), generator=g)
        rows = torch.arange(batch_size)
        x[rows, pos], x[rows, pos + 1], x[rows, pos + 2] = self.MARK, key, val
        x[:, -2], x[:, -1] = self.MARK, key
        y = torch.full_like(x, IGNORE)
        y[:, -1] = val
        return x, y


def build_task(name, **kw):
    if name in ("shakespeare", "wikitext2"):
        return TextTask(name, **kw)
    if name == "copy":
        return CopyTask(**kw)
    if name == "adding":
        return AddingTask(**kw)
    if name == "mqar":
        return MQARTask(**kw)
    if name == "needle":
        return NeedleTask(**kw)
    raise ValueError(name)
