"""Exact char -> token mapping for byte-level BPE ids (decoding tokens one by one drifts on multi-byte characters)."""
import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

_BD = {v: k for k, v in bytes_to_unicode().items()}


def char_index(tok, ids):
    """(text, ends): text = tok.decode(ids); ends[k] = number of CHARS of text covered by tokens 0..k (exact)."""
    raw = []
    for t in tok.convert_ids_to_tokens(ids):
        raw.append(bytes(_BD[c] for c in t) if all(c in _BD for c in t) else t.encode())
    blen = np.cumsum([len(b) for b in raw])
    text = b"".join(raw).decode("utf-8", errors="replace")
    cb = np.cumsum([len(c.encode()) for c in text])                  # bytes covered by chars 0..i
    ends = np.searchsorted(cb, blen, side="left") + 1                # chars fully or partly covered by tokens 0..k
    return text, np.minimum(ends, len(text))


def tok_span(ends, c0, c1):
    """Token range [t0, t1) covering chars [c0, c1)."""
    return int(np.searchsorted(ends, c0, side="right")), int(np.searchsorted(ends, c1, side="left")) + 1
