"""Token units and lexical (BM25) relevance for LayerSieve variants, borrowed from the top-1 team's SpecPrefill proxy
(docs/MOE_VIETTEL_AI_RACE_2026_Technical_Report.pdf, Section 4.5.3): score and drop whole units (lines / sentences /
clauses) instead of single tokens, and protect units that lexically match the question (BM25 over token bigrams).

A unit ends at a boundary token (one containing a newline or ending with . , ; ! ?) and is at most `max_len` tokens long.
Commas count as boundaries because InfiniteBench kv_retrieval puts the whole JSON object on one line.
"""
import torch

_PUNCT = (".", ",", ";", "!", "?")


def boundary_token_ids(tokenizer) -> torch.Tensor:
    """Ids of the vocabulary tokens that end a unit."""
    ids = [i for i in range(len(tokenizer))
           if "\n" in (s := tokenizer.decode([i])) or s.rstrip().endswith(_PUNCT)]
    return torch.tensor(ids, dtype=torch.long)


def unit_ids(input_ids: torch.Tensor, boundary: torch.Tensor, max_len: int = 128) -> torch.Tensor:
    """Unit id (0, 1, ... in prompt order) of every token of a 1-D prompt."""
    T = input_ids.shape[0]
    end = torch.isin(input_ids, boundary).long()
    u = torch.cumsum(end, 0) - end                                         # a boundary token closes its own unit
    ar = torch.arange(T, device=input_ids.device)
    start = torch.full((int(u[-1]) + 1,), T, device=input_ids.device).scatter_reduce(0, u, ar, reduce="amin")
    key = u * (T + 1) + (ar - start[u]) // max_len                         # split units longer than max_len
    return torch.unique(key, return_inverse=True)[1]


def bm25_scores(input_ids: torch.Tensor, unit: torch.Tensor, query_ids: torch.Tensor, k1: float = 1.2,
                b: float = 0.75) -> torch.Tensor:
    """BM25 score of every unit against the query. Terms are token bigrams (a bigram belongs to the unit of its first
    token): single tokens are too common, e.g. Qwen splits a uuid into digits and short hex pieces. Returns [n_units]."""
    U = int(unit.max()) + 1
    V = int(max(input_ids.max(), query_ids.max())) + 1
    doc = input_ids[:-1] * V + input_ids[1:]
    terms = torch.unique(query_ids[:-1] * V + query_ids[1:])
    length = torch.zeros(U, device=unit.device).index_add_(0, unit, torch.ones_like(unit, dtype=torch.float))
    tf = torch.zeros(len(terms), U, device=unit.device).index_add_(1, unit[:-1], (doc[None] == terms[:, None]).float())
    df = (tf > 0).sum(1, keepdim=True)
    idf = torch.log((U - df + 0.5) / (df + 0.5) + 1)
    return (idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / length.mean()))).sum(0)


def bm25_protect(unit: torch.Tensor, scores: torch.Tensor, budget: int, exclude: torch.Tensor) -> torch.Tensor:
    """Token mask of the best-scoring units (score > 0, not touching `exclude` tokens) totalling <= budget tokens."""
    U = scores.shape[0]
    s = scores.clone()
    s[unit[exclude]] = 0
    size = torch.bincount(unit, minlength=U)
    order = torch.argsort(s, descending=True, stable=True)
    pick = order[(torch.cumsum(size[order], 0) <= budget) & (s[order] > 0)]
    chosen = torch.zeros(U, dtype=torch.bool, device=unit.device)
    chosen[pick] = True
    return chosen[unit]


def drop_units(imp: torch.Tensor, unit: torch.Tensor, prunable: torch.Tensor, n_drop: int) -> torch.Tensor:
    """Indices of the tokens of the lowest-scoring whole units (unit score = max token importance): units are dropped in
    score order until at least n_drop tokens are gone (the unit that crosses n_drop is dropped too), so whole-unit
    dropping never keeps more tokens than token-level dropping with the same n_drop. A unit containing any non-prunable
    token is never dropped."""
    uid, inv = torch.unique(unit, return_inverse=True)
    U = uid.shape[0]
    score = torch.full((U,), float("-inf"), device=imp.device).scatter_reduce(0, inv, imp, reduce="amax")
    blocked = torch.zeros(U, dtype=torch.bool, device=imp.device)
    blocked[inv[~prunable]] = True
    score[blocked] = float("inf")
    size = torch.bincount(inv, minlength=U)
    order = torch.argsort(score, stable=True)
    drop = order[(torch.cumsum(size[order], 0) - size[order] < n_drop) & ~blocked[order]]
    chosen = torch.zeros(U, dtype=torch.bool, device=imp.device)
    chosen[drop] = True
    return torch.nonzero(chosen[inv]).squeeze(1)
