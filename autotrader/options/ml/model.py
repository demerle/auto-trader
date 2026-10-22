"""gold_scar's direction model: walk-forward-validated ridge + small causal transformers, in numpy.

The weights ship with the code (`gold_scar.npz`, trained by the research scripts on Alpaca 5-minute bars,
2016-2026). Inference needs only numpy. The model predicts each target ETF's return from the next bar to the
15:20 flatten, in units of the expected move; `predict` returns z-scores (out-of-sample prediction std = 1).
"""

from collections.abc import Mapping
from functools import lru_cache

import numpy as np
import pandas as pd

from autotrader.options.ml import features as F


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def transformer(W: Mapping[str, np.ndarray], p: str, x: np.ndarray, s: np.ndarray, layers: int,
                heads: int) -> np.ndarray:
    """Forward pass of the trained torch model (pre-norm TransformerEncoder, causal) on one window.
    x: steps x BAR_DIM, s: STATIC_DIM -> steps x n_targets."""
    T = len(x)
    h = np.concatenate([x, np.broadcast_to(s, (T, len(s)))], 1) @ W[p + "inp.weight"].T + W[p + "inp.bias"]
    h = h + W[p + "pos"][:T]
    d = h.shape[1]
    dh = d // heads
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(layers):
        q = f"{p}enc.layers.{i}."
        y = _ln(h, W[q + "norm1.weight"], W[q + "norm1.bias"])
        qkv = y @ W[q + "self_attn.in_proj_weight"].T + W[q + "self_attn.in_proj_bias"]
        qh, kh, vh = (qkv[:, j * d:(j + 1) * d].reshape(T, heads, dh).transpose(1, 0, 2) for j in range(3))
        att = qh @ kh.transpose(0, 2, 1) / np.sqrt(dh) + mask
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (att @ vh).transpose(1, 0, 2).reshape(T, d)
        h = h + o @ W[q + "self_attn.out_proj.weight"].T + W[q + "self_attn.out_proj.bias"]
        y = _ln(h, W[q + "norm2.weight"], W[q + "norm2.bias"])
        ff = np.maximum(y @ W[q + "linear1.weight"].T + W[q + "linear1.bias"], 0)
        h = h + ff @ W[q + "linear2.weight"].T + W[q + "linear2.bias"]
    return h @ W[p + "out.weight"].T + W[p + "out.bias"]


@lru_cache(maxsize=1)
def weights() -> dict[str, np.ndarray]:
    """The packaged weights (fixed path: strategies may import this module, so it exposes no file access)."""
    from pathlib import Path

    with np.load(Path(__file__).with_name("gold_scar.npz")) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def predict(bars: Mapping[str, pd.DataFrame], vix: float | None, k: int, today: pd.Timestamp) -> dict[str, float]:
    """z-score per target ETF as of the close of bar k of `today` (an ET date). Needs today's bars and a few
    sessions before: 5-minute bars of features.SYMBOLS (missing cross symbols count as flat) and of ^VIX (its
    prior close; the live `vix` is the fallback). {} if the targets lack today or yesterday."""
    W = weights()
    spy = bars.get("SPY")
    if spy is None or spy.empty:
        return {}
    days, _ = F.sessions(spy)
    today = pd.Timestamp(today).normalize()
    if len(days) < 2 or days[-1] != today:
        return {}
    A = {s: F.sessions(bars.get(s), days)[1] for s in F.SYMBOLS}
    if any(np.isnan(A[u]["c"][-2:, :k + 1]).any() for u in F.TARGETS):
        return {}
    row = len(days) - 1
    prev_vix = F.prior_closes(bars.get(F.VIX), days[row:])
    if not np.isfinite(prev_vix[0]):  # no VIX history: the live level stands in for yesterday's close
        prev_vix = np.array([vix if vix else 18.0], float)
    sig = np.atleast_1d(F.expected_move(prev_vix))
    out = {}
    layers, heads, seeds = (int(W[n]) for n in ("tfm_layers", "tfm_heads", "tfm_seeds"))
    x = F.window(F.bar_raw(A), np.array([row]), sig)[0][:F.SLOTS + k + 1]
    s = F.static_inputs(days[row:], prev_vix)[0]
    tp = np.mean([transformer(W, f"t{i}/", x, s, layers, heads)[-1] for i in range(seeds)], 0)
    for j, u in enumerate(F.TARGETS):
        z = (F.ridge_features(A, u, k, np.full(len(days), sig[0]))[row] - W["ridge_mu"]) / W["ridge_sd"]
        r = float(np.clip(z, -6, 6) @ W["ridge_coef"] + W["ridge_b"]) / float(W["ridge_pred_sd"][k])
        t = float(tp[j]) / float(W["tfm_pred_sd"][k])
        out[u] = (r + t) / 2
    return out
