"""Temperature scaling and Expected Calibration Error (ECE). Owner: member 3.

Temperature scaling (Guo et al. 2017): one scalar T > 0 divides the logits, fitted by
minimising the negative log-likelihood on VALIDATION pixels only (never test). It changes
confidence, not the argmax prediction, so accuracy and F1 are unchanged.

ECE: confidence = max softmax probability; pixels are binned by confidence into ``n_bins``
equal-width bins; ECE = sum over bins of (bin share) * |accuracy - mean confidence|.
Reported here at the pixel level over valid pixels, on a random pixel sample per tile
(``sample_pixels``) so memory stays small.

The fitted T is saved as JSON next to the checkpoint and read by export_evidence.py.
"""
import json
from pathlib import Path

import numpy as np


def softmax(logits, t=1.0):
    z = np.asarray(logits, dtype=np.float64) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, labels, t=1.0):
    p = softmax(logits, t)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits, labels, lo=0.05, hi=20.0, iters=80):
    """T minimising NLL on (logits [N,C], labels [N]). Golden-section search on log T
    (NLL is unimodal in T for fixed logits); no torch needed."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, labels, np.exp(c)), nll(logits, labels, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, labels, np.exp(d))
    return float(np.exp((a + b) / 2))


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error of max-probability confidence (equal-width bins)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def sample_pixels(logits, target, valid, n, rng):
    """Random sample of up to n valid pixels from one tile.
    logits [C,H,W], target [H,W] (0..4), valid [H,W] -> (logits [k,C], labels [k])."""
    keep = np.flatnonzero(np.asarray(valid).ravel() & (np.asarray(target).ravel() <= 4))
    if len(keep) > n:
        keep = rng.choice(keep, n, replace=False)
    c = logits.shape[0]
    return logits.reshape(c, -1)[:, keep].T, np.asarray(target).ravel()[keep]


def calibration_report(logits, labels, n_bins=15):
    """Fit T and report NLL and ECE before and after (pixel level)."""
    lo, hi = 0.05, 20.0
    t = fit_temperature(logits, labels, lo, hi)
    return {
        "temperature_at_search_bound": bool(t < lo * 1.05 or t > hi * 0.95),
        "level": "pixel",
        "pixels": len(labels),
        "temperature": t,
        "nll_before": nll(logits, labels, 1.0),
        "nll_after": nll(logits, labels, t),
        "ece_before": ece(softmax(logits, 1.0), labels, n_bins),
        "ece_after": ece(softmax(logits, t), labels, n_bins),
        "n_bins": n_bins,
    }


def save_temperature(report, path):
    Path(path).write_text(json.dumps(report, indent=2))


def load_temperature(path):
    """T from a JSON written by save_temperature; 1.0 if path is None."""
    if path is None:
        return 1.0
    return float(json.loads(Path(path).read_text())["temperature"])
