"""
Sparse (few-phase) unmixing with shift tolerance
================================================
Real rocks and products usually contain only a few phases. With a large
library, unconstrained NNLS can imitate almost any spectrum (including an
unknown phase) as a sum of many small contributions, which hides unknowns.

This module selects phases greedily (forward selection): at each step it adds
the library spectrum that most reduces the residual, up to `max_phases`, and
stops when the relative improvement is below `min_gain`. Each selected
reference gets +/- first and second derivative columns (first/second-order
Taylor expansion) so small peak shifts and width changes between samples of
the same mineral are absorbed.

Unknown-phase score: the largest SNR of a residual peak that does not lie
within `known_tol_cm` of a clear peak of any selected reference.

Author: Wilmer Gaspar Espinoza Castillo
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks, savgol_filter

from src.mixture.unmixing import _noise_level


@dataclass
class SparseResult:
    fractions: Dict[str, float]           # per label, spectral (not mass) fractions
    unknown_score: float                  # max SNR of an unexplained residual peak
    unexplained_peaks_cm: List[float] = field(default_factory=list)
    residual: np.ndarray = None
    fitted: np.ndarray = None


def _design(cols):
    A = np.column_stack(cols)
    d1 = np.gradient(A, axis=0)
    d2 = np.gradient(d1, axis=0)
    return np.column_stack([A, d1, -d1, d2, -d2])


def sparse_unmix(grid: Sequence[float], y: np.ndarray, refs: Sequence[np.ndarray],
                 labels: Sequence[str], max_phases: int = 3, min_gain: float = 0.02,
                 known_tol_cm: float = 8.0, peak_snr: float = 2.0) -> SparseResult:
    """
    grid, y: measured spectrum on the same grid as refs (scaled to max = 1)
    refs, labels: library spectra and their mineral names (several spectra may
                  share a label; at most one per label is selected)
    """
    grid = np.asarray(grid, float)
    y = np.asarray(y, float)
    base = np.linalg.norm(y) + 1e-12
    chosen: List[int] = []
    current = base
    for _ in range(max_phases):
        used = {labels[c] for c in chosen}
        best = None
        for j in range(len(refs)):
            if labels[j] in used:
                continue
            A = _design([refs[c] for c in chosen + [j]])
            w, _ = nnls(A, y)
            res = np.linalg.norm(y - A @ w)
            if best is None or res < best[0]:
                best = (res, j)
        if best is None or (current - best[0]) / base < min_gain:
            break
        current = best[0]
        chosen.append(best[1])

    if not chosen:
        return SparseResult({}, float("inf"), [], y.copy(), np.zeros_like(y))

    A = _design([refs[c] for c in chosen])
    w, _ = nnls(A, y)
    fitted = A @ w
    r = y - fitted
    k = len(chosen)
    total = float(w[:k].sum()) or 1.0
    fractions = {labels[c]: float(w[i]) / total for i, c in enumerate(chosen)}

    known = []
    for c in chosen:
        rf = savgol_filter(refs[c], 11, 3)
        idx, _ = find_peaks(rf, prominence=0.05)
        known.extend(grid[idx])
    known = np.asarray(known)
    noise = _noise_level(y)
    rs = savgol_filter(r, 9, 3)
    idx, _ = find_peaks(rs, height=peak_snr * noise, distance=4)
    unexplained = [(grid[i], rs[i] / noise) for i in idx
                   if known.size == 0 or np.min(np.abs(known - grid[i])) > known_tol_cm]
    score = max((s for _, s in unexplained), default=0.0)
    return SparseResult(fractions, float(score), [float(p) for p, _ in unexplained], r, fitted)
