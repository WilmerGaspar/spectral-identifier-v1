"""
Reliability flags for a Ramanujan-theta cell fit
================================================
Added after Pushpender's review (Oct 21): the "share of measured peaks
explained" can mislead in two ways.

  N1  noise counted as peaks. In Fe/Mn-bearing minerals the background is
      noisy (consistent with X-ray fluorescence under Cu K-alpha), the peak
      finder returns hundreds of "peaks", and the share drops to ~6 % even
      when the cell is right.
  N2  chance matches. A large cell (vesuvianite, ~650 lines between 5 and
      90 deg) puts some line within the tolerance of ~2/3 of the whole angle
      range, so random peaks "match"; with very few peaks many nearby cells
      score 100 %.

This module does NOT change any fit. It reports, per pattern:

  noise        robust noise level of the pattern (fraction of its max)
  explained    share of detected peaks within tol of a predicted line
  top10        same, for the 10 tallest peaks only (noise cannot fake these)
  chance       share of the measured 2-theta range lying within tol of some
               predicted line = how often a RANDOM peak would match
  score        chance-corrected match, (share - chance) / (1 - chance):
               0 = no better than random, 1 = every peak matched.
               Uses top10 for noisy patterns, explained otherwise.
  sigma        least-squares standard uncertainty of each cell parameter
  flags        "noisy", "few peaks", "unreliable"

Thresholds are round numbers fixed when this was written, not tuned:
  noisy       > 50 detected peaks (clean patterns here give ~4-20)
  few peaks   < 3 peaks per fitted parameter (cell + zero shift)
  unreliable  score < 0.5 (less than halfway from chance to perfect)
"""

from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy.signal import savgol_filter

NOISY_PEAKS = 50
PEAKS_PER_PARAM = 3
RELIABLE_SCORE = 0.5


def noise_level(y: np.ndarray) -> float:
    """Robust (MAD) noise of a max-normalised pattern, as a fraction of its max."""
    y = np.asarray(y, float)
    if y.size < 15 or y.max() <= 0:
        return float("nan")
    r = y / y.max() - savgol_filter(y / y.max(), 11, 3)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def chance_rate(lines: np.ndarray, x0: float, x1: float, tol: float, step: float = 0.005) -> float:
    """Share of [x0, x1] (deg 2-theta) lying within tol of at least one line."""
    lines = np.sort(np.asarray(lines, float))
    lines = lines[(lines > x0 - tol) & (lines < x1 + tol)]
    if lines.size == 0:
        return 0.0
    grid = np.arange(x0, x1, step)
    k = np.clip(np.searchsorted(lines, grid), 1, lines.size - 1)
    d = np.minimum(np.abs(grid - lines[k - 1]), np.abs(grid - lines[k]))
    return float(np.mean(d <= tol))


def _nearest(peaks: np.ndarray, lines: np.ndarray):
    d = peaks[:, None] - lines[None, :]
    j = np.argmin(np.abs(d), axis=1)
    return j, d[np.arange(len(peaks)), j]


def assess(peaks: Sequence[float], heights: Sequence[float], params: Sequence[float],
           lines_of: Callable[[np.ndarray], np.ndarray], zero: float, tol: float,
           x_range: Sequence[float], y: np.ndarray = None,
           names: Sequence[str] = ("a", "c")) -> Dict:
    """
    peaks, heights : detected peak positions (deg) and heights
    params         : fitted cell parameters, e.g. [a] or [a, c]
    lines_of(p)    : predicted 2-theta of every allowed line for parameters p
                     (same order for any p, no zero shift)
    zero           : fitted zero shift (deg)
    """
    pk = np.asarray(peaks, float)
    h = np.asarray(heights, float)
    p = np.asarray(params, float)
    lines = np.asarray(lines_of(p), float)
    ok = np.isfinite(lines)
    lines_ok = lines[ok] + zero
    n = len(pk)
    out = {"n_peaks": n, "noise": noise_level(y) if y is not None else float("nan")}
    if n == 0 or lines_ok.size == 0:
        out.update(explained=0.0, top10=0.0, chance=0.0, score=0.0, sigma={},
                   flags=["unreliable"], verdict="unreliable")
        return out

    j, r = _nearest(pk, lines_ok)
    hit = np.abs(r) <= tol
    top = np.argsort(h)[::-1][:10]
    chance = chance_rate(lines_ok, float(x_range[0]), float(x_range[1]), tol)
    explained, top10 = float(hit.mean()), float(hit[top].mean())

    def corrected(s):
        return (s - chance) / (1 - chance) if chance < 1 else 0.0

    noisy = n > NOISY_PEAKS
    score = corrected(top10 if noisy else explained)

    # least-squares uncertainty of the cell parameters (+ zero shift) on matched peaks
    sigma = {}
    m, k = int(hit.sum()), p.size + 1
    if m > k:
        idx_ok = np.where(ok)[0][j[hit]]          # indices into the full line list
        cols = []
        for i in range(p.size):
            dp = np.zeros_like(p)
            dp[i] = 1e-6 * max(abs(p[i]), 1.0)
            cols.append((lines_of(p + dp)[idx_ok] - lines[idx_ok]) / dp[i])
        J = np.column_stack(cols + [np.ones(m)])
        s2 = float(np.sum(r[hit] ** 2) / (m - k))
        try:
            cov = s2 * np.linalg.inv(J.T @ J)
            sigma = {names[i]: float(np.sqrt(max(cov[i, i], 0.0))) for i in range(p.size)}
        except np.linalg.LinAlgError:
            sigma = {}

    flags: List[str] = []
    if noisy:
        flags.append("noisy")
    if n < PEAKS_PER_PARAM * k:
        flags.append("few peaks")
    if score < RELIABLE_SCORE:
        flags.append("unreliable")
    verdict = "unreliable" if "unreliable" in flags else ("check" if flags else "ok")
    out.update(explained=explained, top10=top10, chance=chance, score=float(score),
               sigma=sigma, flags=flags, verdict=verdict,
               rms_deg=float(np.sqrt(np.mean(r[hit] ** 2))) if m else float("nan"))
    return out
