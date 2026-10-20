"""
Lattice-parameter fit of a measured cubic powder pattern with Ramanujan-theta lines
==================================================================================

Given a measured 2-theta pattern of a cubic phase, find the cell edge `a`
(and the instrument zero shift) whose Ramanujan-theta line positions best
explain the measured peaks.

Lines come from the lattice theta series (theta_cubic: phi and psi of
Ramanujan for P, I, F lattices). Space-group reflection conditions remove
further lines; implemented here for the two space groups of our test minerals,
following the general reflection conditions of International Tables for
Crystallography, Vol. A:

  Ia-3d (No. 230, garnets):      hkl: h+k+l = 2n;  0kl: k, l = 2n;
                                 hhl: 2h+l = 4n;   h00: h = 4n
  Fd-3m (No. 227, spinel, diamond, origin 2): hkl: h+k, h+l, k+l = 2n;
                                 0kl: k+l = 4n;    hhl: h+l = 2n;  h00: h = 4n

(Conditions apply to every permutation/sign of the indices. Atoms on special
positions can remove more lines, e.g. diamond: hkl all even needs h+k+l = 4n.)

Author: Wilmer Gaspar Espinoza Castillo
"""

from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.theta_xrd.ramanujan_xrd import theta_cubic, CU_KA1


# ── reflection conditions ───────────────────────────────────

def ia3d_allowed(h, k, l) -> bool:
    if (h + k + l) % 2:
        return False
    a = [abs(h), abs(k), abs(l)]
    for x, y, z in set(permutations(a)):
        if x == 0 and (y % 2 or z % 2):            # 0kl: k, l = 2n
            return False
        if x == y and (2 * x + z) % 4:              # hhl: 2h + l = 4n
            return False
        if y == 0 and z == 0 and x % 4:             # h00: h = 4n
            return False
    return True


def fd3m_allowed(h, k, l) -> bool:
    if (h + k) % 2 or (h + l) % 2 or (k + l) % 2:
        return False
    a = [abs(h), abs(k), abs(l)]
    for x, y, z in set(permutations(a)):
        if x == 0 and (y + z) % 4:                  # 0kl: k + l = 4n
            return False
        if x == y and (x + z) % 2:                  # hhl: h + l = 2n
            return False
        if y == 0 and z == 0 and x % 4:             # h00: h = 4n
            return False
    return True


def diamond_allowed(h, k, l) -> bool:
    """Fd-3m plus the special-position condition of atoms on 8a."""
    if not fd3m_allowed(h, k, l):
        return False
    if h % 2 == 0 and (h + k + l) % 4:
        return False
    return True


SPACE_GROUPS: Dict[str, Tuple[str, Optional[Callable]]] = {
    "Ia-3d": ("I", ia3d_allowed),
    "Fd-3m": ("F", fd3m_allowed),
    "Fd-3m (diamond, 8a)": ("F", diamond_allowed),
    "Fm-3m": ("F", None),
    "Im-3m": ("I", None),
    "Pm-3m": ("P", None),
}


def allowed_N(space_group: str, n_max: int) -> List[int]:
    """N = h^2+k^2+l^2 with nonzero Ramanujan-theta coefficient AND at least one
    (h,k,l) in that shell allowed by the space group."""
    lattice, rule = SPACE_GROUPS[space_group]
    theta = theta_cubic(lattice, n_max)
    out = []
    r = int(np.sqrt(n_max)) + 1
    for N in range(1, n_max + 1):
        if theta[N] == 0:
            continue
        if rule is None:
            out.append(N)
            continue
        found = False
        for h in range(0, r + 1):
            for k in range(0, h + 1):
                l2 = N - h * h - k * k
                if l2 < 0:
                    continue
                l = int(round(np.sqrt(l2)))
                if l * l == l2 and l <= k and rule(h, k, l):
                    found = True
                    break
            if found:
                break
        if found:
            out.append(N)
    return out


def two_theta_of(N, a, wavelength):
    s = wavelength * np.sqrt(np.asarray(N, float)) / (2 * a)
    return np.degrees(2 * np.arcsin(np.clip(s, 0, 1)))


# ── measured peaks and fit ──────────────────────────────────

def measured_peaks(two_theta, intensity, min_prominence=0.03, min_sep_deg=0.15):
    tt = np.asarray(two_theta, float)
    y = np.asarray(intensity, float)
    y = y - np.min(y)
    y = y / (np.max(y) + 1e-12)
    ys = savgol_filter(y, 11, 3) if len(y) > 11 else y
    step = float(np.median(np.diff(tt)))
    idx, props = find_peaks(ys, prominence=min_prominence, distance=max(1, int(min_sep_deg / step)))
    return tt[idx], ys[idx]


@dataclass
class LatticeFit:
    a: float
    zero_shift: float
    matched: int
    n_measured: int
    rms_deg: float
    max_abs_deg: float
    pairs: List[Tuple[float, int, float]] = field(default_factory=list)  # (2theta_obs, N, 2theta_calc)
    unmatched: List[float] = field(default_factory=list)


def fit_lattice(peaks_2theta: Sequence[float], space_group: str,
                a_range: Tuple[float, float], wavelength: float = CU_KA1,
                tol_deg: float = 0.12, a_step: float = 0.0005,
                weights: Optional[Sequence[float]] = None,
                ka2: Optional[float] = None, ka2_min_split: float = 0.0) -> LatticeFit:
    """
    1. grid search over a: count measured peaks within tol of a predicted line
       (weighted by peak height), allowing a common zero shift;
    2. least-squares refinement of (a, zero shift) on the matched pairs.
    ka2: if given (e.g. 1.5444 for Cu K-alpha2), every predicted line is also
         predicted at this second wavelength, so K-alpha2 peaks at high angle
         are explained instead of pulling the fit to a wrong cell.
    ka2_min_split: predict the K-alpha2 line only where it is at least this far
         (deg) from K-alpha1, i.e. where the doublet can be resolved.
    """
    obs = np.asarray(peaks_2theta, float)
    w = np.ones_like(obs) if weights is None else np.asarray(weights, float)
    tt_max = obs.max() + 1
    n_max = int((2 * a_range[1] * np.sin(np.radians(tt_max / 2)) / min(wavelength, ka2 or wavelength)) ** 2) + 2
    Ns1 = np.array(allowed_N(space_group, n_max))
    lam1 = np.full(len(Ns1), wavelength)

    def lines_for(a):
        if not ka2:
            return Ns1, lam1
        t1 = two_theta_of(Ns1, a, wavelength)
        t2 = two_theta_of(Ns1, a, ka2)
        res = (t2 - t1) >= ka2_min_split
        return (np.concatenate([Ns1, Ns1[res]]),
                np.concatenate([lam1, np.full(int(res.sum()), ka2)]))

    def lines(a):
        N_, L_ = lines_for(a)
        return two_theta_of(N_, a, L_)

    def score(a):
        calc = lines(a)
        d = obs[:, None] - calc[None, :]
        j = np.argmin(np.abs(d), axis=1)
        dd = d[np.arange(len(obs)), j]
        z = np.median(dd[np.abs(dd) < 3 * tol_deg]) if np.any(np.abs(dd) < 3 * tol_deg) else 0.0
        hit = np.abs(dd - z) <= tol_deg
        return float(np.sum(w[hit])) - 1e-3 * float(np.sum(np.abs(dd[hit] - z))), z

    best = None
    for a in np.arange(a_range[0], a_range[1] + a_step, a_step):
        s, z = score(a)
        if best is None or s > best[0]:
            best = (s, a, z)
    _, a, z = best

    # refine (a, z) by Gauss-Newton on matched pairs
    for _ in range(20):
        Ns, lams = lines_for(a)
        calc = lines(a) + z
        d = obs[:, None] - calc[None, :]
        j = np.argmin(np.abs(d), axis=1)
        hit = np.abs(d[np.arange(len(obs)), j]) <= tol_deg
        if hit.sum() < 2:
            break
        N_m, L_m, o_m = Ns[j[hit]], lams[j[hit]], obs[hit]
        f = two_theta_of(N_m, a, L_m) + z - o_m
        da = 1e-5
        dfda = (two_theta_of(N_m, a + da, L_m) - two_theta_of(N_m, a, L_m)) / da
        J = np.column_stack([dfda, np.ones_like(dfda)])
        step, *_ = np.linalg.lstsq(J, -f, rcond=None)
        a, z = a + step[0], z + step[1]
        if np.all(np.abs(step) < [1e-7, 1e-6]):
            break

    Ns, lams = lines_for(a)
    calc = lines(a) + z
    d = obs[:, None] - calc[None, :]
    j = np.argmin(np.abs(d), axis=1)
    dd = d[np.arange(len(obs)), j]
    hit = np.abs(dd) <= tol_deg
    return LatticeFit(
        a=float(a), zero_shift=float(z), matched=int(hit.sum()), n_measured=len(obs),
        rms_deg=float(np.sqrt(np.mean(dd[hit] ** 2))) if hit.any() else float("nan"),
        max_abs_deg=float(np.max(np.abs(dd[hit]))) if hit.any() else float("nan"),
        pairs=[(float(obs[i]), int(Ns[j[i]]), float(calc[j[i]])) for i in np.where(hit)[0]],
        unmatched=[float(obs[i]) for i in np.where(~hit)[0]],
    )
