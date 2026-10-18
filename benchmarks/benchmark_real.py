"""
Benchmarks on REAL RRUFF Raman spectra (data/rruff_subset/raman_subset.npz).

    python util/make_subset.py          # once, on the machine with the RRUFF data
    python benchmarks/benchmark_real.py

1. Identification, leave-one-sample-out: each spectrum is a query against a
   library that does NOT contain its own RRUFF sample (other samples of the
   same mineral stay in). Top-1 / top-5 accuracy for cosine, Pearson
   correlation and SAM on the preprocessed grid.
2. Real two-phase mixtures: query spectra of two minerals mixed 70/30;
   library = the other samples. Checks that the two true minerals are the top
   two contributions.
3. Unknown phase: a third mineral is added (20 %) and ALL its samples are
   removed from the library. Checks the residual flags it, and counts false
   alarms on mixtures without an unknown.
"""

import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.mixture.unmixing import MixtureUnmixer

PATH = os.path.join(ROOT, "data", "rruff_subset", "raman_subset.npz")
# Residual peaks within this distance of an identified phase's peak count as
# sample-to-sample variation, not unknown phases (set a priori: ~2 peak widths).
KNOWN_TOL = 8.0
SHIFT_TOL = True


def cosine(a, B):
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)


def pearson(a, B):
    a0 = a - a.mean()
    B0 = B - B.mean(axis=1, keepdims=True)
    return (B0 @ a0) / (np.linalg.norm(B0, axis=1) * np.linalg.norm(a0) + 1e-12)


def sam(a, B):
    return 1 - np.arccos(np.clip(cosine(a, B), -1, 1)) / np.pi


METRICS = {"cosine": cosine, "pearson": pearson, "SAM": sam}


def identification(S, minerals, ids):
    print("\n1) Identification, leave-one-sample-out")
    res = {m: [0, 0] for m in METRICS}
    n = 0
    for i in range(len(S)):
        lib = ids != ids[i]
        if not np.any(minerals[lib] == minerals[i]):
            continue                    # mineral would be absent from the library
        n += 1
        for name, f in METRICS.items():
            scores = f(S[i], S[lib])
            ranked = []
            for j in np.argsort(-scores):
                m = minerals[lib][j]
                if m not in ranked:
                    ranked.append(m)
                if len(ranked) == 5:
                    break
            res[name][0] += ranked[0] == minerals[i]
            res[name][1] += minerals[i] in ranked
    print(f"   queries: {n} | minerals: {len(set(minerals))}")
    for name, (t1, t5) in res.items():
        print(f"   {name:<8} top-1 {t1 / n:6.1%}   top-5 {t5 / n:6.1%}")


def build_unmixer(grid, S, minerals, keep):
    """Library = one averaged reference per mineral from the kept spectra."""
    um = MixtureUnmixer(grid, peak_snr_threshold=6.0, min_peak_separation_cm=8.0,
                         known_peak_tolerance_cm=KNOWN_TOL, shift_tolerant=SHIFT_TOL)
    groups = defaultdict(list)
    for k in np.where(keep)[0]:
        groups[minerals[k]].append(S[k])
    for m, specs in sorted(groups.items()):
        um.add_reference(m, grid, np.mean(specs, axis=0))
    return um


def mixtures(grid, S, minerals, ids, n_cases=60, seed=0):
    rng = np.random.default_rng(seed)
    names = sorted(set(minerals))
    by = {m: np.where(minerals == m)[0] for m in names}
    ok_pair, fa, n_clean = 0, 0, 0
    flagged, n_unknown = 0, 0
    for _ in range(n_cases):
        a, b, u = rng.choice(names, 3, replace=False)
        qa, qb, qu = rng.choice(by[a]), rng.choice(by[b]), rng.choice(by[u])
        # library excludes the query SAMPLES; the unknown mineral is excluded entirely
        keep = (ids != ids[qa]) & (ids != ids[qb])
        noise = 0.01 * rng.standard_normal(len(grid))

        clean = 0.7 * S[qa] + 0.3 * S[qb] + noise
        r = build_unmixer(grid, S, minerals, keep).unmix(grid, clean)
        top2 = [m for m, _ in sorted(r.fractions.items(), key=lambda t: -t[1])[:2]]
        ok_pair += set(top2) == {a, b}
        fa += r.anomaly
        n_clean += 1

        mixed = 0.56 * S[qa] + 0.24 * S[qb] + 0.20 * S[qu] + noise
        keep_u = keep & (minerals != u)
        r2 = build_unmixer(grid, S, minerals, keep_u).unmix(grid, mixed)
        flagged += r2.anomaly
        n_unknown += 1

    print("\n2) Real two-phase mixtures (70/30, other samples as library)")
    print(f"   both true minerals ranked top-2: {ok_pair}/{n_clean} ({ok_pair / n_clean:.0%})")
    print(f"   false alarms (no unknown present): {fa}/{n_clean} ({fa / n_clean:.0%})")
    print("\n3) Unknown phase (20 %, mineral absent from library)")
    print(f"   flagged: {flagged}/{n_unknown} ({flagged / n_unknown:.0%})")


def main():
    if not os.path.exists(PATH):
        sys.exit(f"Missing {PATH}. Run util/make_subset.py on the machine with RRUFF data.")
    d = np.load(PATH, allow_pickle=False)
    grid, S = d["grid"].astype(float), d["spectra"].astype(float)
    minerals, ids = d["mineral"], d["rruff_id"]
    print(f"Loaded {len(S)} real RRUFF spectra, {len(set(minerals))} minerals, "
          f"grid {grid[0]:.0f}-{grid[-1]:.0f} cm-1")
    identification(S, minerals, ids)
    mixtures(grid, S, minerals, ids)


if __name__ == "__main__":
    main()
