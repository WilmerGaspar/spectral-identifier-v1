"""
Benchmarks on REAL RRUFF Raman spectra (data/rruff_subset/raman_subset.npz:
120 processed spectra, 40 minerals x 3 different RRUFF samples, 532 nm).

    python benchmarks/benchmark_real.py

Protocol (fixed seeds, the query SAMPLE is never in the library):
1. Identification, leave-one-sample-out: top-1 / top-5 for cosine, Pearson, SAM.
2. Two-phase mixtures (70/30 of two query samples + noise), library = all
   other samples. Scored at mineral level and at mineral-GROUP level (groups
   from standard mineralogy, defined before evaluation: garnet, olivine,
   amphibole, feldspar, pyroxene; solid-solution members have near-identical
   Raman spectra).
3. Unknown phase: a third mineral (20 %) whose samples are ALL removed from the
   library. Reported as a ROC (AUC and detection rate at fixed false-alarm
   rates) of the unknown score on mixtures with vs without an unknown.

Methods compared:
  A. NNLS over the whole library (averaged reference per mineral,
     shift-tolerant, phase confirmation) - src/mixture/unmixing.py
  B. Sparse forward selection, at most 3 phases, shift-tolerant -
     src/mixture/sparse.py
"""

import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.mixture.unmixing import MixtureUnmixer
from src.mixture.sparse import sparse_unmix

PATH = os.path.join(ROOT, "data", "rruff_subset", "raman_subset.npz")
N_CASES = 100
SEED = 1

GROUPS = {
    "garnet": ["Almandine", "Pyrope", "Spessartine", "Grossular", "Andradite"],
    "olivine": ["Forsterite", "Fayalite"],
    "amphibole": ["Tremolite", "Actinolite"],
    "feldspar": ["Albite", "Anorthite", "Orthoclase", "Microcline"],
    "pyroxene": ["Diopside", "Enstatite", "Aegirine"],
}
GROUP_OF = {m: g for g, ms in GROUPS.items() for m in ms}


def group(m):
    return GROUP_OF.get(str(m), str(m))


# ── 1. identification ───────────────────────────────────────

def cosine(a, B):
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)


def pearson(a, B):
    a0 = a - a.mean()
    B0 = B - B.mean(axis=1, keepdims=True)
    return (B0 @ a0) / (np.linalg.norm(B0, axis=1) * np.linalg.norm(a0) + 1e-12)


def sam(a, B):
    return 1 - np.arccos(np.clip(cosine(a, B), -1, 1)) / np.pi


def identification(S, minerals, ids):
    print("\n1) Identification, leave-one-sample-out")
    metrics = {"cosine": cosine, "pearson": pearson, "SAM": sam}
    res = {m: [0, 0] for m in metrics}
    n = 0
    for i in range(len(S)):
        lib = ids != ids[i]
        n += 1
        for name, f in metrics.items():
            ranked = []
            for j in np.argsort(-f(S[i], S[lib])):
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


# ── methods for mixtures ────────────────────────────────────

def method_nnls(grid, S, minerals, keep, y):
    um = MixtureUnmixer(grid, peak_snr_threshold=2.0, min_peak_separation_cm=8.0,
                        known_peak_tolerance_cm=8.0, shift_tolerant=True)
    groups = defaultdict(list)
    for k in np.where(keep)[0]:
        groups[minerals[k]].append(S[k])
    for m, specs in sorted(groups.items()):
        um.add_reference(str(m), grid, np.mean(specs, axis=0))
    r = um.unmix(grid, y)
    return r.fractions, max((p.snr for p in r.residual_peaks), default=0.0)


def method_sparse(grid, S, minerals, keep, y):
    idx = np.where(keep)[0]
    r = sparse_unmix(grid, y, [S[k] for k in idx], [str(minerals[k]) for k in idx])
    return r.fractions, r.unknown_score


METHODS = {"A. NNLS, all phases": method_nnls, "B. sparse, <= 3 phases": method_sparse}


def top2_ok(fr, a, b):
    top = [m for m, _ in sorted(fr.items(), key=lambda t: -t[1])[:2]]
    mineral_ok = set(top) == {str(a), str(b)}
    gfr = defaultdict(float)
    for m, f in fr.items():
        gfr[group(m)] += f
    gtop = [m for m, _ in sorted(gfr.items(), key=lambda t: -t[1])[:2]]
    if group(a) != group(b):
        group_ok = set(gtop) == {group(a), group(b)}
    else:
        group_ok = gtop[0] == group(a)
    return mineral_ok, group_ok


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def mixtures(grid, S, minerals, ids):
    rng = np.random.default_rng(SEED)
    names = sorted(set(minerals))
    by = {m: np.where(minerals == m)[0] for m in names}
    cases = []
    for _ in range(N_CASES):
        a, b, u = rng.choice(names, 3, replace=False)
        cases.append((a, b, u, rng.choice(by[a]), rng.choice(by[b]), rng.choice(by[u]),
                      0.01 * rng.standard_normal(len(grid))))

    print(f"\n2) Two-phase mixtures 70/30 and 3) unknown phase (20 %), {N_CASES} cases each")
    print(f"   {'method':<24}{'mineral top-2':>14}{'group top-2':>13}{'unknown AUC':>13}"
          f"{'det@FA5%':>10}{'det@FA10%':>11}{'det@FA20%':>11}")
    for name, method in METHODS.items():
        ok_m = ok_g = 0
        neg, pos = [], []
        for a, b, u, qa, qb, qu, noise in cases:
            keep = (ids != ids[qa]) & (ids != ids[qb])
            fr, s_clean = method(grid, S, minerals, keep, 0.7 * S[qa] + 0.3 * S[qb] + noise)
            m_ok, g_ok = top2_ok(fr, a, b)
            ok_m += m_ok
            ok_g += g_ok
            neg.append(s_clean)
            _, s_unk = method(grid, S, minerals, keep & (minerals != u),
                              0.56 * S[qa] + 0.24 * S[qb] + 0.20 * S[qu] + noise)
            pos.append(s_unk)
        neg, pos = np.array(neg), np.array(pos)
        det = [np.mean(pos > np.quantile(neg, 1 - f)) for f in (0.05, 0.10, 0.20)]
        print(f"   {name:<24}{ok_m / N_CASES:>14.0%}{ok_g / N_CASES:>13.0%}{auc(pos, neg):>13.2f}"
              + "".join(f"{d:>10.0%}" if i == 0 else f"{d:>11.0%}" for i, d in enumerate(det)))
    print("\n   AUC: 0.5 = chance, 1.0 = perfect. det@FA5% = unknowns detected when the")
    print("   threshold allows 5 % false alarms on mixtures without an unknown.")


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
