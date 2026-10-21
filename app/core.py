"""
Logic behind the marimo demo app (app/app.py).

Everything here reuses the repository's own fitting code and the exact
settings of the benchmark scripts, so the numbers the app shows are the same
as in benchmarks/output/ (checked by tests/test_app_core.py).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.theta_xrd.fit import measured_peaks, fit_lattice, allowed_N, two_theta_of
from src.theta_xrd.fit_uniaxial import fit_uniaxial, line_shells, two_theta
from src.theta_xrd.quality import assess
from src.theta_xrd.ramanujan_xrd import CU_KA1
from benchmarks.validate_xrd_real import SEARCH, GARNETS, first_float, header_value
from benchmarks.validate_xrd_real_uniaxial import header_cell, wavelength
from benchmarks.benchmark_real import cosine

XRD_DIR = os.path.join(ROOT, "data", "rruff_xrd_subset")
RAMAN_PATH = os.path.join(ROOT, "data", "rruff_subset", "raman_subset.npz")
SAVED_LITERATURE = os.path.join(ROOT, "src", "ai", "output", "literature_xrd.txt")


# ── X-ray powder diffraction ────────────────────────────────

def load_xrd():
    d = np.load(os.path.join(XRD_DIR, "xrd_subset.npz"))
    with open(os.path.join(XRD_DIR, "xrd_index.json"), encoding="utf-8") as f:
        index = json.load(f)
    return d["grid"].astype(float), d["patterns"].astype(float), index


def xrd_system(info):
    return info.get("space_group") or f"{info['system']}-{info['centering']}"


def fit_xrd(grid, y, info):
    """Fit one measured pattern exactly as the benchmark scripts do.

    Cubic: cell edge searched over a broad range per mineral family (no RRUFF prior).
    Tetragonal / hexagonal: (a, c) refined within +/- 3 % of the RRUFF cell.
    Returns the data to plot (pattern, peaks, predicted theta lines) and the numbers.
    """
    x0, x1 = info["x_range"]
    keep = (grid >= x0) & (grid <= x1)
    tt, yy = grid[keep], y[keep]
    pk, h = measured_peaks(tt, yy)
    out = {"mineral": info["mineral"], "rruff_id": info["rruff_id"], "system": xrd_system(info),
           "two_theta": tt, "intensity": yy, "peaks": pk, "heights": h, "error": None}

    if "system" in info:                                   # tetragonal / hexagonal
        a_ref, c_ref = header_cell(info["header"])
        lam = wavelength(info["header"])
        if a_ref is None or len(pk) < 4:
            return {**out, "error": "No reference cell or too few peaks to fit."}
        f = fit_uniaxial(pk, info["system"], info["centering"], a_ref, c_ref,
                         window=0.03, wavelength=lam, weights=h)
        d_min = lam / (2 * np.sin(np.radians(min(x1 + 1, 179) / 2)))
        A_max = int((a_ref * 1.03 / d_min) ** 2 * (1.0 if info["system"] == "tetragonal" else 0.75)) + 2
        shells = line_shells(info["system"], info["centering"], A_max, int(c_ref * 1.03 / d_min) + 1)

        def lines_of(p):
            return two_theta(shells, p[0], p[1], info["system"], lam)

        params, names, tol = [f.a, f.c], ("a", "c"), 0.08
        out.update(a=f.a, c=f.c, a_ref=a_ref, c_ref=c_ref, garnet=None,
                   protocol="refined within ±3 % of the RRUFF cell")
    else:                                                  # cubic
        sg = info["space_group"]
        lam = first_float(header_value(info["header"], "WAVELENGTH")) or CU_KA1
        a_ref = first_float(header_value(info["header"], "CELL"))
        if len(pk) < 3:
            return {**out, "error": "Too few peaks to fit."}
        f = fit_lattice(pk, sg, SEARCH[sg], wavelength=lam, weights=h)
        n_max = int((2 * SEARCH[sg][1] * np.sin(np.radians((x1 + 1) / 2)) / lam) ** 2) + 2
        Ns = np.array(allowed_N(sg, n_max))

        def lines_of(p):
            return two_theta_of(Ns, p[0], lam)

        params, names, tol = [f.a], ("a",), 0.12
        garnet = min(GARNETS, key=lambda m: abs(GARNETS[m] - f.a)) if sg == "Ia-3d" else None
        out.update(a=f.a, c=None, a_ref=a_ref, c_ref=None, garnet=garnet,
                   protocol=f"searched {SEARCH[sg][0]}–{SEARCH[sg][1]} Å, no RRUFF prior")

    q = assess(pk, h, params, lines_of, f.zero_shift, tol=tol, x_range=(x0, x1), y=yy, names=names)
    lines = np.asarray(lines_of(np.asarray(params, float)), float) + f.zero_shift
    lines = np.unique(np.round(lines[np.isfinite(lines) & (lines >= x0) & (lines <= x1)], 4))
    matched = (np.array([np.min(np.abs(lines - p)) <= tol for p in pk], bool)
               if lines.size else np.zeros(len(pk), bool))
    out.update(lam=lam, zero_shift=f.zero_shift, explained=f.matched / f.n_measured,
               lines=lines, matched=matched, tol=tol, quality=q)
    return out


# ── Raman identification ────────────────────────────────────

def load_raman():
    d = np.load(RAMAN_PATH)
    return d["grid"].astype(float), d["spectra"].astype(float), d["mineral"].astype(str), d["rruff_id"].astype(str)


def identify_raman(S, minerals, ids, i, k=5):
    """Top-k minerals for sample i, cosine similarity, leave-one-sample-out
    (the query sample is never in the library), as in benchmarks/benchmark_real.py.
    Returns [(mineral, score, index of the best-matching library spectrum)]."""
    lib = np.where(ids != ids[i])[0]
    scores = cosine(S[i], S[lib])
    ranked = []
    for j in np.argsort(-scores):
        m = minerals[lib[j]]
        if m not in [r[0] for r in ranked]:
            ranked.append((m, float(scores[j]), int(lib[j])))
        if len(ranked) == k:
            break
    return ranked


# ── Literature (Valency MCP Bond) ───────────────────────────

def literature(phases, topic, token):
    """Live search when a token is given; otherwise (or if the call fails) the saved run."""
    if token:
        try:
            from src.ai.literature import literature_report
            return literature_report(phases, [], token, topic=topic), "live"
        except Exception as e:                  # network / auth problems: fall back, say why
            note = f"Live search failed ({e}); showing the saved run instead.\n\n"
            return note + open(SAVED_LITERATURE, encoding="utf-8").read(), "saved"
    return open(SAVED_LITERATURE, encoding="utf-8").read(), "saved"
