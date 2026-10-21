"""
Ramanujan-theta (a, c) refinement on MEASURED RRUFF powder XRD of tetragonal and
hexagonal/trigonal minerals (entries with "system" in data/rruff_xrd_subset).

Protocol: refinement, not ab-initio indexing. The search window is +/- 3 %
around the cell in the RRUFF header (a few tenths of an angstrom, roughly 100x
the errors we report); inside it, a and c are found from measured line
positions only, using phi(q)^2 (tetragonal) or the Borwein cubic theta a(q)
(hexagonal) times phi(q). Space-group absences are not applied.

Success criteria fixed before running: median |error| < 0.01 A for both a and c,
and >= 70 % of measured peaks explained.

Bug fix (Oct 21, found while checking Pushpender's review): the fit matched each
peak to the nearest line BEFORE applying the zero shift. With a zero shift
comparable to the line spacing (vesuvianite: ~0.09 deg) peaks were assigned to
the wrong lines. After the fix, on the same 68 patterns:
    before (pre-registered run, Oct 20): median |da| 0.0026, |dc| 0.0044 A,
        both < 0.01 A in 69 %, peaks explained 83 %, vesuvianite failed 4/4
        (output kept in benchmarks/output/validate_xrd_real_uniaxial_before_fix.txt)
    after: median |da| 0.0027, |dc| 0.0031 A, both < 0.01 A in 76 %,
        peaks explained 87 %, vesuvianite within 0.007 A on 4/4
Both runs meet the criteria fixed in advance. Cubic results are unchanged.

Reliability flags (src/theta_xrd/quality.py, added Oct 21 after review) are
printed next to each fit, with the least-squares uncertainty of a and c; they
do not change any fitted number.

    python benchmarks/validate_xrd_real_uniaxial.py
"""

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.theta_xrd.fit import measured_peaks
from src.theta_xrd.fit_uniaxial import fit_uniaxial, line_shells, two_theta
from src.theta_xrd.quality import assess
from src.theta_xrd.ramanujan_xrd import CU_KA1

DIR = os.path.join(ROOT, "data", "rruff_xrd_subset")


def header_cell(header):
    """a and c from a RRUFF 'CELL PARAMETERS' string like 'a: 4.91 b: 4.91 c: 5.40 ...'."""
    for k, v in header.items():
        if "CELL" in k.upper():
            a = re.search(r"\ba:\s*([\d.]+)", str(v))
            c = re.search(r"\bc:\s*([\d.]+)", str(v))
            if a and c:
                return float(a.group(1)), float(c.group(1))
    return None, None


def wavelength(header):
    for k, v in header.items():
        if "WAVELENGTH" in k.upper():
            m = re.search(r"\d+\.\d+", str(v))
            if m:
                return float(m.group())
    return CU_KA1


def main():
    path = os.path.join(DIR, "xrd_subset.npz")
    if not os.path.exists(path):
        sys.exit("Missing data/rruff_xrd_subset. Run util/make_xrd_subset.py first.")
    d = np.load(path)
    grid, P = d["grid"].astype(float), d["patterns"].astype(float)
    index = json.load(open(os.path.join(DIR, "xrd_index.json"), encoding="utf-8"))
    rows = []
    print(f"{'mineral':<14}{'RRUFF ID':<10}{'system':<12}{'a fit':>8}{'a RRUFF':>9}{'c fit':>9}"
          f"{'c RRUFF':>9}{'da':>8}{'dc':>8}{'explained':>10}{'sig a':>8}{'sig c':>8}{'score':>7}  flags")
    for y, info in zip(P, index):
        if "system" not in info:
            continue
        a_ref, c_ref = header_cell(info["header"])
        if a_ref is None:
            continue
        x0, x1 = info["x_range"]
        keep = (grid >= x0) & (grid <= x1)
        pk, h = measured_peaks(grid[keep], y[keep])
        if len(pk) < 4:
            continue
        lam = wavelength(info["header"])
        f = fit_uniaxial(pk, info["system"], info["centering"], a_ref, c_ref,
                         window=0.03, wavelength=lam, weights=h)
        # line list over the WHOLE measured range (the chance rate needs every line
        # a random peak could hit, not only those below the last detected peak)
        d_min = lam / (2 * np.sin(np.radians(min(x1 + 1, 179) / 2)))
        A_max = int((a_ref * 1.03 / d_min) ** 2 * (1.0 if info["system"] == "tetragonal" else 0.75)) + 2
        shells = line_shells(info["system"], info["centering"], A_max, int(c_ref * 1.03 / d_min) + 1)
        q = assess(pk, h, [f.a, f.c], lambda p: two_theta(shells, p[0], p[1], info["system"], lam),
                   f.zero_shift, tol=0.08, x_range=(x0, x1), y=y[keep])
        rows.append((info["mineral"], f.a - a_ref, f.c - c_ref, f.matched / f.n_measured, q))
        sa, sc = q["sigma"].get("a", float("nan")), q["sigma"].get("c", float("nan"))
        print(f"{info['mineral']:<14}{info['rruff_id']:<10}{info['system'] + '-' + info['centering']:<12}"
              f"{f.a:>8.4f}{a_ref:>9.4f}{f.c:>9.4f}{c_ref:>9.4f}{f.a - a_ref:>8.4f}{f.c - c_ref:>8.4f}"
              f"{f.matched / f.n_measured:>10.0%}{sa:>8.4f}{sc:>8.4f}{q['score']:>7.2f}  {', '.join(q['flags']) or '-'}")
    if not rows:
        sys.exit("No tetragonal/hexagonal patterns in the subset. Re-run util/make_xrd_subset.py.")
    da = np.abs([r[1] for r in rows])
    dc = np.abs([r[2] for r in rows])
    ex = np.array([r[3] for r in rows])
    print(f"\n{len(rows)} patterns, {len({r[0] for r in rows})} minerals")
    print(f"median |da| {np.median(da):.4f} A, median |dc| {np.median(dc):.4f} A, "
          f"both within 0.01 A: {np.mean((da < 0.01) & (dc < 0.01)):.0%}, "
          f"mean peaks explained {ex.mean():.0%}")
    unrel = np.array(["unreliable" in r[4]["flags"] for r in rows])
    bad = (da >= 0.01) | (dc >= 0.01)
    print(f"\nReliability flags (do not change the numbers above): {unrel.sum()} fits flagged unreliable, "
          f"{sum('noisy' in r[4]['flags'] for r in rows)} noisy, {sum('few peaks' in r[4]['flags'] for r in rows)} few peaks.")
    print(f"fits with an error >= 0.01 A: {bad.sum()}; of these flagged unreliable: {(bad & unrel).sum()}; "
          f"flagged unreliable but both errors < 0.01 A: {(~bad & unrel).sum()}")
    ok = ~unrel
    print(f"fits NOT flagged unreliable ({ok.sum()}): median |da| {np.median(da[ok]):.4f} A, "
          f"median |dc| {np.median(dc[ok]):.4f} A, both within 0.01 A: {np.mean(~bad[ok]):.0%}")


if __name__ == "__main__":
    main()
