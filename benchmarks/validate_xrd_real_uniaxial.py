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
from src.theta_xrd.fit_uniaxial import fit_uniaxial
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
          f"{'c RRUFF':>9}{'da':>8}{'dc':>8}{'explained':>10}")
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
        f = fit_uniaxial(pk, info["system"], info["centering"], a_ref, c_ref,
                         window=0.03, wavelength=wavelength(info["header"]), weights=h)
        rows.append((info["mineral"], f.a - a_ref, f.c - c_ref, f.matched / f.n_measured))
        print(f"{info['mineral']:<14}{info['rruff_id']:<10}{info['system'] + '-' + info['centering']:<12}"
              f"{f.a:>8.4f}{a_ref:>9.4f}{f.c:>9.4f}{c_ref:>9.4f}{f.a - a_ref:>8.4f}{f.c - c_ref:>8.4f}"
              f"{f.matched / f.n_measured:>10.0%}")
    if not rows:
        sys.exit("No tetragonal/hexagonal patterns in the subset. Re-run util/make_xrd_subset.py.")
    da = np.abs([r[1] for r in rows])
    dc = np.abs([r[2] for r in rows])
    ex = np.array([r[3] for r in rows])
    print(f"\n{len(rows)} patterns, {len({r[0] for r in rows})} minerals")
    print(f"median |da| {np.median(da):.4f} A, median |dc| {np.median(dc):.4f} A, "
          f"both within 0.01 A: {np.mean((da < 0.01) & (dc < 0.01)):.0%}, "
          f"mean peaks explained {ex.mean():.0%}")


if __name__ == "__main__":
    main()
