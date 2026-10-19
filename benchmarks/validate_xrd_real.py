"""
Validation of the Ramanujan-theta lattice fit on MEASURED RRUFF powder XRD
(data/rruff_xrd_subset, built by util/make_xrd_subset.py).

For each measured pattern of a cubic mineral:
  * measured peaks are found in the pattern;
  * the cell edge a and the zero shift are fitted using ONLY Ramanujan-theta
    line positions (+ space-group reflection conditions), searched over a
    broad range per mineral family (RRUFF's own cell is not used as a prior);
  * reported: share of measured peaks explained, RMS position error, and the
    fitted a compared with the cell edge RRUFF reports in the file header
    (from RRUFF's own refinement), when present.
Garnets are then classified by the fitted a against the Novak & Gibbs (1971)
end-member values.

    python benchmarks/validate_xrd_real.py
"""

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.theta_xrd.fit import measured_peaks, fit_lattice
from src.theta_xrd.ramanujan_xrd import CU_KA1

DIR = os.path.join(ROOT, "data", "rruff_xrd_subset")
SEARCH = {"Ia-3d": (11.2, 12.3), "Fd-3m": (7.9, 8.6), "Fd-3m (diamond, 8a)": (3.4, 3.7),
          "Fm-3m": (4.0, 6.6)}
GARNETS = {"Pyrope": 11.459, "Almandine": 11.531, "Spessartine": 11.612,
           "Grossular": 11.815, "Andradite": 12.058}


def first_float(text):
    m = re.search(r"[-+]?\d+\.\d+|\d+", str(text))
    return float(m.group()) if m else None


def header_value(header, word):
    for k, v in header.items():
        if word in k.upper():
            return v
    return None


def main():
    if not os.path.exists(os.path.join(DIR, "xrd_subset.npz")):
        sys.exit("Missing data/rruff_xrd_subset. Run util/make_xrd_subset.py first.")
    d = np.load(os.path.join(DIR, "xrd_subset.npz"))
    grid, P = d["grid"].astype(float), d["patterns"].astype(float)
    index = json.load(open(os.path.join(DIR, "xrd_index.json"), encoding="utf-8"))

    print(f"{len(P)} measured RRUFF powder patterns\n")
    print(f"{'mineral':<13}{'RRUFF ID':<10}{'lambda':>8}{'a fit':>9}{'a RRUFF':>9}{'diff':>8}"
          f"{'explained':>11}{'RMS deg':>9}{'garnet guess':>14}")
    rows = []
    for y, info in zip(P, index):
        lam = first_float(header_value(info["header"], "WAVELENGTH")) or CU_KA1
        a_ref = first_float(header_value(info["header"], "CELL"))
        x0, x1 = info["x_range"]
        keep = (grid >= x0) & (grid <= x1)
        pk, h = measured_peaks(grid[keep], y[keep])
        if len(pk) < 3:
            continue
        sg = info["space_group"]
        fit = fit_lattice(pk, sg, SEARCH[sg],
                          wavelength=lam, weights=h)
        guess = (min(GARNETS, key=lambda m: abs(GARNETS[m] - fit.a)) if sg == "Ia-3d" else "")
        diff = fit.a - a_ref if a_ref else float("nan")
        rows.append((info["mineral"], fit, a_ref, guess))
        print(f"{info['mineral']:<13}{info['rruff_id']:<10}{lam:>8.4f}{fit.a:>9.4f}"
              f"{(a_ref if a_ref else float('nan')):>9.4f}{diff:>8.4f}"
              f"{fit.matched / fit.n_measured:>11.0%}{fit.rms_deg:>9.3f}{guess:>14}")

    if not rows:
        sys.exit("No pattern could be fitted.")
    expl = np.mean([r[1].matched / r[1].n_measured for r in rows])
    rms = np.nanmean([r[1].rms_deg for r in rows])
    diffs = [abs(r[1].a - r[2]) for r in rows if r[2]]
    gar = [r for r in rows if r[0] in GARNETS]
    print(f"\nSummary: mean share of measured peaks explained {expl:.0%}, mean RMS {rms:.3f} deg")
    if diffs:
        print(f"         |a fit - a RRUFF|: median {np.median(diffs):.4f} A, max {np.max(diffs):.4f} A "
              f"({len(diffs)} patterns with a header cell)")
    if gar:
        ok = sum(r[0] == r[3] for r in gar)
        print(f"         garnets classified by fitted a: {ok}/{len(gar)} "
              "(real garnets are solid solutions; end-member names are nominal)")


if __name__ == "__main__":
    main()
