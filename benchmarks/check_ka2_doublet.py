"""
Does modelling the Cu K-alpha2 line help the cubic Ramanujan-theta fit?

  part 1  simulated almandine (a = 11.531 A, K-alpha1 + half-intensity K-alpha2),
          30 noisy patterns on a 0.01 deg and a 0.02 deg grid
  part 2  the 42 measured RRUFF cubic patterns (same protocol as validate_xrd_real.py)

    python benchmarks/check_ka2_doublet.py

Result when written (Oct 2026): on simulation, without K-alpha2 the fit fails
(> 0.01 A) on 7/30 patterns at 0.01 deg and 1/30 at 0.02 deg; with K-alpha2
predicted where the doublet is resolvable, 0/30. On the measured RRUFF data it
explains more peaks (78 % -> 85 %) but does NOT improve the cell (median
0.0031 -> 0.0033 A; 40/42 -> 37/42 within 0.01 A; numbers after the Oct 21
zero-shift matching fix). The default therefore stays OFF (ka2=None), which is
the pre-registered method; K-alpha2 is an option.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from benchmarks.simulate_theta_xrd_garnet import simulate
from benchmarks.validate_xrd_real import SEARCH, first_float, header_value
from src.theta_xrd.fit import ia3d_allowed, measured_peaks, fit_lattice
from src.theta_xrd.ramanujan_xrd import CU_KA1

KA2 = 1.5444
VARIANTS = (("K-alpha1 only (default)", None, 0.0),
            ("+ K-alpha2 everywhere", KA2, 0.0),
            ("+ K-alpha2 where resolvable", KA2, 0.15))


def simulated():
    grid02 = np.arange(5, 90.0001, 0.02)
    print("Part 1 - simulated almandine, 30 patterns per row, failure = |error| > 0.01 A")
    for name, ka2, ms in VARIANTS:
        for label, regrid in (("0.01 deg grid", False), ("0.02 deg grid", True)):
            errs = []
            for seed in range(30):
                rng = np.random.default_rng(seed)
                tt, y = simulate(11.531, ia3d_allowed, rng, tt=np.arange(5, 90, 0.01),
                                 zero=rng.uniform(-0.05, 0.05))
                if regrid:
                    y, tt = np.interp(grid02, tt, y), grid02
                pk, h = measured_peaks(tt, y)
                f = fit_lattice(pk, "Ia-3d", (11.2, 12.3), weights=h, ka2=ka2, ka2_min_split=ms)
                errs.append(abs(f.a - 11.531))
            e = np.array(errs)
            print(f"  {name:<30}{label:<15} median {np.median(e):.4f} A  max {e.max():.4f} A  "
                  f"failures {int(np.sum(e > 0.01))}/30")


def measured():
    path = os.path.join(ROOT, "data", "rruff_xrd_subset", "xrd_subset.npz")
    if not os.path.exists(path):
        print("\nPart 2 skipped: data/rruff_xrd_subset missing.")
        return
    d = np.load(path)
    grid, P = d["grid"].astype(float), d["patterns"].astype(float)
    index = json.load(open(os.path.join(ROOT, "data", "rruff_xrd_subset", "xrd_index.json"), encoding="utf-8"))
    print("\nPart 2 - measured RRUFF cubic patterns")
    for name, ka2, ms in VARIANTS:
        diffs, expl = [], []
        for y, info in zip(P, index):
            sg = info.get("space_group")
            if sg not in SEARCH:
                continue
            lam = first_float(header_value(info["header"], "WAVELENGTH")) or CU_KA1
            a_ref = first_float(header_value(info["header"], "CELL"))
            x0, x1 = info["x_range"]
            keep = (grid >= x0) & (grid <= x1)
            pk, h = measured_peaks(grid[keep], y[keep])
            if len(pk) < 3 or not a_ref:
                continue
            f = fit_lattice(pk, sg, SEARCH[sg], wavelength=lam, weights=h, ka2=ka2, ka2_min_split=ms)
            diffs.append(abs(f.a - a_ref))
            expl.append(f.matched / f.n_measured)
        dd = np.array(diffs)
        print(f"  {name:<30} n={len(dd)}  median {np.median(dd):.4f} A  within 0.01 A "
              f"{int(np.sum(dd < 0.01))}/{len(dd)}  max {dd.max():.4f} A  peaks explained {np.mean(expl):.0%}")


if __name__ == "__main__":
    simulated()
    measured()
