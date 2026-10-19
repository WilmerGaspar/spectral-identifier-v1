"""
Validation: Raman-based heat capacity / entropy vs calorimetric reference data.

    python benchmarks/validate_thermo.py

Reference values (298.15 K, per formula unit):
  quartz  SiO2   Cp = 44.58 J/(mol K), S = 41.46 J/(mol K)   NIST-JANAF (Chase 1998)
  calcite CaCO3  Cp ~ 83.5  J/(mol K)                         Robie & Hemingway (USGS, 1995)

The model gives harmonic Cv; measured Cp is ~1-2 % higher near room temperature.
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.reporting.thermo.raman_thermo import find_raman_modes, estimate

CASES = [
    # file, atoms per formula unit, Cp_ref, S_ref
    ("quartz_raman_R040031_514_processed.csv", 3, 44.58, 41.46),
    ("calcite_raman_R040170_514_processed.csv", 5, 83.5, None),
]


def main():
    print(f"{'mineral':<9}{'peak thr':>9}{'modes':>7}{'Cv est':>9}{'Cp ref':>8}{'err':>8}"
          f"{'S est':>8}{'S ref':>8}{'err':>8}")
    for fname, n, cp_ref, s_ref in CASES:
        d = np.genfromtxt(os.path.join(ROOT, "data", "sample", fname), delimiter=",", skip_header=1)
        for prom in (0.01, 0.02, 0.05):
            modes = find_raman_modes(d[:, 0], d[:, 1], prominence=prom)
            e = estimate(modes, n)
            s_txt = f"{e.s:8.1f}{s_ref:8.1f}{(e.s - s_ref) / s_ref:8.0%}" if s_ref else f"{e.s:8.1f}{'-':>8}{'-':>8}"
            print(f"{fname.split('_')[0]:<9}{prom:>9}{len(modes):>7}{e.cv:9.1f}{cp_ref:8.1f}"
                  f"{(e.cv - cp_ref) / cp_ref:8.0%}" + s_txt)
    print("\npeak thr = peak-detection prominence threshold (fraction of max intensity).")


if __name__ == "__main__":
    main()
