"""
Case: XRD mixture analysis with a Ramanujan-theta reference library.

  * Reference patterns of four cubic minerals are generated from the
    Ramanujan lattice theta series (positions) + structure factor (intensities).
  * A "measured" powder pattern of an evaporite-like mixture
    (halite NaCl + sylvite KCl + fluorite CaF2) is simulated with a
    DIFFERENT physical model (angle-dependent scattering, zero shift, noise)
    plus quartz, a hexagonal phase that is NOT in the library.
  * The NNLS unmixer recovers the phases and flags quartz from the residual.

IMPORTANT: the measured pattern is simulated. Real validation needs measured
XRD patterns (e.g. RRUFF powder data).

    python benchmarks/demo_xrd_mixture.py
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.theta_xrd.ramanujan_xrd import cubic_profile, cubic_pattern, hexagonal_pattern
from src.mixture.unmixing import MixtureUnmixer

FCC = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]


def rocksalt(Zc, Za):
    return [(x, y, z, Zc) for x, y, z in FCC] + [((x + .5) % 1, y, z, Za) for x, y, z in FCC]


def fluorite(Zc, Za):
    t = [(.25, .25, .25), (.75, .75, .75), (.25, .75, .75), (.75, .25, .25),
         (.75, .25, .75), (.25, .75, .25), (.75, .75, .25), (.25, .25, .75)]
    return [(x, y, z, Zc) for x, y, z in FCC] + [(x, y, z, Za) for x, y, z in t]


# name: (a [A], lattice, basis)   lattice parameters at room temperature
LIBRARY = {
    "halite NaCl":    (5.6402, "F", rocksalt(11, 17)),
    "sylvite KCl":    (6.2931, "F", rocksalt(19, 17)),
    "fluorite CaF2":  (5.4626, "F", fluorite(20, 9)),
    "periclase MgO":  (4.2112, "F", rocksalt(12, 8)),
}


def quartz_lines_profile(grid, fwhm=0.15):
    """Unknown phase: quartz line positions from Borwein a(q) x phi(q).
    3_1/3_2 screw axis: 00l only with l = 3n. Equal intensity per reflection
    (rough: quartz is NOT in the library, we only need its lines to exist)."""
    rule = lambda A, l: not (A == 0 and l % 3 != 0)
    y = np.zeros_like(grid)
    sig = fwhm / 2.3548
    for line in hexagonal_pattern(4.9134, 5.4052, two_theta_max=grid.max(), l_rule=rule):
        y += line.multiplicity * np.exp(-0.5 * ((grid - line.two_theta) / sig) ** 2)
    return y


def norm(y):
    return y / y.max()


def main():
    grid = np.arange(15.0, 80.0, 0.02)
    rng = np.random.default_rng(7)

    unmixer = MixtureUnmixer(grid, peak_snr_threshold=6.0, min_peak_separation_cm=0.5)
    for name, (a, lat, basis) in LIBRARY.items():
        unmixer.add_reference(name, grid, cubic_profile(grid, a, lat, basis))

    true_w = {"halite NaCl": 0.55, "sylvite KCl": 0.30, "fluorite CaF2": 0.15}
    measured = np.zeros_like(grid)
    for name, w in true_w.items():
        a, lat, basis = LIBRARY[name]
        p = cubic_profile(grid, a, lat, basis, b_factor=1.5, zero_shift=0.02, fwhm=0.17)
        measured += w * norm(p)
    measured += 0.12 * norm(quartz_lines_profile(grid, fwhm=0.17))
    measured += 0.01 * rng.standard_normal(len(grid))
    measured = np.maximum(measured, 0)

    res = unmixer.unmix(grid, measured)
    print("True spectral contributions:", {k: f"{v:.0%}" for k, v in true_w.items()},
          "+ unknown quartz")
    print(res.summary().replace("cm-1", "deg 2theta"))

    # Position test: a residual peak is "unknown" only if it does not sit on any
    # line that the Ramanujan theta series predicts for an identified phase.
    tol = 0.25
    known_lines = []
    for name, frac in res.fractions.items():
        if frac >= 0.05:
            a, lat, _ = LIBRARY[name]
            known_lines += [l.two_theta for l in cubic_pattern(a, lat, two_theta_max=grid.max())]
    unknown = [p for p in res.residual_peaks
               if all(abs(p.wavenumber - k) > tol for k in known_lines)]
    explained = len(res.residual_peaks) - len(unknown)
    print(f"\nResidual peaks: {len(res.residual_peaks)} | on predicted lines of identified "
          f"phases (intensity mismatch): {explained} | UNKNOWN: {len(unknown)}")
    for p in unknown:
        print(f"  unknown line at {p.wavenumber:6.2f} deg  (SNR {p.snr:.1f})")

    q_rule = lambda A, l: not (A == 0 and l % 3 != 0)
    quartz_true = [l.two_theta for l in hexagonal_pattern(4.9134, 5.4052, two_theta_max=grid.max(),
                                                          l_rule=q_rule)]
    tp = [p for p in unknown if any(abs(p.wavenumber - q) < tol for q in quartz_true)]
    print(f"Unknown lines that are real quartz lines: {len(tp)} of {len(unknown)}")
    res.residual_peaks = unknown

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(grid, res.unknown, color="#444444", lw=0.8, label="measured mixture (simulated)")
    a1.plot(grid, res.fitted, color="#2a6fdb", lw=1, label="Ramanujan-theta library fit")
    a1.set_ylabel("normalized intensity")
    a1.set_title("XRD: halite + sylvite + fluorite + unknown phase")
    a1.legend(frameon=False)
    a2.plot(grid, res.residual, color="#888888", lw=0.7)
    a2.axhline(6 * res.noise_level, color="#d1495b", ls="--", lw=0.8, label="alert threshold")
    for p in res.residual_peaks[:4]:
        a2.annotate(f"{p.wavenumber:.2f}°", (p.wavenumber, p.height), textcoords="offset points",
                    xytext=(4, 4), color="#d1495b", fontsize=8)
    a2.set_xlabel(r"2$\theta$ (deg, Cu K$\alpha_1$)")
    a2.set_ylabel("residual")
    a2.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(ROOT, "benchmarks", "out")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "xrd_mixture.png"), dpi=130)
    print("plot saved: benchmarks/out/xrd_mixture.png")


if __name__ == "__main__":
    main()
