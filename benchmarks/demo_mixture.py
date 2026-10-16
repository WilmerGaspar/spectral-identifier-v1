"""
Demo: unmix a quartz + calcite mixture (real RRUFF spectra) and detect
an impurity that is NOT in the reference library.

    python benchmarks/demo_mixture.py
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.mixture.unmixing import MixtureUnmixer


def load_csv(name):
    data = np.genfromtxt(os.path.join(ROOT, "data", "sample", name), delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1]


def gaussian(x, center, fwhm, height):
    s = fwhm / 2.355
    return height * np.exp(-((x - center) ** 2) / (2 * s * s))


def make_mixture(grid, quartz, calcite, wq, wc, impurity=None, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    q = np.interp(grid, *quartz); q /= q.max()
    c = np.interp(grid, *calcite); c /= c.max()
    y = wq * q + wc * c
    if impurity is not None:
        y += impurity
    y += noise * rng.standard_normal(len(grid))
    return np.maximum(y, 0)


def run_case(title, unmixer, grid, y, plot_path=None):
    print(f"\n=== {title} ===")
    res = unmixer.unmix(grid, y)
    print(res.summary())
    if plot_path:
        plot(res, title, plot_path)
    return res


def plot(res, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(res.grid, res.unknown, color="#444444", lw=1, label="unknown spectrum")
    a1.plot(res.grid, res.fitted, color="#2a6fdb", lw=1.2, label="best library fit (NNLS)")
    a1.set_ylabel("normalized intensity")
    a1.set_title(title)
    a1.legend(frameon=False)
    a2.plot(res.grid, res.residual, color="#888888", lw=0.8)
    a2.axhline(6 * res.noise_level, color="#d1495b", ls="--", lw=0.8, label="alert threshold")
    for p in res.residual_peaks:
        a2.annotate(f"{p.wavenumber:.0f} cm$^{{-1}}$", (p.wavenumber, p.height),
                    textcoords="offset points", xytext=(5, 5), color="#d1495b")
    a2.set_ylabel("residual")
    a2.set_xlabel("Raman shift (cm$^{-1}$)")
    a2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"plot saved: {path}")


def main():
    quartz = load_csv("quartz_raman_R040031_514_processed.csv")
    calcite = load_csv("calcite_raman_R040170_514_processed.csv")
    grid = np.arange(150, 1500, 1.0)
    out = os.path.join(ROOT, "benchmarks", "out")
    os.makedirs(out, exist_ok=True)

    lib = MixtureUnmixer(grid)
    lib.add_reference("quartz (R040031)", *quartz)
    lib.add_reference("calcite (R040170)", *calcite)

    # 1) Clean mixture: should recover 70/30 and raise no alert.
    y1 = make_mixture(grid, quartz, calcite, 0.7, 0.3)
    run_case("Case 1: 70% quartz + 30% calcite (spectral)", lib, grid, y1,
             os.path.join(out, "case1_clean_mixture.png"))

    # 2) Same mixture + an unknown phase (sulfate-like band at 1008 cm-1,
    #    deliberately NOT in the library). Should be flagged.
    imp = gaussian(grid, 1008, 8, 0.12)
    y2 = make_mixture(grid, quartz, calcite, 0.7, 0.3, impurity=imp, seed=1)
    run_case("Case 2: same mixture + unknown phase at 1008 cm-1", lib, grid, y2,
             os.path.join(out, "case2_unknown_phase.png"))

    # 3) Sensitivity: smallest impurity that still triggers the alert.
    print("\n=== Case 3: detection limit of the unknown band ===")
    for h in [0.10, 0.06, 0.04, 0.03, 0.02, 0.01]:
        hits = 0
        for s in range(20):
            y = make_mixture(grid, quartz, calcite, 0.7, 0.3,
                             impurity=gaussian(grid, 1008, 8, h), seed=100 + s)
            r = lib.unmix(grid, y)
            hits += any(abs(p.wavenumber - 1008) < 6 for p in r.residual_peaks)
        print(f"  band height {h:.2f} (noise 0.01): detected in {hits}/20 runs")

    # 4) False alarms on clean mixtures with varied proportions.
    fa = 0
    for s in range(50):
        wq = np.random.default_rng(s).uniform(0.1, 0.9)
        fa += lib.unmix(grid, make_mixture(grid, quartz, calcite, wq, 1 - wq, seed=500 + s)).anomaly
    print(f"\n=== Case 4: false alarms on 50 clean mixtures: {fa}/50 ===")


if __name__ == "__main__":
    main()
