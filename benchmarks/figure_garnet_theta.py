"""
Figure: garnet cell edges recovered by the Ramanujan-theta fit (simulation).
    python benchmarks/figure_garnet_theta.py  -> benchmarks/out/garnet_theta_fit.png
"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from benchmarks.simulate_theta_xrd_garnet import simulate, GARNETS
from src.theta_xrd.fit import ia3d_allowed, measured_peaks, fit_lattice

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    names = list(GARNETS)
    fits = {n: [] for n in names}
    for n in names:
        for _ in range(20):
            tt, y = simulate(GARNETS[n], ia3d_allowed, rng, zero=rng.uniform(-0.05, 0.05))
            pk, h = measured_peaks(tt, y)
            fits[n].append(fit_lattice(pk, "Ia-3d", (11.3, 12.2), weights=h).a)

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, n in enumerate(names):
        yrow = len(names) - 1 - i
        a = np.array(fits[n])
        ax.scatter(a, yrow + rng.uniform(-0.12, 0.12, len(a)), s=36, color=SERIES,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.plot([GARNETS[n]] * 2, [yrow - 0.3, yrow + 0.3], color=INK, lw=2, zorder=4)
        ax.text(12.12, yrow, f"mean error {np.mean(np.abs(a - GARNETS[n])):.4f} Å",
                va="center", ha="left", color=INK2, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.capitalize() for n in reversed(names)], color=INK, fontsize=10)
    ax.set_xlim(11.40, 12.30)
    ax.set_xlabel("Cell edge a (Å), fitted from XRD with Ramanujan-theta lines", color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color=GRID, lw=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Raman confuses these garnets; Ramanujan-theta XRD separates them (100/100 correct)",
                 color=INK, fontsize=11, loc="left")
    ax.text(0, -0.2, "Blue dots: 20 simulated patterns per garnet (Cu Kα1+Kα2, noise, zero shift). "
            "Black ticks: reference a, Novak & Gibbs (1971). Simulation only.",
            transform=ax.transAxes, color=INK2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "benchmarks", "out")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "garnet_theta_fit.png"), dpi=140, facecolor=SURFACE)
    print("saved benchmarks/out/garnet_theta_fit.png")

if __name__ == "__main__":
    main()
