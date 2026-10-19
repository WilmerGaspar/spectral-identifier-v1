"""
Figure: Ramanujan-theta cell-edge fit on MEASURED RRUFF powder XRD vs RRUFF's own refined cell.
    python benchmarks/figure_xrd_real.py -> benchmarks/out/xrd_real_fit.png
"""
import json, os, re, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.theta_xrd.fit import measured_peaks, fit_lattice
from src.theta_xrd.ramanujan_xrd import CU_KA1
from benchmarks.validate_xrd_real import SEARCH, header_value, first_float

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FAMILY_COLOR = {"Garnets (Ia-3d)": "#2a78d6", "Spinel group (Fd-3m)": "#eb6834", "Other cubic": "#1baf7a"}

def family(sg):
    return "Garnets (Ia-3d)" if sg == "Ia-3d" else ("Spinel group (Fd-3m)" if sg == "Fd-3m" else "Other cubic")

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(ROOT, "data", "rruff_xrd_subset", "xrd_subset.npz"))
    grid, P = d["grid"].astype(float), d["patterns"].astype(float)
    index = json.load(open(os.path.join(ROOT, "data", "rruff_xrd_subset", "xrd_index.json")))
    pts = []
    for y, info in zip(P, index):
        a_ref = first_float(header_value(info["header"], "CELL"))
        lam = first_float(header_value(info["header"], "WAVELENGTH")) or CU_KA1
        pk, h = measured_peaks(grid, y)
        if a_ref is None or len(pk) < 3:
            continue
        fit = fit_lattice(pk, info["space_group"], SEARCH[info["space_group"]], wavelength=lam, weights=h)
        pts.append((info["mineral"], family(info["space_group"]), a_ref, fit.a - a_ref))
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(-0.01, 0.01, color=GRID, alpha=0.6, zorder=0)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    for fam, col in FAMILY_COLOR.items():
        sel = [p for p in pts if p[1] == fam]
        ax.scatter([p[2] for p in sel], [p[3] for p in sel], s=42, color=col, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=f"{fam} ({len(sel)})")
    ax.text(12.15, 0.0105, "±0.01 Å", color=INK2, fontsize=8, ha="right", va="bottom")
    diffs = np.abs([p[3] for p in pts])
    ax.set_title(f"Ramanujan-theta fit on {len(pts)} measured RRUFF XRD patterns: "
                 f"median |error| {np.median(diffs):.4f} Å", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Cell edge refined by RRUFF (Å)", color=INK2)
    ax.set_ylabel("Theta fit − RRUFF (Å)", color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylim(-0.03, 0.03)
    leg = ax.legend(frameon=False, loc="lower left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(0, -0.2, "17 cubic minerals (garnets, spinels, diamond, halite, sylvite, galena, fluorite). "
            "Cell edge fitted from line positions only; RRUFF's value is not used in the fit.",
            transform=ax.transAxes, color=INK2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "benchmarks", "out")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "xrd_real_fit.png"), dpi=140, facecolor=SURFACE)
    print("saved; n =", len(pts), "median", np.median(diffs), "within 0.01:", np.mean(diffs < 0.01))

if __name__ == "__main__":
    main()
