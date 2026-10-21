"""
Figure: Ramanujan-theta cell refinement on every measured RRUFF pattern in
data/rruff_xrd_subset (cubic, tetragonal, hexagonal/trigonal).

One row per mineral; each marker is one measured pattern. x = |fitted cell -
cell RRUFF reports for the same sample| in angstrom (log scale). Hollow markers:
patterns where the peak finder returns > 50 "peaks" (noisy background; in this
set they are all Fe- or Mn-bearing minerals, consistent with X-ray fluorescence
of Fe/Mn under Cu K-alpha - a hypothesis, not checked against the instruments).

    python benchmarks/figure_xrd_all.py   ->  benchmarks/out/xrd_all_minerals.png
"""

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.theta_xrd.fit import measured_peaks, fit_lattice
from src.theta_xrd.fit_uniaxial import fit_uniaxial
from src.theta_xrd.ramanujan_xrd import CU_KA1
from benchmarks.validate_xrd_real import SEARCH

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL_A, COL_C = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
FLOOR = 1e-4                                  # |error| below this is drawn at the floor
NOISY = 50                                    # detected peaks above this = noisy pattern


def cell_from_header(header):
    for k, v in header.items():
        if "CELL" in k.upper():
            a = re.search(r"\ba:\s*([\d.]+)", str(v))
            c = re.search(r"\bc:\s*([\d.]+)", str(v))
            return (float(a.group(1)) if a else None, float(c.group(1)) if c else None)
    return None, None


def collect():
    d = np.load(os.path.join(ROOT, "data", "rruff_xrd_subset", "xrd_subset.npz"))
    grid, P = d["grid"].astype(float), d["patterns"].astype(float)
    index = json.load(open(os.path.join(ROOT, "data", "rruff_xrd_subset", "xrd_index.json"), encoding="utf-8"))
    rows = []
    for y, info in zip(P, index):
        a_ref, c_ref = cell_from_header(info["header"])
        x0, x1 = info["x_range"]
        keep = (grid >= x0) & (grid <= x1)
        pk, h = measured_peaks(grid[keep], y[keep])
        # same minimum peak counts as validate_xrd_real.py (3) and _uniaxial.py (4)
        if a_ref is None or len(pk) < (3 if "space_group" in info else 4):
            continue
        if "space_group" in info:
            sg = info["space_group"]
            if sg not in SEARCH:
                continue
            f = fit_lattice(pk, sg, SEARCH[sg], wavelength=CU_KA1, weights=h)
            rows.append(dict(mineral=info["mineral"], system="Cubic", da=abs(f.a - a_ref), dc=None,
                             n_peaks=len(pk)))
        else:
            f = fit_uniaxial(pk, info["system"], info["centering"], a_ref, c_ref, window=0.03, weights=h)
            rows.append(dict(mineral=info["mineral"],
                             system="Tetragonal" if info["system"] == "tetragonal" else "Hexagonal / trigonal",
                             da=abs(f.a - a_ref), dc=abs(f.c - c_ref), n_peaks=len(pk)))
    return rows


def main():
    rows = collect()
    order = ["Cubic", "Tetragonal", "Hexagonal / trigonal"]
    minerals = []
    for s in order:
        minerals += [(s, m) for m in sorted({r["mineral"] for r in rows if r["system"] == s})]
    # y positions: a header row above each crystal-system group
    ypos, header_y, y, last = {}, {}, 0, None
    for s, m in minerals:
        if s != last:
            if last is not None:
                y += 0.4
            header_y[s] = y
            y += 1
        ypos[m] = y
        y += 1
        last = s

    fig, ax = plt.subplots(figsize=(7.2, 7.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(0.01, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.0108, header_y["Cubic"], "0.01 Å criterion", color=INK2, fontsize=8, va="center")

    rng = np.random.default_rng(0)
    for r in rows:
        yy = ypos[r["mineral"]]
        hollow = r["n_peaks"] > NOISY
        for val, col, mk, dy in ((r["da"], COL_A, "o", -0.14), (r["dc"], COL_C, "D", 0.14)):
            if val is None:
                continue
            ax.plot(max(val, FLOOR), yy + dy + rng.uniform(-0.05, 0.05), mk, ms=5.2,
                    mfc="none" if hollow else col, mec=col, mew=1.3, zorder=3)

    ax.set_yticks([ypos[m] for _, m in minerals])
    ax.set_yticklabels([m for _, m in minerals], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, 0.5)
    ax.set_xlabel("|fitted cell − RRUFF cell|  (Å, log scale)", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)

    # group headers, left-aligned at the plot edge
    for s in order:
        n = sum(r["system"] == s for r in rows)
        ax.text(FLOOR * 0.85, header_y[s], f"{s.upper()}  ·  {n} patterns", fontsize=8,
                color=INK2, fontweight="bold", va="center", ha="left")
        if s != order[0]:
            ax.axhline(header_y[s] - 0.7, color=GRID, lw=1)

    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=COL_A, mec=COL_A, label="a axis"),
               Line2D([], [], marker="D", ls="", ms=5.5, mfc=COL_C, mec=COL_C, label="c axis (non-cubic)"),
               Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec=INK2, mew=1.3,
                      label=f"noisy pattern (> {NOISY} detected peaks)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.947), ncol=3, frameon=False,
               fontsize=8, labelcolor=INK, handletextpad=0.3, columnspacing=1.2)

    da = np.array([r["da"] for r in rows])
    dc = np.array([r["dc"] for r in rows if r["dc"] is not None])
    fig.suptitle(f"Ramanujan-theta cell refinement on {len(rows)} measured RRUFF patterns "
                 f"({len(minerals)} minerals)", x=0.02, ha="left", fontsize=11, color=INK, y=0.997)
    fig.text(0.02, 0.951, f"Median |error|: a {np.median(da):.4f} Å, c {np.median(dc):.4f} Å. "
             "Reference = cell RRUFF reports for the same sample. Values below 10⁻⁴ Å drawn at 10⁻⁴.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(ROOT, "benchmarks", "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "xrd_all_minerals.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print(f"{len(rows)} patterns, {len(minerals)} minerals -> {path}")
    noisy = sorted({r['mineral'] for r in rows if r['n_peaks'] > NOISY})
    print("noisy patterns (> %d peaks):" % NOISY, noisy)


if __name__ == "__main__":
    main()
