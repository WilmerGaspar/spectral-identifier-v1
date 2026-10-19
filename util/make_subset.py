"""
Build a small, committable RRUFF Raman subset from the parsed pipeline output.

Run AFTER util/main.py (needs data/parsed/processed/...):

    python util/make_subset.py                       # defaults below
    python util/make_subset.py --minerals 40 --per-mineral 3

Selection
  * processed spectra only, from one quality folder (default excellent_unoriented)
  * one laser wavelength (default 532 nm; 514 also common in RRUFF)
  * minerals with at least 2 DIFFERENT RRUFF samples (needed for a fair
    test: the query sample is never in the library)
  * each spectrum resampled to a common grid (150-1500 cm-1, 2 cm-1 step),
    scaled to max = 1, stored as float32

Output: data/rruff_subset/raman_subset.npz (a few hundred KB) + subset_index.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
# New layout: data/parsed/<dataset>/processed (older runs wrote data/parsed/processed)
_NEW = PROJECT / "data" / "parsed" / "raman" / "processed"
PARSED = _NEW if _NEW.exists() else PROJECT / "data" / "parsed" / "processed"
OUT = PROJECT / "data" / "rruff_subset"
GRID = np.arange(150.0, 1500.0 + 1e-9, 2.0)


def load_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape[0] < 50:
        return None
    x, y = data[:, 0], data[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    return x[order], y[order]


def to_grid(x, y):
    if x.min() > GRID[0] + 50 or x.max() < GRID[-1] - 300:
        return None                     # does not cover most of the grid
    g = np.interp(GRID, x, y, left=np.nan, right=np.nan)
    g = np.nan_to_num(g - np.nanmin(g), nan=0.0)
    m = g.max()
    return (g / m).astype(np.float32) if m > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="excellent_unoriented")
    ap.add_argument("--wavelength", default="532")
    ap.add_argument("--minerals", type=int, default=40)
    ap.add_argument("--per-mineral", type=int, default=3)
    args = ap.parse_args()

    meta_file = PARSED / "json" / f"{args.folder}.json"
    csv_dir = PARSED / "csv" / args.folder
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    by_mineral = defaultdict(dict)        # mineral -> {rruff_id: stem}
    for stem, m in meta.items():
        name = (m.get("NAMES") or "").strip()
        rid = (m.get("RRUFFID") or "").strip()
        wl = str(m.get("RAMAN WAVELENGTH") or "").strip()
        if name and rid and wl == args.wavelength and (csv_dir / f"{stem}.csv").exists():
            by_mineral[name].setdefault(rid, stem)   # one spectrum per sample

    eligible = sorted((n for n, s in by_mineral.items() if len(s) >= 2),
                      key=lambda n: (-len(by_mineral[n]), n))

    spectra, rows = [], []
    for name in eligible:
        if len({r["mineral"] for r in rows}) >= args.minerals:
            break
        kept = []
        for rid, stem in sorted(by_mineral[name].items()):
            xy = load_csv(csv_dir / f"{stem}.csv")
            g = to_grid(*xy) if xy else None
            if g is not None:
                kept.append((rid, stem, g))
            if len(kept) >= args.per_mineral:
                break
        if len(kept) >= 2:
            for rid, stem, g in kept:
                spectra.append(g)
                rows.append({"mineral": name, "rruff_id": rid, "stem": stem,
                             "chemistry": meta[stem].get("IDEAL CHEMISTRY", "")})

    if not rows:
        raise SystemExit("No eligible spectra found - check --folder / --wavelength.")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "raman_subset.npz", grid=GRID.astype(np.float32),
                        spectra=np.stack(spectra),
                        mineral=np.array([r["mineral"] for r in rows]),
                        rruff_id=np.array([r["rruff_id"] for r in rows]))
    with open(OUT / "subset_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    size_kb = (OUT / "raman_subset.npz").stat().st_size / 1024
    print(f"{len(rows)} spectra, {len({r['mineral'] for r in rows})} minerals, "
          f"{args.folder}, {args.wavelength} nm -> {OUT / 'raman_subset.npz'} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
