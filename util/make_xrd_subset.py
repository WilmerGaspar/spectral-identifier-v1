"""
Build a small committable subset of MEASURED RRUFF powder XRD patterns of
cubic minerals, for validating the Ramanujan-theta lattice fit.

Run AFTER:
    python util/main.py --dataset powder --only XY_Processed.zip

then:
    python util/make_xrd_subset.py

Uses XY_Processed (measured 2-theta vs intensity). DIF files are NOT used:
they are peak lists derived from the structure refinement, so comparing
against them would not be a test against measurement.

Output: data/rruff_xrd_subset/xrd_subset.npz + xrd_index.json
The index keeps every header field whose name contains WAVELENGTH or CELL,
exactly as written in the RRUFF file, so the benchmark can read them.
"""

import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
PARSED = PROJECT / "data" / "parsed" / "processed"
OUT = PROJECT / "data" / "rruff_xrd_subset"
GRID = np.arange(5.0, 90.0 + 1e-9, 0.02)

# Cubic minerals and their space groups (only groups our fit implements).
CUBIC = {
    "Pyrope": "Ia-3d", "Almandine": "Ia-3d", "Spessartine": "Ia-3d",
    "Grossular": "Ia-3d", "Andradite": "Ia-3d", "Uvarovite": "Ia-3d",
    "Spinel": "Fd-3m", "Magnetite": "Fd-3m", "Chromite": "Fd-3m",
    "Gahnite": "Fd-3m", "Franklinite": "Fd-3m",
    "Diamond": "Fd-3m (diamond, 8a)",
    "Halite": "Fm-3m", "Sylvite": "Fm-3m", "Periclase": "Fm-3m",
    "Galena": "Fm-3m", "Fluorite": "Fm-3m",
}
MAX_PER_MINERAL = 4


def main():
    folders = sorted(p.stem for p in (PARSED / "json").glob("*.json") if "XY" in p.stem.upper())
    if not folders:
        raise SystemExit("No parsed powder folder found. Run: "
                         "python util/main.py --dataset powder --only XY_Processed.zip")
    patterns, index, seen_keys = [], [], set()
    for folder in folders:
        meta = json.loads((PARSED / "json" / f"{folder}.json").read_text(encoding="utf-8"))
        per = {}
        for stem, m in sorted(meta.items()):
            seen_keys.update(m.keys())
            name = (m.get("NAMES") or "").strip()
            if name not in CUBIC or per.get(name, 0) >= MAX_PER_MINERAL:
                continue
            path = PARSED / "csv" / folder / f"{stem}.csv"
            if not path.exists():
                continue
            d = np.genfromtxt(path, delimiter=",", skip_header=1)
            if d.ndim != 2 or len(d) < 200:
                continue
            x, y = d[:, 0], d[:, 1]
            o = np.argsort(x)
            g = np.interp(GRID, x[o], y[o], left=np.nan, right=np.nan)
            g = np.nan_to_num(g - np.nanmin(g), nan=0.0)
            if g.max() <= 0:
                continue
            patterns.append((g / g.max()).astype(np.float32))
            index.append({"mineral": name, "space_group": CUBIC[name],
                          "rruff_id": m.get("RRUFFID", ""), "stem": stem, "folder": folder,
                          "x_range": [float(x.min()), float(x.max())],
                          "header": {k: v for k, v in m.items()
                                     if "WAVELENGTH" in k.upper() or "CELL" in k.upper()}})
            per[name] = per.get(name, 0) + 1
    if not patterns:
        raise SystemExit("No cubic minerals found in the powder data.")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "xrd_subset.npz", grid=GRID.astype(np.float32), patterns=np.stack(patterns))
    (OUT / "xrd_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(patterns)} measured patterns, {len({i['mineral'] for i in index})} cubic minerals "
          f"-> {OUT} ({(OUT / 'xrd_subset.npz').stat().st_size / 1024:.0f} KB)")
    print("Header fields found in the powder files (for reference):", sorted(seen_keys))


if __name__ == "__main__":
    main()
