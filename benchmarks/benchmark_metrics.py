"""
Benchmark: does each similarity metric identify the right material on its own?

Builds a reference library, then creates perturbed "unknown" spectra
(noise, wavenumber shift, baseline slope, intensity jitter) and checks
whether each metric ranks the correct reference first.

Usage:
    python benchmarks/benchmark_metrics.py              # synthetic library
    python benchmarks/benchmark_metrics.py data/rruff   # folder of RRUFF .txt files
"""

import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

from src.pipeline import SpectralIdentifierPipeline
from src.data_loader import DataLoader

METRICS = ["cosine", "sam", "theta", "peak", "combined"]


def perturb(wn, inten, rng, noise=0.03, shift=2.0, slope=0.15, jitter=0.10):
    """Realistic measurement differences between two runs of the same material."""
    y = inten / (np.max(inten) + 1e-12)
    y = np.interp(wn - rng.uniform(-shift, shift), wn, y)          # calibration shift
    y = y * (1 + jitter * rng.standard_normal(len(y))).clip(0.5)  # intensity jitter
    y = y + slope * rng.uniform(0, 1) * (wn - wn[0]) / (wn[-1] - wn[0])  # baseline
    y = y + noise * rng.standard_normal(len(y))                     # detector noise
    return np.maximum(y, 0)


def rank_all(pipe, wn, inten):
    """Score the unknown against every reference with every metric."""
    pre = pipe.preprocessor.process(wn, inten)
    feats = pipe.theta_engine.transform(pre.intensities)
    peaks = pipe.peak_extractor.extract(
        pre.wavenumbers, pre.intensities, theta_intensities=feats.theta_spectrum
    ).peaks
    rows = []
    for ref in pipe.database:
        s = pipe.similarity_engine.combined_score(
            pre.intensities, ref["spectrum"], feats, ref["features"], peaks, ref["peaks"]
        )
        rows.append((ref["name"], s))
    return rows


def main():
    rng = np.random.default_rng(0)
    if len(sys.argv) > 1:
        library = DataLoader.load_rruff_directory(sys.argv[1])
        wn_range = (100, 1500)
    else:
        library = DataLoader.generate_test_database()
        wn_range = (100, 1200)

    pipe = SpectralIdentifierPipeline(config={"wavenumber_range": wn_range})
    pipe.load_database(library)

    n_per_ref = 20
    hits = {m: 0 for m in METRICS}
    margins = {m: [] for m in METRICS}
    total = 0

    for entry in library:
        for _ in range(n_per_ref):
            y = perturb(np.asarray(entry["wavenumbers"]), np.asarray(entry["intensities"]), rng)
            rows = rank_all(pipe, np.asarray(entry["wavenumbers"]), y)
            total += 1
            for m in METRICS:
                scores = [(name, s[m]) for name, s in rows]
                scores.sort(key=lambda t: t[1], reverse=True)
                if scores[0][0] == entry["name"]:
                    hits[m] += 1
                correct = next(v for n, v in scores if n == entry["name"])
                best_wrong = max(v for n, v in scores if n != entry["name"])
                margins[m].append(correct - best_wrong)

    print(f"\nReferences: {len(library)} | perturbed queries: {total}\n")
    print(f"{'metric':<10}{'top-1 acc':>11}{'mean margin':>14}")
    for m in METRICS:
        print(f"{m:<10}{hits[m] / total:>11.1%}{np.mean(margins[m]):>14.3f}")
    print("\nmargin = score(correct) - score(best wrong). Bigger = clearer separation.")


if __name__ == "__main__":
    main()
