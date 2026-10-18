"""
Run GTSF Minimal Core on a real Raman spectrum sample.

This script reads the RRUFF quartz Raman processed sample from:
data/sample/quartz_raman_R040031_514_processed.csv

It then computes the minimal GTSF descriptor using:
- spectral entropy
- arc length
- roughness index
- curvature-like energy
- theta-inspired moments
"""


import csv
import logging
from pathlib import Path

from src.gtsf_minimal import build_gtsf_descriptor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_spectrum_csv(file_path):
    x_values = []
    intensities = []

    with file_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x_values.append(float(row["x"]))
            intensities.append(float(row["intensity"]))

    return x_values, intensities


def main():
    sample_path = Path("data/sample/quartz_raman_R040031_514_processed.csv")

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    logger.info("Loading spectrum from %s", sample_path)

    x_values, intensities = load_spectrum_csv(sample_path)
    descriptor = build_gtsf_descriptor(x_values, intensities)

    logger.info("GTSF Minimal Descriptor")
    logger.info("=======================")
    logger.info("Sample: Quartz RRUFF R040031 Raman 514 nm")
    logger.info("Points: %d", len(x_values))

    for key, value in descriptor.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
