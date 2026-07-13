"""
Compare GTSF Minimal descriptors between two real Raman samples.

Samples:
- Quartz RRUFF R040031 Raman 514 nm
- Calcite RRUFF R040170 Raman 514 nm
"""

import csv
import math
from pathlib import Path

from src.gtsf_minimal import build_gtsf_descriptor


SAMPLES = {
    "quartz_R040031_514": Path(
        "data/sample/quartz_raman_R040031_514_processed.csv"
    ),
    "calcite_R040170_514": Path(
        "data/sample/calcite_raman_R040170_514_processed.csv"
    ),
}


def load_spectrum_csv(file_path):
    x_values = []
    intensities = []

    with file_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x_values.append(float(row["x"]))
            intensities.append(float(row["intensity"]))

    return x_values, intensities


def descriptor_distance(descriptor_a, descriptor_b):
    """
    Compute a simple Euclidean distance between shared numeric descriptor values.
    """
    shared_keys = sorted(set(descriptor_a.keys()) & set(descriptor_b.keys()))

    squared_sum = 0.0

    for key in shared_keys:
        value_a = descriptor_a[key]
        value_b = descriptor_b[key]

        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            squared_sum += (value_a - value_b) ** 2

    return math.sqrt(squared_sum)


def main():
    descriptors = {}

    for sample_name, sample_path in SAMPLES.items():
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        x_values, intensities = load_spectrum_csv(sample_path)
        descriptors[sample_name] = build_gtsf_descriptor(x_values, intensities)

    quartz = descriptors["quartz_R040031_514"]
    calcite = descriptors["calcite_R040170_514"]

    print("GTSF Minimal Descriptor Comparison")
    print("==================================")
    print("Samples: Quartz R040031 vs Calcite R040170")
    print()

    keys = sorted(set(quartz.keys()) & set(calcite.keys()))

    for key in keys:
        print(f"{key}")
        print(f"  quartz : {quartz[key]}")
        print(f"  calcite: {calcite[key]}")
        print(f"  diff   : {abs(quartz[key] - calcite[key])}")
        print()

    distance = descriptor_distance(quartz, calcite)

    print("Overall descriptor distance")
    print("---------------------------")
    print(distance)


if __name__ == "__main__":
    main()
