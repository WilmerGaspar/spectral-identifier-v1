"""
GTSF Minimal Core
Geometric Theta Spectral Framework - lightweight prototype.

This file adds a small research prototype for converting a spectrum
into interpretable mathematical descriptors.

It does not claim experimental validation.
It is intended as a lightweight base for future GTSF development.
"""

import math
from statistics import mean, pstdev


def normalize_spectrum(intensities):
    """
    Normalize intensity values to the range 0-1.
    """
    if not intensities:
        return []

    min_val = min(intensities)
    max_val = max(intensities)

    if max_val == min_val:
        return [0.0 for _ in intensities]

    return [(value - min_val) / (max_val - min_val) for value in intensities]


def spectral_entropy(intensities):
    """
    Compute a simple Shannon-like entropy from intensity values.
    """
    if not intensities:
        return 0.0

    total = sum(abs(value) for value in intensities)

    if total == 0:
        return 0.0

    probabilities = [
        abs(value) / total
        for value in intensities
        if value != 0
    ]

    return -sum(probability * math.log(probability) for probability in probabilities)


def arc_length(x_values, y_values):
    """
    Compute the geometric arc length of a spectrum curve.
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")

    if len(x_values) < 2:
        return 0.0

    length = 0.0

    for index in range(1, len(x_values)):
        dx = x_values[index] - x_values[index - 1]
        dy = y_values[index] - y_values[index - 1]
        length += math.sqrt(dx * dx + dy * dy)

    return length


def roughness_index(y_values):
    """
    Compute a simple roughness index using first differences.
    """
    if len(y_values) < 2:
        return 0.0

    differences = [
        abs(y_values[index] - y_values[index - 1])
        for index in range(1, len(y_values))
    ]

    return mean(differences)


def curvature_energy(y_values):
    """
    Compute a lightweight curvature-like energy using second differences.

    This is not a full differential geometry model.
    It is a first prototype descriptor for curve bending.
    """
    if len(y_values) < 3:
        return 0.0

    second_differences = []

    for index in range(1, len(y_values) - 1):
        value = y_values[index + 1] - 2 * y_values[index] + y_values[index - 1]
        second_differences.append(value * value)

    return mean(second_differences)


def theta_like_moments(y_values):
    """
    Compute simple theta-inspired statistical moments.

    This is not a full Jacobi theta implementation.
    It is a lightweight placeholder for early GTSF descriptors.
    """
    if not y_values:
        return {
            "theta_mean": 0.0,
            "theta_std": 0.0,
            "theta_energy": 0.0,
        }

    return {
        "theta_mean": mean(y_values),
        "theta_std": pstdev(y_values) if len(y_values) > 1 else 0.0,
        "theta_energy": sum(value * value for value in y_values) / len(y_values),
    }


def build_gtsf_descriptor(x_values, intensities):
    """
    Build a minimal GTSF descriptor from raw spectral values.
    """
    normalized = normalize_spectrum(intensities)

    descriptor = {
        "points": len(normalized),
        "spectral_entropy": spectral_entropy(normalized),
        "arc_length": arc_length(x_values, normalized),
        "roughness_index": roughness_index(normalized),
        "curvature_energy": curvature_energy(normalized),
    }

    descriptor.update(theta_like_moments(normalized))

    return descriptor


if __name__ == "__main__":
    demo_x = [100, 200, 300, 400, 500, 600]
    demo_y = [10, 30, 80, 40, 20, 15]

    result = build_gtsf_descriptor(demo_x, demo_y)

    for key, value in result.items():
        print(f"{key}: {value}")
