from src.gtsf_minimal import (
    normalize_spectrum,
    spectral_entropy,
    arc_length,
    roughness_index,
    curvature_energy,
    theta_like_moments,
    build_gtsf_descriptor,
)


def test_normalize_spectrum_range():
    values = [10, 20, 30]
    result = normalize_spectrum(values)

    assert result == [0.0, 0.5, 1.0]


def test_spectral_entropy_positive():
    values = [0.2, 0.3, 0.5]
    result = spectral_entropy(values)

    assert result > 0


def test_arc_length_basic():
    x_values = [0, 1, 2]
    y_values = [0, 1, 0]

    result = arc_length(x_values, y_values)

    assert result > 0


def test_roughness_index_basic():
    values = [0.0, 0.5, 1.0]
    result = roughness_index(values)

    assert result == 0.5


def test_curvature_energy_basic():
    values = [0.0, 1.0, 0.0]
    result = curvature_energy(values)

    assert result > 0


def test_theta_like_moments_keys():
    values = [0.0, 0.5, 1.0]
    result = theta_like_moments(values)

    assert "theta_mean" in result
    assert "theta_std" in result
    assert "theta_energy" in result


def test_build_gtsf_descriptor_keys():
    x_values = [100, 200, 300, 400]
    intensities = [10, 40, 20, 30]

    result = build_gtsf_descriptor(x_values, intensities)

    assert "points" in result
    assert "spectral_entropy" in result
    assert "arc_length" in result
    assert "roughness_index" in result
    assert "curvature_energy" in result
    assert "theta_mean" in result
    assert "theta_std" in result
    assert "theta_energy" in result
