"""
Raman -> Vibrational Thermodynamics (simplified Kieffer-type model)
===================================================================
Estimates heat capacity Cv(T) and vibrational entropy S(T) of a mineral
from the peak positions of its Raman spectrum.

Physics
-------
A crystal with n atoms per formula unit has 3n vibrational modes per
formula unit. Each mode of wavenumber nu behaves (harmonically) like an
oscillator with partition function

    Z(q) = q^(1/2) / (1 - q),     q = exp(-h c nu / k T)

so the thermodynamics is a sum of q-series terms over the mode spectrum.
Following the idea of Kieffer (1979, Rev. Geophys. Space Phys.), the
3n modes are split into:
  * 3 acoustic modes  -> Debye spectrum up to a cutoff nu_ac
  * 3n - 3 optic modes -> placed at the observed Raman frequencies

LIMITATIONS (state these whenever results are shown)
  * Raman does not see every mode: IR-only and silent modes are missing,
    and Raman intensities do not give mode counts. Optic modes are
    therefore distributed over the observed peaks with equal weight.
  * Kieffer used elastic data for the acoustic branch; here the cutoff
    is an assumption (default: lowest observed Raman peak).
  * Harmonic model: gives Cv. Measured Cp is ~1-2 % higher near 300 K.
This is a fast spectroscopic estimate, not a replacement for calorimetry.

Author: Wilmer Gaspar Espinoza Castillo
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.signal import find_peaks

R = 8.314462618          # J/(mol K)
HC_OVER_K = 1.438776877  # cm K  (second radiation constant)


def einstein_cv(nu_cm, T):
    """Heat capacity of one mode (in units of R)."""
    x = HC_OVER_K * np.asarray(nu_cm, float) / T
    x = np.clip(x, 1e-9, 700)
    ex = np.exp(x)
    return x * x * ex / (ex - 1.0) ** 2


def einstein_s(nu_cm, T):
    """Entropy of one mode (in units of R): x/(e^x-1) - ln(1-e^-x)."""
    x = HC_OVER_K * np.asarray(nu_cm, float) / T
    x = np.clip(x, 1e-9, 700)
    return x / np.expm1(x) - np.log(-np.expm1(-x))


def debye_cv(nu_cut, T):
    """Debye heat capacity per mode (units of R) with cutoff nu_cut (cm-1)."""
    xd = HC_OVER_K * nu_cut / T
    f = lambda x: x ** 4 * np.exp(x) / np.expm1(x) ** 2
    return 3.0 / xd ** 3 * quad(f, 1e-9, xd)[0]


def debye_s(nu_cut, T):
    """Debye entropy per mode (units of R)."""
    xd = HC_OVER_K * nu_cut / T
    f = lambda x: x ** 3 / np.expm1(x)
    return 4.0 / xd ** 3 * quad(f, 1e-9, xd)[0] - np.log(-np.expm1(-xd))


def find_raman_modes(wavenumbers, intensities, min_cm=60.0, max_cm=1800.0,
                     prominence=0.02) -> np.ndarray:
    """Peak positions (cm-1) of a Raman spectrum, intensity normalized to max = 1."""
    wn = np.asarray(wavenumbers, float)
    y = np.asarray(intensities, float)
    order = np.argsort(wn)
    wn, y = wn[order], y[order]
    keep = (wn >= min_cm) & (wn <= max_cm)
    wn, y = wn[keep], y[keep]
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    idx, _ = find_peaks(y, prominence=prominence)
    return wn[idx]


@dataclass
class ThermoEstimate:
    T: float
    cv: float                  # J/(mol K), per formula unit
    s: float                   # J/(mol K)
    cv_acoustic: float
    cv_optic: float
    modes_cm: List[float]
    n_atoms: int
    acoustic_cutoff_cm: float


def estimate(modes_cm: Sequence[float], n_atoms: int, T: float = 298.15,
             acoustic_cutoff_cm: Optional[float] = None) -> ThermoEstimate:
    """
    modes_cm: observed Raman peak positions (cm-1)
    n_atoms:  atoms per formula unit (quartz SiO2 = 3, calcite CaCO3 = 5)
    """
    modes = np.sort(np.asarray(modes_cm, float))
    if len(modes) == 0:
        raise ValueError("No Raman modes given.")
    if n_atoms < 2:
        raise ValueError("Need at least 2 atoms per formula unit.")

    nu_ac = float(acoustic_cutoff_cm) if acoustic_cutoff_cm else float(modes[0])
    n_optic = 3 * n_atoms - 3
    w = n_optic / len(modes)  # equal weight per observed peak

    cv_ac = 3 * debye_cv(nu_ac, T)
    s_ac = 3 * debye_s(nu_ac, T)
    cv_op = w * float(np.sum(einstein_cv(modes, T)))
    s_op = w * float(np.sum(einstein_s(modes, T)))

    return ThermoEstimate(
        T=T, cv=R * (cv_ac + cv_op), s=R * (s_ac + s_op),
        cv_acoustic=R * cv_ac, cv_optic=R * cv_op,
        modes_cm=[float(m) for m in modes], n_atoms=n_atoms,
        acoustic_cutoff_cm=nu_ac,
    )
