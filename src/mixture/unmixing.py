"""
Mixture Unmixing + Residual Anomaly Detection
=============================================
Decomposes an unknown spectrum as a non-negative linear combination of
reference spectra (Non-Negative Least Squares), then inspects what the
references cannot explain (the residual) for localized peaks that point
to a phase missing from the library.

Physics: Raman/FTIR intensities from a physical mixture add linearly
(superposition), so  y(x) ~ sum_k w_k * s_k(x) + noise,  with w_k >= 0.

IMPORTANT: the weights are SPECTRAL contributions, not mass or volume
fractions. Each mineral scatters with a different Raman cross-section,
so converting weights to composition needs calibration with known mixtures.

Author: Wilmer Gaspar Espinoza Castillo
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks, savgol_filter


@dataclass
class ResidualPeak:
    wavenumber: float
    height: float          # in units of the normalized unknown spectrum
    snr: float             # height / estimated noise level


@dataclass
class UnmixingResult:
    weights: Dict[str, float]            # raw NNLS weights
    fractions: Dict[str, float]          # weights normalized to sum 1
    fit_quality_r2: float
    residual_ratio: float                # ||r|| / ||y||
    noise_level: float
    residual_peaks: List[ResidualPeak] = field(default_factory=list)
    anomaly: bool = False
    confirmed_phases: List[str] = field(default_factory=list)
    grid: Optional[np.ndarray] = None
    unknown: Optional[np.ndarray] = None
    fitted: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = ["Estimated spectral contributions (not mass fractions):"]
        for name, frac in sorted(self.fractions.items(), key=lambda t: -t[1]):
            if frac >= 0.005:
                lines.append(f"  {name:<20} {frac:6.1%}")
        lines.append(f"Fit R^2: {self.fit_quality_r2:.4f} | residual/signal: {self.residual_ratio:.3f}")
        if self.anomaly:
            lines.append("ANOMALY: residual contains structured peaks not explained by the library:")
            for p in self.residual_peaks:
                lines.append(f"  ~{p.wavenumber:7.1f} cm-1  height {p.height:.3f}  SNR {p.snr:.1f}")
        else:
            lines.append("No unexplained peaks above the noise threshold.")
        return "\n".join(lines)


def _to_grid(wn, inten, grid):
    """Resample onto a common grid and scale to max = 1."""
    wn = np.asarray(wn, float)
    inten = np.asarray(inten, float)
    order = np.argsort(wn)
    y = np.interp(grid, wn[order], inten[order], left=0.0, right=0.0)
    y = np.maximum(y, 0.0)
    m = y.max()
    return y / m if m > 0 else y


def _noise_level(y):
    """Robust noise estimate from the high-frequency part of the signal (MAD)."""
    smooth = savgol_filter(y, 11, 3) if len(y) > 11 else y
    hf = y - smooth
    return 1.4826 * np.median(np.abs(hf - np.median(hf))) + 1e-12


class MixtureUnmixer:
    """
    Usage:
        unmixer = MixtureUnmixer(grid=np.arange(150, 1500, 1.0))
        unmixer.add_reference("quartz", wn, intensity)
        unmixer.add_reference("calcite", wn, intensity)
        result = unmixer.unmix(wn_unknown, intensity_unknown)
        print(result.summary())
    """

    def __init__(self, grid: Sequence[float], peak_snr_threshold: float = 6.0,
                 min_peak_separation_cm: float = 8.0,
                 known_peak_tolerance_cm: Optional[float] = None,
                 min_phase_fraction: float = 0.05,
                 shift_tolerant: bool = False):
        """
        known_peak_tolerance_cm: if set, a residual peak within this distance of
        a peak of an identified phase (fraction >= min_phase_fraction) is treated
        as sample-to-sample variation of a known phase, not as an unknown phase.
        Needed when the library comes from OTHER samples than the measured one.
        shift_tolerant: add +/- first and second derivatives of each reference as
        extra non-negative columns, so small peak shifts and width changes between
        samples are absorbed (first/second-order Taylor expansion of the reference).
        Phase contributions are still read from the main columns only.
        """
        self.grid = np.asarray(grid, float)
        self.snr_threshold = peak_snr_threshold
        self.min_sep = min_peak_separation_cm
        self.known_tol = known_peak_tolerance_cm
        self.shift_tolerant = shift_tolerant
        self.min_phase_fraction = min_phase_fraction
        self.names: List[str] = []
        self._refs: List[np.ndarray] = []
        self._ref_peaks: List[np.ndarray] = []
        self._ref_heights: List[np.ndarray] = []

    def add_reference(self, name: str, wavenumbers, intensities):
        self.names.append(name)
        ref = _to_grid(wavenumbers, intensities, self.grid)
        self._refs.append(ref)
        # Reference peaks: smoothed, and only clear peaks (>= 5 % of max) count,
        # so noise ripples are never mistaken for known peak positions.
        ref_s = savgol_filter(ref, 11, 3) if len(ref) > 11 else ref
        idx, _ = find_peaks(ref_s, prominence=0.05)
        self._ref_peaks.append(self.grid[idx])
        self._ref_heights.append(ref_s[idx])

    def unmix(self, wavenumbers, intensities) -> UnmixingResult:
        if not self._refs:
            raise ValueError("Add at least one reference first.")

        y = _to_grid(wavenumbers, intensities, self.grid)
        M = np.column_stack(self._refs)
        k = M.shape[1]
        if self.shift_tolerant:
            d1 = np.gradient(M, axis=0)
            d2 = np.gradient(d1, axis=0)
            M_full = np.column_stack([M, d1, -d1, d2, -d2])
        else:
            M_full = M
        w_full, _ = nnls(M_full, y)
        w = w_full[:k]

        fitted = M_full @ w_full
        r = y - fitted
        ss_res = float(np.sum(r ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        noise = _noise_level(y)

        # Look for localized positive structure in the (lightly smoothed) residual.
        r_s = savgol_filter(r, 9, 3) if len(r) > 9 else r
        step = float(np.median(np.diff(self.grid)))
        idx, props = find_peaks(r_s, height=self.snr_threshold * noise,
                                distance=max(1, int(self.min_sep / step)))
        peaks = [ResidualPeak(float(self.grid[i]), float(r_s[i]), float(r_s[i] / noise))
                 for i in idx]
        peaks.sort(key=lambda p: -p.height)

        total = float(w.sum()) or 1.0
        confirmed = []
        if self.known_tol is not None:
            # A phase is confirmed only if most of its strongest reference peaks
            # are actually present in the measured spectrum. Phases recruited by
            # the fit to absorb an unknown peak fail this and do not shield it.
            y_s = savgol_filter(y, 11, 3) if len(y) > 11 else y
            meas_idx, _ = find_peaks(y_s, prominence=0.03)
            meas = self.grid[meas_idx]
            known_list = []
            for name, pk, hk, wi in zip(self.names, self._ref_peaks, self._ref_heights, w):
                if wi / total < self.min_phase_fraction or pk.size == 0:
                    continue
                top = pk[np.argsort(-hk)[:5]]
                cover = np.mean([meas.size > 0 and np.min(np.abs(meas - t)) <= self.known_tol
                                 for t in top])
                if cover >= 0.6:
                    confirmed.append(name)
                    known_list.append(pk)
            known = np.concatenate(known_list) if known_list else np.array([])
            peaks = [p for p in peaks
                     if known.size == 0 or np.min(np.abs(known - p.wavenumber)) > self.known_tol]
        return UnmixingResult(
            weights={n: float(v) for n, v in zip(self.names, w)},
            fractions={n: float(v) / total for n, v in zip(self.names, w)},
            fit_quality_r2=1 - ss_res / ss_tot,
            residual_ratio=float(np.linalg.norm(r) / (np.linalg.norm(y) + 1e-12)),
            noise_level=noise,
            residual_peaks=peaks,
            anomaly=len(peaks) > 0,
            confirmed_phases=confirmed,
            grid=self.grid, unknown=y, fitted=fitted, residual=r,
        )
