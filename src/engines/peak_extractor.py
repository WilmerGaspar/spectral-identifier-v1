"""
Peak Extraction Engine
======================
Extracción de picos, shifts e intensidades sobre representación theta.
Incluye caracterización de forma de pico y clasificación.

Autor: Spectral Identifier Team
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings


@dataclass
class Peak:
    """Representación de un pico espectral."""
    position: float           # Posición en wavenumbers
    intensity: float          # Intensidad normalizada [0,1]
    width: float              # Ancho a media altura (FWHM)
    prominence: float         # Prominencia relativa
    area: float               # Área bajo el pico
    shape: str                # 'gaussian', 'lorentzian', 'asymmetric'
    asymmetry: float          # Factor de asimetría
    theta_signature: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {
            'position': float(self.position),
            'intensity': float(self.intensity),
            'width': float(self.width),
            'prominence': float(self.prominence),
            'area': float(self.area),
            'shape': self.shape,
            'asymmetry': float(self.asymmetry)
        }


@dataclass
class PeakExtractionResult:
    """Resultado completo de extracción de picos."""
    peaks: List[Peak]
    wavenumbers: np.ndarray
    intensities: np.ndarray
    theta_intensities: np.ndarray
    extraction_params: dict

    def get_peak_positions(self) -> np.ndarray:
        return np.array([p.position for p in self.peaks])

    def get_peak_intensities(self) -> np.ndarray:
        return np.array([p.intensity for p in self.peaks])

    def get_dominant_peaks(self, n: int = 5) -> List[Peak]:
        """Retorna los n picos más prominentes."""
        sorted_peaks = sorted(self.peaks, key=lambda p: p.prominence, reverse=True)
        return sorted_peaks[:n]


class PeakExtractionEngine:
    """
    Motor de extracción de picos operando sobre representación theta.

    La representación theta resalta picos débiles que métodos estándar
    podrían perder, mejorando la identificación de materiales.
    """

    def __init__(self,
                 min_peak_height: float = 0.05,
                 min_distance: int = 10,
                 prominence_threshold: float = 0.02,
                 width_range: Tuple[float, float] = (1.0, 50.0)):
        self.min_peak_height = min_peak_height
        self.min_distance = min_distance
        self.prominence_threshold = prominence_threshold
        self.width_range = width_range

    # ─────────────────────────────────────────────────────────
    # EXTRACCIÓN PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def extract(self, wavenumbers: np.ndarray,
                intensities: np.ndarray,
                theta_intensities: Optional[np.ndarray] = None) -> PeakExtractionResult:
        """
        Extrae picos del espectro. Si se proporciona theta_intensities,
        usa ambas representaciones para detección robusta.
        """
        # Detección en espectro original
        peaks_orig, properties_orig = find_peaks(
            intensities,
            height=self.min_peak_height,
            distance=self.min_distance,
            prominence=self.prominence_threshold,
            width=self.width_range
        )

        # Si hay representación theta, detectar también ahí
        peaks_theta = np.array([])
        properties_theta = {}

        if theta_intensities is not None:
            peaks_theta, properties_theta = find_peaks(
                theta_intensities,
                height=self.min_peak_height * 0.5,  # Más sensible
                distance=self.min_distance,
                prominence=self.prominence_threshold * 0.5,
                width=self.width_range
            )

        # Fusionar picos: unión de ambos conjuntos con tolerancia
        all_peak_indices = self._merge_peaks(
            peaks_orig, peaks_theta, 
            wavenumbers, tolerance=2.0
        )

        # Caracterizar cada pico
        peaks = []
        for idx in all_peak_indices:
            peak = self._characterize_peak(
                idx, wavenumbers, intensities, 
                theta_intensities, properties_orig
            )
            peaks.append(peak)

        return PeakExtractionResult(
            peaks=peaks,
            wavenumbers=wavenumbers,
            intensities=intensities,
            theta_intensities=theta_intensities if theta_intensities is not None else intensities,
            extraction_params={
                'min_height': self.min_peak_height,
                'min_distance': self.min_distance,
                'prominence': self.prominence_threshold
            }
        )

    def _merge_peaks(self, peaks_a: np.ndarray, peaks_b: np.ndarray,
                     wavenumbers: np.ndarray, tolerance: float = 2.0) -> np.ndarray:
        """Fusiona dos conjuntos de picos eliminando duplicados cercanos."""
        if len(peaks_b) == 0:
            return peaks_a

        all_peaks = np.concatenate([peaks_a, peaks_b])
        all_peaks = np.sort(all_peaks)

        # Eliminar duplicados dentro de tolerancia (en unidades de índice)
        merged = [all_peaks[0]]
        for p in all_peaks[1:]:
            if abs(wavenumbers[int(p)] - wavenumbers[int(merged[-1])]) > tolerance:
                merged.append(p)

        return np.array(merged, dtype=int)

    def _characterize_peak(self, idx: int,
                           wavenumbers: np.ndarray,
                           intensities: np.ndarray,
                           theta_intensities: Optional[np.ndarray],
                           properties: dict) -> Peak:
        """Caracteriza un pico individual."""

        # Posición e intensidad
        position = wavenumbers[idx]
        intensity = intensities[idx]

        # Ancho (FWHM)
        # Aproximación: interpolar entre puntos para mayor precisión
        half_max = intensity / 2.0

        # Buscar izquierda
        left_idx = idx
        while left_idx > 0 and intensities[left_idx] > half_max:
            left_idx -= 1

        # Buscar derecha
        right_idx = idx
        while right_idx < len(intensities) - 1 and intensities[right_idx] > half_max:
            right_idx += 1

        # Interpolar para mayor precisión
        if left_idx < idx and intensities[left_idx] < half_max:
            left_frac = (half_max - intensities[left_idx]) / (intensities[left_idx+1] - intensities[left_idx])
            left_pos = wavenumbers[left_idx] + left_frac * (wavenumbers[left_idx+1] - wavenumbers[left_idx])
        else:
            left_pos = wavenumbers[left_idx]

        if right_idx > idx and intensities[right_idx] < half_max:
            right_frac = (half_max - intensities[right_idx]) / (intensities[right_idx-1] - intensities[right_idx])
            right_pos = wavenumbers[right_idx] + right_frac * (wavenumbers[right_idx-1] - wavenumbers[right_idx])
        else:
            right_pos = wavenumbers[right_idx]

        width = right_pos - left_pos

        # Prominencia
        prominence = properties.get('prominences', [0.1])
        prom_idx = np.where(properties.get('peak_heights', []) == intensity)[0]
        if len(prom_idx) > 0 and prom_idx[0] < len(prominence):
            prom = prominence[prom_idx[0]]
        else:
            prom = intensity * 0.5  # fallback

        # Área (aproximación trapezoidal)
        area = np.trapezoid(intensities[left_idx:right_idx+1], 
                           wavenumbers[left_idx:right_idx+1])

        # Forma y asimetría
        shape, asymmetry = self._fit_peak_shape(
            wavenumbers[left_idx:right_idx+1],
            intensities[left_idx:right_idx+1],
            position, intensity, width
        )

        # Firma theta local (contexto para matching)
        theta_sig = self._extract_theta_signature(
            idx, theta_intensities if theta_intensities is not None else intensities
        )

        return Peak(
            position=position,
            intensity=intensity,
            width=width,
            prominence=prom,
            area=area,
            shape=shape,
            asymmetry=asymmetry,
            theta_signature=theta_sig
        )

    def _fit_peak_shape(self, wn_local: np.ndarray, int_local: np.ndarray,
                        center: float, height: float, width: float) -> Tuple[str, float]:
        """Ajusta gaussiana y lorentziana para determinar forma."""
        if len(wn_local) < 5:
            return "unknown", 0.0

        try:
            # Gaussian: A * exp(-(x-mu)^2 / (2*sigma^2))
            def gaussian(x, A, mu, sigma):
                return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

            # Lorentzian: A * gamma^2 / ((x-mu)^2 + gamma^2)
            def lorentzian(x, A, mu, gamma):
                return A * gamma**2 / ((x - mu)**2 + gamma**2)

            p0_g = [height, center, width/2.355]
            p0_l = [height, center, width/2]

            popt_g, _ = curve_fit(gaussian, wn_local, int_local, p0=p0_g, maxfev=5000)
            popt_l, _ = curve_fit(lorentzian, wn_local, int_local, p0=p0_l, maxfev=5000)

            resid_g = np.sum((int_local - gaussian(wn_local, *popt_g))**2)
            resid_l = np.sum((int_local - lorentzian(wn_local, *popt_l))**2)

            shape = "gaussian" if resid_g < resid_l else "lorentzian"

            # Asimetría: ratio cola izquierda / cola derecha
            left_tail = np.sum(int_local[wn_local < center])
            right_tail = np.sum(int_local[wn_local > center])
            asymmetry = (right_tail - left_tail) / (right_tail + left_tail + 1e-12)

        except Exception:
            shape = "unknown"
            asymmetry = 0.0

        return shape, asymmetry

    def _extract_theta_signature(self, idx: int, theta_intensities: np.ndarray,
                                  window: int = 20) -> np.ndarray:
        """Extrae ventana local de firma theta alrededor del pico."""
        start = max(0, idx - window)
        end = min(len(theta_intensities), idx + window + 1)
        sig = theta_intensities[start:end].copy()

        # Normalizar ventana
        if np.max(sig) > 1e-12:
            sig = sig / np.max(sig)

        # Padding si es necesario
        if len(sig) < 2 * window + 1:
            pad_left = window - idx if idx < window else 0
            pad_right = (idx + window + 1) - len(theta_intensities)
            pad_right = max(0, pad_right)
            sig = np.pad(sig, (pad_left, pad_right), mode='constant')

        return sig

    # ─────────────────────────────────────────────────────────
    # COMPARACIÓN DE PICOS
    # ─────────────────────────────────────────────────────────

    def compare_peak_sets(self, peaks_a: List[Peak], peaks_b: List[Peak],
                          position_tolerance: float = 5.0) -> Dict:
        """
        Compara dos conjuntos de picos y encuentra matches.
        """
        matches = []
        unmatched_a = []
        unmatched_b = list(peaks_b)

        for pa in peaks_a:
            best_match = None
            best_score = -1

            for pb in unmatched_b:
                pos_diff = abs(pa.position - pb.position)
                if pos_diff > position_tolerance:
                    continue

                # Score de match combinado
                pos_score = 1 - (pos_diff / position_tolerance)
                int_score = 1 - abs(pa.intensity - pb.intensity)
                width_score = 1 - min(abs(pa.width - pb.width) / max(pa.width, pb.width, 1), 1)

                score = 0.5 * pos_score + 0.3 * int_score + 0.2 * width_score

                if score > best_score:
                    best_score = score
                    best_match = pb

            if best_match and best_score > 0.5:
                matches.append({
                    'peak_a': pa,
                    'peak_b': best_match,
                    'score': best_score,
                    'position_shift': pa.position - best_match.position,
                    'intensity_ratio': pa.intensity / (best_match.intensity + 1e-12)
                })
                unmatched_b.remove(best_match)
            else:
                unmatched_a.append(pa)

        return {
            'matches': matches,
            'unmatched_a': unmatched_a,
            'unmatched_b': unmatched_b,
            'match_ratio': len(matches) / max(len(peaks_a), 1)
        }
