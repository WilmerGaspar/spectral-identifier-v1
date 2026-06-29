"""
Preprocessing Engine
====================
Limpieza, normalización y alineación de espectros espectroscópicos.
Incluye manejo de metadata y feedback loop para re-procesamiento.

Autor: Spectral Identifier Team
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import warnings


@dataclass
class SpectrumMetadata:
    """Metadatos asociados a un espectro."""
    source: str = "unknown"           # NIST, RRUFF, USGS, own
    instrument: str = "unknown"       # Tipo de instrumento
    laser_wavelength: Optional[float] = None  # nm
    resolution: Optional[float] = None        # cm^-1
    temperature: Optional[float] = None       # K
    sample_name: str = "unknown"
    mineral_class: Optional[str] = None
    operator: str = "unknown"
    date_acquired: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'instrument': self.instrument,
            'laser_wavelength_nm': self.laser_wavelength,
            'resolution_cm-1': self.resolution,
            'temperature_K': self.temperature,
            'sample_name': self.sample_name,
            'mineral_class': self.mineral_class,
            'operator': self.operator,
            'date_acquired': self.date_acquired,
            'notes': self.notes
        }


@dataclass 
class PreprocessedSpectrum:
    """Espectro preprocesado con trazabilidad completa."""
    wavenumbers: np.ndarray
    intensities: np.ndarray
    original_wavenumbers: np.ndarray
    original_intensities: np.ndarray
    metadata: SpectrumMetadata
    processing_log: List[Dict] = field(default_factory=list)
    quality_flags: Dict[str, bool] = field(default_factory=dict)

    def add_log(self, step: str, params: dict, quality_check: dict):
        """Registra un paso de procesamiento."""
        self.processing_log.append({
            'step': step,
            'params': params,
            'quality_check': quality_check,
            'timestamp': str(np.datetime64('now'))
        })

    def get_log(self) -> List[Dict]:
        return self.processing_log


class PreprocessingEngine:
    """
    Motor de preprocesamiento con feedback loop.

    Flujo: clean → normalize → align → quality_check
    Si quality_check falla, puede reintentar con parámetros ajustados.
    """

    def __init__(self,
                 target_resolution: float = 1.0,      # cm^-1
                 wavenumber_range: Tuple[float, float] = (100, 4000),
                 max_iterations: int = 3):
        self.target_resolution = target_resolution
        self.wavenumber_range = wavenumber_range
        self.max_iterations = max_iterations

        # Umbrales de calidad
        self.quality_thresholds = {
            'min_snr': 5.0,           # Señal/ruido mínimo
            'max_baseline_drift': 0.3, # Deriva máxima permitida
            'min_peak_count': 3,       # Mínimo de picos detectables
            'max_saturation': 0.95     # Máximo de saturación permitida
        }

    # ─────────────────────────────────────────────────────────
    # CLEAN: Eliminación de ruido y artefactos
    # ─────────────────────────────────────────────────────────

    def clean(self, wavenumbers: np.ndarray, 
              intensities: np.ndarray,
              method: str = "savgol",
              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Limpia el espectro removiendo ruido y artefactos.

        Methods:
            'savgol': Filtro Savitzky-Golay (preserva picos)
            'gaussian': Suavizado gaussiano
            'wavelet': Umbral wavelet (más agresivo)
        """
        if method == "savgol":
            n = len(intensities)

            if n < 3:
                cleaned = intensities.copy()
            else:
                window = kwargs.get("window", min(51, max(3, (n // 10) * 2 + 1)))

                if window > n:
                    window = n

                if window % 2 == 0:
                    window -= 1

                polyorder = kwargs.get("polyorder", 3)

                if polyorder >= window:
                    polyorder = max(1, window - 1)

                if window >= 3 and window > polyorder:
                    cleaned = savgol_filter(intensities, window, polyorder)
                else:
                    cleaned = intensities.copy()

        elif method == "gaussian":
            sigma = kwargs.get('sigma', 2.0)
            cleaned = gaussian_filter1d(intensities, sigma)

        elif method == "wavelet":
            # Simplificación: umbral en dominio de frecuencia
            fft_spec = np.fft.fft(intensities)
            threshold = kwargs.get('threshold', 0.1)
            magnitude = np.abs(fft_spec)
            fft_spec[magnitude < threshold * np.max(magnitude)] = 0
            cleaned = np.real(np.fft.ifft(fft_spec))

        else:
            cleaned = intensities.copy()

        return wavenumbers, cleaned

    # ─────────────────────────────────────────────────────────
    # NORMALIZE: Corrección de línea base e intensidad
    # ─────────────────────────────────────────────────────────

    def normalize(self, wavenumbers: np.ndarray,
                  intensities: np.ndarray,
                  method: str = "minmax",
                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normaliza el espectro.

        Methods:
            'minmax': [0, 1]
            'area': Área bajo curva = 1
            'vector': Norma L2 = 1
            'baseline': Corrección de línea base + minmax
        """
        if method == "minmax":
            norm = intensities - np.min(intensities)
            max_val = np.max(norm)
            if max_val > 1e-12:
                norm = norm / max_val

        elif method == "area":
            area = np.trapezoid(intensities, wavenumbers)
            if abs(area) > 1e-12:
                norm = intensities / abs(area)
            else:
                norm = intensities.copy()

        elif method == "vector":
            norm_val = np.linalg.norm(intensities)
            if norm_val > 1e-12:
                norm = intensities / norm_val
            else:
                norm = intensities.copy()

        elif method == "baseline":
            # Corrección de línea base polinomial
            degree = kwargs.get('degree', 3)
            baseline = self._fit_baseline(wavenumbers, intensities, degree)
            corrected = intensities - baseline
            corrected = np.maximum(corrected, 0)

            # Luego minmax
            norm = corrected - np.min(corrected)
            max_val = np.max(norm)
            if max_val > 1e-12:
                norm = norm / max_val

        else:
            norm = intensities.copy()

        return wavenumbers, norm

    def _fit_baseline(self, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        """Ajusta línea base usando puntos mínimos locales."""
        # Seleccionar puntos mínimos cada N puntos
        n_segments = max(degree * 2, 10)
        segment_size = len(x) // n_segments

        min_x = []
        min_y = []
        for i in range(n_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(x))
            idx = start + np.argmin(y[start:end])
            min_x.append(x[idx])
            min_y.append(y[idx])

        # Interpolar spline a través de mínimos
        if len(min_x) > degree:
            coeffs = np.polyfit(min_x, min_y, degree)
            baseline = np.polyval(coeffs, x)
        else:
            baseline = np.zeros_like(y)

        return baseline

    # ─────────────────────────────────────────────────────────
    # ALIGN: Interpolación a grid común
    # ─────────────────────────────────────────────────────────

    def align(self, wavenumbers: np.ndarray,
              intensities: np.ndarray,
              target_grid: Optional[np.ndarray] = None,
              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpola el espectro a una cuadrícula estándar.
        """
        if target_grid is None:
            wn_min, wn_max = self.wavenumber_range
            target_grid = np.arange(wn_min, wn_max + self.target_resolution, 
                                    self.target_resolution)

        # Filtrar rango válido
        mask = (wavenumbers >= target_grid[0]) & (wavenumbers <= target_grid[-1])
        wn_valid = wavenumbers[mask]
        int_valid = intensities[mask]

        if len(wn_valid) < 2:
            # Si no hay datos suficientes, rellenar con ceros
            return target_grid, np.zeros_like(target_grid)

        # Interpolación cúbica (suave, preserva picos)
        interpolator = CubicSpline(wn_valid, int_valid)
        aligned = interpolator(target_grid)

        # Manejar extrapolación: rellenar con ceros fuera de rango
        aligned[target_grid < wn_valid[0]] = 0
        aligned[target_grid > wn_valid[-1]] = 0

        return target_grid, aligned

    # ─────────────────────────────────────────────────────────
    # QUALITY CHECK
    # ─────────────────────────────────────────────────────────

    def quality_check(self, wavenumbers: np.ndarray,
                      intensities: np.ndarray) -> Dict[str, any]:
        """
        Evalúa la calidad del espectro preprocesado.
        Devuelve flags y métricas para decisión de re-procesamiento.
        """
        results = {}

        # 1. SNR
        signal_peak = np.max(intensities)
        noise_region = intensities[intensities < np.percentile(intensities, 25)]
        noise_std = np.std(noise_region) if len(noise_region) > 0 else 1e-12
        snr = signal_peak / noise_std
        results['snr'] = float(snr)
        results['snr_pass'] = snr >= self.quality_thresholds['min_snr']

        # 2. Baseline drift
        baseline = self._fit_baseline(wavenumbers, intensities, 1)
        drift = np.max(baseline) - np.min(baseline)
        drift_norm = drift / (np.max(intensities) + 1e-12)
        results['baseline_drift'] = float(drift_norm)
        results['baseline_pass'] = drift_norm <= self.quality_thresholds['max_baseline_drift']

        # 3. Peak count
        peaks, _ = find_peaks(intensities, height=0.1, distance=10)
        results['peak_count'] = len(peaks)
        results['peaks_pass'] = len(peaks) >= self.quality_thresholds['min_peak_count']

        # 4. Saturation
        saturation = np.sum(intensities > self.quality_thresholds['max_saturation']) / len(intensities)
        results['saturation_ratio'] = float(saturation)
        results['saturation_pass'] = saturation < 0.1

        # Overall
        results['overall_pass'] = all([
            results['snr_pass'],
            results['baseline_pass'],
            results['peaks_pass'],
            results['saturation_pass']
        ])

        return results

    # ─────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL CON FEEDBACK LOOP
    # ─────────────────────────────────────────────────────────

    def process(self, wavenumbers: np.ndarray,
                intensities: np.ndarray,
                metadata: Optional[SpectrumMetadata] = None) -> PreprocessedSpectrum:
        """
        Pipeline completo con feedback loop.
        Si quality_check falla, ajusta parámetros y reintenta.
        """
        if metadata is None:
            metadata = SpectrumMetadata()

        original_wn = wavenumbers.copy()
        original_int = intensities.copy()

        result = PreprocessedSpectrum(
            wavenumbers=np.array([]),
            intensities=np.array([]),
            original_wavenumbers=original_wn,
            original_intensities=original_int,
            metadata=metadata
        )

        # Parámetros iniciales
        clean_method = "savgol"
        clean_params = {'window': 51, 'polyorder': 3}
        norm_method = "baseline"
        norm_params = {'degree': 3}

        for iteration in range(self.max_iterations):
            # 1. Clean
            wn, int_clean = self.clean(original_wn, original_int, 
                                       method=clean_method, **clean_params)

            # 2. Normalize
            wn, int_norm = self.normalize(wn, int_clean, 
                                          method=norm_method, **norm_params)

            # 3. Align
            wn_aligned, int_aligned = self.align(wn, int_norm)

            # 4. Quality check
            quality = self.quality_check(wn_aligned, int_aligned)

            result.wavenumbers = wn_aligned
            result.intensities = int_aligned
            result.add_log(
                step=f"iteration_{iteration}",
                params={
                    'clean': {'method': clean_method, **clean_params},
                    'normalize': {'method': norm_method, **norm_params}
                },
                quality_check=quality
            )
            result.quality_flags = quality

            if quality['overall_pass']:
                break

        # Feedback: ajustar parámetros para siguiente iteración
        if not quality['snr_pass']:
            clean_params['window'] = min(clean_params['window'] + 20, max(5, len(intensities)//10*2+1))

        if not quality['baseline_pass']:
            norm_params['degree'] = min(norm_params['degree'] + 1, 5)

        if not quality['peaks_pass']:
            clean_params['window'] = max(clean_params['window'] - 10, 11)

        if not quality['saturation_pass']:
            norm_method = "vector"

        return result
