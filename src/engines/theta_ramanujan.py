"""
Theta-Ramanujan Engine
======================
Motor de transformación espectral basado en funciones theta de Jacobi-Ramanujan.
Incluye optimización FFT para q-map (O(n log n) en lugar de O(n²)).

Autor: Spectral Identifier Team
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings


@dataclass
class ThetaFeatures:
    """Contenedor de features extraídas por el motor theta."""
    q_map: np.ndarray
    phi_signature: np.ndarray
    psi_multiscale: np.ndarray
    theta_spectrum: np.ndarray
    invariants: dict
    q_optimal: float

    def to_vector(self) -> np.ndarray:
        """Vector plano para similarity matching."""
        return np.concatenate([
            self.phi_signature.flatten(),
            self.psi_multiscale.flatten(),
            np.array(list(self.invariants.values()))
        ])


class RamanujanThetaEngine:
    """
    Motor Theta-Ramanujan para análisis espectral de materiales.

    Implementa tres transformaciones:
    - q-map: Mapeo al espacio de parámetro modular q (optimizado FFT)
    - phi: Firma espectral invariante (función theta_3)
    - psi: Análisis multiescala (funciones theta_2, theta_4)
    """

    def __init__(self, 
                 n_terms: int = 50,
                 q_range: Tuple[float, float] = (0.01, 0.99),
                 n_scales: int = 8,
                 n_phases: int = 16,
                 use_fft: bool = True):
        self.n_terms = n_terms
        self.q_min, self.q_max = q_range
        self.n_scales = n_scales
        self.n_phases = n_phases
        self.use_fft = use_fft

        # Precomputar kernels theta para acelerar
        self._theta_cache = {}

    # ─────────────────────────────────────────────────────────
    # FUNCIONES THETA DE JACOBI-RAMANUJAN
    # ─────────────────────────────────────────────────────────

    def theta_3(self, x: np.ndarray, q: float) -> np.ndarray:
        """θ₃(x,q) = 1 + 2Σ q^(n²) cos(2πnx)"""
        result = np.ones_like(x, dtype=np.float64)
        q_safe = min(abs(q), 0.9999)

        for n in range(1, self.n_terms + 1):
            coeff = q_safe ** (n * n)
            result += 2 * coeff * np.cos(2 * np.pi * n * x)
        return result

    def theta_2(self, x: np.ndarray, q: float) -> np.ndarray:
        """θ₂(x,q) = 2Σ q^((n-½)²) cos(2π(n-½)x)"""
        result = np.zeros_like(x, dtype=np.float64)
        q_safe = min(abs(q), 0.9999)

        for n in range(1, self.n_terms + 1):
            nu = n - 0.5
            coeff = q_safe ** (nu * nu)
            result += 2 * coeff * np.cos(2 * np.pi * nu * x)
        return result

    def theta_4(self, x: np.ndarray, q: float) -> np.ndarray:
        """θ₄(x,q) = 1 + 2Σ (-1)^n q^(n²) cos(2πnx)"""
        result = np.ones_like(x, dtype=np.float64)
        q_safe = min(abs(q), 0.9999)

        for n in range(1, self.n_terms + 1):
            coeff = ((-1) ** n) * (q_safe ** (n * n))
            result += 2 * coeff * np.cos(2 * np.pi * n * x)
        return result

    # ─────────────────────────────────────────────────────────
    # Q-MAP OPTIMIZADO CON FFT
    # ─────────────────────────────────────────────────────────

    def _theta_fft_coefficients(self, q: float, N: int) -> np.ndarray:
        """
        Genera coeficientes theta en espacio de Fourier.
        θ₃(x,q) en Fourier tiene picos en frecuencias n con amplitud q^(n²).
        Esto permite correlación O(N log N) en lugar de O(N²).
        """
        q_safe = min(abs(q), 0.9999)
        freqs = np.arange(N)

        # Coeficientes de Fourier de theta_3
        coeffs = np.zeros(N, dtype=np.complex128)
        coeffs[0] = 1.0  # término DC

        max_n = min(self.n_terms, N // 2)
        for n in range(1, max_n + 1):
            coeff = q_safe ** (n * n)
            coeffs[n] = coeff
            if N - n < N:
                coeffs[N - n] = coeff  # simetría hermítica

        return coeffs

    def q_map(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Encuentra q óptimo maximizando correlación espectro-theta.
        Versión FFT: O(M * N log N) donde M = número de q candidatos.
        """
        s_norm = self._normalize_spectrum(spectrum)
        N = len(s_norm)

        # FFT del espectro (una sola vez)
        spectrum_fft = fft(s_norm)
        spectrum_power = np.abs(spectrum_fft) ** 2

        # Grid search sobre q
        q_candidates = np.linspace(self.q_min, self.q_max, 100)
        correlations = []

        for q in q_candidates:
            # Generar theta vía FFT inversa de coeficientes
            theta_coeffs = self._theta_fft_coefficients(q, N)
            theta_signal = np.real(ifft(theta_coeffs))
            theta_signal = theta_signal / (np.linalg.norm(theta_signal) + 1e-12)

            # Correlación en espacio de Fourier (más rápido)
            theta_fft = fft(theta_signal)
            corr = np.real(np.sum(spectrum_fft * np.conj(theta_fft))) / N
            correlations.append(corr)

        correlations = np.array(correlations)
        q_optimal = q_candidates[np.argmax(correlations)]

        # Representación final
        theta_coeffs = self._theta_fft_coefficients(q_optimal, N)
        theta_optimal = np.real(ifft(theta_coeffs))
        q_representation = s_norm * theta_optimal

        return q_representation, float(q_optimal)

    # ─────────────────────────────────────────────────────────
    # PHI SIGNATURE
    # ─────────────────────────────────────────────────────────

    def phi_signature(self, spectrum: np.ndarray, q_optimal: float) -> np.ndarray:
        """Firma phi invariante a traslación y escala."""
        s_norm = self._normalize_spectrum(spectrum)
        N = len(s_norm)
        x = np.linspace(0, 1, N)

        phases = np.linspace(0, 2 * np.pi, self.n_phases, endpoint=False)
        phi_signature = np.zeros(self.n_phases, dtype=np.complex128)

        q_safe = min(abs(q_optimal), 0.9999)

        for i, phase in enumerate(phases):
            theta_complex = np.ones(N, dtype=np.complex128)

            for n in range(1, self.n_terms + 1):
                coeff = q_safe ** (n * n)
                theta_complex += 2 * coeff * np.exp(1j * (2 * np.pi * n * x + phase))

            phi_signature[i] = np.trapezoid(s_norm * np.conj(theta_complex), x)

        phi_signature = phi_signature / (np.linalg.norm(phi_signature) + 1e-12)
        return np.real(phi_signature)

    # ─────────────────────────────────────────────────────────
    # PSI MULTIESCALA
    # ─────────────────────────────────────────────────────────

    def psi_multiscale(self, spectrum: np.ndarray, q_optimal: float) -> np.ndarray:
        """Análisis multiescala con funciones theta."""
        s_norm = self._normalize_spectrum(spectrum)
        N = len(s_norm)
        x = np.linspace(0, 1, N)

        scales = np.logspace(-1.5, 1.5, self.n_scales)
        psi_features = np.zeros(self.n_scales)

        for i, scale in enumerate(scales):
            q_scaled = q_optimal ** (1.0 / max(scale, 0.01))
            q_scaled = np.clip(q_scaled, 0.001, 0.9999)

            theta_detail = self.theta_2(x, q_scaled)
            theta_smooth = self.theta_4(x, q_scaled)

            detail_coeff = np.trapezoid(s_norm * theta_detail, x)
            smooth_coeff = np.trapezoid(s_norm * theta_smooth, x)

            if abs(smooth_coeff) > 1e-10:
                psi_features[i] = detail_coeff / smooth_coeff
            else:
                psi_features[i] = 0.0

        return psi_features

    # ─────────────────────────────────────────────────────────
    # INVARIANTES MODULARES
    # ─────────────────────────────────────────────────────────

    def modular_invariants(self, spectrum: np.ndarray, q_optimal: float) -> dict:
        """Invariantes robustos a ruido y cambios de resolución."""
        s_norm = self._normalize_spectrum(spectrum)
        N = len(s_norm)
        x = np.linspace(0, 1, N)

        theta = self.theta_3(x, q_optimal)
        moment = np.trapezoid(s_norm * theta, x)

        psd = np.abs(fft(s_norm)) ** 2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

        peaks_theta, _ = find_peaks(theta, height=0.1)
        n_peaks_theta = len(peaks_theta)

        # Nuevos invariantes: momentos de orden superior
        mean_val = np.mean(s_norm)
        std_val = np.std(s_norm)
        skewness = np.mean(((s_norm - mean_val) / (std_val + 1e-12)) ** 3)
        kurtosis = np.mean(((s_norm - mean_val) / (std_val + 1e-12)) ** 4) - 3

        return {
            'theta_moment': float(moment),
            'spectral_entropy': float(entropy),
            'theta_peak_count': int(n_peaks_theta),
            'q_complexity': float(q_optimal),
            'bandwidth_ratio': float(std_val / (mean_val + 1e-12)),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'spectral_centroid': float(np.sum(np.arange(N) * s_norm) / (np.sum(s_norm) + 1e-12)),
        }

    # ─────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def transform(self, spectrum: np.ndarray) -> ThetaFeatures:
        """Pipeline completo: espectro crudo → features theta."""
        if len(spectrum) < 10:
            raise ValueError("Espectro demasiado corto (min 10 puntos)")

        q_rep, q_opt = self.q_map(spectrum)
        phi = self.phi_signature(spectrum, q_opt)
        psi = self.psi_multiscale(spectrum, q_opt)
        invariants = self.modular_invariants(spectrum, q_opt)

        x = np.linspace(0, 1, len(spectrum))
        theta_spectrum = self.theta_3(x, q_opt) * self._normalize_spectrum(spectrum)

        return ThetaFeatures(
            q_map=q_rep,
            phi_signature=phi,
            psi_multiscale=psi,
            theta_spectrum=theta_spectrum,
            invariants=invariants,
            q_optimal=q_opt
        )

    # ─────────────────────────────────────────────────────────
    # SIMILARIDAD
    # ─────────────────────────────────────────────────────────

    def theta_similarity(self, features_a: ThetaFeatures, 
                         features_b: ThetaFeatures) -> dict:
        """Métricas de similaridad entre dos espectros."""
        phi_sim = np.corrcoef(features_a.phi_signature, 
                              features_b.phi_signature)[0, 1]

        psi_dist = np.linalg.norm(features_a.psi_multiscale - features_b.psi_multiscale)
        psi_sim = 1 / (1 + psi_dist)

        inv_keys = ['theta_moment', 'spectral_entropy', 'q_complexity', 'skewness', 'kurtosis']
        inv_a = np.array([features_a.invariants[k] for k in inv_keys])
        inv_b = np.array([features_b.invariants[k] for k in inv_keys])
        inv_sim = np.dot(inv_a, inv_b) / (np.linalg.norm(inv_a) * np.linalg.norm(inv_b) + 1e-12)

        # Score combinado con pesos ajustados
        theta_score = 0.45 * phi_sim + 0.25 * psi_sim + 0.30 * inv_sim

        return {
            'phi_similarity': float(phi_sim),
            'psi_similarity': float(psi_sim),
            'invariant_similarity': float(inv_sim),
            'theta_score': float(theta_score),
            'confidence': float(abs(phi_sim))
        }

    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        s = np.array(spectrum, dtype=np.float64)
        s = s - np.min(s)
        max_val = np.max(s)
        if max_val > 1e-12:
            s = s / max_val
        return s
