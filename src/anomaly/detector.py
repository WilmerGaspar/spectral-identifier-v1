"""
Anomaly Detection Engine
========================
Detección de picos desconocidos y ratios raros.
Incluye feedback loop al preprocessing para re-verificación.

Autor: Spectral Identifier Team
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings


@dataclass
class Anomaly:
    """Representación de una anomalía detectada."""
    type: str                    # 'unknown_peak', 'rare_ratio', 'shift', 'shape'
    severity: str                # 'low', 'medium', 'high', 'critical'
    position: Optional[float]    # Posición en wavenumbers (si aplica)
    description: str
    confidence: float            # [0, 1]
    suggested_action: str

    def to_dict(self) -> dict:
        return {
            'type': self.type,
            'severity': self.severity,
            'position': self.position,
            'description': self.description,
            'confidence': round(self.confidence, 4),
            'suggested_action': self.suggested_action
        }


@dataclass
class AnomalyResult:
    """Resultado completo de detección de anomalías."""
    anomalies: List[Anomaly]
    overall_risk: str            # 'none', 'low', 'medium', 'high', 'critical'
    requires_feedback: bool      # ¿Necesita re-procesamiento?
    feedback_reason: str
    statistics: dict

    def get_critical_anomalies(self) -> List[Anomaly]:
        return [a for a in self.anomalies if a.severity in ['high', 'critical']]

    def to_dict(self) -> dict:
        return {
            'overall_risk': self.overall_risk,
            'requires_feedback': self.requires_feedback,
            'feedback_reason': self.feedback_reason,
            'anomaly_count': len(self.anomalies),
            'anomalies': [a.to_dict() for a in self.anomalies],
            'statistics': self.statistics
        }


class AnomalyDetectionEngine:
    """
    Motor de detección de anomalías con feedback loop.

    Detecta:
    - Picos desconocidos (no presentes en referencias cercanas)
    - Ratios de intensidad raros
    - Shifts anómalos en posiciones
    - Formas de pico inusuales
    """

    def __init__(self,
                 unknown_peak_threshold: float = 0.15,
                 rare_ratio_threshold: float = 0.1,
                 shift_tolerance: float = 3.0,
                 confidence_threshold: float = 0.6):
        self.unknown_peak_threshold = unknown_peak_threshold
        self.rare_ratio_threshold = rare_ratio_threshold
        self.shift_tolerance = shift_tolerance
        self.confidence_threshold = confidence_threshold

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE PICOS DESCONOCIDOS
    # ─────────────────────────────────────────────────────────

    def detect_unknown_peaks(self,
                             unknown_peaks: List,
                             reference_peaks: List,
                             position_tolerance: float = 5.0) -> List[Anomaly]:
        """
        Identifica picos en el espectro desconocido que no aparecen
        en las referencias más cercanas.
        """
        anomalies = []

        if not unknown_peaks or not reference_peaks:
            return anomalies

        ref_positions = np.array([p.position for p in reference_peaks])

        for peak in unknown_peaks:
            # Verificar si el pico está cerca de alguna referencia
            distances = np.abs(ref_positions - peak.position)
            min_dist = np.min(distances)

            if min_dist > position_tolerance:
                # Pico desconocido
                severity = self._assess_peak_severity(peak)

                anomaly = Anomaly(
                    type='unknown_peak',
                    severity=severity,
                    position=peak.position,
                    description=f"Pico en {peak.position:.1f} cm⁻¹ no encontrado en referencias "
                               f"(distancia mínima: {min_dist:.1f} cm⁻¹). "
                               f"Intensidad: {peak.intensity:.3f}, Ancho: {peak.width:.1f} cm⁻¹",
                    confidence=min(1.0, min_dist / (position_tolerance * 3)),
                    suggested_action="Verificar si es artefacto de instrumento o nueva fase/material."
                )
                anomalies.append(anomaly)

        return anomalies

    def _assess_peak_severity(self, peak) -> str:
        """Evalúa severidad de un pico desconocido."""
        if peak.intensity > 0.7 and peak.prominence > 0.3:
            return 'critical'
        elif peak.intensity > 0.4 or peak.prominence > 0.2:
            return 'high'
        elif peak.intensity > 0.2:
            return 'medium'
        return 'low'

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE RATIOS RAROS
    # ─────────────────────────────────────────────────────────

    def detect_rare_ratios(self,
                           unknown_peaks: List,
                           reference_peaks: List,
                           database_ratios: Optional[Dict] = None) -> List[Anomaly]:
        """
        Detecta ratios de intensidad entre picos que son estadísticamente
        raros comparados con la base de datos.
        """
        anomalies = []

        if len(unknown_peaks) < 2:
            return anomalies

        # Calcular ratios del espectro desconocido
        intensities = np.array([p.intensity for p in unknown_peaks])
        positions = np.array([p.position for p in unknown_peaks])

        # Ratios entre pares de picos dominantes
        sorted_idx = np.argsort(intensities)[::-1]
        top_n = min(5, len(sorted_idx))

        for i in range(top_n):
            for j in range(i + 1, top_n):
                idx_i = sorted_idx[i]
                idx_j = sorted_idx[j]

                ratio = intensities[idx_i] / (intensities[idx_j] + 1e-12)
                pos_i = positions[idx_i]
                pos_j = positions[idx_j]

                # Verificar contra base de datos de ratios
                if database_ratios:
                    ratio_key = f"{pos_i:.0f}_{pos_j:.0f}"
                    if ratio_key in database_ratios:
                        expected_ratio = database_ratios[ratio_key]['mean']
                        std_ratio = database_ratios[ratio_key]['std']

                        z_score = abs(ratio - expected_ratio) / (std_ratio + 1e-12)

                        if z_score > 3:  # Más de 3 desviaciones estándar
                            severity = 'high' if z_score > 5 else 'medium'

                            anomaly = Anomaly(
                                type='rare_ratio',
                                severity=severity,
                                position=(pos_i + pos_j) / 2,
                                description=f"Ratio de intensidad inusual entre picos "
                                           f"{pos_i:.1f} y {pos_j:.1f} cm⁻¹: "
                                           f"{ratio:.2f} (esperado: {expected_ratio:.2f} ± {std_ratio:.2f}, "
                                           f"z={z_score:.1f}σ)",
                                confidence=min(1.0, z_score / 5),
                                suggested_action="Posible cambio de concentración, orientación cristalina, o nueva fase."
                            )
                            anomalies.append(anomaly)

        return anomalies

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE SHIFTS ANÓMALOS
    # ─────────────────────────────────────────────────────────

    def detect_shifts(self,
                      unknown_peaks: List,
                      reference_peaks: List,
                      position_tolerance: float = 5.0) -> List[Anomaly]:
        """
        Detecta shifts sistemáticos en posiciones de picos que podrían
        indicar estrés, temperatura, o composición diferente.
        """
        anomalies = []

        if not unknown_peaks or not reference_peaks:
            return anomalies

        shifts = []
        matched_pairs = []

        ref_positions = np.array([p.position for p in reference_peaks])

        for peak in unknown_peaks:
            distances = np.abs(ref_positions - peak.position)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist <= position_tolerance:
                shift = peak.position - ref_positions[min_idx]
                shifts.append(shift)
                matched_pairs.append((peak, reference_peaks[min_idx], shift))

        if len(shifts) < 2:
            return anomalies

        # Análisis estadístico de shifts
        mean_shift = np.mean(shifts)
        std_shift = np.std(shifts)

        # Shift sistemático
        if abs(mean_shift) > self.shift_tolerance:
            severity = 'high' if abs(mean_shift) > 10 else 'medium'

            anomaly = Anomaly(
                type='shift',
                severity=severity,
                position=None,
                description=f"Shift sistemático detectado: {mean_shift:.2f} ± {std_shift:.2f} cm⁻¹ "
                           f"en {len(shifts)} picos coincidentes. "
                           f"Posible causa: estrés mecánico, temperatura elevada, o error de calibración.",
                confidence=min(1.0, abs(mean_shift) / 20),
                suggested_action="Verificar calibración del instrumento. Si es real, investigar condiciones de muestra."
            )
            anomalies.append(anomaly)

        # Shifts individuales anómalos
        for peak, ref_peak, shift in matched_pairs:
            if abs(shift - mean_shift) > 2 * std_shift and abs(shift) > self.shift_tolerance:
                anomaly = Anomaly(
                    type='shift',
                    severity='medium',
                    position=peak.position,
                    description=f"Shift anómalo en pico {peak.position:.1f} cm⁻¹: "
                               f"{shift:.2f} cm⁻¹ (media: {mean_shift:.2f} cm⁻¹)",
                    confidence=0.7,
                    suggested_action="Pico específico afectado: posible interacción local o impureza."
                )
                anomalies.append(anomaly)

        return anomalies

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE FORMAS INUSUALES
    # ─────────────────────────────────────────────────────────

    def detect_unusual_shapes(self,
                              unknown_peaks: List,
                              reference_peaks: List) -> List[Anomaly]:
        """
        Detecta formas de pico inusuales (asimetría extrema, ancho anómalo).
        """
        anomalies = []

        if not unknown_peaks:
            return anomalies

        # Estadísticas de forma en picos conocidos
        known_asymmetries = [p.asymmetry for p in reference_peaks] if reference_peaks else [0]
        known_widths = [p.width for p in reference_peaks] if reference_peaks else [10]

        mean_asym = np.mean(np.abs(known_asymmetries))
        std_asym = np.std(known_asymmetries) if len(known_asymmetries) > 1 else 0.1
        mean_width = np.mean(known_widths)
        std_width = np.std(known_widths) if len(known_widths) > 1 else 5

        for peak in unknown_peaks:
            # Asimetría extrema
            z_asym = abs(peak.asymmetry - mean_asym) / (std_asym + 1e-12)
            if z_asym > 3 and abs(peak.asymmetry) > 0.3:
                anomaly = Anomaly(
                    type='shape',
                    severity='medium',
                    position=peak.position,
                    description=f"Forma de pico inusual en {peak.position:.1f} cm⁻¹: "
                               f"asimetría={peak.asymmetry:.3f} (z={z_asym:.1f}σ). "
                               f"Posible superposición de picos o efecto Fano.",
                    confidence=min(1.0, z_asym / 5),
                    suggested_action="Descomponer pico en componentes. Investigar acoplamiento electrón-fonón."
                )
                anomalies.append(anomaly)

            # Ancho anómalo
            z_width = abs(peak.width - mean_width) / (std_width + 1e-12)
            if z_width > 3:
                severity = 'high' if z_width > 5 else 'medium'
                anomaly = Anomaly(
                    type='shape',
                    severity=severity,
                    position=peak.position,
                    description=f"Ancho de pico anómalo en {peak.position:.1f} cm⁻¹: "
                               f"{peak.width:.1f} cm⁻¹ (esperado: {mean_width:.1f} ± {std_width:.1f}, "
                               f"z={z_width:.1f}σ). Posible desorden cristalino o temperatura alta.",
                    confidence=min(1.0, z_width / 5),
                    suggested_action="Verificar temperatura de medición y calidad cristalina."
                )
                anomalies.append(anomaly)

        return anomalies

    # ─────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def detect(self,
               unknown_peaks: List,
               reference_peaks: List,
               database_ratios: Optional[Dict] = None,
               match_confidence: float = 0.5) -> AnomalyResult:
        """
        Pipeline completo de detección de anomalías.

        Args:
            unknown_peaks: Picos del espectro a analizar
            reference_peaks: Picos de la referencia más cercana
            database_ratios: Estadísticas de ratios de la base de datos
            match_confidence: Confianza del match (afecta severidad)
        """
        all_anomalies = []

        # 1. Picos desconocidos
        all_anomalies.extend(
            self.detect_unknown_peaks(unknown_peaks, reference_peaks)
        )

        # 2. Ratios raros
        all_anomalies.extend(
            self.detect_rare_ratios(unknown_peaks, reference_peaks, database_ratios)
        )

        # 3. Shifts
        all_anomalies.extend(
            self.detect_shifts(unknown_peaks, reference_peaks)
        )

        # 4. Formas inusuales
        all_anomalies.extend(
            self.detect_unusual_shapes(unknown_peaks, reference_peaks)
        )

        # Determinar riesgo global
        critical_count = sum(1 for a in all_anomalies if a.severity == 'critical')
        high_count = sum(1 for a in all_anomalies if a.severity == 'high')
        medium_count = sum(1 for a in all_anomalies if a.severity == 'medium')

        if critical_count > 0:
            overall_risk = 'critical'
        elif high_count > 1:
            overall_risk = 'high'
        elif high_count == 1 or medium_count > 2:
            overall_risk = 'medium'
        elif medium_count > 0:
            overall_risk = 'low'
        else:
            overall_risk = 'none'

        # Decidir si requiere feedback
        requires_feedback = (overall_risk in ['high', 'critical'] or 
                            match_confidence < 0.3)

        feedback_reason = ""
        if overall_risk == 'critical':
            feedback_reason = "Anomalías críticas detectadas. Re-verificar preprocesamiento."
        elif match_confidence < 0.3:
            feedback_reason = "Baja confianza en identificación. Revisar calidad de datos."

        # Estadísticas
        stats = {
            'total_anomalies': len(all_anomalies),
            'critical': critical_count,
            'high': high_count,
            'medium': medium_count,
            'low': sum(1 for a in all_anomalies if a.severity == 'low'),
            'mean_confidence': np.mean([a.confidence for a in all_anomalies]) if all_anomalies else 0
        }

        return AnomalyResult(
            anomalies=all_anomalies,
            overall_risk=overall_risk,
            requires_feedback=requires_feedback,
            feedback_reason=feedback_reason,
            statistics=stats
        )
