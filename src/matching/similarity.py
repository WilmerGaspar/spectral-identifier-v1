"""
Similarity + Matching Engine
============================
Métricas de similaridad: cosine, SAM, theta-score.
Incluye matching con pesos por metadata y ranking.

Autor: Spectral Identifier Team
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial.distance import cosine, euclidean
import warnings


@dataclass
class MatchResult:
    """Resultado de matching entre un espectro desconocido y referencia."""
    reference_id: str
    reference_name: str
    reference_metadata: dict

    # Scores individuales
    cosine_score: float
    sam_score: float
    theta_score: float
    peak_match_score: float

    # Score combinado
    combined_score: float
    confidence: float

    # Detalles
    explanation: str = ""
    matching_peaks: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'reference_id': self.reference_id,
            'reference_name': self.reference_name,
            'cosine_score': round(self.cosine_score, 4),
            'sam_score': round(self.sam_score, 4),
            'theta_score': round(self.theta_score, 4),
            'peak_match_score': round(self.peak_match_score, 4),
            'combined_score': round(self.combined_score, 4),
            'confidence': round(self.confidence, 4),
            'explanation': self.explanation
        }


class SimilarityEngine:
    """
    Motor de similaridad con múltiples métricas.

    Implementa:
    - Cosine similarity (tradicional)
    - SAM (Spectral Angle Mapper)
    - Theta-score (nuestra métrica propietaria)
    - Peak matching score
    """

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 metadata_weight: float = 0.1):
        """
        Args:
            weights: Pesos para score combinado
            metadata_weight: Peso adicional por coincidencia de metadata
        """
        self.weights = weights or {
            'cosine': 0.20,
            'sam': 0.20,
            'theta': 0.35,
            'peak': 0.25
        }
        self.metadata_weight = metadata_weight

    # ─────────────────────────────────────────────────────────
    # MÉTRICAS INDIVIDUALES
    # ─────────────────────────────────────────────────────────

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Similaridad coseno: cos(θ) = (a·b) / (||a|| ||b||)
        Rango: [-1, 1], típicamente [0, 1] para espectros positivos.
        """
        # Asegurar no negatividad para espectros
        a = np.maximum(a, 0)
        b = np.maximum(b, 0)

        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def sam_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Spectral Angle Mapper (SAM).
        Mide el ángulo entre vectores espectrales en espacio N-dimensional.

        SAM = arccos( (a·b) / (||a|| ||b||) )
        Similaridad = 1 - SAM/π

        Ventaja: Invariante a ganancia (escala de intensidad).
        """
        a = np.maximum(a, 0)
        b = np.maximum(b, 0)

        # Añadir pequeño valor para evitar division por cero
        a = a + 1e-12
        b = b + 1e-12

        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        cos_angle = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Convertir a similaridad [0, 1]
        return float(1 - angle / np.pi)

    def theta_score(self, features_a, features_b) -> float:
        """
        Theta-score: Métrica propietaria basada en representación theta.
        Usa phi_similarity, psi_similarity e invariant_similarity.
        """
        if hasattr(features_a, 'phi_signature'):
            # Son objetos ThetaFeatures
            from src.engines.theta_ramanujan import RamanujanThetaEngine
            engine = RamanujanThetaEngine()
            sim_dict = engine.theta_similarity(features_a, features_b)
            return sim_dict['theta_score']
        else:
            # Fallback a cosine si no son features theta
            return self.cosine_similarity(features_a, features_b)

    def peak_match_score(self, peaks_a, peaks_b,
                         position_tolerance: float = 5.0) -> float:
        """
        Score basado en coincidencia de picos.
        """
        if len(peaks_a) == 0 or len(peaks_b) == 0:
            return 0.0

        from src.engines.peak_extractor import PeakExtractionEngine
        extractor = PeakExtractionEngine()

        comparison = extractor.compare_peak_sets(
            peaks_a, peaks_b, position_tolerance
        )

        matches = comparison['matches']
        n_a = len(peaks_a)
        n_b = len(peaks_b)

        if n_a == 0:
            return 0.0

        # Score: ratio de matches ponderado por calidad
        match_quality = sum(m['score'] for m in matches) / max(len(matches), 1)
        coverage = len(matches) / n_a

        # Penalizar desbalance (mucho más picos en uno que otro)
        balance = min(n_a, n_b) / max(n_a, n_b)

        return float(match_quality * coverage * balance)

    # ─────────────────────────────────────────────────────────
    # SCORE COMBINADO
    # ─────────────────────────────────────────────────────────

    def combined_score(self,
                       spectrum_a: np.ndarray,
                       spectrum_b: np.ndarray,
                       features_a=None,
                       features_b=None,
                       peaks_a=None,
                       peaks_b=None,
                       metadata_a: Optional[dict] = None,
                       metadata_b: Optional[dict] = None) -> Dict[str, float]:
        """
        Calcula score combinado con todas las métricas.
        """
        scores = {}

        # 1. Cosine
        scores['cosine'] = self.cosine_similarity(spectrum_a, spectrum_b)

        # 2. SAM
        scores['sam'] = self.sam_similarity(spectrum_a, spectrum_b)

        # 3. Theta
        if features_a is not None and features_b is not None:
            scores['theta'] = self.theta_score(features_a, features_b)
        else:
            scores['theta'] = scores['cosine']  # fallback

        # 4. Peak matching
        if peaks_a is not None and peaks_b is not None:
            scores['peak'] = self.peak_match_score(peaks_a, peaks_b)
        else:
            scores['peak'] = 0.5  # neutral si no hay picos

        # Score combinado ponderado
        combined = sum(
            scores[key] * self.weights.get(key, 0.25)
            for key in scores
        )

        # Bonus por metadata coincidente
        metadata_bonus = 0.0
        if metadata_a and metadata_b:
            if metadata_a.get('mineral_class') == metadata_b.get('mineral_class'):
                metadata_bonus = self.metadata_weight
            if metadata_a.get('source') == metadata_b.get('source'):
                metadata_bonus += self.metadata_weight * 0.5

        scores['combined'] = min(combined + metadata_bonus, 1.0)

        # Confianza: acuerdo entre métricas
        metric_values = [scores['cosine'], scores['sam'], scores['theta']]
        std_metrics = np.std(metric_values)
        scores['confidence'] = float(1 - min(std_metrics * 2, 1.0))

        return scores

    # ─────────────────────────────────────────────────────────
    # MATCHING CONTRA BASE DE DATOS
    # ─────────────────────────────────────────────────────────

    def match_against_database(self,
                               unknown_spectrum: np.ndarray,
                               database: List[Dict],
                               unknown_features=None,
                               unknown_peaks=None,
                               unknown_metadata: Optional[dict] = None,
                               top_k: int = 10) -> List[MatchResult]:
        """
        Matchea espectro desconocido contra base de datos de referencia.

        Args:
            database: Lista de dicts con keys:
                'id', 'name', 'spectrum', 'features', 'peaks', 'metadata'
            top_k: Número de mejores matches a retornar
        """
        results = []

        for ref in database:
            ref_spectrum = ref.get('spectrum')
            if ref_spectrum is None:
                continue

            scores = self.combined_score(
                spectrum_a=unknown_spectrum,
                spectrum_b=ref_spectrum,
                features_a=unknown_features,
                features_b=ref.get('features'),
                peaks_a=unknown_peaks,
                peaks_b=ref.get('peaks'),
                metadata_a=unknown_metadata,
                metadata_b=ref.get('metadata')
            )

            # Generar explicación
            explanation = self._generate_explanation(scores, ref)

            result = MatchResult(
                reference_id=ref.get('id', 'unknown'),
                reference_name=ref.get('name', 'Unknown'),
                reference_metadata=ref.get('metadata', {}),
                cosine_score=scores['cosine'],
                sam_score=scores['sam'],
                theta_score=scores['theta'],
                peak_match_score=scores['peak'],
                combined_score=scores['combined'],
                confidence=scores['confidence'],
                explanation=explanation
            )

            results.append(result)

        # Ordenar por score combinado descendente
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results[:top_k]

    def _generate_explanation(self, scores: Dict, reference: Dict) -> str:
        """Genera explicación textual del match."""
        parts = []

        if scores['combined'] > 0.9:
            parts.append("Match excelente")
        elif scores['combined'] > 0.7:
            parts.append("Buen match")
        elif scores['combined'] > 0.5:
            parts.append("Match moderado")
        else:
            parts.append("Match débil")

        if scores['theta'] > scores['cosine']:
            parts.append("La representación theta muestra alta compatibilidad estructural.")

        if scores['peak'] > 0.7:
            parts.append("Coincidencia fuerte en posiciones de picos.")
        elif scores['peak'] < 0.3:
            parts.append("Diferencias significativas en picos detectados.")

        if scores['confidence'] < 0.5:
            parts.append("Baja confianza: las métricas discrepan.")

        return " ".join(parts)
