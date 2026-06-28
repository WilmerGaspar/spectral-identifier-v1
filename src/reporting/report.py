"""
Scientific Report Generator
===========================
Genera reportes científicos con score, ranking y explicación.
Incluye trazabilidad completa y flags de confianza.

Autor: Spectral Identifier Team
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class IdentificationResult:
    """Resultado de identificación de un material."""
    rank: int
    material_name: str
    material_id: str
    match_score: float
    confidence: float
    match_details: dict
    explanation: str

    def to_dict(self) -> dict:
        return {
            'rank': self.rank,
            'material_name': self.material_name,
            'material_id': self.material_id,
            'match_score': round(self.match_score, 4),
            'confidence': round(self.confidence, 4),
            'match_details': self.match_details,
            'explanation': self.explanation
        }


@dataclass
class ScientificReport:
    """Reporte científico completo."""
    sample_id: str
    timestamp: str

    # Identificación
    identifications: List[IdentificationResult]
    best_match: Optional[IdentificationResult] = None

    # Anomalías
    anomalies: List[dict] = field(default_factory=list)
    anomaly_risk: str = "none"

    # Trazabilidad
    processing_trace: List[dict] = field(default_factory=list)
    theta_features: Optional[dict] = None
    peak_summary: Optional[dict] = None

    # Metadata
    sample_metadata: dict = field(default_factory=dict)
    instrument_metadata: dict = field(default_factory=dict)

    # Recomendaciones
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'sample_id': self.sample_id,
            'timestamp': self.timestamp,
            'best_match': self.best_match.to_dict() if self.best_match else None,
            'identifications': [i.to_dict() for i in self.identifications],
            'anomaly_risk': self.anomaly_risk,
            'anomalies': self.anomalies,
            'processing_trace': self.processing_trace,
            'theta_features': self.theta_features,
            'peak_summary': self.peak_summary,
            'sample_metadata': self.sample_metadata,
            'instrument_metadata': self.instrument_metadata,
            'recommendations': self.recommendations
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Genera reporte en formato Markdown legible."""
        lines = []
        lines.append(f"# Reporte de Análisis Espectral")
        lines.append(f"**Muestra:** {self.sample_id}")
        lines.append(f"**Fecha:** {self.timestamp}")
        lines.append("")

        # Mejor match
        if self.best_match:
            lines.append(f"## Mejor Identificación")
            lines.append(f"- **Material:** {self.best_match.material_name}")
            lines.append(f"- **Score:** {self.best_match.match_score:.3f}")
            lines.append(f"- **Confianza:** {self.best_match.confidence:.3f}")
            lines.append(f"- **Explicación:** {self.best_match.explanation}")
            lines.append("")

        # Ranking
        lines.append(f"## Ranking de Identificaciones")
        for ident in self.identifications[:5]:
            lines.append(f"{ident.rank}. **{ident.material_name}** — "
                        f"Score: {ident.match_score:.3f} | "
                        f"Confianza: {ident.confidence:.3f}")
        lines.append("")

        # Anomalías
        lines.append(f"## Anomalías Detectadas")
        lines.append(f"**Riesgo global:** {self.anomaly_risk}")
        if self.anomalies:
            for anom in self.anomalies:
                lines.append(f"- **{anom['type']}** ({anom['severity']}): {anom['description']}")
        else:
            lines.append("No se detectaron anomalías significativas.")
        lines.append("")

        # Recomendaciones
        if self.recommendations:
            lines.append(f"## Recomendaciones")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Trazabilidad
        lines.append(f"## Trazabilidad del Procesamiento")
        for trace in self.processing_trace:
            lines.append(f"- **{trace.get('step', 'unknown')}**: {trace.get('params', {})}")

        return "\n".join(lines)


class ReportGenerator:
    """
    Generador de reportes científicos con trazabilidad completa.
    """

    def __init__(self):
        self.version = "1.0.0"

    def generate(self,
                 sample_id: str,
                 match_results: List,
                 anomaly_result,
                 preprocessed_spectrum,
                 theta_features=None,
                 peak_result=None,
                 sample_metadata: Optional[dict] = None,
                 top_n: int = 10) -> ScientificReport:
        """
        Genera reporte científico completo.

        Args:
            sample_id: Identificador de la muestra
            match_results: Resultados de matching (de SimilarityEngine)
            anomaly_result: Resultado de anomalías (de AnomalyDetectionEngine)
            preprocessed_spectrum: Espectro preprocesado (de PreprocessingEngine)
            theta_features: Features theta (opcional)
            peak_result: Resultado de extracción de picos (opcional)
            sample_metadata: Metadata de la muestra
            top_n: Número de identificaciones a incluir
        """
        timestamp = datetime.now().isoformat()

        # Convertir match results a identificaciones
        identifications = []
        for i, match in enumerate(match_results[:top_n]):
            ident = IdentificationResult(
                rank=i + 1,
                material_name=match.reference_name,
                material_id=match.reference_id,
                match_score=match.combined_score,
                confidence=match.confidence,
                match_details={
                    'cosine': round(match.cosine_score, 4),
                    'sam': round(match.sam_score, 4),
                    'theta': round(match.theta_score, 4),
                    'peak_match': round(match.peak_match_score, 4)
                },
                explanation=match.explanation
            )
            identifications.append(ident)

        best_match = identifications[0] if identifications else None

        # Anomalías
        anomalies = [a.to_dict() for a in anomaly_result.anomalies] if anomaly_result else []
        anomaly_risk = anomaly_result.overall_risk if anomaly_result else "none"

        # Trazabilidad
        processing_trace = preprocessed_spectrum.get_log() if preprocessed_spectrum else []

        # Features theta
        theta_dict = None
        if theta_features:
            theta_dict = {
                'q_optimal': round(theta_features.q_optimal, 6),
                'invariants': {k: round(v, 6) if isinstance(v, float) else v 
                              for k, v in theta_features.invariants.items()},
                'phi_dimension': len(theta_features.phi_signature),
                'psi_scales': len(theta_features.psi_multiscale)
            }

        # Resumen de picos
        peak_summary = None
        if peak_result:
            peak_summary = {
                'total_peaks': len(peak_result.peaks),
                'dominant_peaks': [
                    p.to_dict() for p in peak_result.get_dominant_peaks(5)
                ]
            }

        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            best_match, anomaly_result, identifications
        )

        return ScientificReport(
            sample_id=sample_id,
            timestamp=timestamp,
            identifications=identifications,
            best_match=best_match,
            anomalies=anomalies,
            anomaly_risk=anomaly_risk,
            processing_trace=processing_trace,
            theta_features=theta_dict,
            peak_summary=peak_summary,
            sample_metadata=sample_metadata or {},
            recommendations=recommendations
        )

    def _generate_recommendations(self,
                                   best_match,
                                   anomaly_result,
                                   identifications) -> List[str]:
        """Genera recomendaciones basadas en resultados."""
        recommendations = []

        if best_match:
            if best_match.confidence < 0.5:
                recommendations.append(
                    "Baja confianza en identificación. Considerar medir con mayor resolución "
                    "o en diferentes condiciones experimentales."
                )

            if best_match.match_score < 0.6:
                recommendations.append(
                    "Match débil. El material podría ser una mezcla o una variante no catalogada. "
                    "Considerar análisis complementario (XRD, SEM-EDS)."
                )

            if len(identifications) > 1 and identifications[1].match_score > 0.7:
                recommendations.append(
                    f"Identificación ambigua: {identifications[1].material_name} también tiene "
                    f"alta similaridad ({identifications[1].match_score:.3f}). "
                    "Revisar picos discriminantes específicos."
                )

        if anomaly_result:
            if anomaly_result.overall_risk in ['high', 'critical']:
                recommendations.append(
                    "Anomalías significativas detectadas. Recomendado: (1) Verificar calibración, "
                    "(2) Re-medir muestra, (3) Considerar posible nueva fase o impureza."
                )

            if anomaly_result.requires_feedback:
                recommendations.append(
                    f"Feedback requerido: {anomaly_result.feedback_reason} "
                    "El sistema reintentará preprocesamiento con parámetros ajustados."
                )

        if not recommendations:
            recommendations.append(
                "Identificación confiable. No se detectaron anomalías significativas."
            )

        return recommendations
