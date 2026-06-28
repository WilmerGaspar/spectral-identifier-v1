"""
Spectral Identifier Pipeline
============================
Orquestador que conecta todos los motores con feedback loop.

Flujo corregido:
    Preprocessing → Theta Engine → Peak Extraction → 
    Similarity → Anomaly Detection → [Feedback Loop] → Report

Autor: Spectral Identifier Team
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

# Imports de motores
from src.preprocessing.preprocessor import PreprocessingEngine, SpectrumMetadata
from src.engines.theta_ramanujan import RamanujanThetaEngine
from src.engines.peak_extractor import PeakExtractionEngine
from src.matching.similarity import SimilarityEngine
from src.anomaly.detector import AnomalyDetectionEngine
from src.reporting.report import ReportGenerator


class SpectralIdentifierPipeline:
    """
    Pipeline completo de identificación espectral.

    Con feedback loop: si Anomaly Detection detecta problemas,
    puede reintentar preprocesamiento con parámetros ajustados.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Inicializar motores
        self.preprocessor = PreprocessingEngine(
            target_resolution=self.config.get('target_resolution', 1.0),
            wavenumber_range=self.config.get('wavenumber_range', (100, 4000))
        )

        self.theta_engine = RamanujanThetaEngine(
            n_terms=self.config.get('theta_terms', 50),
            n_scales=self.config.get('theta_scales', 8),
            n_phases=self.config.get('theta_phases', 16)
        )

        self.peak_extractor = PeakExtractionEngine(
            min_peak_height=self.config.get('min_peak_height', 0.05),
            min_distance=self.config.get('peak_min_distance', 10)
        )

        self.similarity_engine = SimilarityEngine(
            weights=self.config.get('similarity_weights')
        )

        self.anomaly_detector = AnomalyDetectionEngine(
            unknown_peak_threshold=self.config.get('unknown_peak_threshold', 0.15)
        )

        self.report_generator = ReportGenerator()

        # Base de datos en memoria
        self.database = []

    # ─────────────────────────────────────────────────────────
    # CARGA DE BASE DE DATOS
    # ─────────────────────────────────────────────────────────

    def load_database(self, database_entries: List[Dict]):
        """
        Carga base de datos de referencia.

        Cada entrada debe tener:
        {
            'id': str,
            'name': str,
            'wavenumbers': np.ndarray,
            'intensities': np.ndarray,
            'metadata': dict (opcional)
        }
        """
        self.database = []

        for entry in database_entries:
            # Preprocesar entrada de referencia
            wn = np.array(entry['wavenumbers'])
            inten = np.array(entry['intensities'])

            metadata = SpectrumMetadata(
                source=entry.get('source', 'database'),
                sample_name=entry.get('name', 'unknown'),
                mineral_class=entry.get('mineral_class')
            )

            preprocessed = self.preprocessor.process(wn, inten, metadata)

            # Calcular features theta
            theta_features = self.theta_engine.transform(preprocessed.intensities)

            # Extraer picos
            peak_result = self.peak_extractor.extract(
                preprocessed.wavenumbers,
                preprocessed.intensities,
                theta_intensities=theta_features.theta_spectrum
            )

            self.database.append({
                'id': entry.get('id', f"ref_{len(self.database)}"),
                'name': entry.get('name', 'Unknown'),
                'spectrum': preprocessed.intensities,
                'wavenumbers': preprocessed.wavenumbers,
                'features': theta_features,
                'peaks': peak_result.peaks,
                'metadata': preprocessed.metadata.to_dict()
            })

        print(f"Base de datos cargada: {len(self.database)} referencias")

    # ─────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def identify(self,
                 wavenumbers: np.ndarray,
                 intensities: np.ndarray,
                 sample_id: str = "unknown",
                 sample_metadata: Optional[dict] = None,
                 max_feedback_loops: int = 2) -> dict:
        """
        Identifica un espectro desconocido.

        Args:
            wavenumbers: Array de números de onda (cm⁻¹)
            intensities: Array de intensidades
            sample_id: Identificador de la muestra
            sample_metadata: Metadata adicional
            max_feedback_loops: Máximo de reintentos por anomalías

        Returns:
            dict con reporte completo
        """
        if len(self.database) == 0:
            raise ValueError("Base de datos vacía. Cargar referencias primero.")

        # Metadata
        metadata = SpectrumMetadata(
            source="unknown",
            sample_name=sample_id,
            **(sample_metadata or {})
        )

        # ── FASE 1: PREPROCESSING ──
        preprocessed = self.preprocessor.process(wavenumbers, intensities, metadata)

        feedback_count = 0
        final_report = None

        while feedback_count <= max_feedback_loops:
            # ── FASE 2: THETA-RAMANUJAN ENGINE ──
            theta_features = self.theta_engine.transform(preprocessed.intensities)

            # ── FASE 3: PEAK EXTRACTION ──
            peak_result = self.peak_extractor.extract(
                preprocessed.wavenumbers,
                preprocessed.intensities,
                theta_intensities=theta_features.theta_spectrum
            )

            # ── FASE 4: SIMILARITY + MATCHING ──
            match_results = self.similarity_engine.match_against_database(
                unknown_spectrum=preprocessed.intensities,
                database=self.database,
                unknown_features=theta_features,
                unknown_peaks=peak_result.peaks,
                unknown_metadata=preprocessed.metadata.to_dict(),
                top_k=10
            )

            # ── FASE 5: ANOMALY DETECTION ──
            best_match = match_results[0] if match_results else None
            reference_peaks = []

            if best_match:
                ref_entry = next(
                    (r for r in self.database if r['id'] == best_match.reference_id),
                    None
                )
                if ref_entry:
                    reference_peaks = ref_entry['peaks']

            anomaly_result = self.anomaly_detector.detect(
                unknown_peaks=peak_result.peaks,
                reference_peaks=reference_peaks,
                match_confidence=best_match.confidence if best_match else 0
            )

            # ── FASE 6: FEEDBACK LOOP ──
            if anomaly_result.requires_feedback and feedback_count < max_feedback_loops:
                feedback_count += 1
                print(f"Feedback loop {feedback_count}: {anomaly_result.feedback_reason}")

                # Ajustar preprocesamiento y reintentar
                preprocessed = self._reprocess_with_adjustments(
                    preprocessed, anomaly_result
                )
                continue

            # ── FASE 7: GENERAR REPORTE ──
            report = self.report_generator.generate(
                sample_id=sample_id,
                match_results=match_results,
                anomaly_result=anomaly_result,
                preprocessed_spectrum=preprocessed,
                theta_features=theta_features,
                peak_result=peak_result,
                sample_metadata=sample_metadata
            )

            final_report = report
            break

        return final_report.to_dict()

    def _reprocess_with_adjustments(self, preprocessed_spectrum, anomaly_result):
        """Re-procesa el espectro con parámetros ajustados según anomalías."""
        # Obtener espectro original
        wn = preprocessed_spectrum.original_wavenumbers
        inten = preprocessed_spectrum.original_intensities
        metadata = preprocessed_spectrum.metadata

        # Ajustar parámetros según tipo de anomalía
        if anomaly_result.overall_risk == 'critical':
            # Más agresivo en limpieza
            self.preprocessor.quality_thresholds['min_snr'] *= 0.8

        # Re-procesar
        return self.preprocessor.process(wn, inten, metadata)

    # ─────────────────────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────────────────────

    def add_reference(self, entry: Dict):
        """Añade una referencia individual a la base de datos."""
        self.load_database([entry])

    def get_database_stats(self) -> dict:
        """Estadísticas de la base de datos cargada."""
        if not self.database:
            return {'count': 0}

        names = [e['name'] for e in self.database]
        sources = set(e['metadata'].get('source', 'unknown') for e in self.database)

        return {
            'count': len(self.database),
            'unique_names': len(set(names)),
            'sources': list(sources)
        }
