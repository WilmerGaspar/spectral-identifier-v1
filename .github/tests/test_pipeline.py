"""Tests del pipeline completo."""
import numpy as np
from src.pipeline import SpectralIdentifierPipeline
from src.data_loader import DataLoader


def test_pipeline_with_synthetic_data():
    """Test end-to-end con datos sintéticos."""
    pipeline = SpectralIdentifierPipeline()

    # Base de datos de prueba
    test_db = DataLoader.generate_test_database()
    pipeline.load_database(test_db)

    # Espectro desconocido (variante de cuarzo)
    unknown = DataLoader.generate_synthetic_spectrum(
        name="unknown_quartz",
        peak_positions=[129, 207, 266, 357, 395, 465, 697, 796, 1065],
        peak_intensities=[0.28, 0.14, 0.38, 0.19, 0.33, 0.95, 0.58, 0.23, 0.09],
        noise_level=0.03
    )

    # Identificar
    result = pipeline.identify(
        wavenumbers=unknown['wavenumbers'],
        intensities=unknown['intensities'],
        sample_id="test_quartz"
    )

    assert result['best_match'] is not None
    assert result['anomaly_risk'] in ['none', 'low', 'medium', 'high', 'critical']
    assert len(result['identifications']) > 0

    # El cuarzo debería ser el mejor match o tener score > 0.5
    top_score = result['identifications'][0]['match_score']
    assert top_score > 0.5, f"Score demasiado bajo: {top_score}"

    print(f"✅ Test passed! Best match: {result['identifications'][0]['material_name']} (score: {top_score:.3f})")


def test_theta_engine():
    """Test del motor theta con espectro simple."""
    from src.engines.theta_ramanujan import RamanujanThetaEngine

    engine = RamanujanThetaEngine(n_terms=30, n_scales=6)

    x = np.linspace(0, 1, 500)
    spectrum = np.exp(-((x - 0.3)**2) / 0.001) + 0.5 * np.exp(-((x - 0.7)**2) / 0.002)

    features = engine.transform(spectrum)

    assert features.q_optimal > 0 and features.q_optimal < 1
    assert len(features.phi_signature) == 16
    assert len(features.psi_multiscale) == 6
    assert 'spectral_entropy' in features.invariants

    print(f"✅ Theta engine test passed! q_optimal={features.q_optimal:.4f}")


def test_preprocessing_feedback():
    """Test del feedback loop de preprocesamiento."""
    from src.preprocessing.preprocessor import PreprocessingEngine, SpectrumMetadata

    preprocessor = PreprocessingEngine()

    wn = np.linspace(100, 1200, 1101)
    # Espectro con ruido y deriva de línea base
    inten = (np.exp(-((wn - 464)**2) / 50) + 
             0.3 * np.exp(-((wn - 128)**2) / 30) +
             0.05 * np.random.randn(1101) +
             0.1 * (wn - 100) / 1100)

    metadata = SpectrumMetadata(source="test", sample_name="feedback_test")
    result = preprocessor.process(wn, inten, metadata)

    assert len(result.intensities) > 0
    assert len(result.processing_log) > 0
    assert result.quality_flags is not None

    print(f"✅ Preprocessing test passed! Quality: {result.quality_flags.get('overall_pass', False)}")


if __name__ == '__main__':
    test_theta_engine()
    test_preprocessing_feedback()
    test_pipeline_with_synthetic_data()
    print("\n🎉 Todos los tests pasaron!")
