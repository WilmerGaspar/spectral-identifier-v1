"""
Spectral Identifier API
=======================
API REST para identificación espectral online.
Endpoints para upload, identificación y reportes.

Autor: Spectral Identifier Team
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile

# Importar pipeline
from src.pipeline import SpectralIdentifierPipeline
from src.data_loader import DataLoader

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'jdx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pipeline global
pipeline = None


def get_pipeline():
    """Singleton del pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = SpectralIdentifierPipeline(config={
            'target_resolution': 1.0,
            'wavenumber_range': (100, 1200),
            'theta_terms': 30,
            'theta_scales': 6,
            'theta_phases': 12
        })

        # Cargar base de datos de prueba
        print("Cargando base de datos de prueba...")
        test_db = DataLoader.generate_test_database()
        pipeline.load_database(test_db)

    return pipeline


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Página principal con interfaz web."""
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check."""
    p = get_pipeline()
    stats = p.get_database_stats()
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'database': stats
    })


@app.route('/api/identify', methods=['POST'])
def identify():
    """
    Identifica un espectro subido.

    Formato multipart:
    - file: archivo de espectro (.txt, .csv, .jdx)
    - sample_id: identificador de muestra (opcional)
    - metadata: JSON string con metadata (opcional)
    """
    try:
        p = get_pipeline()

        # Obtener archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'Format not allowed. Use: {ALLOWED_EXTENSIONS}'}), 400

        # Guardar temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Cargar espectro
        ext = filename.rsplit('.', 1)[1].lower()

        if ext == 'txt':
            spectrum_data = DataLoader.load_rruff(filepath)
        elif ext == 'csv':
            spectrum_data = DataLoader.load_csv(filepath)
        elif ext == 'jdx':
            spectrum_data = DataLoader.load_nist_jdx(filepath)
        else:
            return jsonify({'error': 'Unsupported format'}), 400

        # Metadata
        sample_id = request.form.get('sample_id', spectrum_data['name'])
        metadata_str = request.form.get('metadata', '{}')
        metadata = json.loads(metadata_str)

        # Identificar
        result = p.identify(
            wavenumbers=spectrum_data['wavenumbers'],
            intensities=spectrum_data['intensities'],
            sample_id=sample_id,
            sample_metadata=metadata
        )

        # Limpiar archivo temporal
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/identify/json', methods=['POST'])
def identify_json():
    """
    Identifica un espectro enviado como JSON.

    Body JSON:
    {
        "wavenumbers": [100.0, 101.0, ...],
        "intensities": [0.0, 0.1, ...],
        "sample_id": "muestra_001",
        "metadata": {...}
    }
    """
    try:
        p = get_pipeline()
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        wavenumbers = np.array(data['wavenumbers'])
        intensities = np.array(data['intensities'])
        sample_id = data.get('sample_id', 'unknown')
        metadata = data.get('metadata', {})

        result = p.identify(
            wavenumbers=wavenumbers,
            intensities=intensities,
            sample_id=sample_id,
            sample_metadata=metadata
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/database', methods=['GET'])
def get_database():
    """Obtiene información de la base de datos cargada."""
    p = get_pipeline()
    stats = p.get_database_stats()

    # Lista de materiales
    materials = []
    for entry in p.database:
        materials.append({
            'id': entry['id'],
            'name': entry['name'],
            'metadata': entry['metadata']
        })

    return jsonify({
        'stats': stats,
        'materials': materials
    })


@app.route('/api/database/add', methods=['POST'])
def add_reference():
    """Añade una referencia a la base de datos."""
    try:
        p = get_pipeline()
        data = request.get_json()

        if not data or 'wavenumbers' not in data or 'intensities' not in data:
            return jsonify({'error': 'wavenumbers and intensities required'}), 400

        entry = {
            'id': data.get('id', f"ref_{len(p.database)}"),
            'name': data.get('name', 'Unknown'),
            'wavenumbers': np.array(data['wavenumbers']),
            'intensities': np.array(data['intensities']),
            'source': data.get('source', 'user'),
            'metadata': data.get('metadata', {})
        }

        p.add_reference(entry)

        return jsonify({
            'status': 'added',
            'id': entry['id'],
            'database_size': len(p.database)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_spectra():
    """
    Compara dos espectros directamente.

    Body JSON:
    {
        "spectrum_a": {"wavenumbers": [...], "intensities": [...]},
        "spectrum_b": {"wavenumbers": [...], "intensities": [...]}
    }
    """
    try:
        from src.preprocessing.preprocessor import PreprocessingEngine
        from src.engines.theta_ramanujan import RamanujanThetaEngine
        from src.matching.similarity import SimilarityEngine

        data = request.get_json()

        spec_a = data['spectrum_a']
        spec_b = data['spectrum_b']

        wn_a = np.array(spec_a['wavenumbers'])
        int_a = np.array(spec_a['intensities'])
        wn_b = np.array(spec_b['wavenumbers'])
        int_b = np.array(spec_b['intensities'])

        # Preprocesar
        preprocessor = PreprocessingEngine()
        prep_a = preprocessor.process(wn_a, int_a)
        prep_b = preprocessor.process(wn_b, int_b)

        # Features theta
        theta = RamanujanThetaEngine()
        feat_a = theta.transform(prep_a.intensities)
        feat_b = theta.transform(prep_b.intensities)

        # Similaridad
        sim = SimilarityEngine()
        scores = sim.combined_score(
            prep_a.intensities, prep_b.intensities,
            features_a=feat_a, features_b=feat_b
        )

        return jsonify(scores)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    print(f"Starting Spectral Identifier API on port {port}")
    print(f"Debug mode: {debug}")

    app.run(host='0.0.0.0', port=port, debug=debug)
