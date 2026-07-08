# 🔬 Spectral Identifier v1

**Theta-Ramanujan Spectral Analysis Engine for Raman and FTIR material identification**

Spectral Identifier v1 is an open-source research prototype for identifying and comparing scientific spectra using classical spectral processing combined with Theta-Ramanujan inspired mathematical descriptors.

The current version focuses on Raman and FTIR spectral analysis, material matching, anomaly detection, and scientific report generation.

---

## Project Status

This repository is an early-stage research prototype.

It is not a fully validated commercial or laboratory-certified system. The goal is to explore whether mathematical descriptors based on Theta-Ramanujan representations can support spectral identification, comparison, anomaly detection, and future AI-assisted materials analysis.

---

## Current Architecture

```text
Spectral Data Sources
        ↓
Preprocessing Engine
        ↓
Theta-Ramanujan Engine
        ↓
Peak Extraction
        ↓
Similarity and Matching
        ↓
Anomaly Detection
        ↓
Scientific Report
```

---

## GTSF v2 Development Branch

This branch introduces the first lightweight step toward the **Geometric Theta Spectral Framework (GTSF)**.

GTSF expands the original spectral identifier into a modular mathematical framework where spectra are not only compared by peaks, but transformed into interpretable mathematical descriptors.

The long-term idea is to study spectra as mathematical objects with geometric, statistical, and structural properties.

---

## Minimal GTSF Core

The file:

```text
src/gtsf_minimal.py
```

adds a lightweight descriptor prototype that computes:

- spectral entropy
- arc length
- roughness index
- curvature-like energy
- theta-inspired statistical moments

The current GTSF minimal core follows this simple flow:

```text
Raw spectrum
        ↓
Normalization
        ↓
Mathematical descriptor extraction
        ↓
GTSF descriptor dictionary
```

This is not a full Jacobi theta implementation yet. It is a safe and lightweight foundation for future development.

---

## Scientific Caution

This project generates mathematical descriptors from spectral data.

It does not claim to discover new physical properties by itself. Any physical interpretation must be validated with real spectral databases, experimental references, and comparison against classical spectral analysis methods.

The current objective is to build a reproducible research foundation, not to claim complete experimental validation.

---

## Supported Spectral Inputs

The project is designed for spectral data such as:

- Raman spectra
- FTIR spectra
- text-based spectral files
- CSV spectral data
- JSON spectral input

Future versions may expand toward:

- XRD
- LIBS
- UV-Vis
- additional scientific spectra

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/health` | GET | System health check |
| `/api/identify` | POST | Identify a spectrum from file input |
| `/api/identify/json` | POST | Identify a spectrum from JSON input |
| `/api/database` | GET | List loaded reference materials |
| `/api/database/add` | POST | Add a reference material |
| `/api/compare` | POST | Compare two spectra directly |

---

## Theta-Ramanujan Engine

The current mathematical engine is based on Theta-Ramanujan inspired spectral transformations.

Implemented or planned components include:

- **q-map**: modular parameter mapping
- **phi descriptor**: spectral signature representation
- **psi descriptor**: multiscale spectral analysis
- **theta-inspired moments**: early interpretable descriptors
- **spectral complexity measures**
- **entropy-based descriptors**

---

## GTSF Research Direction

The Geometric Theta Spectral Framework aims to evolve the project into a multilayer mathematical architecture:

```text
Layer 1: Preprocessing
Layer 2: Theta-Ramanujan Representation
Layer 3: Spectral Manifold Construction
Layer 4: Differential Geometry Engine
Layer 5: Structural Descriptor Engine
Layer 6: Decision and AI Engine
```

The objective is to create interpretable descriptors that can support:

- material identification
- spectral comparison
- anomaly detection
- clustering
- ranking
- unknown material detection
- AI-ready scientific features

---

## Quick Start

```bash
git clone https://github.com/WilmerGaspar/spectral-identifier-v1.git
cd spectral-identifier-v1
pip install -r requirements.txt
python run.py
```

Then open:

```text
http://localhost:5000
```

---

## Run the Minimal GTSF Descriptor Demo

```bash
python src/gtsf_minimal.py
```

This runs a tiny demo spectrum and prints the generated mathematical descriptors.

---

## Repository Structure

```text
spectral-identifier-v1/
│
├── frontend/
│   └── templates/
│
├── src/
│   ├── anomaly/
│   ├── engines/
│   ├── matching/
│   ├── preprocessing/
│   ├── reporting/
│   ├── api.py
│   ├── data_loader.py
│   ├── pipeline.py
│   └── gtsf_minimal.py
│
├── requirements.txt
├── run.py
├── Procfile
└── README.md
```

---

## Deployment

### Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn src.api:app
```

### PythonAnywhere

Upload the project files, configure the Flask WSGI entry point, and reload the app.

---

## Roadmap

### v0.1

- Basic Raman/FTIR spectral identifier
- API endpoints
- Web interface
- Reference database support
- Similarity and matching engine

### v0.2

- Minimal GTSF descriptor module
- Entropy descriptor
- Arc length descriptor
- Roughness descriptor
- Curvature-like energy descriptor
- Theta-inspired moments

### v0.3

- Add sample Raman/FTIR datasets
- Add reproducible notebooks
- Compare GTSF descriptors with classical peak matching
- Add basic tests

### v1.0

- Full documentation
- Public demo dataset
- Improved validation
- Exportable scientific reports
- Zenodo release

### GTSF v2

- Real Theta-Ramanujan descriptor implementation
- Spectral manifold construction
- Differential geometry descriptors
- Clustering and anomaly detection
- AI-ready descriptor matrix
- Validation against public spectral databases

---

## Limitations

- This is an early research prototype.
- Current descriptors require validation with real spectral datasets.
- The minimal GTSF module is not a complete mathematical implementation of the full framework.
- Results should not be interpreted as certified laboratory identification.
- Physical meaning must be tested against known references and experimental data.

---

## Author

**Wilmer Gaspar Espinoza Castillo**

Research interests:

- materials informatics
- spectroscopy
- mathematical descriptors
- Theta-Ramanujan representations
- scientific AI
- spectral analysis
- open research tools

---

## License

License to be defined.

---

## Citation

A formal citation will be added after the first stable release and Zenodo DOI.
