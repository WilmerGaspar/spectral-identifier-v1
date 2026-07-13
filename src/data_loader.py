"""
Data Loader
===========
Carga espectros desde múltiples fuentes:
- RRUFF (formato .txt)
- NIST (formato .jdx)
- CSV genérico
- Archivos propios

Autor: Spectral Identifier Team
"""

import numpy as np
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings


class DataLoader:
    """Cargador universal de espectros espectroscópicos."""

    # ─────────────────────────────────────────────────────────
    # RRUFF FORMAT
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def load_rruff(filepath: str) -> Dict:
        """
        Carga espectro en formato RRUFF (.txt).

        Formato RRUFF:
        - Líneas de metadata precedidas por ##
        - Datos: wavenumber intensity
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        metadata = {}
        wavenumbers = []
        intensities = []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Metadata
                if line.startswith('##'):
                    key_value = line[2:].strip()
                    if '=' in key_value:
                        key, value = key_value.split('=', 1)
                        metadata[key.strip()] = value.strip()
                else:
                    # Datos numéricos
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            wn = float(parts[0])
                            inten = float(parts[1])
                            wavenumbers.append(wn)
                            intensities.append(inten)
                        except ValueError:
                            continue

        return {
            'id': metadata.get('RRUFFID', filepath.stem),
            'name': metadata.get('MINERAL', filepath.stem),
            'source': 'RRUFF',
            'wavenumbers': np.array(wavenumbers),
            'intensities': np.array(intensities),
            'metadata': {
                'mineral_class': metadata.get('MINERAL'),
                'chemistry': metadata.get('CHEMISTRY'),
                'locality': metadata.get('LOCALITY'),
                'owner': metadata.get('OWNER'),
                'status': metadata.get('STATUS'),
                'sample_id': metadata.get('SAMPLE_ID'),
                'crystal_system': metadata.get('CRYSTAL_SYSTEM'),
                'orientation': metadata.get('ORIENTATION'),
                'laser_wavelength': metadata.get('LASER_WAVELENGTH'),
                'resolution': metadata.get('RESOLUTION')
            }
        }

    @staticmethod
    def load_rruff_directory(directory: str, limit: Optional[int] = None) -> List[Dict]:
        """Carga todos los archivos RRUFF de un directorio."""
        directory = Path(directory)
        files = list(directory.glob('*.txt'))

        if limit:
            files = files[:limit]

        spectra = []
        for filepath in files:
            try:
                spectrum = DataLoader.load_rruff(str(filepath))
                spectra.append(spectrum)
            except Exception as e:
                warnings.warn(f"Error cargando {filepath}: {e}")

        print(f"Cargados {len(spectra)} espectros RRUFF de {len(files)} archivos")
        return spectra

    # ─────────────────────────────────────────────────────────
    # NIST FORMAT (.jdx)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def load_nist_jdx(filepath: str) -> Dict:
        """
        Carga espectro en formato JCAMP-DX de NIST (.jdx).
        Formato simplificado — para JCAMP completo usar pyjcamp.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        metadata = {}
        wavenumbers = []
        intensities = []

        in_data = False
        x_factor = 1.0
        y_factor = 1.0

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Metadata
                if line.startswith('##'):
                    key_value = line[2:].strip()
                    if '=' in key_value:
                        key, value = key_value.split('=', 1)
                        key = key.strip().upper()
                        value = value.strip()
                        metadata[key] = value

                        if key == 'XFACTOR':
                            x_factor = float(value)
                        elif key == 'YFACTOR':
                            y_factor = float(value)
                        elif key == 'END':
                            in_data = False

                elif line.startswith('##XYDATA'):
                    in_data = True

                elif in_data and not line.startswith('##'):
                    # Datos: puede ser formato XY o X++(Y..Y)
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0]) * x_factor
                            y = float(parts[1]) * y_factor
                            wavenumbers.append(x)
                            intensities.append(y)
                        except ValueError:
                            continue

        return {
            'id': metadata.get('CAS REGISTRY NO', filepath.stem),
            'name': metadata.get('TITLE', filepath.stem),
            'source': 'NIST',
            'wavenumbers': np.array(wavenumbers),
            'intensities': np.array(intensities),
            'metadata': metadata
        }

    # ─────────────────────────────────────────────────────────
    # CSV FORMAT
    # ─────────────────────────────────────────────────────────

       @staticmethod
       def load_csv(
        filepath: str,
        wavenumber_col: str = "wavenumber",
        intensity_col: str = "intensity",
        delimiter: str = ","
    ) -> Dict:
        """
        Carga un espectro desde CSV.

        Acepta encabezados comunes como:
        - wavenumber,intensity
        - x,intensity
        - raman_shift,intensity
        """
        filepath = Path(filepath)

        wavenumbers = []
        intensities = []

        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            if reader.fieldnames is None:
                raise ValueError("CSV file has no header.")

            field_map = {
                field.strip().lower(): field
                for field in reader.fieldnames
                if field is not None
            }

            x_candidates = [
                wavenumber_col,
                "wavenumber",
                "wavenumbers",
                "x",
                "raman_shift",
                "raman shift",
                "shift"
            ]

            y_candidates = [
                intensity_col,
                "intensity",
                "intensities",
                "y"
            ]

            x_column = next(
                (
                    field_map[name.lower()]
                    for name in x_candidates
                    if name.lower() in field_map
                ),
                None
            )

            y_column = next(
                (
                    field_map[name.lower()]
                    for name in y_candidates
                    if name.lower() in field_map
                ),
                None
            )

            if x_column is None or y_column is None:
                raise ValueError(
                    f"Unsupported CSV headers: {reader.fieldnames}. "
                    "Expected spectral position and intensity columns."
                )

            for row in reader:
                try:
                    wn = float(row[x_column])
                    intensity = float(row[y_column])

                    wavenumbers.append(wn)
                    intensities.append(intensity)

                except (ValueError, TypeError, KeyError):
                    continue

        if not wavenumbers or not intensities:
            raise ValueError(
                "CSV contains no valid spectral data points."
            )

        return {
            "id": filepath.stem,
            "name": filepath.stem,
            "source": "CSV",
            "wavenumbers": np.array(wavenumbers, dtype=float),
            "intensities": np.array(intensities, dtype=float),
            "metadata": {
                "filename": str(filepath)
            }
            }
        }

    # ─────────────────────────────────────────────────────────
    # GENERADOR DE ESPECTROS DE PRUEBA
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def generate_synthetic_spectrum(
            name: str = "synthetic",
            peak_positions: Optional[List[float]] = None,
            peak_intensities: Optional[List[float]] = None,
            peak_widths: Optional[List[float]] = None,
            wn_range: Tuple[float, float] = (100, 1200),
            n_points: int = 1101,
            noise_level: float = 0.02,
            baseline_drift: float = 0.1) -> Dict:
        """
        Genera espectro sintético con picos gaussianos.
        Útil para pruebas sin datos reales.
        """
        wavenumbers = np.linspace(wn_range[0], wn_range[1], n_points)
        intensities = np.zeros(n_points)

        # Picos por defecto (simula cuarzo)
        if peak_positions is None:
            peak_positions = [128, 206, 265, 356, 394, 464, 696, 795, 1064]
            peak_intensities = [0.3, 0.15, 0.4, 0.2, 0.35, 1.0, 0.6, 0.25, 0.1]
            peak_widths = [3, 4, 5, 4, 3, 6, 4, 5, 3]

        # Generar picos gaussianos
        for pos, height, width in zip(peak_positions, peak_intensities, peak_widths):
            sigma = width / 2.355  # FWHM = 2.355 * sigma
            intensities += height * np.exp(-((wavenumbers - pos)**2) / (2 * sigma**2))

        # Línea base con deriva
        baseline = baseline_drift * (wavenumbers - wn_range[0]) / (wn_range[1] - wn_range[0])
        intensities += baseline

        # Ruido
        noise = noise_level * np.random.randn(n_points)
        intensities += noise

        # Asegurar no negativo
        intensities = np.maximum(intensities, 0)

        return {
            'id': name,
            'name': name,
            'source': 'synthetic',
            'wavenumbers': wavenumbers,
            'intensities': intensities,
            'metadata': {
                'peak_positions': peak_positions,
                'peak_intensities': peak_intensities,
                'noise_level': noise_level
            }
        }

    @staticmethod
    def generate_test_database(n_samples: int = 5,
                                base_name: str = "mineral") -> List[Dict]:
        """Genera base de datos de prueba con variaciones."""
        database = []

        # Materiales base con picos característicos
        materials = {
            'quartz': {
                'positions': [128, 206, 265, 356, 394, 464, 696, 795, 1064],
                'intensities': [0.3, 0.15, 0.4, 0.2, 0.35, 1.0, 0.6, 0.25, 0.1],
                'widths': [3, 4, 5, 4, 3, 6, 4, 5, 3]
            },
            'calcite': {
                'positions': [156, 282, 714, 1086, 1434],
                'intensities': [0.2, 0.3, 0.4, 1.0, 0.15],
                'widths': [4, 5, 4, 7, 3]
            },
            'gypsum': {
                'positions': [172, 340, 414, 494, 670, 1008, 1134],
                'intensities': [0.25, 0.3, 0.15, 0.2, 0.35, 1.0, 0.4],
                'widths': [3, 4, 3, 4, 5, 6, 4]
            },
            'feldspar': {
                'positions': [180, 290, 480, 510, 760, 815],
                'intensities': [0.3, 0.4, 0.5, 0.6, 0.35, 0.2],
                'widths': [4, 5, 6, 5, 4, 3]
            },
            'olivine': {
                'positions': [235, 420, 550, 820, 915, 960],
                'intensities': [0.4, 0.3, 0.5, 0.25, 0.35, 0.2],
                'widths': [5, 4, 6, 4, 5, 3]
            }
        }

        for i, (name, params) in enumerate(materials.items()):
            # Variar ligeramente para simular muestras reales
            np.random.seed(i * 42)

            pos_var = [p + np.random.normal(0, 1) for p in params['positions']]
            int_var = [max(0, h + np.random.normal(0, 0.05)) for h in params['intensities']]

            spectrum = DataLoader.generate_synthetic_spectrum(
                name=name,
                peak_positions=pos_var,
                peak_intensities=int_var,
                peak_widths=params['widths'],
                noise_level=0.02 + np.random.uniform(0, 0.02)
            )

            spectrum['metadata']['mineral_class'] = name
            database.append(spectrum)

        return database
