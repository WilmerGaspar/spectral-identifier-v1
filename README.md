# 🔬 Spectral Identifier — Theta-Ramanujan Engine

Identificación de materiales mediante espectroscopia Raman/FTIR con motor matemático basado en funciones theta de Jacobi-Ramanujan.

## Arquitectura

```
Spectral Data Sources → Preprocessing Engine → Theta-Ramanujan Engine → Peak Extraction → Similarity + Matching → Anomaly Detection → Scientific Report
```

## Inicio rápido

```bash
# 1. Clonar
git clone https://github.com/tu-usuario/spectral-identifier.git
cd spectral-identifier

# 2. Instalar
pip install -r requirements.txt

# 3. Ejecutar
python run.py

# 4. Abrir http://localhost:5000
```

## API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Interfaz web |
| `/api/health` | GET | Estado del sistema |
| `/api/identify` | POST | Identifica espectro desde archivo (.txt, .csv, .jdx) |
| `/api/identify/json` | POST | Identifica desde JSON directo |
| `/api/database` | GET | Lista materiales de referencia cargados |
| `/api/database/add` | POST | Añade referencia a la base de datos |
| `/api/compare` | POST | Compara dos espectros directamente |

## Motor Theta-Ramanujan

Transformaciones implementadas:
- **q-map**: Optimización del parámetro modular q (FFT-accelerated, O(N log N))
- **phi**: Firma espectral invariante a traslación y escala (función theta₃)
- **psi**: Análisis multiescala con funciones theta₂ y theta₄

## Deploy

### Render (gratis)
1. Sube a GitHub
2. Conecta repo en [render.com](https://render.com)
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn src.api:app`

### PythonAnywhere (alternativa gratis)
- Upload files → Configurar WSGI para Flask → Reload app

## Licencia

MIT
