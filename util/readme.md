# Utilities

The `util/` directory contains scripts for acquiring RRUFF spectra, converting the source text
files into tables and metadata, and building compact Raman and powder XRD benchmark subsets. These
are data-preparation utilities; they are separate from the Flask application and can be run on
their own.

## Scripts

| Script | Purpose |
|---|---|
| `main.py` | Runs the complete download, parse, and optional source-cleanup workflow for one RRUFF dataset. |
| `extract_data_rruff.py` | Downloads selected RRUFF zip files, extracts them under `data/source/`, and records completed downloads. |
| `process_data.py` | Converts each source `.txt` spectrum into a CSV and groups its header metadata into JSON. Also provides guarded cleanup of verified source files. |
| `make_subset.py` | Builds a small, normalized Raman reference subset for repeatable identification benchmarks. |
| `make_xrd_subset.py` | Builds a small, normalized measured powder XRD subset for theta/lattice-fit validation. |

## Requirements and location

Use Python 3.8 or newer. From the repository root, install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

The pipeline requires network access to `https://rruff.net/zipped_data_files/`. It uses `requests`
for downloads. The subset builders require NumPy, which is also in `requirements.txt`.

Run commands from the repository root (the directory containing `util/`). Data paths are resolved
relative to the scripts, but `main.py` writes `pipeline.log` in the current working directory.

## Download and parse

The main command downloads any unseen archives for a dataset, extracts them, then parses every `.txt`
file currently under that dataset's source directory:

```powershell
python util/main.py
```

By default this processes Raman. Other examples:

```powershell
python util/main.py --dataset powder --only XY_Processed.zip
python util/main.py --dataset infrared
python util/main.py --dataset raman --only excellent_unoriented.zip fair_unoriented.zip
```

`--only` accepts one or more exact `.zip` names listed by RRUFF for that dataset. The available
dataset choices are `raman`, `powder`, and `infrared`.

| Option | Effect |
|---|---|
| `--dataset NAME` | Choose the RRUFF dataset; defaults to `raman`. |
| `--only ZIP [ZIP ...]` | Restrict downloading to the named archive(s). Existing source files are still parsed. |
| `--clean` | After parsing, delete files under `data/source/<dataset>/` only if every source `.txt` has a CSV and there are no recorded parse failures. |

The pipeline returns exit code `0` when downloads and parsing report no problems, or `1` if a
download failed or at least one source file failed to parse. A refused cleanup is reported in the
log; it does not remove the parsed output. `--clean` is destructive to the unzipped originals, so
inspect the parsed results before using it:

```powershell
python util/main.py --dataset raman --clean
```

This command still performs download and parsing before attempting cleanup; it is not a standalone
cleanup-only operation.

### Pipeline directories

The scripts create these directories as needed:

```text
data/
├── .downloaded.json
├── source/<dataset>/<archive folder>/*.txt
└── parsed/<dataset>/
    ├── raw/
    │   ├── csv/<source folder>/<spectrum>.csv
    │   └── json/<source folder>.json
    ├── processed/
    │   ├── csv/<source folder>/<spectrum>.csv
    │   └── json/<source folder>.json
    └── failed/json/<source folder>.json
pipeline.log
```

Raman archive folders may encode quality and orientation, such as `excellent_unoriented`; the
metadata parser extracts those values when the folder matches that naming pattern. Other folders,
including `LR-Raman`, do not receive quality/orientation values.

`.downloaded.json` records archive names already handled, scoped by dataset. Do not delete it unless
you intend to allow previously downloaded archives to be fetched again. `pipeline.log` contains
download, parsing, and cleanup messages.

## Build benchmark subsets

Run these only after the corresponding source data has been parsed.

### Raman

First download and parse Raman data, then build a compact subset:

```powershell
python util/main.py --dataset raman
python util/make_subset.py
```

Defaults select processed spectra from `excellent_unoriented` at 532 nm. Eligible minerals need at
least two distinct RRUFF sample IDs. Spectra are interpolated to a shared 150-1500 cm-1 grid at
2 cm-1 intervals, baseline-shifted to a zero minimum, and scaled to a maximum of one. The default
limit is 40 minerals and three samples per mineral.

Options allow changing the source folder, wavelength, and limits:

```powershell
python util/make_subset.py --folder excellent_unoriented --wavelength 514 --minerals 40 --per-mineral 3
```

Outputs:

```text
data/rruff_subset/raman_subset.npz
data/rruff_subset/subset_index.csv
```

The NPZ stores the shared `grid`, stacked `spectra`, mineral names, and RRUFF IDs. The CSV index also
includes the source stem and ideal chemistry when present.

### Powder XRD

Download and parse measured `XY_Processed` powder patterns, then build the subset:

```powershell
python util/main.py --dataset powder --only XY_Processed.zip
python util/make_xrd_subset.py
```

The builder selects a configured list of cubic minerals, keeps up to four patterns per mineral,
interpolates to a 5-90 degree 2-theta grid at 0.02 degree intervals, shifts the minimum to zero, and
scales the maximum to one. It uses measured XY patterns, not `DIF` peak-list files.

Outputs:

```text
data/rruff_xrd_subset/xrd_subset.npz
data/rruff_xrd_subset/xrd_index.json
```

The NPZ stores `grid` and stacked `patterns`. The JSON index records each mineral, configured space
group, RRUFF ID, source stem/folder, observed x range, and available wavelength/cell metadata.

## Parsed file formats

Each CSV represents one spectrum and has two columns. The first column depends on dataset:

| Dataset | First column | Second column |
|---|---|---|
| Raman | `wavenumber` | `intensity` |
| Infrared | `wavenumber` | `intensity` |
| Powder | `two_theta` | `intensity` |

Values are written as strings from the source numeric rows; the parser does not resample, sort, or
round them. Each JSON file contains one entry per spectrum, keyed by the corresponding CSV stem.
Original header fields are retained alongside `dataset`, `source_folder`, `quality`, `orientation`,
`source_file`, and `n_points`. Failed files are listed with an error message under
`parsed/<dataset>/failed/json/` and do not prevent other files from being processed.

To load an example with Python:

```python
import json
import pandas as pd

folder = "excellent_unoriented"
name = "spectrum_name"
metadata_path = f"data/parsed/raman/processed/json/{folder}.json"
csv_path = f"data/parsed/raman/processed/csv/{folder}/{name}.csv"

with open(metadata_path, encoding="utf-8") as file:
    metadata = json.load(file)
spectrum = pd.read_csv(csv_path)
print(metadata[name].get("RAMAN WAVELENGTH"), len(spectrum))
```

Pandas is only needed for this example; it is not required by the utilities.

## Parsing and download behavior

- A file is classified as raw when its `FILETYPE` header contains `raw` as a separate word (including underscores). If there is no `FILETYPE`, a standalone uppercase `RAW` in the filename is used. All other spectra are processed.
- Header fields are read from `##KEY=value` lines. An `##END=` marker is optional. Numeric data may be comma-, tab-, space-, or semicolon-separated; rows without two numeric values are skipped with a warning.
- A source file fails only if it has no usable numeric x/y rows or another processing error occurs. Failure JSON is cleared and regenerated for the latest split run.
- Long trailing hex IDs are removed from output stems when possible. If this would cause a filename collision within a source folder, the full stems are retained to prevent overwriting.
- Every pipeline run re-processes all `.txt` files still present in `data/source/<dataset>/`; it does not only parse newly downloaded archives.
- The download record stores archive names, not remote modification times. Updated server-side archives are not automatically detected. To fetch one again, remove its dataset/archive entry from `data/.downloaded.json`.
- Failed downloads are not marked complete and can be retried on a later run. Parsing still runs on source files already present when the download step fails.
- Cleanup verifies the expected CSVs and checks that no failures are listed before removing source contents. It never removes `data/parsed/` or `.downloaded.json`.

## Run individual steps

To parse already-extracted files without downloading, run:

```powershell
python util/process_data.py
```

This standalone entry point parses Raman only. The processing functions can also be imported and
called for a selected dataset from a Python script; multiprocessing requires the usual main guard:

```python
import logging

from util.extract_data_rruff import BASE_URL, sync
from util.process_data import clean_source, split_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync(BASE_URL + "powder/", only=["XY_Processed.zip"])
    counts = split_all("powder")
    print(counts)  # raw, processed, and failed counts
    clean_source("powder")
```

For most users, `util/main.py` is the recommended entry point because it configures logging and
provides the dataset/selection flags.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `No new files` | Usually means all listed archives are already in `.downloaded.json`; the pipeline still processes source files on disk. |
| No files are parsed | Confirm source `.txt` files exist under `data/source/<dataset>/`. |
| A file appears in `failed/` | Read its message in `data/parsed/<dataset>/failed/json/<folder>.json`. |
| Cleanup is refused | Resolve listed failures or rerun parsing so every source `.txt` has a matching CSV. |
| A subset command reports no eligible spectra | Check that the parsed output exists, and that the requested Raman folder/wavelength or supported XRD mineral set is present. |
| Download fails | Check network access and RRUFF availability; parsing will still process source files already on disk. |

The source labels `raw` and `processed` describe RRUFF file metadata, not guaranteed experimental
preparation. For scientific comparisons, inspect intensity ranges, sampling intervals, and baseline
behavior before treating spectra as directly comparable. The configured powder XRD space groups and
Raman selection defaults are intended for these research benchmarks, not as independent validation
of mineral identity.