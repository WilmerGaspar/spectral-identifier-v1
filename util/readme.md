# RRUFF Raman Data Pipeline

Downloads the zipped Raman spectra from RRUFF, splits every spectrum into a
**CSV** (the numbers) and a **JSON** (the metadata), and sorts them into **raw**
and **processed** folders. Re-running only downloads zips it has not seen before.

## What it does

```
RRUFF server  ->  data/source/   ->  data/parsed/
   (zips)        (unzipped .txt)     (CSV + JSON, sorted raw / processed)
```

1. **Download** (`util/extract_data_rruff.py`): finds the `.zip` files listed on the RRUFF page,
   downloads the ones not downloaded before, unzips each into `data/source/`, then deletes the zip.
2. **Split** (`util/process_data.py`): reads every `.txt` in `data/source/` and writes the
   CSV and JSON files into `data/parsed/`.
3. **Clean** (optional): deletes the unzipped originals once the split has been verified.

The folder names under `source/` and `csv/` are the zip names from RRUFF. They carry the
quality rating (`excellent`, `fair`, `poor`, `unrated`) and orientation (`oriented`,
`unoriented`). `LR-Raman` has neither.

## Usage

Run from the project root (the folder that contains `util/`):

```bash
python util/main.py            # download new zips, then split everything
python util/main.py --clean    # also delete data/source/ if the split checks out
```

The exit code is `0` if everything worked and `1` if a download or any file failed.

## Output format

### CSV

One file per spectrum. The numbers are copied exactly as written in the original file
(nothing is rounded, sorted or cleaned).

```csv
wavenumber,intensity
147.1911,18520.79
149.1196,19625.49
```

The first column is `wavenumber` for Raman files and `x` for anything else.

### JSON

One file per source folder. Each key is a spectrum name (the same name as its CSV file),
and the value holds every header field from the original file plus these added fields:

| Field | Meaning |
|---|---|
| `source_folder` | Zip folder it came from, e.g. `excellent_oriented` |
| `quality` | `excellent`, `fair`, `poor` or `unrated`; `null` for folders that do not follow that pattern (e.g. `LR-Raman`) |
| `orientation` | `oriented` or `unoriented`; `null` as above |
| `source_file` | Original path under `data/source/`, including the long id in the name |
| `n_points` | Number of data rows in the CSV |

```json
{
  "Zykaite__R070477__Raman__785__0__unoriented__Raman_Data_RAW": {
    "NAMES": "Zykaite",
    "RRUFFID": "R070477",
    "FILETYPE": "Raman RAW",
    "RAMAN WAVELENGTH": "785",
    "source_folder": "excellent_unoriented",
    "quality": "excellent",
    "orientation": "unoriented",
    "source_file": "excellent_unoriented/Zykaite__...__RAW__c7ee56937fa26d4a89736f652793.txt",
    "n_points": 2181
  }
}
```

### Reading it back

```python
import json
import pandas as pd

meta = json.load(open("data/parsed/raw/json/excellent_unoriented.json"))
name = "Zykaite__R070477__Raman__785__0__unoriented__Raman_Data_RAW"
spectrum = pd.read_csv(f"data/parsed/raw/csv/excellent_unoriented/{name}.csv")
print(meta[name]["RAMAN WAVELENGTH"], len(spectrum))
```

## Rules the scripts follow

**Raw or processed.** A spectrum goes to `raw/` if its `FILETYPE` header contains the word
"raw". If there is no `FILETYPE` header, a standalone uppercase `RAW` in the file name is used.
Everything else goes to `processed/`.

**File names.** The long hex id at the end of RRUFF file names
(`..._Raman_Data_RAW__c7ee56937fa26d4a89736f652793`) is removed from the CSV names and JSON
keys. If removing it would give two files in the same source folder the same name, those files
keep their full names so that neither overwrites the other (a warning is logged). The original
name is always kept in `source_file`.

**Reading files.**
- Header lines start with `##KEY=value`. An `##END=` line is optional.
- Columns can be separated by commas, tabs, spaces or semicolons (RRUFF uses both commas and tabs).
- Lines that are not an `x y` pair, such as a lone number at the end, are skipped with a warning.
- A file fails only if it has no usable `x y` rows at all.

**Failures.** A failed file does not stop the others. It is listed with its error message in
`data/parsed/failed/json/<source folder>.json`. That folder is cleared at the start of each
split, so it only describes the latest run.

**Cleaning.** `clean_source()` deletes the contents of `data/source/` only when every
`.txt` has a matching CSV and nothing is listed in `failed/`. Otherwise it deletes nothing and
logs why. It never touches `data/parsed/` or `data/.downloaded.json`.

## Things to know

- **Downloads are all-or-nothing per zip, with no resume or retry.** If a download fails, the
  zip is removed and tried again on the next run.
- **Updated files on the server are not detected.** The record only stores zip names. To
  download a zip again, remove its name from `data/.downloaded.json`.
- **Every run re-splits everything currently in `data/source/`.** After cleaning, only newly
  downloaded zips are processed.
- **Rebuilding.** If the originals are still in `source/`, delete `data/parsed/` and run again.
  If you already cleaned them, delete `data/.downloaded.json` as well so the zips are downloaded again.
- **"RAW" in the header is not a guarantee.** Some spectra labelled `Raman RAW` are
  already scaled (for example to a maximum of 100). Check intensity ranges before treating a
  file as unprocessed.
- **Raw spectra often have a large background.** Fluorescence can dominate, and point spacing
  is not always even, so resample onto a common grid and subtract a baseline before comparing spectra.
- **Do not call `split_all()` from inside a web server** (Flask, gunicorn). It starts worker
  processes and must be run from a script with the `if __name__ == "__main__":` guard.

## Troubleshooting

| Symptom | Check |
|---|---|
| `No new files` every run | Expected when nothing changed on the server. |
| A file is in `failed/` | Open `data/parsed/failed/json/<folder>.json` for the reason. |
| `Not cleaning: N of M files have no CSV yet` | Run the split step first, or look for failures. |
| `Not cleaning: failures are listed in ...` | Fix or inspect the failed files, then run the split again. |
| `skipped N line(s) that were not an 'x y' pair` | Harmless for stray lines; a problem only if `n_points` is far too low. |
| Download step fails | Check your internet connection and that RRUFF is reachable; the split step still runs on files already in `source/`. |