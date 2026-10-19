"""
Split every RRUFF .txt file of one dataset (raman, powder, infrared, ...) into:
    - one CSV per spectrum, in the same folder layout as the downloaded zips
    - one consolidated JSON per source folder with the metadata
    - one consolidated JSON per source folder listing files that failed

    data/source/<dataset>/<zip folder>/*.txt          (input, written by extract_data_rruff.py)
    data/parsed/<dataset>/{raw,processed}/csv/<zip folder>/<spectrum>.csv
    data/parsed/<dataset>/{raw,processed}/json/<zip folder>.json
    data/parsed/<dataset>/failed/json/<zip folder>.json
"""

import json
import logging
import re
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
SOURCE_ROOT = DATA_DIR / "source"     # source/<dataset>/...   (written by extract_data_rruff.py)
OUT_ROOT = DATA_DIR / "parsed"        # parsed/<dataset>/...   (the dataset you keep)

log = logging.getLogger("process_data")

# Name of the first CSV column for each dataset. Anything else is called "x".
X_NAMES = {"raman": "wavenumber", "infrared": "wavenumber", "powder": "two_theta"}

# Columns may be separated by commas, tabs, spaces or semicolons (RRUFF uses both
# "x, y" and "x<TAB>y" in different files).
SEPARATOR = re.compile(r"[,\t; ]+")

# "raw" as its own word, also when joined with underscores (XY_RAW, Raman_Data_RAW),
# but not inside other words (Crawfordite).
RAW_WORD = re.compile(r"(?<![A-Za-z])raw(?![A-Za-z])", re.IGNORECASE)

# RRUFF file names end in a long hex id, e.g. "..._Raman_Data_RAW__c7ee56937fa26d4a89736f652793".
# Outputs are saved without it; the full original name stays in the JSON as "source_file".
HASH_SUFFIX = re.compile(r"__[0-9a-fA-F]{16,}$")


def split_file(txt: Path):
    """Return (meta dict, list of (x, y) strings) for one file."""
    meta, rows = {}, []
    in_header = True
    skipped = 0

    for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue

        if in_header and line.startswith("##"):
            key, _, value = line[2:].partition("=")
            cleaned_key = key.strip()
            if cleaned_key == "END":
                in_header = False
                continue
            if cleaned_key:
                meta[cleaned_key] = value.strip()
            continue

        in_header = False

        if line.startswith("##"):
            continue

        # Take the first two values on the line, whatever separates them.
        tokens = [t for t in SEPARATOR.split(line) if t]
        if len(tokens) < 2:
            skipped += 1            # e.g. a lone trailing number: not an x, y pair
            continue
        x, y = tokens[0], tokens[1]
        try:
            float(x), float(y)
        except ValueError:
            skipped += 1
            continue
        rows.append((x, y))

    if skipped:
        log.warning("%s: skipped %d line(s) that were not an 'x y' pair", txt.name, skipped)

    return meta, rows


def quality_orientation(folder: str):
    m = re.fullmatch(r"(excellent|fair|poor|unrated)_(oriented|unoriented)", folder, re.IGNORECASE)
    return (m.group(1).lower(), m.group(2).lower()) if m else (None, None)


def is_raw(meta: dict, txt: Path) -> bool:
    filetype = meta.get("FILETYPE", "")
    if filetype:                                    # the header decides when present
        return RAW_WORD.search(filetype) is not None
    # fallback: a standalone uppercase RAW in the file name
    return re.search(r"(?<![A-Za-z])RAW(?![A-Za-z])", txt.name) is not None


def top_folder(f: Path, source_dir: Path) -> str:
    parts = f.relative_to(source_dir).parts
    return parts[0] if len(parts) > 1 else "root"


def plan_names(files, source_dir: Path):
    """
    Map each source file to the name its outputs are saved under: the file name without
    the trailing hex id. If removing it would make two files in the same source folder
    share a name, those files keep their full name, so nothing is ever overwritten.
    Returns {Path: output_stem}.
    """
    groups = defaultdict(list)
    for f in files:
        groups[(top_folder(f, source_dir), HASH_SUFFIX.sub("", f.stem))].append(f)

    names = {}
    for (folder, short), members in groups.items():
        if len(members) > 1:
            log.warning("%s: %d files would all be saved as %r; keeping their full names",
                        folder, len(members), short)
            for f in members:
                names[f] = f.stem
        else:
            names[members[0]] = short
    return names


def process_single_file(job: tuple) -> tuple:
    """
    Processes a single file.
    job = (source path, name to save outputs under, source dir, output dir, dataset)
    Returns: (source_folder_name, output_name, status_kind, result_data)
    """
    txt, filename_stem, source_dir, out_dir, dataset = job
    rel = txt.relative_to(source_dir)
    folder = top_folder(txt, source_dir)

    try:
        meta, rows = split_file(txt)
        if not rows:
            raise ValueError("no numeric data rows")

        kind = "raw" if is_raw(meta, txt) else "processed"
        csv_dir = out_dir / kind / "csv" / rel.parent
        csv_dir.mkdir(parents=True, exist_ok=True)

        xname = X_NAMES.get(dataset, "x")
        with open(csv_dir / f"{filename_stem}.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"{xname},intensity\n")
            f.writelines(f"{x},{y}\n" for x, y in rows)

        # Build clean metadata structure
        meta["dataset"] = dataset
        meta["source_folder"] = folder
        meta["quality"], meta["orientation"] = quality_orientation(folder)
        meta["source_file"] = rel.as_posix()
        meta["n_points"] = len(rows)

        return folder, filename_stem, kind, meta
    except Exception as e:
        log.error("%s failed: %s", txt.name, e)
        return folder, filename_stem, "failed", str(e)


def _write_groups(groups, out_dir: Path):
    for folder, entries in groups.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{folder}.json").write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def split_all(dataset: str = "raman"):
    """Split every .txt under data/source/<dataset>/ concurrently."""
    source_dir = SOURCE_ROOT / dataset
    out_dir = OUT_ROOT / dataset

    counts = {"raw": 0, "processed": 0, "failed": 0}
    files = sorted(source_dir.rglob("*.txt"))
    total_files = len(files)

    if not files:
        log.warning("No .txt files found under %s", source_dir)
        return counts

    # Failure files describe the latest run only: remove ones left by earlier runs.
    shutil.rmtree(out_dir / "failed", ignore_errors=True)

    workers = cpu_count()
    log.info("[%s] Starting processing on %d workers for %d files...", dataset, workers, total_files)

    raw_groups = defaultdict(dict)
    processed_groups = defaultdict(dict)
    failed_groups = defaultdict(dict)

    names = plan_names(files, source_dir)
    jobs = [(f, names[f], source_dir, out_dir, dataset) for f in files]

    with Pool(processes=workers) as pool:
        chunksize = max(1, total_files // (workers * 4))
        results = pool.imap_unordered(process_single_file, jobs, chunksize=chunksize)

        for idx, (folder, filename_stem, kind, data) in enumerate(results, 1):
            counts[kind] += 1
            {"raw": raw_groups, "processed": processed_groups, "failed": failed_groups}[kind][folder][filename_stem] = data

            if idx % 100 == 0 or idx == total_files:
                log.info("[%s] Progress: %d/%d files completed...", dataset, idx, total_files)

    _write_groups(raw_groups, out_dir / "raw" / "json")
    _write_groups(processed_groups, out_dir / "processed" / "json")
    _write_groups(failed_groups, out_dir / "failed" / "json")

    log.info("[%s] Done: %d raw, %d processed, %d failed",
             dataset, counts["raw"], counts["processed"], counts["failed"])
    return counts


def clean_source(dataset: str = "raman"):
    """
    Delete the unzipped originals in data/source/<dataset>/ once the split is verified.

    Deletes only when every .txt has a matching CSV under parsed/<dataset>/raw or /processed
    and nothing is listed in parsed/<dataset>/failed. Only the contents of the dataset's
    source folder are removed: parsed/ and .downloaded.json are never touched.
    Returns True if anything was deleted.
    """
    source_dir = SOURCE_ROOT / dataset
    out_dir = OUT_ROOT / dataset

    if source_dir.parent.name != "source":                # safety guard against a bad path edit
        raise RuntimeError(f"Refusing to clean {source_dir}: not inside a 'source' folder")

    files = list(source_dir.rglob("*.txt"))
    if not files:
        log.warning("Nothing to clean: no .txt files under %s", source_dir)
        return False

    failed_dir = out_dir / "failed"
    if failed_dir.exists() and any(failed_dir.rglob("*.json")):
        log.error("Not cleaning: failures are listed in %s", failed_dir)
        return False

    names = plan_names(files, source_dir)
    missing = []
    for txt in files:
        rel_csv = txt.relative_to(source_dir).parent / f"{names[txt]}.csv"
        if not any((out_dir / kind / "csv" / rel_csv).exists() for kind in ("raw", "processed")):
            missing.append(txt.relative_to(source_dir).as_posix())
    if missing:
        log.error("Not cleaning: %d of %d files have no CSV yet, e.g. %s",
                  len(missing), len(files), missing[:3])
        return False

    for child in source_dir.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    log.info("Cleaned %s: removed %d original files", source_dir, len(files))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    split_all()