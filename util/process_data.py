"""
Split every RRUFF .txt file into:
    - Individual CSV files matching the original folder layout.
    - One single consolidated JSON file per source folder for metadata.
    - One single consolidated JSON file per source folder for errors/failures.
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
SOURCE_DIR = DATA_DIR / "source"     # unzipped originals (written by extract_data_rruff.py)
OUT_DIR = DATA_DIR / "parsed"        # the dataset you keep

log = logging.getLogger("split_rruff")

# Columns may be separated by commas, tabs, spaces or semicolons (RRUFF uses both
# "x, y" and "x<TAB>y" in different files).
SEPARATOR = re.compile(r"[,\t; ]+")


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
    if filetype:
        return re.search(r"\braw\b", filetype, re.IGNORECASE) is not None
    return re.search(r"(?<![A-Za-z])RAW(?![A-Za-z])", txt.name) is not None


# RRUFF file names end in a long hex id, e.g. "..._Raman_Data_RAW__c7ee56937fa26d4a89736f652793".
# Outputs are saved without it; the full original name stays in the JSON as "source_file".
HASH_SUFFIX = re.compile(r"__[0-9a-fA-F]{16,}$")


def plan_names(files):
    """
    Map each source file to the name its outputs are saved under: the file name without
    the trailing hex id. If removing it would make two files in the same source folder
    share a name, those files keep their full name, so nothing is ever overwritten.
    Returns {Path: output_stem}.
    """
    def top_folder(f):
        parts = f.relative_to(SOURCE_DIR).parts
        return parts[0] if len(parts) > 1 else "root"

    groups = defaultdict(list)
    for f in files:
        groups[(top_folder(f), HASH_SUFFIX.sub("", f.stem))].append(f)

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
    Processes a single file. job = (source path, name to save the outputs under).
    Returns: (source_folder_name, output_name, status_kind, result_data)
    """
    txt, filename_stem = job
    rel = txt.relative_to(SOURCE_DIR)

    # Extract the topmost subfolder name under data/ dynamically
    folder = rel.parts[0] if len(rel.parts) > 1 else "root"

    try:
        meta, rows = split_file(txt)
        if not rows:
            raise ValueError("no numeric data rows")

        kind = "raw" if is_raw(meta, txt) else "processed"
        sub = rel.parent
        csv_dir = OUT_DIR / kind / "csv" / sub

        csv_dir.mkdir(parents=True, exist_ok=True)
        name = filename_stem

        xname = "wavenumber" if "raman" in meta.get("FILETYPE", "").lower() else "x"
        with open(csv_dir / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"{xname},intensity\n")
            f.writelines(f"{x},{y}\n" for x, y in rows)

        # Build clean metadata structure
        meta["source_folder"] = folder
        meta["quality"], meta["orientation"] = quality_orientation(folder)
        meta["source_file"] = rel.as_posix()
        meta["n_points"] = len(rows)

        return folder, filename_stem, kind, meta
    except Exception as e:
        log.error("%s failed: %s", txt.name, e)
        return folder, filename_stem, "failed", str(e)


def split_all():
    """Split every .txt under SOURCE_DIR concurrently using a pooled map strategy."""
    counts = {"raw": 0, "processed": 0, "failed": 0}
    files = sorted(SOURCE_DIR.rglob("*.txt"))
    total_files = len(files)

    if not files:
        log.warning("No .txt files found under %s", SOURCE_DIR)
        return counts

    # Failure files describe the latest run only: remove ones left by earlier runs.
    shutil.rmtree(OUT_DIR / "failed", ignore_errors=True)

    workers = cpu_count()
    log.info("Starting processing on %d workers for %d files...", workers, total_files)

    # Dynamic groupings for ALL folders found in data/
    raw_groups = defaultdict(dict)
    processed_groups = defaultdict(dict)
    failed_groups = defaultdict(dict)

    with Pool(processes=workers) as pool:
        chunksize = max(1, total_files // (workers * 4))
        names = plan_names(files)
        jobs = [(f, names[f]) for f in files]
        results = pool.imap_unordered(process_single_file, jobs, chunksize=chunksize)

        for idx, (folder, filename_stem, kind, data) in enumerate(results, 1):
            counts[kind] += 1

            if kind == "raw":
                raw_groups[folder][filename_stem] = data
            elif kind == "processed":
                processed_groups[folder][filename_stem] = data
            elif kind == "failed":
                failed_groups[folder][filename_stem] = data

            if idx % 100 == 0 or idx == total_files:
                log.info("Progress: %d/%d files completed...", idx, total_files)

    # Save consolidated raw metadata files dynamically per folder
    for folder, meta_dict in raw_groups.items():
        json_dir = OUT_DIR / "raw" / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        (json_dir / f"{folder}.json").write_text(
            json.dumps(meta_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Save consolidated processed metadata files dynamically per folder
    for folder, meta_dict in processed_groups.items():
        json_dir = OUT_DIR / "processed" / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        (json_dir / f"{folder}.json").write_text(
            json.dumps(meta_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Save consolidated failure log files dynamically per folder
    for folder, error_dict in failed_groups.items():
        error_dir = OUT_DIR / "failed" / "json"
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / f"{folder}.json").write_text(
            json.dumps(error_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    log.info("Done: %(raw)d raw, %(processed)d processed, %(failed)d failed", counts)
    return counts


def clean_source():
    """Delete the contents of data/source/ while keeping the directory."""
    if SOURCE_DIR.name != "source":
        raise RuntimeError(
            f"Refusing to clean {SOURCE_DIR}: not a 'source' folder"
        )

    if not SOURCE_DIR.exists():
        log.warning("Source directory does not exist: %s", SOURCE_DIR)
        return False

    for child in SOURCE_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    log.info("Cleaned source directory: %s", SOURCE_DIR)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    split_all()