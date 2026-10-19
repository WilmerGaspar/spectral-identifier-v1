"""
Download every .zip from a RRUFF zipped_data_files folder, unzip it, delete the zip.
Files already downloaded on a previous run are skipped.

Each dataset (raman, powder, infrared, ...) gets its own folder, so zips with the same
name in different datasets never collide:

    data/source/raman/excellent_oriented/...
    data/source/powder/XY_Processed/...

    import logging
    from extract_data_rruff import sync

    logging.basicConfig(level=logging.INFO)
    sync()
"""

import json
import logging
import re
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://rruff.net/zipped_data_files/"
URL = BASE_URL + "raman/"   # default dataset
# This file lives in <project>/util/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # <project>/data/
SOURCE_DIR = DATA_DIR / "source"                             # unzipped originals: source/<dataset>/<zip name>/

log = logging.getLogger("extract_data_rruff")


def sync(url: str = URL, only=None):
    """Download and unzip new files from `url` (a RRUFF zipped_data_files folder).
    `only`: optional list of zip names to download (e.g. ["XY_Processed.zip"]).
    Returns the list of names handled this run."""
    if not url.endswith("/"):
        url += "/"
    dataset = url.rstrip("/").rsplit("/", 1)[-1]        # "raman", "powder", "infrared", ...
    dataset_dir = SOURCE_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    record = DATA_DIR / ".downloaded.json"    # kept outside source/ so cleanup never removes it
    loaded = json.loads(record.read_text()) if record.exists() else []
    # Entries are "<dataset>/<zip name>". Old records held bare names, which were all raman.
    done = {n if "/" in n else f"raman/{n}" for n in loaded}

    # Find the file names on the server
    page = requests.get(url, timeout=30)
    page.raise_for_status()
    names = sorted(set(re.findall(r'href="([^"?/]+\.zip)"', page.text)))
    if only:
        names = [n for n in names if n in set(only)]
    new = [n for n in names if f"{dataset}/{n}" not in done]

    if not new:
        log.info("No new files in %s (%d already downloaded).", dataset, len(names))
        return []

    handled = []
    for name in new:
        zip_path = dataset_dir / name
        try:
            log.info("Downloading %s/%s", dataset, name)
            with requests.get(url + name, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 256):
                        f.write(chunk)

            log.info("Unzipping %s/%s", dataset, name)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dataset_dir / zip_path.stem)
            zip_path.unlink()

            done.add(f"{dataset}/{name}")
            record.write_text(json.dumps(sorted(done)))
            handled.append(name)
        except Exception as e:
            log.error("%s/%s failed: %s", dataset, name, e)
            zip_path.unlink(missing_ok=True)   # it will be tried again next run

    return handled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sync()