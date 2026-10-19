"""
Download every .zip from a web directory listing, unzip it, delete the zip.
Files already downloaded on a previous run are skipped.

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
SOURCE_DIR = DATA_DIR / "source"                             # unzipped originals go here

log = logging.getLogger("extract_data_rruff")


def sync(url: str = URL, only=None):
    """Download and unzip new files from `url` (a RRUFF zipped_data_files folder).
    `only`: optional list of zip names to download (e.g. ["XY_Processed.zip"]).
    Returns the list of names handled this run."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    record = DATA_DIR / ".downloaded.json"    # kept outside source/ so cleanup never removes it
    done = set(json.loads(record.read_text())) if record.exists() else set()

    # Find the file names on the server
    page = requests.get(url, timeout=30)
    page.raise_for_status()
    names = sorted(set(re.findall(r'href="([^"?/]+\.zip)"', page.text)))
    if only:
        names = [n for n in names if n in set(only)]
    new = [n for n in names if n not in done]

    if not new:
        log.info("No new files (%d already downloaded).", len(names))
        return []

    handled = []
    for name in new:
        zip_path = SOURCE_DIR / name
        try:
            log.info("Downloading %s", name)
            with requests.get(url + name, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 256):
                        f.write(chunk)

            log.info("Unzipping %s", name)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(SOURCE_DIR / zip_path.stem)
            zip_path.unlink()

            done.add(name)
            record.write_text(json.dumps(sorted(done)))
            handled.append(name)
        except Exception as e:
            log.error("%s failed: %s", name, e)
            zip_path.unlink(missing_ok=True)   # it will be tried again next run

    return handled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sync()