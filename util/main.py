"""
Run the whole pipeline: download -> split -> (optionally) delete the originals.

Run from the project root:

    python util/main.py            # download new Raman files, then split them
    python util/main.py --clean    # ...and delete data/source/raman/ if the split checks out
    python util/main.py --dataset powder --only XY_Processed.zip   # measured powder XRD
"""

import argparse
import logging
import sys

from extract_data_rruff import sync, BASE_URL
from process_data import split_all, clean_source

log = logging.getLogger("pipeline")


def main(clean: bool = False, dataset: str=None, only=None) -> int:
    """Returns 0 on success, 1 if anything failed."""
    problems = 0

    # 1. Download and unzip new files into data/source/<dataset>/
    try:
        new = sync(BASE_URL + dataset + "/", only=only)
        log.info("Download step done: %d new zip(s)", len(new))
    except Exception as e:
        # Keep going: files from an earlier run may still be waiting to be split.
        log.error("Download step failed: %s", e)
        problems += 1

    # 2. Split every .txt in data/source/<dataset>/ into data/parsed/<dataset>/
    counts = split_all(dataset)
    log.info("Split step done: %s", counts)
    problems += counts["failed"]

    # 3. Optionally remove the originals (only deletes if everything was split cleanly)
    if clean:
        clean_source(dataset)   # logs its own reason if it refuses; split failures are counted above

    log.info("Pipeline finished with %d problem(s)", problems)
    return 1 if problems else 0


if __name__ == "__main__":      # this guard is required: split_all() starts worker processes
    parser = argparse.ArgumentParser(description="Download, split and optionally clean RRUFF data")
    parser.add_argument("--clean", action="store_true",
                        help="delete data/source/<dataset>/ after a verified split")
    parser.add_argument("--dataset", default="raman", choices=["raman", "powder", "infrared"],
                        help="RRUFF zipped_data_files folder to download")
    parser.add_argument("--only", nargs="+", default=None,
                        help="only these zip names, e.g. XY_Processed.zip")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log", encoding="utf-8")],
    )
    sys.exit(main(clean=args.clean, dataset=args.dataset, only=args.only))