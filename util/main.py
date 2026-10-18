"""
Run the whole pipeline: download -> split -> (optionally) delete the originals.

Put this file in the project root, next to the utill/ folder, then run:

    python run_pipeline.py            # download new files, then split everything
    python run_pipeline.py --clean    # ...and delete data/source/ if the split checks out
"""

import argparse
import logging
import sys

from extract_data_rruff_copy import sync
from process_data_copy import split_all, clean_source

log = logging.getLogger("pipeline")


def main(clean: bool = False) -> int:
    """Returns 0 on success, 1 if anything failed."""
    problems = 0

    # 1. Download and unzip new files into data/source/
    try:
        new = sync()
        log.info("Download step done: %d new zip(s)", len(new))
    except Exception as e:
        # Keep going: files from an earlier run may still be waiting to be split.
        log.error("Download step failed: %s", e)
        problems += 1

    # 2. Split every .txt in data/source/ into data/parsed/
    counts = split_all()
    log.info("Split step done: %s", counts)
    problems += counts["failed"]

    # 3. Optionally remove the originals (only deletes if everything was split cleanly)
    if clean:
        clean_source()      # logs its own reason if it refuses; split failures are counted above

    log.info("Pipeline finished with %d problem(s)", problems)
    return 1 if problems else 0


if __name__ == "__main__":      # this guard is required: split_all() starts worker processes
    parser = argparse.ArgumentParser(description="Download, split and optionally clean RRUFF data")
    parser.add_argument("--clean", action="store_true",
                        help="delete data/source/ after a verified split")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log", encoding="utf-8")],
    )
    sys.exit(main(clean=args.clean))