"""CLI: extract Drebin features from one or more APKs into JSON files."""

import argparse
import json
import logging
from pathlib import Path

from .extract import extract_apk


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="drebin-extract")
    parser.add_argument("inputs", nargs="+", help="APK files or directories")
    parser.add_argument("-o", "--outdir", default="out", help="output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    # androguard logs through loguru at DEBUG by default, which is unusable in bulk.
    from loguru import logger

    logger.remove()
    logger.add(lambda m: None, level="ERROR")

    paths: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        paths.extend(sorted(p.rglob("*.apk")) if p.is_dir() else [p])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    for path in paths:
        record = extract_apk(path)
        (outdir / f"{record['sha256']}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        if record["errors"]:
            failed += 1
            logging.warning("%s: %s", path.name, "; ".join(record["errors"]))
        else:
            ok += 1
        logging.info("%s -> %d features", path.name, record.get("n_features", 0))

    print(f"{ok} ok, {failed} with errors -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())