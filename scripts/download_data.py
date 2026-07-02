#!/usr/bin/env python
"""Verify (and optionally fetch) the Spider dataset.

Spider is distributed via Google Drive and usually needs a manual download or
`gdown`. This script checks the expected layout and, if given a Drive id or URL,
tries to fetch + unzip it.

  python scripts/download_data.py --out data/raw/spider
  python scripts/download_data.py --out data/raw/spider --gdrive-id <DRIVE_FILE_ID>
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

EXPECTED = ["tables.json", "dev.json", "train_spider.json", "database"]

INSTRUCTIONS = """\
Spider was not found. Download it (Google Drive) and unzip into {out}/ so it looks like:

  {out}/tables.json
  {out}/dev.json
  {out}/train_spider.json
  {out}/train_others.json
  {out}/database/<db_id>/<db_id>.sqlite

Get it from https://yale-lily.github.io/spider (the "Spider Dataset" link), or
with gdown once you have the file id:

  pip install gdown
  python scripts/download_data.py --out {out} --gdrive-id <DRIVE_FILE_ID>

For test-suite execution accuracy, also fetch the distilled test-suite databases
(github.com/taoyds/test-suite-sql-eval) and set data.test_suite_root in the config.
"""


def is_present(out: Path) -> bool:
    return all((out / p).exists() for p in EXPECTED)


def try_fetch(out: Path, gdrive_id: str | None, url: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "spider.zip"
    if gdrive_id:
        import gdown

        gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
    elif url:
        import urllib.request

        urllib.request.urlretrieve(url, zip_path)
    else:
        return
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    zip_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/spider")
    ap.add_argument("--gdrive-id", default=None)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    if is_present(out):
        print(f"Spider present at {out}")
        return

    if args.gdrive_id or args.url:
        try_fetch(out, args.gdrive_id, args.url)

    if is_present(out):
        print(f"Spider ready at {out}")
    else:
        raise SystemExit(INSTRUCTIONS.format(out=out))


if __name__ == "__main__":
    main()