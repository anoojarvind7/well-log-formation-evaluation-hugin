"""Download the public input data and verify checksums.

Raw data is never committed to git. This script recreates ``data/raw/`` from
public sources:

* FORCE 2020 well logs + lithofacies (CC BY 4.0). Default source is the
  competition GitHub repository, pinned to a commit so the bytes never change.
  The same data is archived on Zenodo (doi:10.5281/zenodo.4351156).
* Norwegian core data, RealPore por/perm/lithology from 1,240 wells
  (CC BY 4.0, GeoProvider, doi:10.5281/zenodo.4419060). Downloaded from Zenodo
  with ``--core``; if Zenodo is unreachable, copy the xlsx in by hand with
  ``--core-file PATH`` and the checksum is still verified.

Usage::

    python scripts/fetch_data.py                 # FORCE logs only
    python scripts/fetch_data.py --core          # + core plugs from Zenodo
    python scripts/fetch_data.py --core-file ~/Downloads/RealPore*.xlsx
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

FORCE_COMMIT = "c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1"
FORCE_BASE = (
    "https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/"
    f"{FORCE_COMMIT}/lithology_competition/data/"
)

# path relative to FORCE_BASE -> md5
FORCE_FILES = {
    "train.zip": "d935a5161fb47421a07f0a51abfc3d46",
    "leaderboard_test_features.csv": "b562ce84d315de526e5983bfe85aaf30",
    "leaderboard_test_target.csv": "50d873b9ee892ff5f0d8b8075c474482",
    "hidden_test.csv": "1ea8d4310de00f1beb7d5ba7e7006c26",
    # Per-well LAS files: used only for curve units / header checks (step 1-2)
    "las_files_Lithostrat_data/15_9-13.las": "82b4c3f8a37d42fb62e31ede269eca83",
    "las_files_Lithostrat_data/25_5-3.las": "298389e88021a1e146642b7ce0c908da",
    "las_files_Lithostrat_data/25_2-14.las": "77261b7e51e0fb6dda41c0b2509ed258",
    "las_files_Lithostrat_data/26_4-1.las": "6c2e45e6cb35ff66ea5b1f153ed2958e",
}

RETRIES = 4

CORE_NAME = "RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx"
CORE_URL = "https://zenodo.org/records/4419060/files/" + urllib.parse.quote(CORE_NAME) + "?download=1"
CORE_MD5 = "ff2167f56a63b80cb87e20748cbcfc50"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_md5: str) -> None:
    if dest.exists() and md5(dest) == expected_md5:
        print(f"ok (cached)  {dest.relative_to(RAW)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Large files are sometimes cut short in transit (seen through HTTP proxies),
    # which shows up as an md5 mismatch; retry a few times before giving up.
    for attempt in range(1, RETRIES + 1):
        print(f"download     {dest.relative_to(RAW)}" + (f" (attempt {attempt})" if attempt > 1 else ""))
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except OSError as e:
            got = f"error: {e}"
        else:
            got = md5(tmp)
            if got == expected_md5:
                tmp.replace(dest)
                print(f"ok (md5)     {dest.relative_to(RAW)}")
                return
        tmp.unlink(missing_ok=True)
        if attempt < RETRIES:
            time.sleep(2 ** attempt)
    raise SystemExit(f"md5 mismatch for {dest.name} after {RETRIES} attempts: got {got}, expected {expected_md5}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", action="store_true", help="also download the core xlsx from Zenodo")
    p.add_argument("--core-file", type=Path, help="copy a locally downloaded core xlsx into data/raw")
    a = p.parse_args(argv)

    for rel, digest in FORCE_FILES.items():
        url = FORCE_BASE + urllib.parse.quote(rel)
        fetch(url, RAW / "force2020" / rel, digest)

    core_dest = RAW / "core" / CORE_NAME
    if a.core_file:
        core_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.core_file, core_dest)
        if md5(core_dest) != CORE_MD5:
            print(f"WARNING: {a.core_file} md5 {md5(core_dest)} != published {CORE_MD5}", file=sys.stderr)
        else:
            print(f"ok (md5)     {core_dest.relative_to(RAW)}")
    elif a.core:
        fetch(CORE_URL, core_dest, CORE_MD5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
