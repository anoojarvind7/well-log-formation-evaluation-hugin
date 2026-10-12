"""Loading the FORCE 2020 data.

The CSVs are the primary source because they carry the NPD group/formation
labels and the bit size; the per-well LAS files are read with lasio only for
their curve units and header (the CSVs have no units).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

from . import config as C

NULL_VALUES = [-999.25, -999.0, -9999.0]


def sniff_csv(path: Path, nbytes: int = 4096) -> dict:
    """Report delimiter and header of a CSV instead of assuming them."""
    with open(path, "rb") as f:
        head = f.read(nbytes).decode("utf-8", errors="replace")
    first = head.splitlines()[0]
    delim = max([";", ",", "\t"], key=first.count)
    return {"file": path.name, "delimiter": delim, "columns": first.split(delim)}


def _read(path_or_buf, sep: str) -> pd.DataFrame:
    return pd.read_csv(path_or_buf, sep=sep, na_values=NULL_VALUES, low_memory=False)


def load_force(wells: list[str] | None = None, cache: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """All FORCE samples (train + leaderboard test + hidden test), one frame.

    Returns the data and a list of format checks (one per source file).
    The leaderboard-test lithology labels ship in a separate target file; they
    are joined back on (WELL, DEPTH_MD) after checking row alignment.
    """
    d = C.FORCE_DIR
    checks = []
    cache_file = C.INTERIM / "force_all.parquet"
    if cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        checks = pd.read_json(C.INTERIM / "format_checks.json").to_dict("records")
    else:
        with zipfile.ZipFile(d / "train.zip") as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as f:
                first = f.readline().decode()
            sep = max([";", ","], key=first.count)
            checks.append({"file": f"train.zip:{name}", "delimiter": sep, "n_columns": len(first.split(sep))})
            with z.open(name) as f:
                train = _read(f, sep)
        train["SPLIT"] = "train"

        info = sniff_csv(d / "leaderboard_test_features.csv")
        lb = _read(d / "leaderboard_test_features.csv", info["delimiter"])
        tgt = _read(d / "leaderboard_test_target.csv", sniff_csv(d / "leaderboard_test_target.csv")["delimiter"])
        aligned = (len(lb) == len(tgt) and np.allclose(lb.DEPTH_MD, tgt.DEPTH_MD)
                   and (lb.WELL.values == tgt.WELL.values).all())
        if not aligned:
            raise ValueError("leaderboard target rows do not align with features")
        lb["FORCE_2020_LITHOFACIES_LITHOLOGY"] = tgt["FORCE_2020_LITHOFACIES_LITHOLOGY"].values
        lb["SPLIT"] = "leaderboard_test"
        checks.append({"file": info["file"], "delimiter": info["delimiter"], "n_columns": len(info["columns"]),
                       "note": "lithology joined from leaderboard_test_target.csv (rows aligned)"})

        info = sniff_csv(d / "hidden_test.csv")
        hid = _read(d / "hidden_test.csv", info["delimiter"])
        hid["SPLIT"] = "hidden_test"
        checks.append({"file": info["file"], "delimiter": info["delimiter"], "n_columns": len(info["columns"])})

        df = pd.concat([train, lb, hid], ignore_index=True)
        df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
        df["LITHO"] = df["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(C.FORCE_LITHO)
        df["TVDSS"] = -df["Z_LOC"]
        if cache:
            C.INTERIM.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file)
            pd.DataFrame(checks).to_json(C.INTERIM / "format_checks.json")
    if wells is not None:
        df = df[df.WELL.isin(wells)].reset_index(drop=True)
    return df, checks


def las_path(well: str) -> Path:
    return C.FORCE_DIR / "las_files_Lithostrat_data" / (well.replace("/", "_") + ".las")


def las_units(well: str) -> dict[str, str]:
    """Curve mnemonic -> unit string from the well's LAS header."""
    p = las_path(well)
    if not p.exists():
        return {}
    las = lasio.read(p, ignore_data=True)
    return {c.mnemonic: c.unit for c in las.curves}
