"""Core plug data (RealPore, Zenodo 4419060).

The column layout is detected from the header rather than assumed, and the
detection result is printed so it can be checked on first load.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

_PATTERNS = {
    "well": [r"^well", r"wellbore", r"well.?name"],
    "depth": [r"^depth", r"md", r"depth.*m"],
    "porosity": [r"^por", r"porosity", r"phi"],
    "permeability": [r"perm", r"^k.?h", r"^kair", r"^k\b"],
    "lithology": [r"lith", r"rock.?type", r"descr"],
}

_CACHE: dict = {}


def norm_well(name) -> str:
    """'NO 15/9-13 ', '15/9-13 S' etc -> '15/9-13', '15/9-13 S' (uppercase, single spaces)."""
    s = str(name).upper().strip()
    s = re.sub(r"^(NO|N)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _pick(cols: list[str], key: str) -> str | None:
    for pat in _PATTERNS[key]:
        for c in cols:
            if re.search(pat, c.strip().lower()):
                return c
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = [str(c) for c in df.columns]
    out = {}
    for key in _PATTERNS:
        out[key] = _pick([c for c in cols if c not in out.values()], key)
    return out


def load_core(path=None, verbose: bool = False) -> pd.DataFrame | None:
    """Standardised plug table: well, depth_m, phi (v/v), perm_md, lith.
    Returns None when the file has not been downloaded."""
    path = path or C.CORE_FILE
    if not path.exists():
        return None
    if path in _CACHE:
        return _CACHE[path]
    raw = pd.read_excel(path, sheet_name=0)
    cols = detect_columns(raw)
    if verbose:
        print("core columns detected:", cols)
    if cols["well"] is None or cols["depth"] is None or cols["porosity"] is None:
        raise ValueError(f"could not identify well/depth/porosity columns in {path.name}: {list(raw.columns)}")
    out = pd.DataFrame({
        "well": raw[cols["well"]].map(norm_well),
        "depth_m": pd.to_numeric(raw[cols["depth"]], errors="coerce"),
        "phi": pd.to_numeric(raw[cols["porosity"]], errors="coerce"),
        "perm_md": pd.to_numeric(raw[cols["permeability"]], errors="coerce") if cols["permeability"] else np.nan,
        "lith": raw[cols["lithology"]] if cols["lithology"] else None,
    }).dropna(subset=["depth_m", "phi"])
    # Porosity reported in percent -> fraction
    if out.phi.quantile(0.9) > 1.0:
        out["phi"] = out.phi / 100.0
    out.attrs["columns"] = cols
    _CACHE[path] = out
    return out


def plugs_in_interval(core_df: pd.DataFrame | None, well: str, top: float, base: float) -> int | None:
    if core_df is None:
        return None
    w = norm_well(well)
    sel = (core_df.well == w) & core_df.depth_m.between(top, base)
    return int(sel.sum())
