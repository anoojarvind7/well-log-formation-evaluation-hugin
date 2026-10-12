"""Core plug data (RealPore, Zenodo 4419060).

Checked on first load (sheet "Combined all wells", 387,467 plugs, 1,259 wells):
well names use "_" for "/" (e.g. "15_9-13", "1_3-7_T3"), depths are
"Measured Depth" in m, porosity is in percent and numeric columns are stored as
text. Several porosity and permeability columns exist; the preferred ones are
listed in PREFERRED and the regex patterns are only a fallback for other files.
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

# First matching column wins. Helium porosity is the standard plug measurement;
# "best of available" fills plugs that have no helium value.
PREFERRED = {
    "well": ["Well Name"],
    "depth": ["Measured Depth"],
    "porosity": ["Porosity helium"],
    "porosity_fallback": ["porosity best of available"],
    "permeability": ["Permeability horizontal ka  (air) "],
    "lithology": ["main lithology"],
}

_CACHE: dict = {}


def norm_well(name) -> str:
    """Normalise well names from FORCE and RealPore to one form:
    'NO 15/9-13 ' -> '15/9-13', '15_9-13' -> '15/9-13', '1_2 -1 ' -> '1/2-1',
    '1_3-7_T3' -> '1/3-7 T3', '25/2-13  t4' -> '25/2-13 T4'."""
    s = str(name).upper().strip()
    s = re.sub(r"^(NO|N)\s+", "", s)
    s = s.replace("_", "/", 1).replace("_", " ")
    s = re.sub(r"\s*-\s*", "-", s)
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
    out: dict[str, str | None] = {}
    for key in _PATTERNS:
        pref = next((c for c in PREFERRED.get(key, []) if c in cols), None)
        out[key] = pref or _pick([c for c in cols if c not in out.values()], key)
    out["porosity_fallback"] = next((c for c in PREFERRED["porosity_fallback"] if c in cols), None)
    return out


def load_core(path=None, verbose: bool = False) -> pd.DataFrame | None:
    """Standardised plug table: well, depth_m, phi (v/v), perm_md, lith.
    Returns None when the file has not been downloaded."""
    path = path or C.CORE_FILE
    if not path.exists():
        return None
    if path in _CACHE:
        return _CACHE[path]
    # The xlsx takes minutes to parse; keep a parquet copy beside the other
    # intermediate data, rebuilt whenever the xlsx is newer.
    cache = C.INTERIM / "core_plugs.parquet"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        out = pd.read_parquet(cache)
        _CACHE[path] = out
        return out
    raw = pd.read_excel(path, sheet_name=0)
    cols = detect_columns(raw)
    if verbose:
        print("core columns detected:", cols)
    if cols["well"] is None or cols["depth"] is None or cols["porosity"] is None:
        raise ValueError(f"could not identify well/depth/porosity columns in {path.name}: {list(raw.columns)}")
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")  # noqa: E731
    phi = num(cols["porosity"])
    phi_src = pd.Series(cols["porosity"], index=raw.index)
    if cols["porosity_fallback"]:
        fb = phi.isna()
        phi = phi.where(~fb, num(cols["porosity_fallback"]))
        phi_src = phi_src.where(~fb, cols["porosity_fallback"])
    out = pd.DataFrame({
        "well": raw[cols["well"]].map(norm_well),
        "depth_m": num(cols["depth"]),
        "phi": phi,
        "phi_source": phi_src,
        "perm_md": num(cols["permeability"]) if cols["permeability"] else np.nan,
        "lith": raw[cols["lithology"]] if cols["lithology"] else None,
    }).dropna(subset=["depth_m", "phi"])
    # Porosity reported in percent -> fraction
    if out.phi.quantile(0.9) > 1.0:
        out["phi"] = out.phi / 100.0
    out["lith"] = out["lith"].astype("string")
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_parquet(cache)
    _CACHE[path] = out
    return out


def plugs_in_interval(core_df: pd.DataFrame | None, well: str, top: float, base: float) -> int | None:
    if core_df is None:
        return None
    w = norm_well(well)
    sel = (core_df.well == w) & core_df.depth_m.between(top, base)
    return int(sel.sum())
