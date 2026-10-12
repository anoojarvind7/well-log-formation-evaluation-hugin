"""Step 1 - load and inventory.

One row per well that has a Hugin Fm label: Hugin depth range, which curves
exist across it and how complete they are, and core-plug overlap. This
table is what the well selection in config.WELLS is argued from.
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import core

CURVES = ["GR", "CALI", "BS", "RHOB", "DRHO", "NPHI", "PEF", "RDEP", "RMED", "DTC", "DTS", "SP"]


def hugin_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.FORMATION == C.RESERVOIR]


def inventory(df: pd.DataFrame) -> pd.DataFrame:
    h = hugin_rows(df)
    core_df = core.load_core()
    rows = []
    for well, g in h.groupby("WELL"):
        r = {
            "well": well,
            "split": g.SPLIT.iloc[0],
            "hugin_top_md": g.DEPTH_MD.min(),
            "hugin_base_md": g.DEPTH_MD.max(),
            "hugin_md_m": g.DEPTH_MD.max() - g.DEPTH_MD.min(),
            "hugin_top_tvdss": g.TVDSS.min(),
            "n_samples": len(g),
        }
        for c in CURVES:
            r[f"{c}_pct"] = round(100 * g[c].notna().mean(), 1) if c in g else 0.0
        r["core_plugs_in_hugin"] = (
            core.plugs_in_interval(core_df, well, r["hugin_top_md"] - 5, r["hugin_base_md"] + 5)
            if core_df is not None else None
        )
        need = ["GR", "RHOB", "NPHI", "RDEP"]
        r["missing_core_curves"] = ",".join(c for c in need if r[f"{c}_pct"] < 90) or ""
        rows.append(r)
    inv = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    inv["selected"] = inv.well.isin(C.WELLS)
    return inv

