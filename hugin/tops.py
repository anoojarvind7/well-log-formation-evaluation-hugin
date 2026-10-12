"""Step 3 - formation tops.

Tops come from the NPD lithostratigraphy labels shipped with FORCE (the
FORMATION column). They are checked, not replaced: for the Hugin top the
module finds the sharpest downward GR drop (shale -> sand) within +-8 m of the
labelled top and reports the offset. A large offset is a prompt to look at the
tops figure, not an automatic correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def formation_tops(w: pd.DataFrame) -> pd.DataFrame:
    f = w.dropna(subset=["FORMATION"])
    t = f.groupby("FORMATION", sort=False).agg(top_md=("DEPTH_MD", "min"), base_md=("DEPTH_MD", "max"),
                                                top_tvdss=("TVDSS", "min"))
    return t.sort_values("top_md").reset_index()


def gr_break(w: pd.DataFrame, label_top: float, search_m: float = 8.0, smooth_m: float = 1.0) -> float:
    step = float(np.median(np.diff(w.DEPTH_MD)))
    k = max(3, int(round(smooth_m / step)))
    gr = w.GR.interpolate(limit=3).rolling(k, center=True, min_periods=1).mean()
    d = gr.diff(k) / (k * step)   # API per m, looking down
    sel = w.DEPTH_MD.between(label_top - search_m, label_top + search_m)
    if not sel.any() or d[sel].isna().all():
        return np.nan
    i = d[sel].idxmin()
    # the derivative at i spans [i-k, i]; place the break mid-way
    return float(w.DEPTH_MD.iloc[i] - 0.5 * k * step)


def check_reservoir_top(w: pd.DataFrame, well: str) -> dict:
    h = w[w.FORMATION == C.RESERVOIR]
    top, base = h.DEPTH_MD.min(), h.DEPTH_MD.max()
    pick = gr_break(w, top)
    above = w[w.DEPTH_MD < top].FORMATION.dropna()
    below = w[w.DEPTH_MD > base].FORMATION.dropna()
    return dict(
        well=well, hugin_top_md=top, hugin_base_md=base, gross_md_m=round(base - top, 2),
        gross_tvd_m=round(h.TVDSS.max() - h.TVDSS.min(), 2),
        overlying=above.iloc[-1] if len(above) else None,
        underlying=below.iloc[0] if len(below) else None,
        gr_break_md=round(pick, 2), offset_m=round(pick - top, 2),
    )
