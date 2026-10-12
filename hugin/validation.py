"""Step 10 - validation against core plugs and published values.

Core-to-log depth shift: core depths are driller's depths, logs are logger's
depths, and they rarely agree. The shift is found by sliding the log total
porosity over +-MAX_SHIFT_M and keeping the shift with the highest correlation
to plug porosity. It is reported, and a shift at the edge of the search window
is flagged as unreliable rather than used silently.

Plug porosity (helium, ambient) is compared with log total porosity PHIT, the
like-for-like quantity; PHIE is reported alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import core

MAX_SHIFT_M = 5.0


def best_shift(depth_log, phi_log, depth_core, phi_core, max_shift=MAX_SHIFT_M, step=0.152):
    shifts = np.arange(-max_shift, max_shift + step / 2, step)
    best = (0.0, -np.inf, 0)
    for s in shifts:
        lp = np.interp(depth_core + s, depth_log, phi_log, left=np.nan, right=np.nan)
        ok = np.isfinite(lp) & np.isfinite(phi_core)
        if ok.sum() < 5:
            continue
        r = np.corrcoef(lp[ok], phi_core[ok])[0, 1]
        if r > best[1]:
            best = (float(s), float(r), int(ok.sum()))
    return best


def core_comparison(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (matched plug table, per-well stats) or None without core data."""
    cd = core.load_core(verbose=True)
    if cd is None:
        return None
    matched, stats = [], []
    for well, g in curves.groupby("WELL"):
        g = g.sort_values("DEPTH_MD")
        top, base = g.DEPTH_MD.min(), g.DEPTH_MD.max()
        c = cd[(cd.well == core.norm_well(well)) & cd.depth_m.between(top - MAX_SHIFT_M, base + MAX_SHIFT_M)]
        row = dict(well=well, n_plugs=len(c))
        if len(c) >= 5:
            phit = g.PHIT.interpolate(limit=3).to_numpy()
            s, r, n = best_shift(g.DEPTH_MD.to_numpy(), phit, c.depth_m.to_numpy(), c.phi.to_numpy())
            edge = abs(abs(s) - MAX_SHIFT_M) < 0.2
            d = c.depth_m.to_numpy() + s
            m = pd.DataFrame({
                "well": well, "core_depth_m": c.depth_m.to_numpy(), "log_depth_m": d,
                "phi_core": c.phi.to_numpy(), "perm_md": c.perm_md.to_numpy(),
                "phit_log": np.interp(d, g.DEPTH_MD, phit, left=np.nan, right=np.nan),
                "phie_log": np.interp(d, g.DEPTH_MD, g.PHIE.interpolate(limit=3), left=np.nan, right=np.nan),
            })
            m = m[m.log_depth_m.between(top, base)]
            diff = (m.phit_log - m.phi_core).dropna()
            row.update(shift_m=round(s, 2), shift_r=round(r, 2), shift_at_edge=edge, n_compared=len(diff),
                       bias_phit=round(diff.mean(), 3), rmse_phit=round(float(np.sqrt((diff ** 2).mean())), 3),
                       mean_core_phi=round(m.phi_core.mean(), 3), mean_log_phit=round(m.phit_log.mean(), 3),
                       within_0p03=round(float((diff.abs() <= 0.03).mean()), 2))
            matched.append(m)
        stats.append(row)
    return (pd.concat(matched) if matched else pd.DataFrame()), pd.DataFrame(stats)


def published_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ref in C.PUBLISHED:
        for w in ref["area_wells"]:
            s = summary[summary.well == w]
            if s.empty:
                continue
            phi = float(s.phie_net.iloc[0])
            rows.append(dict(well=w, source=ref["source"], what=ref["what"],
                             published=f"{ref['phi_low']:.2f}-{ref['phi_high']:.2f}",
                             this_work_phie_net=round(phi, 3),
                             inside=ref["phi_low"] <= phi <= ref["phi_high"]))
    return pd.DataFrame(rows)
