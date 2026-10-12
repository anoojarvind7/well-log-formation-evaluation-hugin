"""Step 10 - validation against core plugs and published values.

Core-to-log depth shift: core depths are driller's depths, logs are logger's
depths, and they rarely agree. The shift is found by sliding log density
porosity PHID over +-MAX_SHIFT_M and keeping the shift with the highest
correlation to plug porosity. PHID is used for the tie because it is a single
sharp measurement that does not depend on the neutron unit assumption. The
shift is reported; a weak correlation (r < MIN_R) or a shift at the edge of the
search window is flagged rather than trusted silently.

Plug porosity (helium, on cleaned and dried plugs) is compared with both log
total porosity PHIT and shale-corrected PHIE. In clean sand they should agree;
in shaly sand dried-plug helium porosity usually sits between the two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import core

MAX_SHIFT_M = 5.0
MIN_R = 0.5


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
            interp = {k: g[k].interpolate(limit=3).to_numpy() for k in ("PHID", "PHIT", "PHIE", "VSH")}
            s, r, n = best_shift(g.DEPTH_MD.to_numpy(), interp["PHID"], c.depth_m.to_numpy(), c.phi.to_numpy())
            edge = abs(abs(s) - MAX_SHIFT_M) < 0.2
            d = c.depth_m.to_numpy() + s
            at = lambda k: np.interp(d, g.DEPTH_MD, interp[k], left=np.nan, right=np.nan)  # noqa: E731
            m = pd.DataFrame({
                "well": well, "core_depth_m": c.depth_m.to_numpy(), "log_depth_m": d,
                "phi_core": c.phi.to_numpy(), "perm_md": c.perm_md.to_numpy(),
                "phit_log": at("PHIT"), "phie_log": at("PHIE"), "phid_log": at("PHID"), "vsh_log": at("VSH"),
            })
            m = m[m.log_depth_m.between(top, base)]
            row.update(shift_m=round(s, 2), shift_r=round(r, 2), shift_reliable=bool(r >= MIN_R and not edge),
                       n_compared=int(m.phit_log.notna().sum()), mean_core_phi=round(m.phi_core.mean(), 3))
            for k in ("phit", "phie"):
                diff = (m[f"{k}_log"] - m.phi_core).dropna()
                row[f"mean_log_{k}"] = round(m[f"{k}_log"].mean(), 3)
                row[f"bias_{k}"] = round(diff.mean(), 3)
                row[f"rmse_{k}"] = round(float(np.sqrt((diff ** 2).mean())), 3)
                row[f"within_0p03_{k}"] = round(float((diff.abs() <= 0.03).mean()), 2)
            matched.append(m)
        stats.append(row)
    return (pd.concat(matched) if matched else pd.DataFrame()), pd.DataFrame(stats)


VSH_CLASSES = ([-0.01, 0.25, 0.40, 1.01], ["clean (Vsh < 0.25)", "shaly (0.25-0.40)", "shale (Vsh > 0.40)"])


def bias_by_vsh(matched: pd.DataFrame) -> pd.DataFrame:
    """Log-minus-core porosity split by the log's own Vsh at each plug."""
    m = matched.dropna(subset=["phit_log", "vsh_log"]).copy()
    m["vsh_class"] = pd.cut(m.vsh_log, *VSH_CLASSES[:1], labels=VSH_CLASSES[1])
    m["d_phit"] = m.phit_log - m.phi_core
    m["d_phie"] = m.phie_log - m.phi_core
    agg = dict(n=("d_phit", "size"), mean_core_phi=("phi_core", "mean"),
               bias_phit=("d_phit", "mean"), bias_phie=("d_phie", "mean"))
    per = m.groupby(["well", "vsh_class"], observed=True).agg(**agg).reset_index()
    pooled = m.groupby("vsh_class", observed=True).agg(**agg).reset_index()
    pooled.insert(0, "well", "ALL")
    return pd.concat([per, pooled], ignore_index=True).round(3)


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
