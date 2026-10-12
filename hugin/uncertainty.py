"""Step 9 - Monte Carlo uncertainty.

Draws m, n, an Rw multiplier, matrix and fluid density and shifts of both GR
baselines (ranges in config.MC), and re-runs the full step 5-8 chain for every
draw. Rw is re-derived from the water leg inside each draw with that draw's m
and densities, so Rw and m stay consistent the way a real Pickett fit would.
Cutoffs are held at the base case; their effect is shown separately by the
step 8 sweep so the two sources of spread are not mixed.

Reporting convention (industry, exceedance): P90 is the low case (90 % chance
the true value is at least this), P50 the median, P10 the high case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .evaluation import WellInputs, evaluate

METRICS = ["phie_net", "sw_pay", "net_md", "pay_md", "hc_column_m", "rw"]


def draw_params(n: int, rng: np.random.Generator) -> dict:
    mc = C.MC
    tri = lambda lo_mode_hi: rng.triangular(*lo_mode_hi, size=n)  # noqa: E731
    return dict(
        m=tri(mc["m"]), n=tri(mc["n"]), a=np.full(n, C.ARCHIE["a"]), rw_factor=tri(mc["rw_factor"]),
        rho_matrix=tri(mc["rho_matrix"]), rho_fluid=tri(mc["rho_fluid"]),
        gr_clean_shift=tri(mc["gr_clean_shift"]), gr_shale_shift=tri(mc["gr_shale_shift"]),
        nphi_shift=tri(mc["nphi_shift"]),
    )


def run(inp: WellInputs, n: int | None = None, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = n or C.MC["n_draws"]
    rng = np.random.default_rng(seed if seed is not None else C.MC["seed"])
    p = draw_params(n, rng)
    r = evaluate(inp, p)
    draws = pd.DataFrame({**{k: v for k, v in p.items() if k != "a"}, **{k: r[k] for k in METRICS}})
    draws.insert(0, "well", inp.well)
    rows = []
    for k in METRICS:
        x = draws[k].to_numpy()
        fin = x[np.isfinite(x)]
        rows.append(dict(
            well=inp.well, metric=k,
            P90=np.percentile(fin, 10) if len(fin) else np.nan,
            P50=np.percentile(fin, 50) if len(fin) else np.nan,
            P10=np.percentile(fin, 90) if len(fin) else np.nan,
            mean=fin.mean() if len(fin) else np.nan,
            n_valid=len(fin),
        ))
    return draws, pd.DataFrame(rows)


def sensitivity(draws: pd.DataFrame, metric: str = "pay_md") -> pd.Series:
    """Rank correlation of each input with a metric (a quick tornado)."""
    inputs = ["m", "n", "rw_factor", "rho_matrix", "rho_fluid", "gr_clean_shift", "gr_shale_shift", "nphi_shift"]
    d = draws[inputs + [metric]].dropna()
    if d[metric].nunique() < 2:
        return pd.Series(0.0, index=inputs)
    return d[inputs].rank().corrwith(d[metric].rank()).sort_values(key=np.abs, ascending=False)
