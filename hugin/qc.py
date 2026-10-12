"""Step 2 - QC.

What FORCE already did (Bormann et al. 2020, competition readme): logs were
"lightly cleaned and partly despiked" and resampled to 0.152 m. Nothing below
claims that work. What this module adds, per well and over the evaluation
window (reservoir +- CONTEXT_M):

* bad-hole flag from caliper vs bit size and from |DRHO|
* spike removal against a running median (robust sigma)
* physical range checks (catches % vs v/v and unit mix-ups)
* LAS unit check against the expected units
* curve depth-alignment check (lagged correlation against GR)

Nothing is dropped silently: every masked sample ends up in
``results/qc_masked_intervals.csv`` with its reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import io

EXPECTED_UNITS = {
    "GR": {"gapi"}, "RHOB": {"g/cm3", "g/cc"}, "NPHI": {"m3/m3", "v/v", "dec"},
    "RDEP": {"ohm.m", "ohmm"}, "RMED": {"ohm.m", "ohmm"}, "CALI": {"in"},
    "DTC": {"us/ft"}, "PEF": {"b/e"}, "DRHO": {"g/cm3", "g/cc"},
}
DENSITY_CURVES = ["RHOB", "NPHI", "PEF"]


def window(df: pd.DataFrame, well: str) -> pd.DataFrame:
    g = df[df.WELL == well]
    h = g[g.FORMATION == C.RESERVOIR]
    top, base = h.DEPTH_MD.min(), h.DEPTH_MD.max()
    return g[g.DEPTH_MD.between(top - C.CONTEXT_M, base + C.CONTEXT_M)].copy().reset_index(drop=True)


def runs(mask: np.ndarray, depth: np.ndarray) -> list[tuple[float, float, int]]:
    """Contiguous True runs -> (top, base, n_samples)."""
    out = []
    m = np.asarray(mask, bool)
    if not m.any():
        return out
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        out.append((float(depth[s]), float(depth[e - 1]), int(e - s)))
    return out


def spike_mask(x: pd.Series, win: int, k: float) -> np.ndarray:
    """A spike departs from the running median of its neighbours by more than
    k robust sigmas, where sigma is estimated from the sample-to-sample
    differences of the whole window (MAD), so ordinary bed boundaries are not
    flagged but isolated one- or two-sample excursions are."""
    med = x.rolling(win, center=True, min_periods=3).median()
    dx = x.diff()
    sigma = 1.4826 * np.nanmedian(np.abs(dx - np.nanmedian(dx)))
    if not np.isfinite(sigma) or sigma == 0:
        return np.zeros(len(x), bool)
    with np.errstate(invalid="ignore"):
        return ((x - med).abs() > k * sigma).fillna(False).to_numpy()


def alignment_lag(a: pd.Series, b: pd.Series, max_lag: int = 6) -> tuple[int, float]:
    """Lag (samples) of b relative to a giving the largest |correlation|.
    Positive = b would need shifting down to match a."""
    best = (0, 0.0)
    for lag in range(-max_lag, max_lag + 1):
        r = a.corr(b.shift(lag))
        if np.isfinite(r) and abs(r) > abs(best[1]) + 1e-9:
            best = (lag, r)
    return best


def qc_well(w: pd.DataFrame, well: str) -> tuple[pd.DataFrame, list[dict], dict]:
    """Returns the cleaned window, masked-interval records and a summary."""
    cfg = C.QC
    depth = w.DEPTH_MD.to_numpy()
    log: list[dict] = []
    summ: dict = {"well": well}

    # Bit size
    if w.BS.notna().mean() > 0.5:
        bs = w.BS
        summ["bit_size_source"] = "BS curve"
    else:
        bs = pd.Series(C.BIT_SIZE_FALLBACK_IN.get(well, np.nan), index=w.index)
        summ["bit_size_source"] = f"assumed {C.BIT_SIZE_FALLBACK_IN.get(well)} in (no BS curve)"
    w["BS_USED"] = bs

    # Range checks first, so spikes are judged on plausible data
    for c, (lo, hi) in cfg["ranges"].items():
        if c not in w:
            continue
        bad = (w[c] < lo) | (w[c] > hi)
        for t, b, n in runs(bad.to_numpy(), depth):
            log.append(dict(well=well, curve=c, reason=f"outside {lo}-{hi}", top_md=t, base_md=b, n=n))
        w.loc[bad, c] = np.nan

    # Spikes
    n_spk = 0
    for c in cfg["spike_curves"]:
        if c not in w or w[c].notna().sum() < 20:
            continue
        x = np.log10(w[c]) if c in ("RDEP", "RMED") else w[c]
        m = spike_mask(x, cfg["spike_window"], cfg["spike_k"])
        n_spk += int(m.sum())
        for t, b, n in runs(m, depth):
            log.append(dict(well=well, curve=c, reason="spike", top_md=t, base_md=b, n=n))
        w.loc[m, c] = np.nan
    summ["spike_samples_masked"] = n_spk

    # Bad hole
    wash = (w.CALI - w.BS_USED) > cfg["washout_in"]
    drho = w.DRHO.abs() > cfg["drho_max"]
    bad = (wash | drho).fillna(False).to_numpy()
    w["BADHOLE"] = bad
    for t, b, n in runs(bad, depth):
        why = []
        sl = slice(np.searchsorted(depth, t), np.searchsorted(depth, b) + 1)
        if wash.iloc[sl].any():
            why.append("washout")
        if drho.iloc[sl].any():
            why.append("|DRHO|")
        log.append(dict(well=well, curve="+".join(DENSITY_CURVES), reason="bad hole: " + ",".join(why),
                        top_md=t, base_md=b, n=n))
    for c in DENSITY_CURVES:
        w.loc[bad, c] = np.nan

    hug = w.FORMATION == C.RESERVOIR
    summ["hugin_badhole_pct"] = round(100 * w.loc[hug, "BADHOLE"].mean(), 2)
    summ["hugin_cali_minus_bs_p95_in"] = round(float(np.nanpercentile((w.CALI - w.BS_USED)[hug], 95)), 2)
    summ["hugin_pef_median"] = round(float(w.loc[hug, "PEF"].median()), 2) if w.PEF.notna().any() else None

    # Depth alignment vs GR in the window. Only a strong correlation can show
    # an offset; a weak one says nothing either way and is reported as such.
    for c in ["RHOB", "NPHI", "RDEP"]:
        lag, r = alignment_lag(w.GR, w[c] if c != "RDEP" else np.log10(w[c]))
        summ[f"lag_{c}_samples"] = lag
        summ[f"lag_{c}_r"] = round(r, 2)
        summ[f"align_{c}"] = ("inconclusive (r < 0.5)" if abs(r) < 0.5
                              else "aligned (+-1 sample)" if abs(lag) <= 1 else f"offset {lag} samples")
    return w, log, summ


def unit_check(well: str) -> list[dict]:
    units = io.las_units(well)
    out = []
    for c, ok in EXPECTED_UNITS.items():
        u = units.get(c)
        if u is None:
            continue
        out.append(dict(well=well, curve=c, las_unit=u, ok=u.lower().replace(" ", "") in ok))
    return out
