"""Step 4 - lithology from logs, then a measured comparison with FORCE.

The call is deliberately simple and rule-based so every sample's class can be
explained: coal and carbonate stringers are recognised on density-neutron,
then sand vs shale is split on Larionov Vsh. PEF is shown on the crossplot but
not used in the rules: across the Hugin it reads 3.5-5.5 b/e in sandstone
(quartz is ~1.8), which points at barite in the mud, so it would misclassify
clean sand as carbonate.

The FORCE labels are an independent interpretation by FORCE geoscientists.
They are only read here for the comparison, never used to tune the rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CLASSES = ["Clean sand", "Shaly sand", "Shale", "Carbonate", "Coal"]

# Approximate density-neutron chart lines (NPHI in limestone units), after
# Schlumberger CP-1. Good to ~1-2 p.u.; for display only.
_PHI = np.linspace(0, 0.40, 41)
CHART_LINES = {
    "Sandstone": (_PHI - 0.03 * (1 - _PHI / 0.4), 2.65 - 1.65 * _PHI),
    "Limestone": (_PHI, 2.71 - 1.71 * _PHI),
    "Dolomite": (_PHI + 0.06 * (1 - _PHI / 0.4) + 0.01, 2.87 - 1.87 * _PHI),
}


def classify(vsh: np.ndarray, rhob: np.ndarray, nphi: np.ndarray) -> np.ndarray:
    r = C.LITHO_RULES
    out = np.full(vsh.shape, None, dtype=object)
    with np.errstate(invalid="ignore"):
        coal = (rhob < r["coal_rhob_max"]) & (nphi > r["coal_nphi_min"])
        carb = (rhob > r["carb_rhob_min"]) & (nphi < r["carb_nphi_max"]) & (vsh < r["carb_vsh_max"])
        out[vsh <= r["clean_vsh_max"]] = "Clean sand"
        out[(vsh > r["clean_vsh_max"]) & (vsh < r["shale_vsh_min"])] = "Shaly sand"
        out[vsh >= r["shale_vsh_min"]] = "Shale"
        out[carb] = "Carbonate"
        out[coal] = "Coal"
    out[np.isnan(vsh) | np.isnan(rhob) | np.isnan(nphi)] = None
    return out


def force_class(litho_names: pd.Series) -> pd.Series:
    return litho_names.map(C.FORCE_TO_CLASS)


def confusion(mine: pd.Series, force: pd.Series) -> pd.DataFrame:
    ok = mine.notna() & force.notna()
    t = pd.crosstab(pd.Categorical(mine[ok], CLASSES), pd.Categorical(force[ok], CLASSES),
                    rownames=["this workflow"], colnames=["FORCE"], dropna=False)
    return t


def agreement(t: pd.DataFrame) -> float:
    n = t.to_numpy().sum()
    return float(np.trace(t.to_numpy()) / n) if n else float("nan")
