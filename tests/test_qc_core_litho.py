import numpy as np
import pandas as pd

from hugin import core, lithology, qc, validation


def test_spike_mask_finds_isolated_spike_not_bed_boundary():
    x = pd.Series(np.r_[np.full(50, 2.3), np.full(50, 2.55)] + np.random.default_rng(0).normal(0, 0.005, 100))
    x.iloc[20] = 3.0
    m = qc.spike_mask(x, 5, 6.0)
    assert m[20] and m.sum() == 1


def test_runs():
    d = np.arange(10) * 0.152
    r = qc.runs(np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1], bool), d)
    assert [n for *_, n in r] == [2, 1, 2]


def test_norm_well():
    assert core.norm_well(" NO 15/9-13 ") == "15/9-13"
    assert core.norm_well("25/2-13  t4") == "25/2-13 T4"
    # RealPore spellings
    assert core.norm_well("15_9-13") == "15/9-13"
    assert core.norm_well("1_2 -1 ") == "1/2-1"
    assert core.norm_well("1_3-7_T3") == "1/3-7 T3"


def test_detect_columns():
    df = pd.DataFrame(columns=["Well name", "Depth (m)", "Porosity (%)", "Permeability Kh (mD)", "Lithology"])
    c = core.detect_columns(df)
    assert c == {"well": "Well name", "depth": "Depth (m)", "porosity": "Porosity (%)",
                 "permeability": "Permeability Kh (mD)", "lithology": "Lithology", "porosity_fallback": None}


def test_detect_columns_prefers_realpore_names():
    df = pd.DataFrame(columns=["seq numb", "Well Name", "Measured Depth", "Permeability horizontal ka  (air) ",
                               "Porosity. Horizontal PLUG", "Porosity helium", "porosity best of available",
                               "main lithology"])
    c = core.detect_columns(df)
    assert c["porosity"] == "Porosity helium" and c["porosity_fallback"] == "porosity best of available"
    assert c["depth"] == "Measured Depth" and c["well"] == "Well Name"


def test_classify():
    vsh = np.array([0.1, 0.4, 0.8, 0.1, 0.2])
    rhob = np.array([2.3, 2.4, 2.5, 2.62, 1.6])
    nphi = np.array([0.2, 0.25, 0.35, 0.06, 0.5])
    assert lithology.classify(vsh, rhob, nphi).tolist() == ["Clean sand", "Shaly sand", "Shale", "Carbonate", "Coal"]


def test_best_shift_recovers_known_offset():
    d = np.arange(2000, 2050, 0.152)
    phi = 0.2 + 0.05 * np.sin(d / 1.7)
    core_d = np.arange(2010, 2040, 0.5)
    core_phi = np.interp(core_d + 1.52, d, phi)   # core is 1.52 m shallow
    s, r, n = validation.best_shift(d, phi, core_d, core_phi)
    assert abs(s - 1.52) < 0.16 and r > 0.99
