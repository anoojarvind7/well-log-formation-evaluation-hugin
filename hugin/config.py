"""Every interpretation parameter in one place, each with its reason.

Anything marked ASSUMPTION is a choice not measured in these wells. The Monte
Carlo ranges in ``MC`` are where those assumptions are allowed to vary.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FORCE_DIR = RAW / "force2020"
CORE_FILE = RAW / "core" / "RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

RESERVOIR = "Hugin Fm."
# Depth window either side of the reservoir kept for context (tops check, plots)
CONTEXT_M = 40.0

# The four wells picked in step 1 (see results/SUMMARY.md, "Well selection").
WELLS = ["15/9-13", "25/5-3", "25/2-14", "26/4-1"]

# Offset well used only to anchor Rw for 15/9-13 (no clean water leg of its own).
OFFSET_WELLS = ["15/9-14"]

# --- Step 2: QC --------------------------------------------------------------
QC = dict(
    # Bad hole: caliper more than this over bit size, in inches. 1.5 in on an
    # 8.5 in hole is ~18 % enlargement, where pad-type density reads low.
    washout_in=1.5,
    # |DRHO| above this means the density correction is too large to trust.
    drho_max=0.15,
    # Spike: sample differs from a 5-sample running median by > k robust sigmas
    spike_window=5,
    spike_k=6.0,
    spike_curves=["GR", "RHOB", "NPHI", "RDEP", "RMED", "PEF", "DTC"],
    # Physical plausibility ranges (value outside -> unit or data problem)
    ranges={
        "GR": (0, 1000), "RHOB": (1.0, 3.2), "NPHI": (-0.05, 1.0),
        "RDEP": (0.05, 5000), "RMED": (0.05, 5000), "PEF": (0, 15),
        "CALI": (4, 30), "DTC": (35, 200),
    },
)

# ASSUMPTION: 15/9-13 and 15/9-14 have no BS curve. Their Hugin calipers read
# 8.2-9.0 in (medians 8.46 and 8.97 in), consistent with a standard 8 1/2 in
# reservoir section.
BIT_SIZE_FALLBACK_IN = {"15/9-13": 8.5, "15/9-14": 8.5}

# --- Step 5: shale volume ----------------------------------------------------
# GR clean = 5th percentile of Hugin GR in that well (the cleanest sand it has).
# GR shale = median GR of the nearest *non-radioactive* marine mudstone. The
# Draupne Fm is excluded everywhere: it is an organic-rich hot shale (GR 150-250)
# and would understate Vsh. Heather Fm is used where it is thick; 15/9-13 has
# only 1.8 m of Heather (mixed with Draupne), so the underlying Smith Bank Fm
# mudstone is used instead.
GR_SHALE_SOURCE = {
    "15/9-13": "Smith Bank Fm.",
    "25/5-3": "Heather Fm.",
    "25/2-14": "Heather Fm.",
    "26/4-1": "Heather Fm.",
    "15/9-14": "Heather Fm.",
}
GR_CLEAN_PCTL = 5

# --- Step 6: porosity --------------------------------------------------------
RHO_MATRIX = 2.65   # quartz; Hugin is a quartz-rich shoreface sandstone
# ASSUMPTION: invaded-zone fluid is mud filtrate at ~1.0 g/cc. Mud type is not
# in the dataset; with oil-based mud 0.85 would be closer (MC covers 0.9-1.1).
RHO_FLUID = 1.0
# ASSUMPTION: NPHI is in limestone units (usual NCS convention, not stated in
# the LAS header). In a sandstone the tool reads ~0.04 v/v low, so +0.04 converts
# to sandstone units (approximation of Schlumberger chart Por-5).
NPHI_SS_SHIFT = 0.04

# --- Step 7: saturation ------------------------------------------------------
# ASSUMPTION: a = 1, m = 2, n = 2. No SCAL is available. These are the textbook
# values for consolidated sandstone; the MC varies m and n around them.
ARCHIE = dict(a=1.0, m=2.0, n=2.0)

# Rw source per well. "water_leg": fitted on that well's own Pickett plot over
# the stated MD interval (clean, wet sand). "offset": taken from another well's
# water leg and temperature-corrected with Arps.
RW_SOURCE = {
    "25/5-3": ("water_leg", (2426.0, 2431.0)),   # below the 2424-2426 m transition
    # 25/2-14 reads wet throughout, so its whole Hugin is the water leg. That is
    # circular for this well (it cannot show pay by construction near Rw), so
    # its Rw is cross-checked against the other wells at a common temperature.
    "25/2-14": ("water_leg", (3135.0, 3156.0)),
    "26/4-1": ("water_leg", (2765.0, 2810.0)),
    "15/9-14": ("water_leg", (3248.0, 3254.0)),
    "15/9-13": ("offset", "15/9-14"),
}
# Temperature for the Arps correction. ASSUMPTION: 4 degC at seabed and a
# 35 degC/km gradient below it (typical North Sea; no BHT in the dataset).
SEABED_T_C = 4.0
GEOTHERMAL_C_PER_KM = 35.0

# --- Step 8: cutoffs ---------------------------------------------------------
# Samples called coal or carbonate in step 4 are never net reservoir: coal
# reads as very high "porosity" on density-neutron, which would otherwise pass
# the porosity cutoff (seen at the Hugin base in 26/4-1).
# Common NCS sandstone screening values. They are judgement, not physics; the
# sweep in step 8 shows how much the answer depends on them.
CUTOFFS = dict(vsh=0.40, phi=0.10, sw=0.60)
SWEEP = dict(
    vsh=[0.30, 0.35, 0.40, 0.45, 0.50],
    phi=[0.06, 0.08, 0.10, 0.12, 0.14],
    sw=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
)

# --- Step 9: Monte Carlo -----------------------------------------------------
# Triangular (low, mode, high) unless stated. Rw and GR baselines are scaled
# or shifted relative to each well's base-case pick.
MC = dict(
    n_draws=5000,
    seed=20201,
    m=(1.8, 2.0, 2.2),
    n=(1.8, 2.0, 2.4),
    rw_factor=(0.75, 1.0, 1.30),
    rho_matrix=(2.64, 2.65, 2.68),
    rho_fluid=(0.90, 1.00, 1.10),
    gr_clean_shift=(-5.0, 0.0, 5.0),      # API
    gr_shale_shift=(-15.0, 0.0, 15.0),    # API
    nphi_shift=(0.0, 0.04, 0.06),         # 0 = NPHI already in sandstone units
)

# --- Step 4: lithology mapping ----------------------------------------------
FORCE_LITHO = {
    30000: "Sandstone", 65030: "Sandstone/Shale", 65000: "Shale", 80000: "Marl",
    74000: "Dolomite", 70000: "Limestone", 70032: "Chalk", 88000: "Halite",
    86000: "Anhydrite", 99000: "Tuff", 90000: "Coal", 93000: "Basement",
}
# FORCE classes collapsed to the classes this workflow can call from logs
FORCE_TO_CLASS = {
    "Sandstone": "Clean sand", "Sandstone/Shale": "Shaly sand", "Shale": "Shale",
    "Coal": "Coal", "Limestone": "Carbonate", "Dolomite": "Carbonate",
    "Marl": "Carbonate", "Chalk": "Carbonate",
}
LITHO_RULES = dict(
    coal_rhob_max=2.1,          # coal: very low density (2.1 also catches the
    coal_nphi_min=0.28,         # ... bed-shoulder samples) with high neutron
    carb_rhob_min=2.55,         # cemented stringer: dense,
    carb_nphi_max=0.12,         # low neutron,
    carb_vsh_max=0.35,          # and not shaly
    clean_vsh_max=0.25,
    shale_vsh_min=0.55,
)

# Published comparison values (step 10). Only values we could check are listed.
PUBLISHED = [
    dict(
        source="Otoo & Hodgetts (2021), Geosci. Model Dev. 14, 2075-2095, "
               "doi:10.5194/gmd-14-2075-2021",
        what="Volve reservoir (Hugin) log porosity, 'large proportion between'",
        area_wells=["15/9-13"],
        phi_low=0.18, phi_high=0.24,
    ),
]
