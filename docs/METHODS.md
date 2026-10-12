# Methods and reasoning

Why each step is done the way it is. Parameter values live in `hugin/config.py`; this file is the
argument for them.

## 1. Data and inventory

The FORCE 2020 competition CSVs are the working source because they carry the NPD formation labels
and the bit size; the per-well LAS files are read with `lasio` for curve units only. The CSV
delimiter (`;`) is detected from the header, not assumed, and the leaderboard-test lithology labels
(shipped in a separate target file) are joined back only after checking the rows align one-to-one.

22 FORCE wells have a Hugin label. The four used are the plan's strongest core candidates with full
GR/density/neutron/resistivity/PEF coverage, and between them they give two wells with pay and two
with a clear water leg. 15/9-14 is used only as the Rw offset for 15/9-13.

## 2. QC

FORCE already lightly cleaned and partly despiked the logs; that is their work. This repository adds:

- **Bad hole**: caliper more than 1.5 in over bit size, or |DRHO| > 0.15 g/cc. Density, neutron and
  PEF are masked there. None of the Hugin is affected in the four wells; the washouts are in the
  surrounding shales.
- **Spikes**: a sample more than 6 robust sigmas from its 5-sample running median, with sigma from
  the MAD of sample-to-sample differences over the window. A first version estimated sigma from the
  local residuals and flagged ~25 % of samples at ordinary bed boundaries; the difference-based
  sigma flags only isolated excursions (a handful per well), which is what one expects after FORCE's
  despiking.
- **Units and ranges**: LAS units checked against expected units; values outside physical ranges
  are masked (this is what would catch NPHI in percent).
- **Depth alignment**: lagged correlation of each curve against GR. Only strong correlations can
  reveal an offset, so weak ones are reported as inconclusive and no shift is applied.

Every masked sample is listed in `results/02_qc_masked_intervals.csv` with its reason.

## 3. Tops

The NPD formation labels are used as tops. As a check, the steepest GR drop within 8 m of the
labelled Hugin top is found; all four wells agree within 1.6 m, and the tops figures show the
labelled top at the Heather shale to Hugin sand transition.

## 4. Lithology

A transparent rule set (coal: RHOB < 2.1 and NPHI > 0.28; carbonate/cemented: RHOB > 2.55,
NPHI < 0.12, Vsh < 0.35; then clean sand / shaly sand / shale on Larionov Vsh at 0.25 and 0.55).
PEF is plotted but not used: it reads 3.5-5.4 b/e in Hugin sandstone against ~1.8 for quartz,
most likely a barite mud effect. FORCE labels are read only for the confusion table, never to tune
the rules. Coal and carbonate samples are excluded from net reservoir.

## 5. Shale volume

Linear GR index between a clean baseline (5th percentile of Hugin GR in that well) and a shale
baseline (median GR of the nearest non-radioactive marine mudstone: Heather Fm; Smith Bank Fm in
15/9-13 where the Heather is 1.8 m thick). The Draupne Fm is never used: it is an organic-rich hot
shale (GR 150-250) that would understate Vsh. The Larionov older-rocks correction is applied because
the Hugin is Middle Jurassic and consolidated; it lowers mean Vsh by 28-44 % versus linear.

## 6. Porosity

Density porosity with a quartz matrix (2.65 g/cc) and filtrate density 1.0 g/cc (mud type is not in
the dataset). NPHI is assumed to be in limestone units and shifted +0.04 v/v to sandstone units.
Density and neutron are combined as the root-mean-square, which damps the light-hydrocarbon effect
seen as density-neutron crossover in the pay of 15/9-13 and 25/5-3. Effective porosity shale-corrects
density and neutron separately with the shale point read in the same baseline shale, then combines.

## 7. Water saturation

Rw is read from a Pickett plot (m = 2) as the median apparent Rw (Rt * PHIT^m) of clean, porous
samples in a water leg:

| Well | Rw source |
|---|---|
| 25/5-3 | own water leg 2426-2431 m MD, below a 2424-2426 m transition |
| 26/4-1 | own water leg 2765-2810 m MD |
| 25/2-14 | whole Hugin (wet throughout; circular for this well, so cross-checked) |
| 15/9-13 | no clean water leg of its own: 15/9-14 water leg (3248-3254 m), moved to 15/9-13 temperature with Arps |

Temperatures assume 4 degC at seabed and 35 degC/km; they are only used to move Rw between wells.
At a common 100 degC the four Rw values span 0.056-0.087 ohm.m, and the Monte Carlo Rw factor
(0.75-1.30) covers that spread. As a further check each well is re-run with the median Rw of the
other three; no well changes from wet to pay or back.

Archie (a = 1, m = 2, n = 2, total porosity) and Simandoux (effective porosity, Rsh from the baseline
shale) are both computed. Simandoux is the base case because most of the Hugin carries some shale;
in clean sand the two agree by construction (tested).

## 8. Cutoffs and net pay

Vsh <= 0.40, PHIE >= 0.10, Sw <= 0.60: common screening values, not calibrated to any field. The
sweep over 150 combinations moves net pay by about -25 % to +30 % in the two paying wells,
and most of that comes from the Sw cutoff.

## 9. Uncertainty

5000 Monte Carlo draws per well with triangular distributions on m, n, an Rw multiplier, matrix and
fluid density, the neutron unit shift, and shifts of both GR baselines. Every draw re-runs the whole
step 5-8 chain, including re-fitting Rw on the water leg with that draw's m and densities, so Rw and
m stay consistent the way a real Pickett fit would. Cutoffs are held fixed so their effect (step 8)
and parameter uncertainty are not mixed. P90 is the low case (exceedance convention).

What the Monte Carlo does not cover: the choice of saturation model, the choice of Rw interval, and
the cutoffs. Those are shown separately (Archie vs Simandoux curves, regional-Rw check, sweep).

## 10. Validation

- **Core plugs** (Zenodo 4419060): matched by well name; plug porosity compared with log total
  porosity after a core-to-log depth shift found by maximising correlation within +-5 m. The shift is
  reported, and one at the edge of the search window is flagged rather than used. Not yet run in
  `results/`: the core file was not available when they were generated.
- **Published values**: 15/9-13 average PHIE is compared with the 0.18-0.24 log porosity range
  reported for the Volve reservoir by Otoo & Hodgetts (2021). This is a regional comparison only.
- **FORCE lithology**: the step 4 confusion table.
