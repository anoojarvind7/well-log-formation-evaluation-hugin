"""Run steps 1-11 end to end:  python -m hugin.pipeline

Writes tables to results/, figures to results/figures/, and results/SUMMARY.md.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from . import evaluation, inventory, io, lithology, petro, plots, qc, report, tops, uncertainty, validation


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=C.MC["n_draws"])
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    C.MC["n_draws"] = a.draws
    C.RESULTS.mkdir(exist_ok=True)
    R = {}

    # 1. Load + inventory --------------------------------------------------
    df, checks = io.load_force(cache=not a.no_cache)
    R["format_checks"] = checks
    inv = inventory.inventory(df)
    inv.to_csv(C.RESULTS / "01_inventory_hugin_wells.csv", index=False)
    R["inventory"] = inv
    print(f"[1] {df.WELL.nunique()} wells loaded, {len(inv)} with a Hugin label")

    # 2. QC ------------------------------------------------------------------
    all_wells = C.WELLS + C.OFFSET_WELLS
    windows, masked, qcs, ucheck = {}, [], [], []
    for w in all_wells:
        win, log, s = qc.qc_well(qc.window(df, w), w)
        windows[w] = win
        masked += log
        qcs.append(s)
        ucheck += qc.unit_check(w)
    masked = pd.DataFrame(masked)
    masked.to_csv(C.RESULTS / "02_qc_masked_intervals.csv", index=False)
    qcs = pd.DataFrame(qcs)
    qcs.to_csv(C.RESULTS / "02_qc_summary.csv", index=False)
    ucheck = pd.DataFrame(ucheck)
    ucheck.to_csv(C.RESULTS / "02_qc_units.csv", index=False)
    R.update(masked=masked, qc=qcs, units=ucheck)
    print(f"[2] QC: {len(masked)} masked intervals logged")

    # 3. Tops ---------------------------------------------------------------
    tchk, tops_tbl = [], {}
    for w in all_wells:
        t = tops.formation_tops(windows[w])
        tops_tbl[w] = t
        c = tops.check_reservoir_top(windows[w], w)
        tchk.append(c)
        if w in C.WELLS:
            plots.tops_check(windows[w], w, t, c)
    tchk = pd.DataFrame(tchk)
    tchk.to_csv(C.RESULTS / "03_tops_check.csv", index=False)
    R["tops"] = tchk
    print("[3] tops checked")

    # 5-8. Evaluation (needed before 4, which uses Vsh) ---------------------
    inputs, curves, summ, sweeps = {}, {}, [], []
    for w in C.WELLS:
        inp = evaluation.build_inputs(windows, w)
        inputs[w] = inp
        cv, s = evaluation.base_case(inp)
        cv["RT"] = inp.rt
        curves[w] = cv
        summ.append(s)
        sweeps.append(evaluation.cutoff_sweep(cv, inp.dz_md, inp.exclude))
        plots.pickett(inp, cv, s["rw"], C.ARCHIE["m"], C.ARCHIE["n"])
    summ = pd.DataFrame(summ)
    # Regional Rw cross-check: rerun each well with the median Rw of the
    # *other* wells (compared at 100 degC, moved to this well's temperature).
    reg = []
    for w in C.WELLS:
        s = summ.set_index("well").loc[w]
        others = summ[summ.well != w].rw_at_100c.median()
        rw_reg = float(petro.arps_rw(others, 100.0, s.t_reservoir_c))
        p = evaluation.base_params()
        p["rw_factor"] = rw_reg / s.rw
        r = evaluation.evaluate(inputs[w], p)
        reg.append(dict(well=w, rw_own=s.rw, rw_regional=round(rw_reg, 4), pay_md_own=s.pay_md,
                        pay_md_regional_rw=round(float(r["pay_md"]), 2)))
    reg = pd.DataFrame(reg)
    reg.to_csv(C.RESULTS / "07_rw_regional_check.csv", index=False)
    R["rw_regional"] = reg
    summ.to_csv(C.RESULTS / "08_base_case_summary.csv", index=False)
    sweep = pd.concat(sweeps)
    sweep.to_csv(C.RESULTS / "08_cutoff_sweep.csv", index=False)
    allcurves = pd.concat(curves.values())
    allcurves.round(4).to_csv(C.RESULTS / "08_curves_hugin.csv", index=False)
    plots.sweep_plot(sweep)
    R.update(summary=summ, sweep=sweep)
    print("[5-8] base case done")

    # 4. Lithology -----------------------------------------------------------
    conf_all, agree = [], []
    mine_by_well = {}
    for w in C.WELLS:
        win = windows[w]
        win["LITHO_CLASS"] = lithology.force_class(win.LITHO)
        cv = curves[w]
        h = win[win.FORMATION == C.RESERVOIR].reset_index(drop=True)
        mine = lithology.classify(cv.VSH.to_numpy(), h.RHOB.to_numpy(), h.NPHI.to_numpy())
        mine_by_well[w] = mine
        cv["LITHO_THIS_WORK"] = mine
        cv["LITHO_FORCE"] = h.LITHO_CLASS.to_numpy()
        t = lithology.confusion(pd.Series(mine), h.LITHO_CLASS)
        t2 = t.copy()
        t2.insert(0, "well", w)
        conf_all.append(t2.reset_index())
        agree.append(dict(well=w, agreement=round(lithology.agreement(t), 3), n=int(t.to_numpy().sum())))
        plots.dn_crossplot(h, w)
    pooled = sum(c.drop(columns=["well"]).set_index("this workflow") for c in conf_all)
    agree.append(dict(well="ALL", agreement=round(lithology.agreement(pooled), 3), n=int(pooled.to_numpy().sum())))
    pd.concat(conf_all).to_csv(C.RESULTS / "04_lithology_confusion_by_well.csv", index=False)
    pooled.to_csv(C.RESULTS / "04_lithology_confusion_pooled.csv")
    agree = pd.DataFrame(agree)
    agree.to_csv(C.RESULTS / "04_lithology_agreement.csv", index=False)
    R.update(confusion=pooled, agreement=agree)
    print("[4] lithology agreement:", agree.set_index("well").agreement.to_dict())

    # 9. Monte Carlo ---------------------------------------------------------
    draws, pct, sens = [], [], []
    for w in C.WELLS:
        d, p = uncertainty.run(inputs[w], n=a.draws)
        draws.append(d)
        pct.append(p)
        s = uncertainty.sensitivity(d, "pay_md").rename("rank_corr_with_pay").reset_index()
        s.insert(0, "well", w)
        sens.append(s.rename(columns={"index": "input"}))
    draws, pct, sens = pd.concat(draws), pd.concat(pct), pd.concat(sens)
    pct.round(4).to_csv(C.RESULTS / "09_monte_carlo_percentiles.csv", index=False)
    sens.round(3).to_csv(C.RESULTS / "09_monte_carlo_sensitivity.csv", index=False)
    plots.mc_plot(draws)
    R.update(mc=pct, sens=sens)
    print("[9] Monte Carlo done")

    # 10. Validation ---------------------------------------------------------
    cc = validation.core_comparison(allcurves)
    if cc is not None:
        matched, cstats = cc
        matched.to_csv(C.RESULTS / "10_core_vs_log_matched.csv", index=False)
        cstats.to_csv(C.RESULTS / "10_core_vs_log_stats.csv", index=False)
        if len(matched):
            plots.core_plot(matched)
        R["core"] = cstats
    else:
        R["core"] = None
    pub = validation.published_comparison(summ)
    pub.to_csv(C.RESULTS / "10_published_comparison.csv", index=False)
    R["published"] = pub
    print("[10] validation:", "core file present" if cc is not None else "core file NOT present")

    # 11. Outputs ------------------------------------------------------------
    for w in C.WELLS:
        plots.composite(windows[w], curves[w], w, tops_tbl[w], mine_by_well[w])
    plots.correlation_panel([(w, windows[w], curves[w]) for w in C.WELLS])
    report.write_summary(R)
    print("[11] results/SUMMARY.md written")


if __name__ == "__main__":
    main()
