"""Steps 5-8 - shale volume, porosity, water saturation, cutoffs and net pay.

The whole chain lives in one vectorised function, ``evaluate``, so the base case
(step 5-8) and the Monte Carlo (step 9) run exactly the same code: the base case
is simply the Monte Carlo with every parameter at its mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from . import config as C
from . import lithology, petro


@dataclass
class WellInputs:
    well: str
    depth: np.ndarray            # MD, m (Hugin samples)
    tvdss: np.ndarray
    dz_md: np.ndarray            # sample thickness, m
    dz_tvd: np.ndarray
    gr: np.ndarray
    rhob: np.ndarray
    nphi: np.ndarray
    rt: np.ndarray
    gr_clean: float
    gr_shale: float
    rhob_sh: float
    nphi_sh: float
    rsh: float
    wl_rt: np.ndarray            # water-leg samples used for Rw
    wl_rhob: np.ndarray
    wl_nphi: np.ndarray
    rw_temp_ratio: float = 1.0   # Arps factor (1 for a well's own water leg)
    exclude: np.ndarray | None = None   # coal / carbonate samples, never net
    notes: dict = field(default_factory=dict)


def _thickness(x: np.ndarray) -> np.ndarray:
    d = np.diff(x)
    d = np.concatenate([d, d[-1:]])
    return np.abs(d)


def shale_samples(w: pd.DataFrame, well: str) -> pd.DataFrame:
    src = C.GR_SHALE_SOURCE[well]
    return w[w.FORMATION == src]


def water_leg_samples(w: pd.DataFrame, well: str) -> pd.DataFrame:
    kind, arg = C.RW_SOURCE[well]
    assert kind == "water_leg"
    top, base = arg
    s = w[w.DEPTH_MD.between(top, base) & (w.FORMATION == C.RESERVOIR)]
    return s


def build_inputs(windows: dict[str, pd.DataFrame], well: str) -> WellInputs:
    w = windows[well]
    h = w[w.FORMATION == C.RESERVOIR]
    sh = shale_samples(w, well)
    gr_clean = float(np.nanpercentile(h.GR, C.GR_CLEAN_PCTL))
    gr_shale = float(sh.GR.median())
    notes = {"gr_shale_source": C.GR_SHALE_SOURCE[well], "n_shale_samples": len(sh)}

    kind, arg = C.RW_SOURCE[well]
    if kind == "water_leg":
        wl_src, wl_well, ratio = water_leg_samples(w, well), well, 1.0
    else:
        wl_well = arg
        ow = windows[wl_well]
        wl_src = water_leg_samples(ow, wl_well)
        t_off = petro.formation_temperature(wl_src.TVDSS.mean(), C.SEABED_T_C, C.GEOTHERMAL_C_PER_KM)
        t_here = petro.formation_temperature(h.TVDSS.mean(), C.SEABED_T_C, C.GEOTHERMAL_C_PER_KM)
        ratio = float((t_off + 21.5) / (t_here + 21.5))
        notes.update(rw_offset_well=wl_well, t_offset_c=round(float(t_off), 1), t_here_c=round(float(t_here), 1))
    # keep only clean, porous water-leg points (base-case parameters)
    ow = windows[wl_well]
    wl_gr_clean = float(np.nanpercentile(ow.loc[ow.FORMATION == C.RESERVOIR, "GR"], C.GR_CLEAN_PCTL))
    wl_gr_sh = float(shale_samples(windows[wl_well], wl_well).GR.median())
    vsh_wl = petro.larionov_older(petro.gr_index(wl_src.GR.to_numpy(), wl_gr_clean, wl_gr_sh))
    phit_wl = petro.density_neutron_porosity(
        petro.density_porosity(wl_src.RHOB.to_numpy(), C.RHO_MATRIX, C.RHO_FLUID),
        wl_src.NPHI.to_numpy() + C.NPHI_SS_SHIFT)
    keep = (vsh_wl < 0.3) & (phit_wl > 0.10) & np.isfinite(wl_src.RDEP.to_numpy())
    wl = wl_src[keep]
    notes.update(rw_source=kind, rw_well=wl_well, n_water_leg=int(keep.sum()),
                 water_leg_md=f"{C.RW_SOURCE[wl_well][1][0]}-{C.RW_SOURCE[wl_well][1][1]}")

    vsh_base = petro.larionov_older(petro.gr_index(h.GR.to_numpy(), gr_clean, gr_shale))
    cls = lithology.classify(vsh_base, h.RHOB.to_numpy(), h.NPHI.to_numpy())
    exclude = np.isin(cls, ["Coal", "Carbonate"])
    notes["n_excluded_coal_carb"] = int(exclude.sum())

    return WellInputs(
        exclude=exclude, well=well, depth=h.DEPTH_MD.to_numpy(), tvdss=h.TVDSS.to_numpy(),
        dz_md=_thickness(h.DEPTH_MD.to_numpy()), dz_tvd=_thickness(h.TVDSS.to_numpy()),
        gr=h.GR.to_numpy(), rhob=h.RHOB.to_numpy(), nphi=h.NPHI.to_numpy(), rt=h.RDEP.to_numpy(),
        gr_clean=gr_clean, gr_shale=gr_shale, rhob_sh=float(sh.RHOB.median()),
        nphi_sh=float(sh.NPHI.median()), rsh=float(sh.RDEP.median()),
        wl_rt=wl.RDEP.to_numpy(), wl_rhob=wl.RHOB.to_numpy(), wl_nphi=wl.NPHI.to_numpy(),
        rw_temp_ratio=ratio, notes=notes,
    )


def base_params() -> dict:
    return dict(
        m=C.ARCHIE["m"], n=C.ARCHIE["n"], a=C.ARCHIE["a"], rw_factor=1.0,
        rho_matrix=C.RHO_MATRIX, rho_fluid=C.RHO_FLUID, gr_clean_shift=0.0, gr_shale_shift=0.0,
        nphi_shift=C.NPHI_SS_SHIFT,
    )


def evaluate(inp: WellInputs, p: dict, cutoffs: dict | None = None, curves: bool = False) -> dict:
    """Run steps 5-8. Parameters may be scalars or arrays of shape (N,);
    sample arrays broadcast along the last axis, so results are (N, S)."""
    cut = cutoffs or C.CUTOFFS
    col = {k: np.asarray(v, float)[..., None] if np.ndim(v) else float(v) for k, v in p.items()}

    igr = petro.gr_index(inp.gr, inp.gr_clean + col["gr_clean_shift"], inp.gr_shale + col["gr_shale_shift"])
    vsh = petro.larionov_older(igr)
    phid = petro.density_porosity(inp.rhob, col["rho_matrix"], col["rho_fluid"])
    phin = inp.nphi + col["nphi_shift"]
    phit = np.clip(petro.density_neutron_porosity(phid, phin), 0, 0.45)
    phid_sh = petro.density_porosity(inp.rhob_sh, col["rho_matrix"], col["rho_fluid"])
    phin_sh = inp.nphi_sh + col["nphi_shift"]
    phie = petro.effective_porosity(phid, phin, vsh, phid_sh, phin_sh)

    phit_wl = petro.density_neutron_porosity(
        petro.density_porosity(inp.wl_rhob, col["rho_matrix"], col["rho_fluid"]), inp.wl_nphi + col["nphi_shift"])
    rwa_wl = petro.rwa(inp.wl_rt, phit_wl, col["a"], col["m"])
    rw = np.median(rwa_wl, axis=-1, keepdims=True) * inp.rw_temp_ratio * col["rw_factor"]

    sw_ar = petro.archie_sw(inp.rt, phit, rw, col["a"], col["m"], col["n"])
    sw_sim = petro.simandoux_sw(inp.rt, phie, vsh, rw, inp.rsh, col["a"], col["m"], col["n"])
    net, pay = petro.flags(vsh, phie, sw_sim, cut["vsh"], cut["phi"], cut["sw"])
    if inp.exclude is not None:
        net = net & ~inp.exclude
        pay = pay & ~inp.exclude

    dz, dzt = inp.dz_md, inp.dz_tvd
    netm = (net * dz).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = dict(
            rw=np.squeeze(rw, -1),
            gross_md=dz.sum(),
            net_md=netm,
            pay_md=(pay * dz).sum(-1),
            pay_tvd=(pay * dzt).sum(-1),
            ntg=netm / dz.sum(),
            phie_net=np.nansum(phie * net * dz, -1) / netm,
            sw_pay=np.nansum(sw_sim * phie * pay * dz, -1) / np.nansum(phie * pay * dz, -1),
            hc_column_m=np.nansum(phie * (1 - sw_sim) * pay * dz, -1),
            unevaluated_md=(np.isnan(phie) * dz).sum(-1) if np.ndim(phie) else 0.0,
        )
    if curves:
        out.update(VSH_LIN=igr, VSH=vsh, PHID=phid, PHIN=phin, PHIT=phit, PHIE=phie,
                   SW_ARCHIE=sw_ar, SW_SIM=sw_sim, NET=net, PAY=pay, RWA_WL=rwa_wl)
    return out


def base_case(inp: WellInputs) -> tuple[pd.DataFrame, dict]:
    r = evaluate(inp, base_params(), curves=True)
    curves = pd.DataFrame({
        "WELL": inp.well, "DEPTH_MD": inp.depth, "TVDSS": inp.tvdss,
        **{k: r[k] for k in ["VSH_LIN", "VSH", "PHID", "PHIN", "PHIT", "PHIE", "SW_ARCHIE", "SW_SIM", "NET", "PAY"]},
    })
    with np.errstate(invalid="ignore"):
        summ = dict(
            well=inp.well, gr_clean=round(inp.gr_clean, 1), gr_shale=round(inp.gr_shale, 1),
            gr_shale_source=inp.notes["gr_shale_source"], rsh=round(inp.rsh, 2),
            rhob_sh=round(inp.rhob_sh, 3), nphi_sh=round(inp.nphi_sh, 3),
            rw=round(float(r["rw"]), 4), rw_source=inp.notes["rw_source"], rw_well=inp.notes["rw_well"],
            water_leg_md=inp.notes["water_leg_md"], n_water_leg=inp.notes["n_water_leg"],
            gross_md=round(float(r["gross_md"]), 2), net_md=round(float(r["net_md"]), 2),
            pay_md=round(float(r["pay_md"]), 2), pay_tvd=round(float(r["pay_tvd"]), 2),
            ntg=round(float(r["ntg"]), 3), phie_net=round(float(r["phie_net"]), 3),
            sw_pay=round(float(r["sw_pay"]), 3) if np.isfinite(r["sw_pay"]) else None,
            hc_column_m=round(float(r["hc_column_m"]), 2),
            vsh_lin_mean=round(float(np.nanmean(r["VSH_LIN"])), 3),
            vsh_larionov_mean=round(float(np.nanmean(r["VSH"])), 3),
            phit_mean=round(float(np.nanmean(r["PHIT"])), 3), phie_mean=round(float(np.nanmean(r["PHIE"])), 3),
            sw_archie_mean=round(float(np.nanmean(r["SW_ARCHIE"])), 3),
            sw_sim_mean=round(float(np.nanmean(r["SW_SIM"])), 3),
            unevaluated_md=round(float(r["unevaluated_md"]), 2),
        )
        t_here = petro.formation_temperature(np.nanmean(inp.tvdss), C.SEABED_T_C, C.GEOTHERMAL_C_PER_KM)
        summ["t_reservoir_c"] = round(float(t_here), 1)
        # Rw at a common 100 degC so wells can be compared (Arps)
        summ["rw_at_100c"] = round(float(petro.arps_rw(float(r["rw"]), t_here, 100.0)), 4)
        pay = r["PAY"].astype(bool)
        summ["lowest_pay_md"] = round(float(inp.depth[pay].max()), 2) if pay.any() else None
        summ["highest_wet_below_md"] = None
        # Density-neutron crossover (PHID > PHIN_ss by > 0.05) in pay: a light
        # hydrocarbon indicator. It does not say gas vs light oil.
        sep = (r["PHID"] - r["PHIN"]) > 0.05
        summ["dn_crossover_pay_frac"] = round(float(sep[pay].mean()), 2) if pay.any() else None
        if pay.any():
            below = (inp.depth > inp.depth[pay].max()) & (r["SW_SIM"] > 0.9) & (r["VSH"] < 0.3)
            summ["highest_wet_below_md"] = round(float(inp.depth[below].min()), 2) if below.any() else None
        if "rw_offset_well" in inp.notes:
            summ["rw_note"] = (f"from {inp.notes['rw_offset_well']} water leg, Arps "
                               f"{inp.notes['t_offset_c']} -> {inp.notes['t_here_c']} degC")
    return curves, summ


def cutoff_sweep(curves: pd.DataFrame, dz: np.ndarray, exclude: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    keep = ~exclude if exclude is not None else True
    v, p, s = curves.VSH.to_numpy(), curves.PHIE.to_numpy(), curves.SW_SIM.to_numpy()
    for vc, pc, sc in product(C.SWEEP["vsh"], C.SWEEP["phi"], C.SWEEP["sw"]):
        net, pay = petro.flags(v, p, s, vc, pc, sc)
        net, pay = net & keep, pay & keep
        rows.append(dict(well=curves.WELL.iloc[0], vsh_cut=vc, phi_cut=pc, sw_cut=sc,
                         net_md=float((net * dz).sum()), pay_md=float((pay * dz).sum())))
    return pd.DataFrame(rows)
