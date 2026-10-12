"""Step 11 figures (plus the QC/diagnostic plots used in steps 3, 4, 7, 8, 9, 10)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import config as C  # noqa: E402
from .lithology import CHART_LINES, CLASSES  # noqa: E402

# Validated reference palette (categorical slots, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LITHO_COLORS = {"Clean sand": YELLOW, "Shaly sand": ORANGE, "Shale": "#8a8984",
                "Carbonate": BLUE, "Coal": "#2b2b2b", None: "#ffffff"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 8.5, "axes.titlesize": 9.5, "lines.linewidth": 1.1, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _slug(well: str) -> str:
    return well.replace("/", "_").replace(" ", "")


def _save(fig, name: str):
    C.FIGS.mkdir(parents=True, exist_ok=True)
    p = C.FIGS / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def _tops(ax, tops: pd.DataFrame, label: bool = False):
    for _, t in tops.iterrows():
        ax.axhline(t.top_md, color=INK2, lw=0.6, ls=":")
        if label:
            ax.text(ax.get_xlim()[0], t.top_md, " " + t.FORMATION.replace(" Fm.", ""), va="bottom",
                    ha="left", fontsize=7, color=INK2)


def _strip(ax, depth, classes, title):
    cols = [LITHO_COLORS.get(c, "#ffffff") for c in classes]
    step = np.median(np.diff(depth))
    ax.barh(depth, 1, height=step * 1.02, color=cols, align="center", lw=0)
    ax.set_xticks([])
    ax.set_title(title, fontsize=8)
    ax.grid(False)


def tops_check(w: pd.DataFrame, well: str, tops: pd.DataFrame, chk: dict):
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 7), sharey=True)
    axs[0].plot(w.GR, w.DEPTH_MD, color=GREEN)
    axs[0].set_xlabel("GR (gAPI)")
    axs[0].axhline(chk["gr_break_md"], color=RED, lw=1.2, label=f"GR break ({chk['offset_m']:+.1f} m)")
    axs[0].legend(loc="lower right", fontsize=7)
    a = axs[1]
    a.plot(w.RHOB, w.DEPTH_MD, color=RED, label="RHOB")
    a.set_xlim(1.95, 2.95)
    a.set_xlabel("RHOB (g/cc)")
    b = a.twiny()
    b.plot(w.NPHI, w.DEPTH_MD, color=BLUE, ls="--", label="NPHI")
    b.set_xlim(0.45, -0.15)
    b.set_xlabel("NPHI (v/v, LS)")
    b.grid(False)
    for ax in axs:
        _tops(ax, tops)
    _tops(axs[0], tops, label=True)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("MD (m)")
    fig.suptitle(f"{well}: FORCE/NPD tops vs logs", fontsize=10)
    return _save(fig, f"tops_{_slug(well)}.png")


def dn_crossplot(df: pd.DataFrame, well: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(df.NPHI, df.RHOB, c=df.GR, s=9, cmap="viridis", vmin=20, vmax=120, lw=0)
    for name, (x, y) in CHART_LINES.items():
        ax.plot(x, y, color=INK2, lw=1)
        ax.text(x[-1], y[-1], " " + name, fontsize=7, color=INK2, va="center")
    ax.set_xlim(-0.05, 0.6)
    ax.set_ylim(2.95, 1.7)
    ax.set_xlabel("NPHI (v/v, limestone units)")
    ax.set_ylabel("RHOB (g/cc)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("GR (gAPI)")
    ax.set_title(f"{well}: Hugin density-neutron (chart lines approximate)")
    return _save(fig, f"dn_crossplot_{_slug(well)}.png")


def pickett(inp, curves: pd.DataFrame, rw: float, m: float, n: float):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(np.clip(curves.RT, 0.05, None), curves.PHIT, c=curves.VSH, s=9, cmap="cividis_r",
                    vmin=0, vmax=1, lw=0)
    wl_phi = np.sqrt(((C.RHO_MATRIX - inp.wl_rhob) / (C.RHO_MATRIX - C.RHO_FLUID)) ** 2 / 2
                     + (inp.wl_nphi + C.NPHI_SS_SHIFT) ** 2 / 2)
    ax.scatter(inp.wl_rt, wl_phi, s=16, facecolors="none", edgecolors=RED, lw=0.8, label="Rw points")
    phi = np.logspace(-1.3, -0.35, 20)
    for sw, ls in [(1.0, "-"), (0.5, "--"), (0.25, ":")]:
        ax.plot(rw / (phi ** m * sw ** n), phi, color=INK, ls=ls, lw=1, label=f"Sw={sw:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 200)
    ax.set_ylim(0.04, 0.45)
    ax.set_xlabel("Rt (ohm.m)")
    ax.set_ylabel("PHIT (v/v)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Vsh (Larionov)")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title(f"{inp.well}: Pickett plot, Rw={rw:.3f} ohm.m, m={m:g}, n={n:g}")
    return _save(fig, f"pickett_{_slug(inp.well)}.png")


def composite(w: pd.DataFrame, curves: pd.DataFrame, well: str, tops: pd.DataFrame, mine: np.ndarray):
    fig, axs = plt.subplots(1, 7, figsize=(13, 9), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.4, 1.4, 0.35, 0.35, 0.35], wspace=0.08))
    d = w.DEPTH_MD
    a = axs[0]
    a.plot(w.GR, d, color=GREEN, label="GR")
    a.set_xlim(0, 200)
    a.set_xlabel("GR (gAPI)", color=GREEN)
    b = a.twiny()
    b.plot(w.CALI, d, color=INK2, lw=0.8, label="CALI")
    b.plot(w.BS_USED, d, color=INK2, lw=0.8, ls="--", label="Bit")
    b.set_xlim(6, 16)
    b.set_xlabel("CALI / bit (in)", color=INK2)
    b.grid(False)
    a = axs[1]
    a.semilogx(w.RDEP, d, color=RED, label="RDEP")
    a.semilogx(w.RMED, d, color=VIOLET, lw=0.8, label="RMED")
    a.set_xlim(0.2, 200)
    a.set_xlabel("Resistivity (ohm.m)")
    a.legend(loc="upper right", fontsize=6.5)
    a = axs[2]
    a.plot(w.RHOB, d, color=RED)
    a.set_xlim(1.95, 2.95)
    a.set_xlabel("RHOB (g/cc)", color=RED)
    b = a.twiny()
    b.plot(w.NPHI, d, color=BLUE, ls="--")
    b.set_xlim(0.45, -0.15)
    b.set_xlabel("NPHI (v/v)", color=BLUE)
    b.grid(False)
    a = axs[3]
    cd = curves.DEPTH_MD
    a.fill_betweenx(cd, 0, curves.VSH, color="#8a8984", alpha=0.45, lw=0, label="Vsh")
    a.plot(curves.PHIE, cd, color=BLUE, label="PHIE")
    a.plot(curves.SW_SIM, cd, color=AQUA, label="Sw (Simandoux)")
    a.plot(curves.SW_ARCHIE, cd, color=AQUA, lw=0.7, ls=":", label="Sw (Archie)")
    a.set_xlim(0, 1)
    a.set_xlabel("Vsh / PHIE / Sw (v/v)")
    a.legend(loc="lower right", fontsize=6.5)
    a = axs[4]
    step = np.median(np.diff(cd))
    a.barh(cd[curves.NET.astype(bool)], 1, height=step, color=MAGENTA, lw=0)
    a.barh(cd[curves.PAY.astype(bool)], 1, height=step, color=RED, lw=0)
    a.set_xticks([])
    a.set_title("Net / pay", fontsize=8)
    a.grid(False)
    _strip(axs[5], curves.DEPTH_MD.to_numpy(), mine, "This\nwork")
    _strip(axs[6], d.to_numpy(), w.LITHO_CLASS.to_numpy(), "FORCE")
    for ax in axs[:4]:
        _tops(ax, tops)
    _tops(axs[0], tops, label=True)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("MD (m)")
    handles = [Patch(color=LITHO_COLORS[c], label=c) for c in CLASSES] + [
        Patch(color=MAGENTA, label="Net reservoir"), Patch(color=RED, label="Net pay")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{well}: Hugin Fm composite (FORCE 2020 logs; interpretation this work)", fontsize=11, y=0.995)
    return _save(fig, f"composite_{_slug(well)}.png")


def correlation_panel(items: list[tuple[str, pd.DataFrame, pd.DataFrame]]):
    """GR and PHIE/pay per well, hung on the Hugin top (depth = MD below top)."""
    n = len(items)
    fig, axs = plt.subplots(1, 2 * n, figsize=(2.2 * n * 1.6, 7.5), sharey=True,
                            gridspec_kw=dict(width_ratios=[1, 0.8] * n, wspace=0.12))
    for i, (well, w, curves) in enumerate(items):
        top = w.loc[w.FORMATION == C.RESERVOIR, "DEPTH_MD"].min()
        base = w.loc[w.FORMATION == C.RESERVOIR, "DEPTH_MD"].max()
        a, b = axs[2 * i], axs[2 * i + 1]
        a.plot(w.GR, w.DEPTH_MD - top, color=GREEN)
        a.set_xlim(0, 160)
        a.set_xticks([0, 100])
        a.set_title(well, fontsize=9, loc="left")
        a.set_xlabel("GR")
        rel = curves.DEPTH_MD - top
        b.fill_betweenx(rel, 0, curves.PHIE.where(curves.PAY.astype(bool)), color=RED, lw=0, alpha=0.8)
        b.plot(curves.PHIE, rel, color=BLUE, lw=0.9)
        b.set_xlim(0.35, 0)
        b.set_xticks([0.3, 0.1])
        b.set_xlabel("PHIE")
        for ax in (a, b):
            ax.axhline(0, color=INK, lw=1.2)
            ax.axhline(base - top, color=INK2, lw=0.8, ls="--")
    axs[0].set_ylim(80, -25)
    axs[0].set_ylabel("m MD below top Hugin (datum)")
    fig.legend(handles=[Patch(color=RED, label="PHIE in net pay"),
                        plt.Line2D([], [], color=INK, label="Top Hugin (datum)"),
                        plt.Line2D([], [], color=INK2, ls="--", label="Base Hugin")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Hugin Fm correlation panel, hung on top Hugin", fontsize=11)
    return _save(fig, "correlation_panel.png")


def sweep_plot(sweep: pd.DataFrame):
    base = C.CUTOFFS
    wells = list(dict.fromkeys(sweep.well))
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (key, col) in zip(axs, [("vsh", "vsh_cut"), ("phi", "phi_cut"), ("sw", "sw_cut")]):
        others = {k: v for k, v in [("vsh_cut", base["vsh"]), ("phi_cut", base["phi"]), ("sw_cut", base["sw"])]
                  if k != col}
        for wi, well in enumerate(wells):
            s = sweep[(sweep.well == well)]
            for k, v in others.items():
                s = s[np.isclose(s[k], v)]
            ax.plot(s[col], s.pay_md, marker="o", ms=4, color=colors[wi], label=well)
        ax.axvline(base[key], color=INK2, ls=":", lw=1)
        ax.set_xlabel(f"{key} cutoff")
    axs[0].set_ylabel("Net pay (m MD)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Cutoff sensitivity: one cutoff varied, others at base case (dotted)", fontsize=10)
    return _save(fig, "cutoff_sweep.png")


def mc_plot(draws: pd.DataFrame):
    wells = list(dict.fromkeys(draws.well))
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
    metrics = [("phie_net", "Avg PHIE in net (v/v)"), ("sw_pay", "Avg Sw in pay (v/v)"),
               ("pay_md", "Net pay (m MD)")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (k, lab) in zip(axs, metrics):
        for wi, well in enumerate(wells):
            x = np.sort(draws.loc[draws.well == well, k].dropna().to_numpy())
            if len(x) == 0:
                continue
            ax.plot(x, 1 - np.arange(len(x)) / len(x), color=colors[wi], label=well, lw=1.4)
        for p in (0.9, 0.5, 0.1):
            ax.axhline(p, color=GRID, lw=0.8)
        ax.set_xlabel(lab)
    axs[0].set_ylabel("Probability of exceeding")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Monte Carlo ({C.MC['n_draws']} draws per well): exceedance curves; "
                 "P90/P50/P10 at the grey lines", fontsize=10)
    return _save(fig, "monte_carlo.png")


def core_plot(matched: pd.DataFrame):
    wells = list(dict.fromkeys(matched.well))
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for wi, well in enumerate(wells):
        m = matched[matched.well == well]
        ax.scatter(m.phi_core, m.phit_log, s=12, color=colors[wi % 4], label=well, lw=0)
    ax.plot([0, 0.4], [0, 0.4], color=INK, lw=1)
    ax.fill_between([0, 0.4], [-0.03, 0.37], [0.03, 0.43], color=GRID, alpha=0.6, lw=0, label="+-0.03")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("Core plug porosity (v/v)")
    ax.set_ylabel("Log PHIT (v/v), depth-shifted")
    ax.legend(fontsize=7)
    ax.set_title("Log vs core porosity, Hugin Fm")
    return _save(fig, "core_vs_log.png")
