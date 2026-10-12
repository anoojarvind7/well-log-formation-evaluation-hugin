"""Petrophysical equations (steps 5-7). Pure numpy, broadcasting-friendly so the
Monte Carlo in step 9 can call them with parameter arrays."""
from __future__ import annotations

import numpy as np


# --- Step 5: shale volume ----------------------------------------------------
def gr_index(gr, gr_clean, gr_shale):
    """Linear GR index, clipped to [0, 1]."""
    return np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)


def larionov_older(igr):
    """Larionov (1969) correction for older (pre-Tertiary) consolidated rocks:
    Vsh = 0.33 * (2**(2*IGR) - 1). Always <= IGR on [0, 1]."""
    return 0.33 * (np.power(2.0, 2.0 * igr) - 1.0)


# --- Step 6: porosity --------------------------------------------------------
def density_porosity(rhob, rho_matrix=2.65, rho_fluid=1.0):
    return (rho_matrix - rhob) / (rho_matrix - rho_fluid)


def density_neutron_porosity(phid, phin):
    """Root-mean-square combination. Reduces the light-hydrocarbon effect
    (density reads high, neutron low) compared with an arithmetic mean."""
    return np.sqrt((np.square(phid) + np.square(phin)) / 2.0)


def shale_corrected(phi, vsh, phi_shale):
    """Remove the shale's apparent porosity from a log porosity."""
    return phi - vsh * phi_shale


def effective_porosity(phid, phin, vsh, phid_sh, phin_sh):
    """Shale-correct density and neutron porosity separately, then combine.
    Negative corrected values are set to 0 before combining."""
    d = np.clip(shale_corrected(phid, vsh, phid_sh), 0.0, None)
    n = np.clip(shale_corrected(phin, vsh, phin_sh), 0.0, None)
    return np.clip(density_neutron_porosity(d, n), 0.0, 0.45)


# --- Step 7: water saturation ------------------------------------------------
def archie_sw(rt, phi, rw, a=1.0, m=2.0, n=2.0):
    phi = np.clip(phi, 1e-4, None)
    sw = np.power(a * rw / (np.power(phi, m) * rt), 1.0 / n)
    return np.clip(sw, 0.0, 1.0)


def simandoux_sw(rt, phi, vsh, rw, rsh, a=1.0, m=2.0, n=2.0, iters=60):
    """Simandoux (1963) in the form
        1/Rt = phi^m Sw^n / (a Rw) + Vsh Sw / Rsh
    solved for Sw by bisection (left side is monotone in Sw), so any n works.
    Reduces to Archie when Vsh = 0."""
    rt, phi, vsh, rw, rsh, a, m, n = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (rt, phi, vsh, rw, rsh, a, m, n)))
    phi = np.clip(phi, 1e-4, None)
    k1 = np.power(phi, m) / (a * rw)
    k2 = vsh / rsh
    target = 1.0 / rt
    lo = np.zeros_like(rt)
    hi = np.ones_like(rt)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = k1 * np.power(mid, n) + k2 * mid - target
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    sw = 0.5 * (lo + hi)
    return np.where(np.isfinite(rt) & np.isfinite(phi), sw, np.nan)


def rwa(rt, phi, a=1.0, m=2.0):
    """Apparent water resistivity, Rt * phi^m / a (= Rw where Sw = 1)."""
    return rt * np.power(phi, m) / a


def arps_rw(rw1, t1_c, t2_c):
    """Arps (1953): move Rw from temperature t1 to t2 (degC)."""
    return rw1 * (t1_c + 21.5) / (t2_c + 21.5)


def formation_temperature(tvdss_m, seabed_c=4.0, grad_c_per_km=35.0, water_depth_m=0.0):
    return seabed_c + grad_c_per_km * np.clip(tvdss_m - water_depth_m, 0, None) / 1000.0


# --- Step 8: cutoffs ---------------------------------------------------------
def flags(vsh, phi, sw, vsh_cut, phi_cut, sw_cut):
    """Return (net reservoir, net pay) boolean arrays. NaN fails every test."""
    with np.errstate(invalid="ignore"):
        net = (vsh <= vsh_cut) & (phi >= phi_cut)
        pay = net & (sw <= sw_cut)
    return net, pay
