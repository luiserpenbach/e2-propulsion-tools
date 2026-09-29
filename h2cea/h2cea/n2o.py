"""Nitrous oxide as coolant: states, jacket pressure model, first CHF estimate.

These are first-order tools for the handbook (section 10). The coupled channel
solution with regime switching lives in RESA.
"""
import numpy as np
from CoolProp.CoolProp import PropsSI

from . import config as C

F = "NitrousOxide"
T_CRIT = PropsSI("Tcrit", F)
P_CRIT = PropsSI("pcrit", F) / 1e5


def psat(T):
    return PropsSI("P", "T", T, "Q", 0, F) / 1e5


def tsat(p_bar):
    return PropsSI("T", "P", p_bar * 1e5, "Q", 0, F)


def sat_props(p_bar):
    P = p_bar * 1e5
    return {"T": PropsSI("T", "P", P, "Q", 0, F),
            "rho_l": PropsSI("D", "P", P, "Q", 0, F),
            "rho_v": PropsSI("D", "P", P, "Q", 1, F),
            "h_l": PropsSI("H", "P", P, "Q", 0, F),
            "h_v": PropsSI("H", "P", P, "Q", 1, F),
            "sigma": PropsSI("I", "P", P, "Q", 0, F)}


def h_TP(T, p_bar):
    return PropsSI("H", "T", T, "P", p_bar * 1e5, F)


def state_hp(h, p_bar):
    P = p_bar * 1e5
    T = PropsSI("T", "H", h, "P", P, F)
    rho = PropsSI("D", "H", h, "P", P, F)
    out = {"T": T, "rho": rho, "x": None}
    if p_bar < P_CRIT:
        s = sat_props(p_bar)
        out["x"] = (h - s["h_l"]) / (s["h_v"] - s["h_l"])
    return out


def pseudo_critical_T(p_bar, T_lo=None, T_hi=400.0, n=2000):
    T_lo = T_lo or T_CRIT - 5
    Ts = np.linspace(T_lo, T_hi, n)
    cp = [PropsSI("C", "T", T, "P", p_bar * 1e5, F) for T in Ts]
    i = int(np.argmax(cp))
    return Ts[i], cp[i]


# ---------------------------------------------------------------- jacket pressure
def jacket_pressures(mox, pc_bar, mox_nom, pc_nom=C.PC_MAX,
                     p_inj_nom=C.P_INJ_IN_NOM, dp_jacket_nom=25.0):
    """First-order jacket inlet/outlet pressure at part load.

    Oxidiser injector (gas-like fluid): p_in^2 - pc^2 scales with mox^2 (isothermal
    compressible orifice, injector inlet temperature held constant).
    Jacket: dp scales with mox^2 (fixed geometry, fixed mean density).
    Both simplifications are stated in the handbook; RESA replaces them.
    """
    r = mox / mox_nom
    p_out = np.sqrt(pc_bar ** 2 + (p_inj_nom ** 2 - pc_nom ** 2) * r * r)
    p_in = p_out + dp_jacket_nom * r * r
    return p_in, p_out


# ---------------------------------------------------------------- CHF estimate
def chf_hall_mudawar(G, D, p_bar, h_local):
    """Hall and Mudawar (2000) subcooled/low-quality CHF, outlet-conditions form.

    Bo = C1 We^C2 (rho_f/rho_g)^C3 [1 - C4 (rho_f/rho_g)^C5 x]
    Developed for water (1-200 bar, D 0.25-15 mm, G 300-30 000 kg/m2 s). Written in
    dimensionless groups, so it is used here as a fluid-to-fluid ESTIMATE for N2O.
    Near the critical pressure (p/pc > 0.8) it is an extrapolation.
    """
    s = sat_props(p_bar)
    hfg = s["h_v"] - s["h_l"]
    x = (h_local - s["h_l"]) / hfg
    We = G * G * D / (s["rho_l"] * s["sigma"])
    rr = s["rho_l"] / s["rho_v"]
    Bo = 0.0722 * We ** -0.312 * rr ** -0.644 * (1 - 0.900 * rr ** 0.724 * x)
    return Bo * G * hfg, x
