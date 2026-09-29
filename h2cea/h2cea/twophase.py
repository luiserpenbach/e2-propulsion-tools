"""Boiling and near-critical heat transfer to nitrous oxide in the jacket.

The E2 jacket runs at 0.3 to 0.8 of the N2O critical pressure, with mass fluxes of
2 000 to 19 000 kg/(m2 s) and coolant-side heat fluxes of 1 to 22 MW/m2. No flow
boiling correlation has been validated in that range. This module provides the
pieces needed to identify the regime and to bracket the coolant-side heat transfer
coefficient with physically different models; `regen.march(..., tp_model=...)`
uses them in the coupled wall solution.

Regime limits
    T_superheat_limit   Lienhard (1976) limiting liquid superheat,
                        T_sl/Tc = 0.905 + 0.095 (Tsat/Tc)^8. A wall hotter than
                        about T_sl cannot stay wetted by liquid; the minimum film
                        boiling (rewetting) temperature is of this order.
    q_nb_max            highest heat flux nucleate boiling can carry before the
                        wall reaches T_sl, from the Cooper correlation.

Coolant-side heat transfer models (tp_model in regen.march)
    'liquid_only'   Gnielinski with the whole flow as liquid (h2cea default so far)
    'nucleate'      asymptotic sum of liquid-only convection and Cooper (1984)
                    nucleate boiling, sqrt(h_lo^2 + h_nb^2); ignores the superheat
                    limit, so it is an unreachable optimistic bound here
    'miropolskii'   film boiling, Miropolskii (1963), water 40-220 bar:
                    Nu_v = 0.023 [Re_v (x + rho_v/rho_l (1-x))]^0.8 Pr_vw^0.8 Y
    'groeneveld'    film boiling, Groeneveld (1973), tubes:
                    Nu_v = 1.09e-3 [Re_v (x + rho_v/rho_l (1-x))]^0.989 Pr_vw^1.41 Y^-1.15
                    with Y = 1 - 0.1 (rho_l/rho_v - 1)^0.4 (1 - x)^0.4
    'jackson'       near-critical, homogeneous fluid with a gas-like wall layer:
                    Jackson (2002) supercritical form with homogeneous bulk,
                    Nu_b = 0.0183 Re_b^0.82 Pr_bar^0.5 (rho_w/rho_b)^0.3,
                    Pr_bar from the integrated specific heat (h_w - h_b)/(T_w - T_b)

The film boiling and near-critical models need the wall temperature, so they are
evaluated inside the wall iteration. They are applied for 0 <= x < 1 (and, with
film_subcooled, also to subcooled liquid when the wall is above T_sl).
"""
import numpy as np
from CoolProp.CoolProp import PropsSI

M_N2O = 44.013


# --------------------------------------------------------------------------- limits
def T_superheat_limit(Tsat, Tc):
    """Lienhard (1976) limiting liquid superheat temperature (K)."""
    return Tc * (0.905 + 0.095 * (Tsat / Tc) ** 8)


def cooper_coeff(pr, M=M_N2O, Rp_um=1.0):
    """h_nb = C q^0.67 (Cooper 1984); Rp surface roughness in micrometres."""
    return 55.0 * pr ** (0.12 - 0.2 * np.log10(Rp_um)) * (-np.log10(pr)) ** -0.55 * M ** -0.5


def h_cooper(q, pr, **kw):
    return cooper_coeff(pr, **kw) * max(q, 1.0) ** 0.67


def q_nb_max(pr, dT_max, **kw):
    """Heat flux at which Cooper nucleate boiling needs a wall superheat dT_max."""
    return (cooper_coeff(pr, **kw) * max(dT_max, 0.0)) ** (1.0 / 0.33)


def y_factor(rr, x):
    """Groeneveld / Miropolskii Y; rr = rho_l / rho_v."""
    return 1.0 - 0.1 * (rr - 1.0) ** 0.4 * (1.0 - min(max(x, 0.0), 1.0)) ** 0.4


# --------------------------------------------------------------------------- properties
class Props:
    """Saturated and wall-state properties the film boiling models need."""

    def __init__(self, cool):
        self.cool = cool
        self._cache = {}

    def sat(self, p):
        key = round(p, -2)
        if key not in self._cache:
            s = self.cool.sat(p)
            f = self.cool.fluid
            hv = s["h_v"]
            v = self.cool.state(hv + 50.0, p)                     # just into the vapour
            l = self.cool.state(s["h_l"] - 50.0, p)               # just into the liquid
            s = dict(s, mu_v=v.mu, k_v=v.k, cp_v=v.cp, mu_l=l.mu, k_l=l.k, cp_l=l.cp,
                     pr=p / self.cool.pcrit, Tc=self.cool.Tcrit)
            s["T_sl"] = T_superheat_limit(s["T"], self.cool.Tcrit)
            self._cache[key] = s
        return self._cache[key]

    T_PROP_MAX = 780.0     # K, upper end of the N2O equation of state in CoolProp

    def at_T(self, T, p):
        """Coolant state at wall temperature T (vapour side if above saturation).
        Wall states above T_PROP_MAX use the properties at T_PROP_MAX."""
        h = PropsSI("H", "T", min(T, self.T_PROP_MAX), "P", p, self.cool.fluid)
        return self.cool.state(h, p)


# --------------------------------------------------------------------------- models
def h_miropolskii(G, D, x, sat, w):
    rr = sat["rho_l"] / sat["rho_v"]
    xe = min(max(x, 0.0), 1.0)
    Re = G * D / sat["mu_v"] * (xe + (1 - xe) / rr)
    Nu = 0.023 * Re ** 0.8 * w.Pr ** 0.8 * y_factor(rr, xe)
    return Nu * sat["k_v"] / D


def h_groeneveld(G, D, x, sat, w):
    rr = sat["rho_l"] / sat["rho_v"]
    xe = min(max(x, 0.0), 1.0)
    Re = G * D / sat["mu_v"] * (xe + (1 - xe) / rr)
    Nu = 1.09e-3 * Re ** 0.989 * w.Pr ** 1.41 * y_factor(rr, xe) ** -1.15
    return Nu * sat["k_v"] / D


def h_jackson_homogeneous(G, D, st, sat, w):
    xe = min(max(st.x, 0.0), 1.0)
    mu_b = 1.0 / (xe / sat["mu_v"] + (1 - xe) / sat["mu_l"])     # McAdams
    k_b = xe * sat["k_v"] + (1 - xe) * sat["k_l"]
    Re = G * D / mu_b
    cp_bar = max((w.h - st.h) / max(w.T - st.T, 1e-3), 1.0)
    Pr_bar = cp_bar * mu_b / k_b
    Nu = 0.0183 * Re ** 0.82 * Pr_bar ** 0.5 * (w.rho / st.rho) ** 0.3
    return Nu * k_b / D


def x_dryout_incipience(G, D, sat, q_H, PH_PF):
    """Kim and Mudawar (2013, Part I) dryout incipience quality for saturated flow
    boiling in mini/micro-channels (database: 13 fluids incl. CO2, P_R 0.005-0.78,
    G 29-2303 kg/m2s, Bo up to 44e-4). A negative value means the heat flux is
    too high for any wetted annular film, i.e. the boiling crisis occurs at once."""
    hfg = sat["h_v"] - sat["h_l"]
    Bo = q_H / (G * hfg)
    We = G * G * D / (sat["rho_l"] * sat["sigma"])
    Ca = sat["mu_l"] * G / (sat["rho_l"] * sat["sigma"])
    return (1.4 * We ** 0.03 * sat["pr"] ** 0.08
            - 15.0 * (Bo * PH_PF) ** 0.15 * Ca ** 0.35 * (sat["rho_v"] / sat["rho_l"]) ** 0.06)


def h_kim_mudawar(G, D, x, sat, q_H, PH_PF):
    """Kim and Mudawar (2013, Part II) pre-dryout saturated flow boiling coefficient
    (database: 18 fluids, P_R 0.005-0.69, G 19-1608 kg/m2s)."""
    hfg = sat["h_v"] - sat["h_l"]
    xe = min(max(x, 1e-4), 0.999)
    Bo = q_H / (G * hfg)
    We = G * G * D / (sat["rho_l"] * sat["sigma"])
    Re_f = G * (1 - xe) * D / sat["mu_l"]
    Pr_f = sat["cp_l"] * sat["mu_l"] / sat["k_l"]
    h_f = 0.023 * Re_f ** 0.8 * Pr_f ** 0.4 * sat["k_l"] / D
    Xtt = (sat["mu_l"] / sat["mu_v"]) ** 0.1 * ((1 - xe) / xe) ** 0.9 * (sat["rho_v"] / sat["rho_l"]) ** 0.5
    h_nb = 2345.0 * (Bo * PH_PF) ** 0.70 * sat["pr"] ** 0.38 * (1 - xe) ** -0.51 * h_f
    h_cb = (5.2 * (Bo * PH_PF) ** 0.08 * We ** -0.54 + 3.5 * (1 / Xtt) ** 0.94
            * (sat["rho_v"] / sat["rho_l"]) ** 0.25) * h_f
    return float(np.hypot(h_nb, h_cb))


MODELS = ("liquid_only", "nucleate", "miropolskii", "groeneveld", "jackson", "regime")


def coolant_htc(model, G, D, st, sat, props, Twc, q_c, h_lo, PH_PF=0.6):
    """Coolant-side coefficient (W/m2K, referred to the wetted surface) for one
    station; h_lo is the liquid-only Gnielinski value.

    'regime' is the regime-switching best estimate: Kim-Mudawar pre-dryout flow
    boiling while x is below the dryout incipience quality and the wall stays below
    the limiting liquid superheat, Miropolskii film boiling otherwise."""
    if model == "regime":
        if 0 <= st.x < x_dryout_incipience(G, D, sat, q_c, PH_PF):
            h = h_kim_mudawar(G, D, st.x, sat, q_c, PH_PF)
            if sat["T"] + q_c / h <= sat["T_sl"]:
                return h
        model = "miropolskii"
    if model == "liquid_only":
        return h_lo
    if model == "nucleate":
        return float(np.hypot(h_lo, h_cooper(q_c, sat["pr"])))
    w = props.at_T(max(Twc, sat["T"] + 0.5), st.p)
    if model == "miropolskii":
        return h_miropolskii(G, D, st.x, sat, w)
    if model == "groeneveld":
        return h_groeneveld(G, D, st.x, sat, w)
    if model == "jackson":
        return h_jackson_homogeneous(G, D, st, sat, w)
    raise ValueError(model)
