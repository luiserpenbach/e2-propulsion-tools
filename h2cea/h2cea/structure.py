"""First-order stresses in the channel wall (handbook 11.2 to 11.4, calculation note 12).

hot wall      clamped beam over the channel:   sigma_b = dp/2 (w/t_w)^2
rib           tension from one channel pitch:  sigma_rib = dp (w + t_rib)/t_rib
closeout      hoop, ribs and hot wall ignored: sigma_h = p R/t_co
              bending between ribs:           sigma = p/2 (w_co/t_co)^2
thermal       fully restrained plate, linear gradient:
                                               sigma_th = E alpha dT_w / (2 (1 - nu))
"""
import numpy as np

from . import materials as mat
from .jacket import Jacket, channels


def pressure_stresses(r, j: Jacket, dp_bar, p_coolant_bar=None):
    """Stresses (Pa) at gas-side radius r for a coolant-to-chamber pressure
    difference dp_bar across the hot wall; closeout loaded by p_coolant_bar
    (defaults to dp_bar, i.e. chamber at ambient)."""
    g = channels(r, j)
    w = g["w"]
    dp = dp_bar * 1e5
    p = (p_coolant_bar if p_coolant_bar is not None else dp_bar) * 1e5
    R_co = r + j.t_wall + j.h
    w_co = 2 * np.pi * R_co / j.n - j.t_rib
    return {"hot_wall_bending": dp / 2 * (w / j.t_wall) ** 2,
            "rib_tension": dp * (w + j.t_rib) / j.t_rib,
            "closeout_hoop": p * R_co / j.t_closeout,
            "closeout_bending": p / 2 * (w_co / j.t_closeout) ** 2}


def thermal_stress(res):
    """Thermal gradient stress along the channel from a RegenResult (Pa)."""
    m = mat.get(res.jacket.material)
    a = res.arrays
    Tm = 0.5 * (a["Twg"] + a["Twc"])
    dT = a["Twg"] - a["Twc"]
    s = mat.E(m, Tm) * mat.alpha(m, Tm) * dT / (2 * (1 - m["nu"]))
    return {"sigma_th": s, "dT_wall": dT, "T_mean": Tm,
            "yield_hot_face": mat.yield_strength(m, a["Twg"]),
            "ratio_to_yield": s / mat.yield_strength(m, a["Twg"])}


def load_cases(r_cyl, r_throat, j: Jacket, cases):
    """cases: {name: (dp_bar across hot wall, coolant pressure bar)} -> table rows (MPa)."""
    rows = []
    for name, (dp, pcool) in cases.items():
        c = pressure_stresses(r_cyl, j, dp, pcool)
        t = pressure_stresses(r_throat, j, dp, pcool)
        rows.append({"case": name, "dp_bar": dp, "p_coolant_bar": pcool,
                     "hot_wall_cyl_MPa": c["hot_wall_bending"] / 1e6,
                     "hot_wall_throat_MPa": t["hot_wall_bending"] / 1e6,
                     "rib_cyl_MPa": c["rib_tension"] / 1e6,
                     "closeout_hoop_MPa": c["closeout_hoop"] / 1e6,
                     "closeout_bending_MPa": c["closeout_bending"] / 1e6})
    return rows
