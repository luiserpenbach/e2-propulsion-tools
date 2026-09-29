"""Boiling-regime analysis of the N2O-cooled E2-REG-1 jacket (calculation note
'Nitrous oxide boiling regime').

    python run_boiling.py        # ~3 min -> out/boiling/

For each throttle point the jacket is marched with the same inlet state and
injector-side outlet pressure as run_regen.py, once per coolant-side model in
twophase.MODELS, and the regime indicators (reduced pressure, mass flux, boiling
number, Weber and confinement numbers, homogeneous Mach number, limiting liquid
superheat, Cooper nucleate-boiling limit, Hall-Mudawar CHF) are tabulated along
the channel.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

from h2cea import config as C
from h2cea import n2o
from h2cea.contour import e2_reg1_asbuilt, wall_stations
from h2cea.jacket import e2_reg1
from h2cea.operating_line import state
from h2cea.propellants import n2o_card
from h2cea.regen import march_to_outlet_pressure, chf_hall_mudawar
from h2cea import twophase as TP
from h2cea import boiling_plots as BP
from h2cea.coolant import Coolant

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "boiling")
os.makedirs(OUT, exist_ok=True)
POINTS = {"100": 0.874, "81": 0.715, "74": 0.658, "50": 0.470}
F = "NitrousOxide"
PROPS = TP.Props(Coolant("n2o"))


def hem_sound_speed(p, h):
    s = PropsSI("S", "P", p, "H", h, F)
    dp = 2000.0
    r1 = PropsSI("D", "P", p - dp, "S", s, F)
    r2 = PropsSI("D", "P", p + dp, "S", s, F)
    return float(np.sqrt(2 * dp / (r2 - r1)))


def regime_table(res):
    a = res.arrays
    pc = PropsSI("pcrit", F)
    rows = []
    for i in range(len(a["x"]) - 1, -1, -1):
        p, xq = a["p"][i], a["xq"][i]
        rl = PropsSI("D", "P", p, "Q", 0, F); rv = PropsSI("D", "P", p, "Q", 1, F)
        hfg = PropsSI("H", "P", p, "Q", 1, F) - PropsSI("H", "P", p, "Q", 0, F)
        sg = PropsSI("I", "P", p, "Q", 0, F)
        G, D, qc = a["G"][i], a["Dh"][i], a["q_c"][i]
        c = hem_sound_speed(p, a["h"][i]) if 0 < xq < 1 else np.nan
        satd = {"h_v": hfg + 0.0, "h_l": 0.0, "rho_l": rl, "rho_v": rv, "sigma": sg}
        chf = chf_hall_mudawar(G, D, satd, max(xq, 0.0)) if xq <= 0.05 else np.nan
        w = a["w"][i]
        ph_pf = (w + 2 * res.jacket.h) / (2 * w + 2 * res.jacket.h)
        xdi = TP.x_dryout_incipience(G, D, PROPS.sat(p), qc, ph_pf) if 0 < xq < 1 else np.nan
        rows.append(dict(x_mm=a["x"][i] * 1e3, regime=a["regime"][i], p_bar=p / 1e5, pr=p / pc, G=G,
                         q_gas_MW=a["q"][i] / 1e6, q_c_MW=qc / 1e6, x_eq=xq, hfg_kJ=hfg / 1e3,
                         rho_ratio=rl / rv, sigma_mN=sg * 1e3, Bo=qc / (G * hfg),
                         We=G * G * D / (rl * sg), Co=np.sqrt(sg / (9.81 * (rl - rv))) / D,
                         v=a["v"][i], c_hem=c, Mach_hem=a["v"][i] / c if np.isfinite(c) else np.nan,
                         Tb_C=a["Tb"][i] - 273.15, Tsat_C=a["Tsat"][i] - 273.15, T_sl_C=a["T_sl"][i] - 273.15,
                         q_nb_max_MW=a["q_nb_max"][i] / 1e6, chf_hm_MW=chf / 1e6, x_di=xdi,
                         Twc_C=a["Twc"][i] - 273.15, Twg_C=a["Twg"][i] - 273.15))
    return pd.DataFrame(rows)


def main(models=TP.MODELS):
    t0 = time.time()
    contour = e2_reg1_asbuilt()
    wall = wall_stations(contour)
    jac = e2_reg1()
    At = np.pi * C.R_T ** 2
    liq = n2o_card(C.T_OX_NOM, 70.0)
    ops = {k: state(m, At, eps=C.EPS_E2, ox=liq) for k, m in POINTS.items()}
    h_tank = n2o.h_TP(C.T_OX_NOM, C.P_TANK)
    summary, profiles = [], {}
    for k, s in ops.items():
        _, p_out = n2o.jacket_pressures(s["mox"], s["pc"], POINTS["100"])
        for mdl in models:
            for fs in ((False, True) if mdl in ("miropolskii", "jackson") else (False,)):
                r = march_to_outlet_pressure(wall, s["pt"], s["pc"], jac, "n2o", s["mox"], p_out, h_tank,
                                             p_guess_bar=p_out + 15, case=f"n2o {k} % {mdl}",
                                             tp_model=mdl, film_subcooled=fs)
                tag = mdl + ("+sub" if fs else "")
                profiles[(k, tag)] = r
                a = r.arrays
                it = int(np.nanargmax(a["Twg"]))
                tp = a["regime"] == "two-phase"
                summary.append(dict(point=k, model=tag, Q_kW=r["Q"] / 1e3, p_in=r["p_in_bar"], dp=r["dp_bar"],
                                    T_out_C=r["T_out"] - 273.15, Twg_max_C=r["Twg_max"] - 273.15,
                                    x_Twg_max_mm=r["x_Twg_max"] * 1e3, Twc_max_C=r["Twc_max"] - 273.15,
                                    hc_tp_min_kW=float(np.nanmin(a["hc"][tp]) / 1e3) if tp.any() else np.nan,
                                    hc_tp_max_kW=float(np.nanmax(a["hc"][tp]) / 1e3) if tp.any() else np.nan,
                                    q_max_MW=r["q_max"] / 1e6, energy_err=r["energy_balance_err"]))
                pd.DataFrame(r.dataframe()).to_csv(os.path.join(OUT, f"n2o_{k}_{tag}.csv"), index=False)
                print(f"{k:>3} % {tag:<16} Twg_max {r['Twg_max']-273.15:7.1f} C  Twc_max {r['Twc_max']-273.15:7.1f} C  "
                      f"Q {r['Q']/1e3:6.1f} kW  ({time.time()-t0:.0f} s)", flush=True)
        regime_table(profiles[(k, "liquid_only")]).to_csv(os.path.join(OUT, f"regime_{k}.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    xt = contour["xt"] * 1e3
    for k in ("100", "50"):
        BP.wall_by_model(profiles, k, os.path.join(OUT, f"B0{1 if k == '100' else 2}_wall_by_model_{k}.png"), xt)
    tabs = {f"{k} %": pd.read_csv(os.path.join(OUT, f"regime_{k}.csv")) for k in ("100", "74", "50")}
    BP.regime_map(tabs, os.path.join(OUT, "B03_regime_map.png"), xt)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return profiles, pd.DataFrame(summary)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or TP.MODELS)
