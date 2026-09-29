"""Regenerative cooling analyses of E2-REG-1 (replaces the resa-v3 runs for E2).

    python run_regen.py             # water and N2O cases, figures, CSVs, results_regen.json (~2 min)
    python run_regen.py --viewer    # also 2D sections along the channel and the 3D viewer (+ ~6 min)

Everything is written into out/regen/. Inputs come from h2cea/config.py (engine,
jacket, water test conditions) and the operating line (operating_line.state with
the as-built throat and area ratio).

Cases
    water      E2-REG-1-A test configuration: 1.0 kg/s, 30 bar, 20 C, at 100 % and 50 %
    n2o        design case: full oxidiser flow through the jacket at 100/81/74/50 %,
               jacket inlet enthalpy = tank state (isenthalpic valves and venturi),
               jacket outlet pressure = oxidiser injector inlet (n2o.jacket_pressures)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from h2cea import config as C
from h2cea import n2o
from h2cea import regen_plots as RP
from h2cea import section2d as S2
from h2cea import structure as ST
from h2cea.contour import e2_reg1_asbuilt, wall_stations
from h2cea.coolant import h_from_Tp
from h2cea.jacket import e2_reg1, Jacket
from h2cea.operating_line import state
from h2cea.propellants import n2o_card
from h2cea.regen import march, march_to_outlet_pressure, checks, verify

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "regen")
os.makedirs(OUT, exist_ok=True)
LIQ = n2o_card(C.T_OX_NOM, 70.0)
POINTS = {"100": 0.874, "81": 0.715, "74": 0.658, "50": 0.470}   # oxidiser flow, kg/s
R = {}


def f(name):
    return os.path.join(OUT, name)


def log(*a):
    print(*a, flush=True)


def K2C(T):
    return float(T) - 273.15


def scalars(res):
    sc = {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
          for k, v in res.scalars.items()}
    sc["checks"] = {k: {"value": float(v[0]), "limit": float(v[1]), "ok": bool(v[2])}
                    for k, v in checks(res).items()}
    return sc


def main(viewer=False):
    t0 = time.time()
    contour = e2_reg1_asbuilt()
    wall = wall_stations(contour)
    jac = e2_reg1()
    At = np.pi * C.R_T ** 2
    xt = contour["xt"]
    ops = {k: state(m, At, eps=C.EPS_E2, ox=LIQ) for k, m in POINTS.items()}
    R["operating_points"] = {k: {"mox": s["mox"], "MR": s["MR"], "pc_bar": s["pc"], "F_N": s["F"],
                                 "Tc_K": s["Tc"]} for k, s in ops.items()}

    # ------------------------------------------------------------------ water
    h_w = h_from_Tp("water", C.WATER_T_IN, C.WATER_P_IN * 1e5)
    water = {}
    for k in ("50", "100"):
        s = ops[k]
        water[k] = march(wall, s["pt"], s["pc"], jac, "water", C.WATER_MDOT, C.WATER_P_IN, h_w,
                         case=f"water {k} %")
        water[k].dataframe().to_csv(f(f"water_{k}_profile.csv"), index=False)
    R["water"] = {k: scalars(v) for k, v in water.items()}
    R["verification_water_100"] = verify(lambda n: wall_stations(contour, n), ops["100"]["pt"], ops["100"]["pc"],
                                         jac, "water", C.WATER_MDOT, C.WATER_P_IN, h_w)
    log(f"water done ({time.time()-t0:.0f} s)")

    # 2D sections at throat and cylinder
    secs, sec_rows = {}, []
    for k in ("50", "100"):
        for lab, xx in (("throat", xt), ("cylinder", 0.040)):
            i = water[k].at_x(xx)
            for bl in (False, True):
                sec = S2.section_at(water[k], i, blocked=bl, d=0.02e-3)
                secs[(k, lab, bl)] = sec
                sm = S2.summary(sec)
                sec_rows.append({"case": f"water {k} %", "station": lab, "blocked": bl, "x_mm": water[k]["x"][i] * 1e3,
                                 "Twg_1D_C": K2C(water[k]["Twg"][i]), "Ts_max_C": K2C(sm["Ts_max"]),
                                 "Ts_over_channel_C": K2C(sm["Ts_channel1"]), "Ts_over_rib_C": K2C(sm["Ts_rib"]),
                                 "Ts_centre_C": K2C(sm["Ts_centre"]), "q_mean_MWm2": sm["q_mean"] / 1e6,
                                 "q_1D_MWm2": water[k]["q"][i] / 1e6, "T_closeout_max_C": K2C(sm["T_closeout_max"])})
    pd.DataFrame(sec_rows).to_csv(f("water_sections.csv"), index=False)
    R["water_sections"] = sec_rows
    RP.cross_sections([[secs[("50", "throat", False)], secs[("50", "cylinder", False)]],
                       [secs[("100", "throat", False)], secs[("100", "cylinder", False)]]],
                      [["Throat, 50 %", "Cylinder, 50 %"], ["Throat, 100 %", "Cylinder, 100 %"]],
                      f("R02_water_cross_sections.png"), npitch=1,
                      suptitle="Water-cooled: temperature in one channel pitch (true scale)")
    RP.cross_sections([[secs[("50", "throat", True)]], [secs[("50", "cylinder", True)]],
                       [secs[("100", "throat", True)]], [secs[("100", "cylinder", True)]]],
                      [["Throat, 50 %"], ["Cylinder, 50 %"], ["Throat, 100 %"], ["Cylinder, 100 %"]],
                      f("R03_water_blocked_channel.png"), npitch=3, vmax=1300, true_scale=False,
                      suptitle="Water-cooled: centre channel blocked (three pitches, stretched horizontally)")
    log(f"water sections done ({time.time()-t0:.0f} s)")

    # flow sweep and sensitivities
    sweep = []
    for k in ("50", "100"):
        s = ops[k]
        for m in np.round(np.arange(0.4, 1.61, 0.2), 2):
            r = march(wall, s["pt"], s["pc"], jac, "water", float(m), C.WATER_P_IN, h_w)
            sweep.append({"case": f"{k} %", "mdot": m, "Twg_max": r["Twg_max"], "margin_sat_min": r["margin_sat_min"],
                          "dp_bar": r["dp_bar"], "T_out_C": K2C(r["T_out"]), "Q_kW": r["Q"] / 1e3})
    pd.DataFrame(sweep).to_csv(f("water_flow_sweep.csv"), index=False)
    RP.flow_sweep(sweep, f("R04_water_flow_sweep.png"), title="water flow")
    sens = []
    variants = {"baseline": ({}, jac), "bartz +20 %": ({"bartz_factor": 1.2}, jac),
                "bartz -20 %": ({"bartz_factor": 0.8}, jac),
                "roughness 40 um": ({}, Jacket(roughness=40e-6)), "hot wall 0.6 mm": ({}, Jacket(t_wall=0.6e-3))}
    for k in ("50", "100"):
        s = ops[k]
        for name, (kw, jj) in variants.items():
            r = march(wall, s["pt"], s["pc"], jj, "water", C.WATER_MDOT, C.WATER_P_IN, h_w, **kw)
            i = r.at_x(xt)
            sens.append({"case": f"{k} %", "variant": name, "Twg_throat_C": K2C(r["Twg"][i]),
                         "Twc_throat_C": K2C(r["Twc"][i]), "dp_bar": r["dp_bar"], "Q_kW": r["Q"] / 1e3})
    pd.DataFrame(sens).to_csv(f("water_sensitivities.csv"), index=False)
    R["water_sensitivities"] = sens
    log(f"water sweeps done ({time.time()-t0:.0f} s)")

    # ------------------------------------------------------------------ N2O
    h_tank = n2o.h_TP(C.T_OX_NOM, C.P_TANK)
    nox = {}
    for k, s in ops.items():
        _, p_out = n2o.jacket_pressures(s["mox"], s["pc"], POINTS["100"])
        nox[k] = march_to_outlet_pressure(wall, s["pt"], s["pc"], jac, "n2o", s["mox"], p_out, h_tank,
                                          p_guess_bar=p_out + 15, case=f"n2o {k} %")
        nox[k].dataframe().to_csv(f(f"n2o_{k}_profile.csv"), index=False)
        log(f"n2o {k} % done ({time.time()-t0:.0f} s)")
    R["n2o"] = {k: scalars(v) for k, v in nox.items()}
    R["n2o_note"] = ("Two-phase: liquid-only Gnielinski (no boiling credit). Hall-Mudawar CHF only for x <= 0.05; "
                     "dryout CHF in the saturated region is not evaluated. N2O transport by CO2 corresponding states.")
    R["venturi_max_outlet_bar"] = C.VENTURI_RECOVERY * C.P_VENTURI_IN

    # ------------------------------------------------------------------ structure
    i_c, i_t = water["100"].at_x(0.040), water["100"].at_x(xt)
    r_c, r_t = water["100"]["r"][i_c], water["100"]["r"][i_t]
    load = ST.load_cases(r_c, r_t, jac, {
        "water on, before ignition (30 bar)": (C.WATER_P_IN, C.WATER_P_IN),
        "water proof 1.5 x 30 bar": (1.5 * C.WATER_P_IN, 1.5 * C.WATER_P_IN),
        "N2O jacket MEOP 115 bar (primed)": (115.0, 115.0),
        "N2O jacket proof 172.5 bar": (172.5, 172.5)})
    pd.DataFrame(load).to_csv(f("structure_pressure_cases.csv"), index=False)
    R["structure_pressure"] = load
    th = {}
    for name, res in list(("water " + k, v) for k, v in water.items()) + list(("n2o " + k, v) for k, v in nox.items()):
        t = ST.thermal_stress(res)
        i = int(np.nanargmax(t["sigma_th"]))
        th[name] = {"sigma_th_max_MPa": float(t["sigma_th"][i] / 1e6), "x_mm": float(res["x"][i] * 1e3),
                    "dT_wall_K": float(t["dT_wall"][i]), "ratio_to_yield": float(t["ratio_to_yield"][i])}
    R["structure_thermal"] = th

    # ------------------------------------------------------------------ figures
    sec_line = None
    if viewer:
        from h2cea.viewer import sections_along, write_viewer
        prog = lambda case, bl, k, n: (k % 30 == 0) and log(f"  sections {case} {'blocked' if bl else 'normal'} {k}/{n}")
        cases = [water["50"], water["100"], nox["100"], nox["50"]]
        alongs = [sections_along(r, d=0.05e-3, progress=prog) for r in cases]
        sec_line = {r.case: {"x": [r["x"][i] for i in a["idx"]], "nom": a["nom"]["Tsmax"], "blk": a["blk"]["Tsmax"]}
                    for r, a in zip(cases, alongs)}
        write_viewer(f("channel_viewer.html"), cases, alongs, {
            "title": "E2-REG-1-A cooling channel", "page_title": "REG-1-A Cooling Channel",
            "subtitle": ("Fifteen of the 50 channels from the injector face to the section you pick. "
                         "Water: 1.0 kg/s at 30 bar, 20 °C. N2O: full oxidiser flow from the tank state."),
            "labels": ["Water 50 %", "Water 100 %", "N2O 100 %", "N2O 50 %"],
            "subs": [f"pc {r['pc_bar']:.1f} bar, coolant {r['mdot']:.2f} kg/s" for r in cases],
            "titleblock": [["Article", "E2-REG-1-A, SN1/SN2"], ["Material", "IN718, LPBF"],
                           ["Channels", f"{jac.n}, counterflow"],
                           ["Hot wall / rib / height", f"{jac.t_wall*1e3:.2f} / {jac.t_rib*1e3:.2f} / {jac.h*1e3:.2f} mm"],
                           ["Closeout", f"{jac.t_closeout*1e3:.2f} mm (TBC)"], ["Model", "h2cea 1D march + 2D sections"]]})
        R["sections_along"] = {r.case: {"Ts_max_nom_C": K2C(max(a["nom"]["Tsmax"])),
                                        "Ts_max_blk_C": K2C(max(a["blk"]["Tsmax"]))} for r, a in zip(cases, alongs)}
        log(f"viewer done ({time.time()-t0:.0f} s)")
    RP.axial_profiles([water["50"], water["100"]], ["Water 1.0 kg/s, 50 %", "Water 1.0 kg/s, 100 %"],
                      f("R01_water_axial.png"), sections=sec_line,
                      suptitle="E2-REG-1-A, water-cooled test configuration")
    RP.axial_profiles([nox["100"], nox["50"]], ["N2O coolant, 100 %", "N2O coolant, 50 %"],
                      f("R05_n2o_axial.png"), sections=sec_line,
                      suptitle="E2-REG-1, nitrous oxide as coolant (design case)")
    RP.n2o_ph_paths([nox[k] for k in ("100", "81", "74", "50")], ["100 %", "81 %", "74 %", "50 %"],
                    f("R06_n2o_ph_path.png"))

    with open(f("results_regen.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
    log(f"all done in {time.time()-t0:.0f} s -> {OUT}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="2D sections along the channel and the 3D viewer")
    main(viewer=ap.parse_args().viewer)
