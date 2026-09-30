"""Design and test-point calculations for the igniter app (app.py).

Wraps the models in igniter_thermal_analysis.py (CEA, Bartz, energy budget,
heat soak), so the app and the scripts give the same numbers. Everything here
works on plain dicts with the keys of a config file.
"""
from __future__ import annotations

import math

import numpy as np

import igniter_config as IC
import igniter_thermal_analysis as T

G0 = 9.80665


def _setup(raw: dict):
    """Case, CEA objects and nozzle for one config dict. Sets the module
    globals of igniter_thermal_analysis (its functions read them)."""
    cfg = IC.from_dict(raw)
    T.apply_case(cfg)
    cea_b, cea_u = T.build_cea()
    return cfg, cea_b, cea_u


def pc_from_flow(cea_u, mdot_kg_s: float, of: float, eta: float, at_m2: float,
                 pc_guess: float = 20.0) -> float:
    """Chamber pressure (bar) for a mass flow through a throat: pc = mdot eta c*(pc) / At."""
    pc = pc_guess
    for _ in range(60):
        new = mdot_kg_s * cea_u.get_Cstar(Pc=pc, MR=of) * eta / at_m2 / 1e5
        if abs(new - pc) < 1e-4:
            return new
        pc = 0.6 * pc + 0.4 * new
    return pc


# ----------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------
def analyse(raw: dict, n_nozzle: int = 300, sweep: bool = True) -> dict:
    """Operating points at every O/F of the case (plus the nominal one), axial
    heat-flux profiles, heat soak per wall material, and an O/F sweep."""
    cfg, cea_b, cea_u = _setup(raw)
    nozzle = T.build_nozzle(n_pts=n_nozzle)
    ae_m2 = cfg.eps * cfg.at_m2
    ofs = sorted(set(cfg.of_points) | {cfg.of_nominal})
    points, profiles = [], {}
    for of in ofs:
        pc = T.calc_pc(cea_u, of, cfg.eta_cstar)
        gp = T.get_gas_props(cea_b, cea_u, of, pc)
        hf = T.compute_heat_flux(gp, nozzle)
        eb = T.energy_budget(gp, hf["P_wall"])
        i_t = int(np.argmin(np.abs(nozzle["x"] - nozzle["x_throat"])))
        q_thr = float(hf["q"][i_t])
        isp_vac = cfg.eta_cstar * gp.Isp_vac
        row = {
            "O/F": of, "nominal": abs(of - cfg.of_nominal) < 1e-9,
            "pc_bar": pc, "Tc_K": gp.Tc, "cstar_ideal_m_s": gp.cstar,
            "cstar_del_m_s": cfg.eta_cstar * gp.cstar,
            "isp_vac_s": isp_vac,
            "isp_amb_s": isp_vac - cfg.pa_bar * 1e5 * ae_m2 / (cfg.mdot_kg_s * G0),
            "mdot_ox_g_s": 1e3 * cfg.mdot_kg_s * of / (1 + of),
            "mdot_fuel_g_s": 1e3 * cfg.mdot_kg_s / (1 + of),
            "q_throat_MW_m2": q_thr / 1e6, "q_max_MW_m2": float(np.max(hf["q"])) / 1e6,
            "P_wall_kW": hf["P_wall"] / 1e3, "wall_loss_pct": 100 * eb["eta_wall"],
        }
        for name, mat in T.MATERIALS.items():
            row[f"t_limit_{name}_s"] = T.t_to_limit(q_thr, mat)
        points.append(row)
        profiles[of] = {"x_mm": nozzle["x"] * 1e3, "q_MW_m2": hf["q"] / 1e6,
                        "Taw_K": hf["Taw"], "hg_kW_m2K": hf["hg"] / 1e3}
    out = {
        "case": cfg,
        "points": points,
        "profiles": profiles,
        "contour": {"x_mm": nozzle["x"] * 1e3, "r_mm": nozzle["r"] * 1e3,
                    "x_throat_mm": nozzle["x_throat"] * 1e3},
        "t_slab_valid_s": {name: T.t_slab_valid(mat) for name, mat in T.MATERIALS.items()},
        "materials": {name: dict(mat) for name, mat in T.MATERIALS.items()},
    }
    if sweep:
        rows = []
        for of in np.linspace(1.0, 4.0, 31):
            try:
                pc = T.calc_pc(cea_u, float(of), cfg.eta_cstar)
                rows.append({"O/F": float(of), "pc_bar": pc,
                             "Tc_K": float(cea_u.get_Tcomb(Pc=pc, MR=float(of))),
                             "cstar_del_m_s": cfg.eta_cstar * float(cea_u.get_Cstar(Pc=pc, MR=float(of)))})
            except Exception:
                pass
        out["sweep"] = rows
    return out


def nominal_point(result: dict) -> dict:
    return next(p for p in result["points"] if p["nominal"])


def geometry_summary(cfg) -> dict:
    return {"Dt_mm": cfg.dt_mm, "Dc_mm": cfg.dc_mm, "De_mm": cfg.de_mm,
            "contraction_ratio": cfg.eps_c, "L_star_m": cfg.lstar_m,
            "L_cyl_mm": cfg.lcyl_mm, "L_conv_mm": cfg.lconv_mm, "L_div_mm": cfg.ldiv_mm,
            "L_total_mm": cfg.lcyl_mm + cfg.lconv_mm + cfg.ldiv_mm,
            "L_cyl_over_Dc": cfg.lcyl_mm / cfg.dc_mm, "V_chamber_cm3": cfg.vc_cm3}


# ----------------------------------------------------------------------------
# Test points
# ----------------------------------------------------------------------------
TEST_COLUMNS = ["test_id", "date", "mdot_ox_g_s", "mdot_fuel_g_s", "pc_bar", "thrust_N",
                "ignition", "notes"]


def analyse_test_points(raw: dict, rows: list[dict], dt_mm: float | None = None,
                        pc_gauge: bool = False) -> list[dict]:
    """Measured c* and c* efficiency for each test point, and the model's
    chamber pressure at the measured flows with the design eta_c*.

    rows: dicts with mdot_ox_g_s, mdot_fuel_g_s, pc_bar and optionally thrust_N.
    dt_mm: measured throat diameter (default: the config's).
    pc_gauge: chamber pressure is gauge; the config's ambient pressure is added.
    """
    cfg, _cea_b, cea_u = _setup(raw)
    dt = dt_mm or cfg.dt_mm
    at = math.pi / 4 * (dt * 1e-3) ** 2
    out = []
    for r in rows:
        try:
            mo, mf, pc = float(r["mdot_ox_g_s"]) / 1e3, float(r["mdot_fuel_g_s"]) / 1e3, float(r["pc_bar"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (mo > 0 and mf > 0 and pc > 0) or any(map(math.isnan, (mo, mf, pc))):
            continue
        if pc_gauge:
            pc += cfg.pa_bar
        md, of = mo + mf, mo / mf
        cs_meas = pc * 1e5 * at / md
        cs_ideal = float(cea_u.get_Cstar(Pc=pc, MR=of))
        pc_pred = pc_from_flow(cea_u, md, of, cfg.eta_cstar, at, pc_guess=pc)
        res = {"test_id": r.get("test_id"), "O/F": of, "mdot_g_s": md * 1e3, "pc_abs_bar": pc,
               "cstar_meas_m_s": cs_meas, "cstar_ideal_m_s": cs_ideal,
               "eta_cstar_meas": cs_meas / cs_ideal, "eta_cstar_design": cfg.eta_cstar,
               "pc_model_bar": pc_pred, "pc_error_pct": 100 * (pc / pc_pred - 1),
               "Tc_ideal_K": float(cea_u.get_Tcomb(Pc=pc, MR=of))}
        F = r.get("thrust_N")
        try:
            F = float(F)
        except (TypeError, ValueError):
            F = float("nan")
        if F == F and F > 0:
            res["Cf_meas"] = F / (pc * 1e5 * at)
            res["isp_meas_s"] = F / (md * G0)
        out.append(res)
    return out


def of_line(raw: dict, eta: float, of_min: float = 1.0, of_max: float = 4.0, n: int = 31) -> list[dict]:
    """Model c* along O/F at the config's chamber pressure level, for plots."""
    cfg, _cea_b, cea_u = _setup(raw)
    pc = T.calc_pc(cea_u, cfg.of_nominal, cfg.eta_cstar)
    return [{"O/F": float(of), "cstar_ideal_m_s": float(cea_u.get_Cstar(Pc=pc, MR=float(of))),
             "cstar_model_m_s": eta * float(cea_u.get_Cstar(Pc=pc, MR=float(of)))}
            for of in np.linspace(of_min, of_max, n)]
