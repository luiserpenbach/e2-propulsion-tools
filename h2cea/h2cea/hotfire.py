"""Hot-fire test points of the E2 thrust chamber compared with the h2cea models.

Used by the E2 Test Explorer (test_explorer.py), and kept free of any UI so the same
numbers can be produced from a script or ported to hopper-test-lab later.

One steady-state test point (window means of chamber pressure, flows, thrust and,
if measured, injector and water-jacket data) gives

    measured      O/F, c* = pc At / mdot, Cf = F / (pc At), Isp
    CEA ideal     at the measured pc and O/F, liquid N2O card at the measured oxidiser
                  temperature (same thermochemistry as the engine table)
    efficiencies  eta_c* = c*_meas / c*_ideal, eta_Cf,vac = (F + pa Ae) / (pc At Cf_vac),
                  the same split as baseline eta_cstar / eta_cf_vac
    model         the baseline engine model at the measured flows (operating_line.pc_at
                  with the assumed efficiencies): predicted pc, thrust and Isp
    nozzle        exit pressure, Summerfield check (same criterion as the engine table)
                  and the Schmucker separation point along the contour
    injector      pressure drop, stiffness dp/pc and effective CdA per circuit
    water jacket  calorimetric heat Q = mdot_w (h_out - h_in), the regen.march heat at
                  the measured operating point, and the Bartz factor that matches Q

Units are part of every key name. Pressures are absolute in the results; inputs can be
gauge (`pressures_gauge`), in which case the ambient pressure is added.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from . import config as C
from .cea_si import point
from .contour import e2_contour, e2_reg1_asbuilt
from .gasdyn import mach_from_area, p_ratio, schmucker_psep_over_pa
from .operating_line import pc_at, size_throat
from .propellants import ethanol_card, n2o_card

F_RATED = C.F_MAX
SUMMERFIELD = 0.4          # pe / pa below which the flow separates (as in run_engine_table.py)
STIFFNESS_MIN = 0.15       # dp / pc, injector stiffness rule of thumb against chugging
MR_RANGE = (1.0, 8.0)      # O/F range accepted for the CEA lookups


# ----------------------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Hardware:
    key: str
    label: str
    throat_d_mm: float
    eps: float
    regen_model: bool      # the E2-REG-1 jacket model (regen.march) applies

    @property
    def At(self):
        return math.pi * (self.throat_d_mm * 1e-3) ** 2 / 4

    def contour(self, throat_d_mm=None):
        Rt = 0.5e-3 * (throat_d_mm or self.throat_d_mm)
        if self.key == "E2-REG-1":
            return e2_reg1_asbuilt() if throat_d_mm is None else \
                e2_contour(eps=C.EPS_E2, Rt=Rt, th_n=C.THETA_N_E2, th_e=C.THETA_E_E2, L_bell=C.L_BELL_E2)
        return e2_contour(eps=self.eps, Rt=Rt)


@lru_cache(maxsize=None)
def _flight_throat_d_mm():
    _, At = size_throat(ox=n2o_card(C.T_OX_NOM, C.P_TANK), fuel=ethanol_card(C.ETHANOL_WT))
    return round(2e3 * math.sqrt(At / math.pi), 3)


@lru_cache(maxsize=None)
def hardware_presets():
    return {
        "E2-REG-1": Hardware("E2-REG-1", "E2-REG-1 as built (eps 2.99, water / N2O jacket)",
                             round(2e3 * C.R_T, 3), round(C.EPS_E2, 3), True),
        "E2 flight": Hardware("E2 flight", f"E2 flight baseline (eps {C.EPS:g})",
                              _flight_throat_d_mm(), C.EPS, False),
    }


# ----------------------------------------------------------------------------------------
# Input and result fields (one place for the form, the files and the tables)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Field:
    key: str
    label: str
    unit: str
    group: str
    default: float | None = None
    help: str = ""
    step: float = 0.01
    fmt: str = "{:.2f}"


INPUT_FIELDS = [
    Field("pc_bar", "Chamber pressure", "bar", "op", None, "Steady-state mean.", 0.1),
    Field("ox_flow_kg_s", "Oxidiser flow", "kg/s", "op", None, "N2O mass flow, steady-state mean.", 0.005, "{:.3f}"),
    Field("fuel_flow_kg_s", "Fuel flow", "kg/s", "op", C.MDOT_FUEL, "Ethanol mass flow (venturi design value as default).", 0.005, "{:.3f}"),
    Field("thrust_N", "Thrust", "N", "op", None, "Load cell, steady-state mean. Optional.", 5.0, "{:.0f}"),
    Field("duration_s", "Burn duration", "s", "op", None, "Optional.", 0.1, "{:.1f}"),
    Field("p_amb_bar", "Ambient pressure", "bar", "op", C.P_AMB, "", 0.001, "{:.3f}"),
    Field("ox_temp_C", "N2O temperature", "°C", "inj", C.T_OX_NOM - 273.15, "Liquid N2O upstream of the engine (run tank or feed line); sets the CEA reactant enthalpy and the N2O density.", 0.5, "{:.1f}"),
    Field("p_ox_inj_bar", "Ox injector inlet", "bar", "inj", None, "Manifold pressure. Optional.", 0.1),
    Field("p_fuel_inj_bar", "Fuel injector inlet", "bar", "inj", None, "Manifold pressure. Optional.", 0.1),
    Field("throat_d_mm", "Throat diameter", "mm", "inj", None, "Measured throat (post-test erosion). Empty = hardware nominal.", 0.01, "{:.2f}"),
    Field("water_flow_kg_s", "Water flow", "kg/s", "water", None, "Jacket coolant flow.", 0.01, "{:.3f}"),
    Field("water_T_in_C", "Water inlet temperature", "°C", "water", None, "", 0.1, "{:.1f}"),
    Field("water_T_out_C", "Water outlet temperature", "°C", "water", None, "", 0.1, "{:.1f}"),
    Field("water_p_in_bar", "Water inlet pressure", "bar", "water", None, "", 0.1),
    Field("water_p_out_bar", "Water outlet pressure", "bar", "water", None, "Optional; gives the jacket pressure drop.", 0.1),
]
INPUTS = {f.key: f for f in INPUT_FIELDS}
REQUIRED = ("pc_bar", "ox_flow_kg_s", "fuel_flow_kg_s")
PRESSURE_KEYS = ("pc_bar", "p_ox_inj_bar", "p_fuel_inj_bar", "water_p_in_bar", "water_p_out_bar")

OUTCOMES = {"pass": "Nominal", "anomaly": "Off-nominal", "fail": "Failed", "void": "Invalid"}   # as hopper-test-lab

# key: (label, unit, format) for tables and plots
RESULT_LABELS = {
    "OF": ("O/F", "-", "{:.2f}"),
    "mdot_kg_s": ("Total flow", "kg/s", "{:.3f}"),
    "pc_abs_bar": ("pc measured", "bar", "{:.2f}"),
    "pc_model_bar": ("pc model", "bar", "{:.2f}"),
    "pc_dev_pct": ("pc vs model", "%", "{:+.1f}"),
    "cstar_meas_m_s": ("c* measured", "m/s", "{:.0f}"),
    "cstar_ideal_m_s": ("c* CEA", "m/s", "{:.0f}"),
    "eta_cstar": ("η c*", "-", "{:.3f}"),
    "Tc_K": ("Tc CEA", "K", "{:.0f}"),
    "thrust_N": ("Thrust measured", "N", "{:.0f}"),
    "thrust_model_N": ("Thrust model", "N", "{:.0f}"),
    "thrust_dev_pct": ("Thrust vs model", "%", "{:+.1f}"),
    "throttle_pct": ("Throttle", "% of 2250 N", "{:.0f}"),
    "cf_meas": ("Cf measured", "-", "{:.3f}"),
    "cf_ideal": ("Cf ideal", "-", "{:.3f}"),
    "eta_cf_vac": ("η Cf,vac", "-", "{:.3f}"),
    "isp_meas_s": ("Isp measured", "s", "{:.1f}"),
    "isp_ideal_s": ("Isp ideal", "s", "{:.1f}"),
    "isp_model_s": ("Isp model", "s", "{:.1f}"),
    "eta_isp": ("η Isp", "-", "{:.3f}"),
    "pe_bar": ("Exit pressure", "bar", "{:.3f}"),
    "pe_over_pa": ("pe / pa", "-", "{:.2f}"),
    "x_sep_mm": ("Separation at x", "mm", "{:.1f}"),
    "dp_ox_bar": ("Ox injector dp", "bar", "{:.2f}"),
    "stiff_ox_pct": ("Ox stiffness", "% pc", "{:.0f}"),
    "CdA_ox_mm2": ("Ox CdA", "mm²", "{:.2f}"),
    "dp_fuel_bar": ("Fuel injector dp", "bar", "{:.2f}"),
    "stiff_fuel_pct": ("Fuel stiffness", "% pc", "{:.0f}"),
    "CdA_fuel_mm2": ("Fuel CdA", "mm²", "{:.2f}"),
    "Q_meas_kW": ("Jacket heat measured", "kW", "{:.1f}"),
    "Q_model_kW": ("Jacket heat model", "kW", "{:.1f}"),
    "bartz_fit": ("Bartz factor (fit)", "-", "{:.2f}"),
    "Twg_max_fit_K": ("Hot wall max (fit)", "K", "{:.0f}"),
    "dp_jacket_meas_bar": ("Jacket dp measured", "bar", "{:.2f}"),
    "dp_jacket_model_bar": ("Jacket dp model", "bar", "{:.2f}"),
}


# ----------------------------------------------------------------------------------------
# Test points
# ----------------------------------------------------------------------------------------
def new_test(**kw) -> dict:
    """A test record: metadata plus the input values (None = not measured)."""
    t = {"test_id": "", "date": str(dt.date.today()), "hardware": "E2-REG-1", "outcome": "pass",
         "pressures_gauge": False, "notes": "", "source": "",
         "inputs": {f.key: f.default for f in INPUT_FIELDS}}
    for k, v in kw.items():
        if k in INPUTS:
            t["inputs"][k] = v
        else:
            t[k] = v
    return t


def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _ox_card(T_K, p_bar):
    # rounded so that a session does not register hundreds of RocketCEA cards
    return n2o_card(round(T_K * 2) / 2, float(max(5, round(p_bar / 5) * 5)))


def _rho_n2o(T_K, p_Pa):
    psat = PropsSI("P", "T", T_K, "Q", 0, "NitrousOxide")
    if p_Pa > psat:
        return PropsSI("D", "T", T_K, "P", p_Pa, "NitrousOxide"), True
    return PropsSI("D", "T", T_K, "Q", 0, "NitrousOxide"), False


def gamma_matching_exit(pt):
    """Constant gamma that gives the CEA (shifting) exit pressure at the CEA area ratio,
    for the wall-pressure distribution along the nozzle."""
    target = pt.pe / pt.pc

    def f(g):
        return p_ratio(mach_from_area(pt.eps, g, True), g) - target
    try:
        return brentq(f, 1.05, 1.45)
    except ValueError:
        return pt.gam_fr_c


def nozzle_wall(contour, pt, pa_bar):
    """Wall pressure along the divergent part and the Schmucker separation point.

    Returns x, r (mm), p_wall and p_sep (bar), and x_sep_mm (None if attached)."""
    g = gamma_matching_exit(pt)
    x, r, xt, Rt = contour["x"], contour["r"], contour["xt"], contour["Rt"]
    m = x >= xt
    xd, rd = x[m], r[m]
    eps_loc = np.maximum((rd / Rt) ** 2, 1.0 + 1e-9)
    M = np.array([mach_from_area(e, g, True) for e in eps_loc])
    p_wall = pt.pc * p_ratio(M, g)
    p_sep = pa_bar * np.array([schmucker_psep_over_pa(Mi) if 1.88 * Mi > 1.0 + 1e-6 else np.inf for Mi in M])
    sep = np.nonzero(p_wall < p_sep)[0]
    return {"x_mm": xd * 1e3, "r_mm": rd * 1e3, "M": M, "p_wall_bar": p_wall, "p_sep_bar": p_sep,
            "x_sep_mm": float(xd[sep[0]] * 1e3) if len(sep) else None, "gamma": g,
            "x_throat_mm": xt * 1e3, "x_exit_mm": float(xd[-1] * 1e3)}


def analyse(test: dict, eta_cstar_model=C.ETA_CSTAR, eta_cf_model=C.ETA_CF_VAC) -> dict:
    """All results of one test point. Raises ValueError if a required input is missing."""
    v = {k: _num(test["inputs"].get(k)) for k in INPUTS}
    missing = [INPUTS[k].label for k in REQUIRED if not v[k] or v[k] <= 0]
    if missing:
        raise ValueError("missing: " + ", ".join(missing))
    hw = hardware_presets()[test.get("hardware", "E2-REG-1")]
    pa = v["p_amb_bar"] or C.P_AMB
    add = pa if test.get("pressures_gauge") else 0.0
    P = {k: (v[k] + add if v[k] is not None else None) for k in PRESSURE_KEYS}
    pc, mox, mf, F = P["pc_bar"], v["ox_flow_kg_s"], v["fuel_flow_kg_s"], v["thrust_N"]
    T_ox = (v["ox_temp_C"] if v["ox_temp_C"] is not None else C.T_OX_NOM - 273.15) + 273.15
    dt_mm = v["throat_d_mm"] or hw.throat_d_mm
    At = math.pi * (dt_mm * 1e-3) ** 2 / 4
    eps = hw.eps * (hw.throat_d_mm / dt_mm) ** 2      # the exit stays, the throat changes
    Ae = eps * At
    mdot = mox + mf
    OF = mox / mf
    if not MR_RANGE[0] <= OF <= MR_RANGE[1]:
        raise ValueError(f"O/F {OF:.2f} is outside {MR_RANGE[0]}-{MR_RANGE[1]}; check the flows")
    # Reactant enthalpy: compressed liquid at the supply pressure. The feed throttles
    # isenthalpically, so h at the injector is the liquid h upstream even where the N2O
    # flashes in the jacket or the manifold (a card at the injector pressure would be vapour).
    ox = _ox_card(T_ox, C.P_TANK)
    fuel = ethanol_card(C.ETHANOL_WT)
    pt = point(round(pc, 3), round(OF, 4), round(eps, 4), pa, ox, fuel)

    R = {"hardware": hw.key, "At_mm2": At * 1e6, "eps": eps, "throat_d_mm": dt_mm, "pa_bar": pa,
         "OF": OF, "mdot_kg_s": mdot, "pc_abs_bar": pc, "T_ox_K": T_ox}
    # c*
    R["cstar_meas_m_s"] = pc * 1e5 * At / mdot
    R["cstar_ideal_m_s"] = pt.cstar
    R["eta_cstar"] = R["cstar_meas_m_s"] / pt.cstar
    R["Tc_K"] = pt.Tc
    # model at the measured flows (baseline efficiencies)
    pcm = pc_at(mdot, OF, At, eps, eta_cstar_model, ox, fuel)
    ptm = point(round(pcm, 3), round(OF, 4), round(eps, 4), pa, ox, fuel)
    R["pc_model_bar"] = pcm
    R["pc_dev_pct"] = 100 * (pc / pcm - 1)
    R["thrust_model_N"] = eta_cf_model * ptm.cf_vac * pcm * 1e5 * At - pa * 1e5 * Ae
    R["isp_model_s"] = R["thrust_model_N"] / (mdot * C.G0)
    R["isp_ideal_s"] = pt.isp_amb
    R["cf_ideal"] = pt.cf_amb
    R["cf_vac_ideal"] = pt.cf_vac
    # thrust
    if F:
        R["thrust_N"] = F
        R["thrust_dev_pct"] = 100 * (F / R["thrust_model_N"] - 1)
        R["cf_meas"] = F / (pc * 1e5 * At)
        R["eta_cf_vac"] = (F + pa * 1e5 * Ae) / (pc * 1e5 * At) / pt.cf_vac
        R["isp_meas_s"] = F / (mdot * C.G0)
        R["eta_isp"] = R["isp_meas_s"] / pt.isp_amb
        R["throttle_pct"] = 100 * F / F_RATED
    else:
        R["throttle_pct"] = 100 * R["thrust_model_N"] / F_RATED
    # nozzle
    R["pe_bar"] = pt.pe
    R["pe_over_pa"] = pt.pe / pa
    R["summerfield_sep"] = bool(pt.pe < SUMMERFIELD * pa)
    nz = nozzle_wall(hw.contour(dt_mm if v["throat_d_mm"] else None), pt, pa)
    R["x_sep_mm"] = nz["x_sep_mm"]
    R["nozzle"] = nz
    # injector
    if P["p_ox_inj_bar"]:
        dp = P["p_ox_inj_bar"] - pc
        rho, liquid = _rho_n2o(T_ox, P["p_ox_inj_bar"] * 1e5)
        R.update(dp_ox_bar=dp, stiff_ox_pct=100 * dp / pc, ox_liquid_at_inlet=liquid, rho_ox=rho)
        if dp > 0:
            R["CdA_ox_mm2"] = 1e6 * mox / math.sqrt(2 * rho * dp * 1e5)
    if P["p_fuel_inj_bar"]:
        dp = P["p_fuel_inj_bar"] - pc
        rho = PropsSI("D", "T", 293.15, "P", P["p_fuel_inj_bar"] * 1e5, "Ethanol")
        R.update(dp_fuel_bar=dp, stiff_fuel_pct=100 * dp / pc, rho_fuel=rho)
        if dp > 0:
            R["CdA_fuel_mm2"] = 1e6 * mf / math.sqrt(2 * rho * dp * 1e5)
    # water calorimetry
    w = {k: v[k] for k in ("water_flow_kg_s", "water_T_in_C", "water_T_out_C")}
    if all(x is not None for x in w.values()) and w["water_flow_kg_s"] > 0:
        p_in = P["water_p_in_bar"] or C.WATER_P_IN
        p_out = P["water_p_out_bar"] or p_in
        h_in = PropsSI("H", "T", w["water_T_in_C"] + 273.15, "P", p_in * 1e5, "Water")
        h_out = PropsSI("H", "T", w["water_T_out_C"] + 273.15, "P", p_out * 1e5, "Water")
        R["Q_meas_kW"] = w["water_flow_kg_s"] * (h_out - h_in) / 1e3
        R["q_per_ox_kJ_kg"] = R["Q_meas_kW"] / mox
        if P["water_p_out_bar"] and P["water_p_in_bar"]:
            R["dp_jacket_meas_bar"] = P["water_p_in_bar"] - P["water_p_out_bar"]
    R["pt"] = pt
    R["ox_card"] = ox
    return R


# ----------------------------------------------------------------------------------------
# Water jacket: regen.march at the measured point, Bartz factor fitted to the calorimetry
# ----------------------------------------------------------------------------------------
def water_jacket(test: dict, R: dict, n_stations=200, fit=True) -> dict | None:
    """Model jacket heat at the measured point and the Bartz factor that matches the
    calorimetric heat. E2-REG-1 only. About 1 s per march, 3-4 marches with the fit."""
    if "Q_meas_kW" not in R or not hardware_presets()[R["hardware"]].regen_model:
        return None
    from .contour import wall_stations
    from .jacket import e2_reg1
    from .regen import march

    v = {k: _num(test["inputs"].get(k)) for k in INPUTS}
    add = R["pa_bar"] if test.get("pressures_gauge") else 0.0
    p_in = (v["water_p_in_bar"] + add) if v["water_p_in_bar"] is not None else C.WATER_P_IN
    h_in = PropsSI("H", "T", v["water_T_in_C"] + 273.15, "P", p_in * 1e5, "Water")
    wall = wall_stations(e2_reg1_asbuilt(), n_stations)
    jac = e2_reg1()
    Qm = R["Q_meas_kW"] * 1e3

    def run(bf):
        return march(wall, R["pt"], R["pc_abs_bar"], jac, "water", v["water_flow_kg_s"], p_in, h_in,
                     bartz_factor=bf, eta_cstar=R["eta_cstar"])
    r1 = run(1.0)
    out = {"model": r1, "Q_model_kW": r1["Q"] / 1e3, "dp_jacket_model_bar": float(r1["dp_bar"]),
           "T_out_model_C": r1["T_out"] - 273.15, "Twg_max_K": r1["Twg_max"]}
    if not fit or Qm <= 0:
        return out
    # Q is close to proportional to the Bartz factor (the wall stays far below Taw); secant
    b0, Q0, b1 = 1.0, r1["Q"], Qm / r1["Q"]
    rf = r1
    for _ in range(4):
        b1 = float(np.clip(b1, 0.2, 3.0))
        rf = run(b1)
        if abs(rf["Q"] - Qm) < 0.005 * Qm:
            break
        b0, Q0, b1 = b1, rf["Q"], b1 + (Qm - rf["Q"]) * (b1 - b0) / (rf["Q"] - Q0)
    out.update(fit=rf, bartz_fit=b1, Twg_max_fit_K=rf["Twg_max"], Twc_max_fit_K=rf["Twc_max"],
               Q_fit_kW=rf["Q"] / 1e3)
    return out


# ----------------------------------------------------------------------------------------
# Model lines for the plots
# ----------------------------------------------------------------------------------------
def model_line(hw_key, mf, T_ox_K, pa=C.P_AMB, throat_d_mm=None, eta_c=C.ETA_CSTAR, eta_f=C.ETA_CF_VAC,
               of_range=(1.6, 5.2), n=36):
    """Throttle line of the hardware at a fixed fuel flow: delivered (assumed efficiencies)
    and ideal (eta = 1) pc, thrust, c* and Isp against O/F."""
    hw = hardware_presets()[hw_key]
    dt_mm = throat_d_mm or hw.throat_d_mm
    At = math.pi * (dt_mm * 1e-3) ** 2 / 4
    eps = hw.eps * (hw.throat_d_mm / dt_mm) ** 2
    ox, fuel = _ox_card(T_ox_K, C.P_TANK), ethanol_card(C.ETHANOL_WT)
    rows = []
    for of in np.linspace(*of_range, n):
        mdot = mf * (1 + of)
        row = {"OF": of, "ox_flow_kg_s": mf * of, "mdot_kg_s": mdot}
        for tag, ec, ef in (("model", eta_c, eta_f), ("ideal", 1.0, 1.0)):
            pc = pc_at(mdot, of, At, eps, ec, ox, fuel)
            p = point(round(pc, 3), round(of, 4), round(eps, 4), pa, ox, fuel)
            F = ef * p.cf_vac * pc * 1e5 * At - pa * 1e5 * eps * At
            row.update({f"pc_{tag}": pc, f"thrust_{tag}": F, f"cstar_{tag}": ec * p.cstar,
                        f"isp_{tag}": F / (mdot * C.G0), f"Tc_{tag}": p.Tc})
        rows.append(row)
    return rows


def cea_map(eps, T_ox_K=C.T_OX_NOM, pa=C.P_AMB, pcs=None, ofs=None):
    """Ideal CEA grid over pc and O/F for a model add-on that cannot run CEA itself
    (hopper-test-lab is TypeScript). Bilinear interpolation of this grid reproduces
    the point() values to well under 0.5 % inside the grid."""
    pcs = list(pcs if pcs is not None else np.arange(4.0, 36.01, 2.0))
    ofs = list(ofs if ofs is not None else np.round(np.arange(1.5, 6.01, 0.1), 2))
    ox, fuel = _ox_card(T_ox_K, C.P_TANK), ethanol_card(C.ETHANOL_WT)
    keys = ("cstar", "Tc", "cf_vac", "isp_vac", "pe", "gam_fr_c", "mw_c")
    grid = {k: [] for k in keys}
    for pc in pcs:
        rows = {k: [] for k in keys}
        for of in ofs:
            p = point(float(pc), float(of), round(eps, 4), pa, ox, fuel)
            for k in keys:
                rows[k].append(round(float(getattr(p, k)), 5))
        for k in keys:
            grid[k].append(rows[k])
    import e2_baseline
    return {
        "kind": "e2-cea-map", "version": 1,
        "description": "Ideal CEA (shifting equilibrium) at chamber pressure pc [bar, rows] and O/F [columns]. "
                       "cstar m/s, Tc K, cf_vac -, isp_vac s, pe bar (exit pressure at this pc), gam_fr_c, mw_c kg/kmol. "
                       "Delivered: c* = eta_cstar c*_ideal; F = eta_cf_vac cf_vac pc At - pa Ae.",
        "eps": round(float(eps), 4), "ox_card": ox, "ox_temperature_K": T_ox_K, "fuel_card": fuel,
        "pa_bar": pa, "baseline": e2_baseline.describe(C.BASELINE),
        "assumed": {"eta_cstar": C.ETA_CSTAR, "eta_cf_vac": C.ETA_CF_VAC},
        "pc_bar": [float(x) for x in pcs], "OF": [float(x) for x in ofs], "grid": grid,
    }


# ----------------------------------------------------------------------------------------
# Time series
# ----------------------------------------------------------------------------------------
CHANNEL_HINTS = {   # input key -> name patterns (case-insensitive) for the CSV column guess
    "time": [r"^t$", r"time", r"^t[_ ]?s"],
    "pc_bar": [r"pc\b", r"p_?c\b", r"chamber", r"pt-?cc", r"\bcc\b"],
    "ox_flow_kg_s": [r"ox.*flow", r"mdot.*ox", r"fm-?ox", r"m_?ox", r"n2o.*(flow|mdot)"],
    "fuel_flow_kg_s": [r"fu(el)?.*flow", r"mdot.*fu", r"fm-?fu", r"m_?f(uel)?\b", r"eth.*(flow|mdot)"],
    "thrust_N": [r"thrust", r"\blc", r"load", r"force"],
    "p_ox_inj_bar": [r"p.*ox.*inj", r"pt-?ox", r"inj.*ox", r"p_?ox"],
    "p_fuel_inj_bar": [r"p.*fu.*inj", r"pt-?fu", r"inj.*fu", r"p_?fu"],
    "ox_temp_C": [r"t.*ox", r"tt-?ox", r"n2o.*temp"],
    "water_flow_kg_s": [r"water.*flow", r"fm-?w", r"cool.*flow"],
    "water_T_in_C": [r"t.*w.*in", r"tt-?w.*in", r"water.*t.*in"],
    "water_T_out_C": [r"t.*w.*out", r"tt-?w.*out", r"water.*t.*out"],
    "water_p_in_bar": [r"p.*w.*in", r"pt-?w.*in"],
    "water_p_out_bar": [r"p.*w.*out", r"pt-?w.*out"],
}


def guess_columns(columns) -> dict:
    """Best-guess mapping input key -> CSV column from the column names."""
    out, used = {}, set()
    for key, pats in CHANNEL_HINTS.items():
        for pat in pats:
            hit = next((c for c in columns if c not in used and re.search(pat, str(c), re.I)), None)
            if hit is not None:
                out[key] = hit
                used.add(hit)
                break
    return out


def burn_window(t, pc, frac_on=0.5, trim_start=0.3, trim_end=0.1, min_trim_s=0.3):
    """Burn and steady-state window from a chamber pressure trace.

    The burn is the longest run with pc above frac_on x the 98th percentile of pc
    (relative to the pre-fire level). The steady window drops trim_start of the burn
    (at least min_trim_s) at the start for the start transient and trim_end at the end.
    Returns (burn_t0, burn_t1, steady_t0, steady_t1) or None if there is no burn."""
    t, pc = np.asarray(t, float), np.asarray(pc, float)
    ok = np.isfinite(t) & np.isfinite(pc)
    t, pc = t[ok], pc[ok]
    if len(t) < 10:
        return None
    base = np.percentile(pc, 5)
    top = np.percentile(pc, 98)
    if top - base <= 0:
        return None
    on = pc > base + frac_on * (top - base)
    edges = np.diff(np.r_[0, on.astype(int), 0])
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    if not len(starts):
        return None
    i = int(np.argmax(t[ends] - t[starts]))
    b0, b1 = t[starts[i]], t[ends[i]]
    L = b1 - b0
    s0 = b0 + max(trim_start * L, min(min_trim_s, 0.4 * L))
    s1 = b1 - trim_end * L
    if s1 <= s0:
        s0, s1 = b0, b1
    return float(b0), float(b1), float(s0), float(s1)


def window_means(df, mapping: dict, t0: float, t1: float, scale: dict | None = None) -> dict:
    """Means of the mapped columns of a DataFrame between t0 and t1."""
    scale = scale or {}
    tcol = mapping["time"]
    sel = df[(df[tcol] >= t0) & (df[tcol] <= t1)]
    out = {}
    for key, col in mapping.items():
        if key == "time" or col is None or col not in df:
            continue
        val = float(sel[col].mean()) * scale.get(key, 1.0)
        out[key] = None if math.isnan(val) else val
    out["_n"] = len(sel)
    return out


def eta_cstar_trace(t, pc_bar, mox, mf, hw_key, T_ox_K, pa=C.P_AMB, throat_d_mm=None, n_max=300):
    """Time-resolved O/F, measured c* and eta_c* during the burn (flow-meter lag makes
    the start and shutdown transients meaningless; read only the burn)."""
    hw = hardware_presets()[hw_key]
    dt_mm = throat_d_mm or hw.throat_d_mm
    At = math.pi * (dt_mm * 1e-3) ** 2 / 4
    eps = hw.eps * (hw.throat_d_mm / dt_mm) ** 2
    ox, fuel = _ox_card(T_ox_K, C.P_TANK), ethanol_card(C.ETHANOL_WT)
    t, pc_bar, mox, mf = (np.asarray(a, float) for a in (t, pc_bar, mox, mf))
    step = max(1, len(t) // n_max)
    t, pc_bar, mox, mf = t[::step], pc_bar[::step], mox[::step], mf[::step]
    rows = []
    for ti, p, mo, mu in zip(t, pc_bar, mox, mf):
        if not (p > 2.0 and mo > 0.02 and mu > 0.02):
            continue
        of = mo / mu
        if not MR_RANGE[0] <= of <= MR_RANGE[1]:
            continue
        cs = point(round(p * 5) / 5, round(of * 50) / 50, round(eps, 4), pa, ox, fuel).cstar
        cm = p * 1e5 * At / (mo + mu)
        rows.append({"t": ti, "OF": of, "cstar_meas": cm, "cstar_ideal": cs, "eta_cstar": cm / cs})
    return rows


def demo_trace(eta_c=0.93, eta_f=0.965, mox_ss=0.62, mf_ss=0.20, burn_s=6.0, rate=200, seed=1,
               hw_key="E2-REG-1", water=True):
    """Synthetic DAQ trace (columns as a test-stand export would have) built from the
    engine model with lags and noise. For trying the app only; marked as demo."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    hw = hardware_presets()[hw_key]
    t = np.arange(-1.5, burn_s + 2.0, 1 / rate)
    ramp = np.clip(t / 0.35, 0, 1) * np.clip((burn_s - t) / 0.25, 0, 1)
    ramp = np.convolve(ramp, np.ones(15) / 15, mode="same")
    drift = 1 - 0.012 * np.clip(t / burn_s, 0, 1)                   # small ox drift (tank cooling)
    mox = mox_ss * ramp * drift + rng.normal(0, 0.004, len(t))
    mf = mf_ss * np.clip(ramp * 1.05, 0, 1) + rng.normal(0, 0.0015, len(t))
    ox, fuel = _ox_card(C.T_OX_NOM, C.P_TANK), ethanol_card(C.ETHANOL_WT)
    pc = np.full(len(t), 0.0)
    F = np.full(len(t), 0.0)
    for i in range(0, len(t), 5):
        if ramp[i] > 0.05 and mox[i] > 0.05 and mf[i] > 0.03:
            of = max(1.2, min(7.5, mox[i] / mf[i]))
            p = pc_at(mox[i] + mf[i], of, hw.At, hw.eps, eta_c, ox, fuel)
            pt = point(round(p, 1), round(of, 2), round(hw.eps, 4), C.P_AMB, ox, fuel)
            pc[i:i + 5] = p
            F[i:i + 5] = eta_f * pt.cf_vac * p * 1e5 * hw.At - C.P_AMB * 1e5 * hw.eps * hw.At
    pc = np.maximum(pc, 0) + rng.normal(0, 0.06, len(t))
    pc += 0.25 * np.sin(2 * np.pi * 180 * t) * (ramp > 0.5)          # a little combustion noise
    F = np.maximum(F, 0) + rng.normal(0, 4.0, len(t))
    df = pd.DataFrame({
        "time_s": t, "PT-CC-01 [bar]": pc,
        "FM-OX [kg/s]": mox, "FM-FU [kg/s]": mf, "LC-01 [N]": F,
        "PT-OX-INJ [bar]": pc + (8.0 * (mox / mox_ss) ** 2) * ramp + rng.normal(0, 0.08, len(t)),
        "PT-FU-INJ [bar]": pc + (6.5 * (mf / mf_ss) ** 2) * ramp + rng.normal(0, 0.08, len(t)),
        "TT-OX [C]": 18.0 + rng.normal(0, 0.1, len(t)),
    })
    if water:
        heat = np.cumsum(ramp) / rate
        df["FM-W [kg/s]"] = 1.0 + rng.normal(0, 0.005, len(t))
        df["TT-W-IN [C]"] = 18.0 + rng.normal(0, 0.05, len(t))
        df["TT-W-OUT [C]"] = 18.0 + 36.0 * (1 - np.exp(-heat / 0.4)) * (ramp > 0.02) + rng.normal(0, 0.1, len(t))
        df["PT-W-IN [bar]"] = 30.0 + rng.normal(0, 0.05, len(t))
        df["PT-W-OUT [bar]"] = 26.0 + rng.normal(0, 0.05, len(t))
    return df


def demo_tests(n=6, seed=7):
    """A synthetic campaign on E2-REG-1 built from the engine model (efficiencies, a slowly
    eroding throat, water calorimetry). Test IDs start with DEMO; never saved."""
    from .operating_line import state
    rng = np.random.default_rng(seed)
    hw = hardware_presets()["E2-REG-1"]
    ox = _ox_card(C.T_OX_NOM, C.P_TANK)
    tests = []
    for i in range(n):
        mox = [0.47, 0.55, 0.62, 0.70, 0.62, 0.78, 0.84, 0.62][i % 8]
        eta_c = 0.915 + 0.006 * i + rng.normal(0, 0.004)
        eta_f = 0.962 + rng.normal(0, 0.004)
        s = state(mox, hw.At, eps=hw.eps, eta_c=eta_c, eta_f=eta_f, ox=ox)
        cda_ox, cda_fu = 17.5e-6 * (1 + rng.normal(0, 0.01)), 6.2e-6 * (1 + rng.normal(0, 0.01))
        rho_ox = PropsSI("D", "T", C.T_OX_NOM, "Q", 0, "NitrousOxide")
        rho_fu = PropsSI("D", "T", 293.15, "P", 30e5, "Ethanol")
        Q = 1e3 * (110 + 70 * (s["pc"] - 13) / 12) * (0.86 + rng.normal(0, 0.02))
        tests.append(new_test(
            test_id=f"DEMO-HF-{i + 1:03d}", date=str(dt.date(2026, 10, 1) + dt.timedelta(days=2 * i)),
            hardware="E2-REG-1", demo=True, notes="Synthetic demo point (model + noise), not a test.",
            pc_bar=s["pc"] * (1 + rng.normal(0, 0.003)), ox_flow_kg_s=mox, fuel_flow_kg_s=0.2,
            thrust_N=s["F"] * (1 + rng.normal(0, 0.004)), duration_s=round(5 + 2 * rng.random(), 1),
            ox_temp_C=round(C.T_OX_NOM - 273.15 + rng.normal(0, 1.5), 1),
            p_ox_inj_bar=s["pc"] + (mox / cda_ox) ** 2 / (2 * rho_ox) / 1e5,
            p_fuel_inj_bar=s["pc"] + (0.2 / cda_fu) ** 2 / (2 * rho_fu) / 1e5,
            water_flow_kg_s=1.0, water_T_in_C=18.0, water_T_out_C=18.0 + Q / 4180.0,
            water_p_in_bar=30.0, water_p_out_bar=26.1))
    return tests


# ----------------------------------------------------------------------------------------
# Storage: one YAML file per test in hotfire_tests/
# ----------------------------------------------------------------------------------------
TEST_DIR = Path(__file__).resolve().parents[1] / "hotfire_tests"


def _safe_name(test_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", test_id).strip("_") or "test"


def save_test(test: dict, folder: Path = TEST_DIR) -> Path:
    if not test.get("test_id"):
        raise ValueError("a test needs a test ID")
    folder.mkdir(parents=True, exist_ok=True)
    rec = {k: test[k] for k in ("test_id", "date", "hardware", "outcome", "pressures_gauge", "notes", "source")
           if k in test}
    rec["inputs"] = {k: _num(test["inputs"].get(k)) for k in INPUTS}
    import e2_baseline
    rec["saved"] = {"at": dt.datetime.now().isoformat(timespec="seconds"),
                    "baseline": e2_baseline.describe(C.BASELINE)}
    path = folder / f"{_safe_name(test['test_id'])}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# E2 hot-fire test point (E2 Test Explorer). Inputs only; results are recomputed\n"
                 "# with the current model when the file is loaded.\n")
        yaml.safe_dump(rec, fh, sort_keys=False, allow_unicode=True)
    return path


def load_tests(folder: Path = TEST_DIR) -> list[dict]:
    tests = []
    for p in sorted(folder.glob("*.yaml")):
        rec = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if "inputs" not in rec:
            continue
        t = new_test()
        t.update({k: v for k, v in rec.items() if k != "inputs"})
        t["date"] = str(t.get("date", ""))
        t["inputs"].update(rec["inputs"])
        t["_file"] = p.name
        tests.append(t)
    return tests


def delete_test(test_id: str, folder: Path = TEST_DIR):
    p = folder / f"{_safe_name(test_id)}.yaml"
    if p.exists():
        p.unlink()


def flat(test: dict, R: dict) -> dict:
    """One table row: metadata, inputs and scalar results."""
    row = {"test_id": test["test_id"], "date": test.get("date"), "hardware": test.get("hardware"),
           "outcome": OUTCOMES.get(test.get("outcome"), test.get("outcome")), "demo": bool(test.get("demo"))}
    row.update({k: _num(test["inputs"].get(k)) for k in INPUTS})
    for k in RESULT_LABELS:
        if k in R and R[k] is not None and not isinstance(R[k], (dict, bool)):
            row[f"r_{k}"] = float(R[k])
    row["summerfield_sep"] = R.get("summerfield_sep")
    row["notes"] = test.get("notes", "")
    return row


# ----------------------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------------------
def observations(R: dict, W: dict | None = None, eta_cstar_model=C.ETA_CSTAR, eta_cf_model=C.ETA_CF_VAC,
                 reference: dict | None = None) -> list[tuple[str, str]]:
    """Short findings, each (level, text) with level good / warn / crit / info.
    reference: optional mean results of earlier tests (keys as R) for drift checks."""
    out = []
    e = R["eta_cstar"]
    d = e - eta_cstar_model
    lvl = "good" if d >= -0.01 else ("warn" if d >= -0.04 else "crit")
    if e > 1.0:
        lvl = "warn"
    out.append((lvl, f"η c* {e:.3f} vs {eta_cstar_model:.3f} assumed ({100 * d:+.1f} pts); "
                     f"pc {R['pc_dev_pct']:+.1f} % against the model at the measured flows."
                + (" Above 1: check the flow meters, the throat diameter and gauge/absolute pressure." if e > 1.0 else "")))
    if "eta_cf_vac" in R:
        d = R["eta_cf_vac"] - eta_cf_model
        lvl = "good" if d >= -0.015 else "warn"
        if R["summerfield_sep"] or R.get("x_sep_mm"):
            lvl = "info"
        out.append((lvl, f"η Cf,vac {R['eta_cf_vac']:.3f} vs {eta_cf_model:.3f} assumed; thrust "
                         f"{R['thrust_dev_pct']:+.1f} % against the model. Isp {R['isp_meas_s']:.1f} s."))
    if R["summerfield_sep"] or R.get("x_sep_mm") is not None:
        where = f" Schmucker: separation at x = {R['x_sep_mm']:.0f} mm (exit {R['nozzle']['x_exit_mm']:.0f} mm)." \
            if R.get("x_sep_mm") is not None else ""
        out.append(("warn", f"Nozzle flow likely separated: pe {R['pe_bar']:.2f} bar = {R['pe_over_pa']:.2f} pa."
                            f"{where} Thrust and Cf are not representative of attached flow."))
    elif R["pe_over_pa"] < 0.55:
        out.append(("info", f"Overexpanded, pe/pa {R['pe_over_pa']:.2f}; attached by Summerfield and Schmucker."))
    for c, name in (("ox", "Ox"), ("fuel", "Fuel")):
        k = f"stiff_{c}_pct"
        if k in R:
            s = R[k]
            if R[f"dp_{c}_bar"] <= 0:
                out.append(("crit", f"{name} injector dp {R[f'dp_{c}_bar']:.1f} bar: manifold below chamber pressure; check the sensors."))
            else:
                lvl = "good" if s >= 100 * STIFFNESS_MIN else "warn"
                out.append((lvl, f"{name} injector stiffness {s:.0f} % of pc (dp {R[f'dp_{c}_bar']:.1f} bar), "
                                 f"CdA {R.get(f'CdA_{c}_mm2', float('nan')):.2f} mm²."
                                 + ("" if lvl == "good" else f" Below {100 * STIFFNESS_MIN:.0f} %: low margin against chugging.")))
    if "ox_liquid_at_inlet" in R and not R["ox_liquid_at_inlet"]:
        out.append(("info", "N2O at the injector inlet is below its saturation pressure: two-phase inflow. "
                            "The ox CdA uses the saturated liquid density; compare it between tests only."))
    if W:
        q = R["Q_meas_kW"] / W["Q_model_kW"]
        out.append(("info", f"Jacket heat {R['Q_meas_kW']:.0f} kW measured vs {W['Q_model_kW']:.0f} kW model "
                            f"(×{q:.2f}). Bartz factor that matches: {W.get('bartz_fit', float('nan')):.2f}."))
        if "Twg_max_fit_K" in W:
            T = W["Twg_max_fit_K"]
            lvl = "good" if T < C.T_WG_LIMIT - 50 else ("warn" if T < C.T_WG_LIMIT else "crit")
            out.append((lvl, f"Hot-wall maximum {T:.0f} K with the fitted Bartz factor "
                             f"(IN718 limit {C.T_WG_LIMIT:.0f} K)."))
        if "dp_jacket_meas_bar" in R:
            out.append(("info", f"Jacket dp {R['dp_jacket_meas_bar']:.2f} bar measured vs "
                                f"{W['dp_jacket_model_bar']:.2f} bar model (manifolds not modelled)."))
    if reference:
        for k, name, tol in (("CdA_ox_mm2", "Ox CdA", 0.05), ("CdA_fuel_mm2", "Fuel CdA", 0.05),
                             ("eta_cstar", "η c*", 0.02)):
            if k in R and reference.get(k):
                rel = R[k] / reference[k] - 1
                if abs(rel) > tol:
                    out.append(("warn", f"{name} {100 * rel:+.1f} % against the mean of the earlier tests "
                                        f"on this hardware ({reference[k]:.3g})."))
    return out
