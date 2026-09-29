#!/usr/bin/env python3
"""
LCSC injector element sizing and analysis
=========================================

Liquid-centred swirl coaxial (LCSC) element: a closed swirl chamber injects the
liquid fuel as a conical film, and the gaseous (or two-phase) oxidizer leaves
through a coaxial annulus around the swirl nozzle and breaks up the film.

Usage
-----
    python lcsc_sizing.py config.yaml            # size + analyse, writes report/json/csv
    python lcsc_sizing.py config.yaml -o out/    # choose output folder
    python lcsc_sizing.py --check                # run the built-in validation cases

All flows and geometry are per element. The YAML file lists the inputs; see the
example configs next to this script.

What the script does
--------------------
1. Fuel swirl element
   * method "bazarov" (default): Abramovich maximum-flow theory with Bazarov's
     viscous correction (friction on the swirl chamber wall via an equivalent
     geometric characteristic A_eq, plus inlet losses). The geometry is iterated
     until the predicted spray half-angle and flow coefficient match the
     targets at the design pressure drop.
   * method "nardi": the procedure used in the vom Schemm thesis (and for P03/P04):
     X = 0.0042 alpha^1.2714, inviscid C_D, Nardi inlet coefficient.
   In both cases the final geometry is analysed with the viscous model, so the
   predicted pressure drop, spray angle and film are always from the same model.
2. Oxidizer annulus: exit area from a chosen exit velocity, exit state at
   chamber pressure and stagnation enthalpy minus v^2/2. Recess set for critical
   recess flow (recess angle = 2 x spray half-angle) or from a recess ratio.
3. Oxidizer metering holes: homogeneous equilibrium model (HEM, isentropic
   equilibrium expansion with real-fluid properties from CoolProp, choking
   included). Works for gas, supercritical, liquid and two-phase inlet states.
   "spi" (single-phase incompressible) is available for comparison.
4. Throttle sweep of the (rounded or as-built) geometry: fuel pressure drop,
   oxidizer feed pressure, stiffness, film thickness, momentum flux ratio J,
   recess regime and warnings.

Main references
---------------
* M. vom Schemm, Design und Charakterisierung von Koaxial-Swirl-Injektoren fuer
  die Anwendung in einem drosselbaren Lachgas-Ethanol-Triebwerk, MA, RWTH 2024.
* V. Bazarov, V. Yang, P. Puri, Design and Dynamics of Jet and Swirl Injectors,
  in: Liquid Rocket Thrust Chambers, AIAA Prog. Astro. Aero. 200, 2004.
* Suyari & Lefebvre (1986), Rizk & Lefebvre (1985), Fu et al. (2011) for film
  thickness correlations.

Model validity and calibration
------------------------------
* Swirl flow coefficient: the thesis measured 1.5x less flow than the Nardi
  method predicted. The viscous model predicts 0.073 for that thesis geometry
  (water, 20 g/s) against about 0.057 measured, i.e. still ~25 % high for the
  SLA-printed parts. Set calibration.swirl_cd_factor from cold-flow or hot-fire
  data (measured C_D / predicted C_D).
* Spray angle: the model predicts ~61 deg for the thesis geometry, measured was
  50-55 deg at J = 0. With gas flow the spray contracts to 40-45 deg (near
  field) and 20-25 deg (far field). Use calibration.spray_half_angle_measured_deg
  for the recess once measured.
* Film thickness is not validated in the thesis; J is therefore reported for
  several film models. Thesis data: SMD falls steeply up to J ~ 0.5 and levels
  out at 20-25 um from J ~ 1 (water/N2, atmospheric).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
import yaml
import CoolProp.CoolProp as CP
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline"))
import e2_baseline  # noqa: E402

BAR = 1e5
MM = 1e-3

# ----------------------------------------------------------------------------
# Default configuration (anything given in the YAML overrides these)
# ----------------------------------------------------------------------------
DEFAULTS = {
    "name": "lcsc",
    "elements": 1,
    "operating_point": {"chamber_pressure_bar": None, "mdot_total_kg_s": None, "of_ratio": None},
    "fuel": {"fluid": "Ethanol", "temperature_K": 300.0, "pressure_drop_bar": None,
             "density_kg_m3": None, "viscosity_Pa_s": None},
    "oxidizer": {"fluid": "N2O", "temperature_K": None, "quality": None, "pressure_drop_bar": None},
    "swirl": {"method": "bazarov", "spray_half_angle_deg": 60.0, "n_inlets": 3,
              "swirl_chamber_to_outlet_diameter": 3.3,
              "swirl_chamber_length_to_outlet_diameter": 3.3,
              "outlet_length_to_diameter": 0.5,
              "inlet_length_to_diameter": 3.0,
              "inlet_entry_loss": 0.5},
    "gas_annulus": {"exit_velocity_m_s": 100.0, "post_thickness_mm": 0.5,
                    "recess": "critical", "recess_ratio": None,
                    "recess_angle_source": "predicted"},
    "ox_holes": {"n_holes": 6, "model": "hem", "discharge_coefficient": 0.75},
    "manufacturing": {"min_hole_diameter_mm": 0.5, "round_to_mm": 0.01},
    "calibration": {"swirl_cd_factor": 1.0, "spray_half_angle_measured_deg": None},
    "limits": {"min_stiffness": 0.15, "min_J": 0.5},
    "throttle_points": [{"fraction": 1.0}],
    "as_built": {},
}

GEOMETRY_KEYS_MM = ["d_inlet", "l_inlet", "d_out", "l_out", "d_swirl_chamber", "l_swirl_chamber",
                    "post_thickness", "d_gas_out", "recess", "d_ox_hole"]
GEOMETRY_KEYS_INT = ["n_inlets", "n_ox_holes"]


def deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = deep_merge(DEFAULTS, yaml.safe_load(f) or {})
    if cfg.get("baseline"):
        apply_baseline(cfg, path)
    missing = []
    for sec, key in [("operating_point", "chamber_pressure_bar"), ("operating_point", "mdot_total_kg_s"),
                     ("operating_point", "of_ratio"), ("fuel", "pressure_drop_bar"),
                     ("oxidizer", "pressure_drop_bar")]:
        if cfg[sec][key] is None:
            missing.append(f"{sec}.{key}")
    if cfg["oxidizer"]["temperature_K"] is None and cfg["oxidizer"]["quality"] is None:
        missing.append("oxidizer.temperature_K or oxidizer.quality")
    if missing:
        raise ValueError("Missing required inputs: " + ", ".join(missing))
    unknown = set(cfg["as_built"]) - set(GEOMETRY_KEYS_MM) - set(GEOMETRY_KEYS_INT)
    if unknown:
        raise ValueError(f"Unknown as_built keys: {sorted(unknown)}. "
                         f"Allowed: {GEOMETRY_KEYS_MM + GEOMETRY_KEYS_INT}")
    return cfg


def apply_baseline(cfg: dict, path: str) -> None:
    """Operating point and throttle points from the engine table of the shared
    baseline. The rated row gives the nominal point; each throttle point's
    `fraction` is a fraction of rated THRUST and takes chamber pressure, O/F
    and total flow from the table (fuel flow fixed, oxidizer throttled).
    Values set in the config win; differences from the baseline are reported."""
    bl = e2_baseline.load(e2_baseline.resolve(cfg["baseline"], path))
    table = e2_baseline.engine_table(bl)
    rated = e2_baseline.table_point(table, 1.0)
    e2_baseline.fill(cfg, {"operating_point.chamber_pressure_bar": rated["chamber_pressure_bar"],
                           "operating_point.mdot_total_kg_s": rated["total_flow_kg_s"],
                           "operating_point.of_ratio": rated["mixture_ratio"]},
                     label=os.path.basename(path))
    for pt in cfg["throttle_points"]:
        row = e2_baseline.table_point(table, pt.get("fraction", 1.0))
        pt.setdefault("chamber_pressure_bar", row["chamber_pressure_bar"])
        pt.setdefault("of_ratio", row["mixture_ratio"])
        pt.setdefault("mdot_total_kg_s", row["total_flow_kg_s"])
    cfg["baseline_info"] = e2_baseline.describe(bl)


# ----------------------------------------------------------------------------
# Fluid properties (CoolProp)
# ----------------------------------------------------------------------------
_AS_CACHE: dict = {}


def state(fluid: str) -> CP.AbstractState:
    if fluid not in _AS_CACHE:
        _AS_CACHE[fluid] = CP.AbstractState("HEOS", fluid)
    return _AS_CACHE[fluid]


def fuel_props(cfg: dict, p: float) -> tuple[float, float]:
    """Liquid fuel density and dynamic viscosity at feed pressure."""
    fc = cfg["fuel"]
    rho, mu = fc["density_kg_m3"], fc["viscosity_Pa_s"]
    if rho is None or mu is None:
        s = state(fc["fluid"])
        s.update(CP.PT_INPUTS, p, fc["temperature_K"])
        rho = rho if rho is not None else s.rhomass()
        mu = mu if mu is not None else s.viscosity()
    return rho, mu


@dataclass
class OxInlet:
    p: float
    T: float
    rho: float
    h: float
    s: float
    quality: float  # -1 if single phase (CoolProp convention)
    phase: str


def ox_inlet_state(cfg: dict, p: float, T: float | None = None, quality: float | None = None) -> OxInlet:
    oc = cfg["oxidizer"]
    s = state(oc["fluid"])
    if T is None and quality is None:
        T, quality = oc["temperature_K"], oc["quality"]
    if T is not None:
        try:
            s.update(CP.PT_INPUTS, p, T)
        except ValueError:  # p exactly on the saturation line: step off it
            s.update(CP.PT_INPUTS, p * (1 + 1e-5), T)
    else:
        s.update(CP.PQ_INPUTS, p, quality)
    q = s.Q()
    ph = s.phase()
    phase = {CP.iphase_liquid: "liquid", CP.iphase_gas: "gas", CP.iphase_twophase: "two-phase",
             CP.iphase_supercritical: "supercritical", CP.iphase_supercritical_gas: "supercritical gas",
             CP.iphase_supercritical_liquid: "supercritical liquid"}.get(ph, str(ph))
    return OxInlet(p=p, T=s.T(), rho=s.rhomass(), h=s.hmass(), s=s.smass(), quality=q, phase=phase)


def ox_state_ph(cfg: dict, p: float, h: float) -> float:
    s = state(cfg["oxidizer"]["fluid"])
    s.update(CP.HmassP_INPUTS, h, p)
    return s.rhomass()


def ox_quality_ph(cfg: dict, p: float, h: float) -> float:
    """Vapour quality at (p, h); -1 for single phase."""
    s = state(cfg["oxidizer"]["fluid"])
    s.update(CP.HmassP_INPUTS, h, p)
    return s.Q()


def ox_saturation_pressure(cfg: dict, T: float) -> float | None:
    s = state(cfg["oxidizer"]["fluid"])
    if T >= s.T_critical():
        return None
    s.update(CP.QT_INPUTS, 0.0, T)
    return s.p()


# ----------------------------------------------------------------------------
# Oxidizer orifice flow: HEM and SPI
# ----------------------------------------------------------------------------
def _hem_flux_at(fluid: str, s0: float, h0: float, p: float) -> float:
    st = state(fluid)
    try:
        st.update(CP.PSmass_INPUTS, p, s0)
        dh = h0 - st.hmass()
        return st.rhomass() * math.sqrt(2.0 * dh) if dh > 0 else 0.0
    except ValueError:
        return float("nan")


def hem_mass_flux(fluid: str, inlet: OxInlet, p_down: float) -> tuple[float, bool, float]:
    """Homogeneous equilibrium mass flux [kg/m^2 s] from inlet stagnation state to p_down.

    The flux is the maximum of rho*sqrt(2(h0-h)) along the isentrope between
    p_down and p0; if the maximum lies above p_down the orifice is choked.
    Returns (G, choked, throat pressure).
    """
    p0 = inlet.p
    if p_down >= p0:
        return 0.0, False, p0
    f = lambda p: _hem_flux_at(fluid, inlet.s, inlet.h, p)
    if p0 - p_down < 1e-3 * p0:  # tiny pressure drop: no choking possible
        G = f(p_down)
        return (G if math.isfinite(G) else 0.0), False, p_down
    grid = np.linspace(p_down, p0 * (1 - 1e-5), 40)
    vals = np.array([f(p) for p in grid])
    vals = np.where(np.isfinite(vals), vals, -1.0)
    i = int(np.argmax(vals))
    if i == 0:
        return float(vals[0]), False, p_down
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda p: -f(p), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4 * p0})
    if math.isfinite(res.fun) and -res.fun > vals[i]:
        return float(-res.fun), True, float(res.x)
    return float(vals[i]), True, float(grid[i])


def spi_mass_flux(inlet: OxInlet, p_down: float) -> float:
    return math.sqrt(2.0 * inlet.rho * max(inlet.p - p_down, 0.0))


def ox_mass_flux(cfg: dict, inlet: OxInlet, p_down: float) -> tuple[float, bool]:
    if cfg["ox_holes"]["model"].lower() == "spi":
        return spi_mass_flux(inlet, p_down), False
    G, choked, _ = hem_mass_flux(cfg["oxidizer"]["fluid"], inlet, p_down)
    return G, choked


# ----------------------------------------------------------------------------
# Swirl element theory (Abramovich / Bazarov)
# ----------------------------------------------------------------------------
# phi: filling coefficient of the nozzle (1 - X), A: geometric characteristic,
# mu: flow coefficient referred to the nozzle (outlet) area.

def A_of_phi(phi: float) -> float:
    return (1.0 - phi) * math.sqrt(2.0) / phi ** 1.5


def mu_of_phi(phi: float) -> float:
    return phi ** 1.5 / math.sqrt(2.0 - phi)


def phi_of_A(A: float) -> float:
    return brentq(lambda p: A_of_phi(p) - A, 1e-8, 1.0 - 1e-12)


def spray_half_angle(phi: float, A: float, mu: float) -> float:
    """Spray half-angle [deg], Abramovich/Bazarov: tan a = 2 mu A / sqrt((1+S)^2 - 4 mu^2 A^2)."""
    S = math.sqrt(1.0 - phi)
    arg = (1.0 + S) ** 2 - 4.0 * mu ** 2 * A ** 2
    return math.degrees(math.atan2(2.0 * mu * A, math.sqrt(max(arg, 1e-12))))


def phi_of_angle(alpha_deg: float) -> float:
    f = lambda p: spray_half_angle(p, A_of_phi(p), mu_of_phi(p)) - alpha_deg
    return brentq(f, 1e-4, 1.0 - 1e-6)


def friction_factor(Re: float) -> float:
    """Bazarov: log10(lambda) = 25.8 / (log10 Re)^2.58 - 2."""
    Re = max(Re, 1000.0)
    return 10.0 ** (25.8 / math.log10(Re) ** 2.58 - 2.0)


@dataclass
class SwirlResult:
    A: float          # geometric characteristic R_in r_n / (n r_in^2)
    A_eq: float       # equivalent characteristic with wall friction
    C: float          # R_in / r_n
    Re_in: float
    friction: float
    phi: float        # filling coefficient at nozzle (with friction)
    mu_ideal: float   # inviscid flow coefficient for geometric A
    mu_model: float   # viscous model incl. inlet losses
    mu: float         # used value = model x calibration factor
    alpha_deg: float  # predicted spray half-angle
    dp: float         # pressure drop for the given flow [Pa]
    film_theory: float  # film thickness at nozzle exit from phi [m]


def swirl_analyse(g: dict, mdot: float, rho: float, visc: float, cfg: dict) -> SwirlResult:
    sw, cal = cfg["swirl"], cfg["calibration"]
    n = g["n_inlets"]
    r_in, r_n, r_sk = g["d_inlet"] / 2, g["d_out"] / 2, g["d_swirl_chamber"] / 2
    R = r_sk - r_in
    A = R * r_n / (n * r_in ** 2)
    C = R / r_n
    Re = 2.0 * mdot / (math.pi * n * r_in * visc)
    lam = friction_factor(Re)
    A_eq = R * r_n / (n * r_in ** 2 + 0.5 * lam * R * (R - r_n))
    phi = phi_of_A(A_eq)
    mu_eq = mu_of_phi(phi)
    xi = sw["inlet_entry_loss"] + lam * g["l_inlet"] / g["d_inlet"]
    mu_model = mu_eq / math.sqrt(1.0 + xi * mu_eq ** 2 * A ** 2 / C ** 2)
    mu = mu_model * cal["swirl_cd_factor"]
    dp = (mdot / (mu * math.pi * r_n ** 2)) ** 2 / (2.0 * rho)
    return SwirlResult(A=A, A_eq=A_eq, C=C, Re_in=Re, friction=lam, phi=phi,
                       mu_ideal=mu_of_phi(phi_of_A(A)), mu_model=mu_model, mu=mu,
                       alpha_deg=spray_half_angle(phi, A_eq, mu_eq), dp=dp,
                       film_theory=r_n * (1.0 - math.sqrt(1.0 - phi)))


def _swirl_proportions(g: dict, cfg: dict) -> dict:
    sw = cfg["swirl"]
    g["l_out"] = sw["outlet_length_to_diameter"] * g["d_out"]
    g["l_swirl_chamber"] = sw["swirl_chamber_length_to_outlet_diameter"] * g["d_out"]
    g["l_inlet"] = sw["inlet_length_to_diameter"] * g["d_inlet"]
    return g


def size_swirl_nardi(mdot: float, rho: float, dp: float, cfg: dict) -> dict:
    """Thesis procedure (vom Schemm section 5.2, Nardi et al.)."""
    sw = cfg["swirl"]
    X = 0.0042 * sw["spray_half_angle_deg"] ** 1.2714
    cd = math.sqrt((1 - X) ** 3 / (1 + X))
    cd_in = math.sqrt(X ** 3 / (2 - X))
    q = math.sqrt(2 * rho * dp)
    d_out = math.sqrt(4 * mdot / (cd * q) / math.pi)
    d_in = math.sqrt(4 * mdot / (cd_in * q) / (math.pi * sw["n_inlets"]))
    g = {"n_inlets": sw["n_inlets"], "d_inlet": d_in, "d_out": d_out,
         "d_swirl_chamber": sw["swirl_chamber_to_outlet_diameter"] * d_out}
    return _swirl_proportions(g, cfg)


def size_swirl_bazarov(mdot: float, rho: float, visc: float, dp: float, cfg: dict) -> dict:
    """Viscous sizing: iterate geometry until predicted angle and flow match targets."""
    sw = cfg["swirl"]
    n, k_sk = sw["n_inlets"], sw["swirl_chamber_to_outlet_diameter"]
    phi_t = phi_of_angle(sw["spray_half_angle_deg"])
    A_eq_t = A_of_phi(phi_t)
    A, mu = A_eq_t, mu_of_phi(phi_t) * cfg["calibration"]["swirl_cd_factor"]
    g = {"n_inlets": n}
    for _ in range(200):
        r_n = math.sqrt(mdot / (math.pi * mu * math.sqrt(2 * rho * dp)))
        r_sk = 0.5 * k_sk * 2 * r_n
        # n A r_in^2 + r_n r_in - r_sk r_n = 0  (from A = (r_sk - r_in) r_n / (n r_in^2))
        r_in = (-r_n + math.sqrt(r_n ** 2 + 4 * n * A * r_sk * r_n)) / (2 * n * A)
        g.update(d_out=2 * r_n, d_inlet=2 * r_in, d_swirl_chamber=2 * r_sk)
        _swirl_proportions(g, cfg)
        res = swirl_analyse(g, mdot, rho, visc, cfg)
        A_max = 2.0 / (res.friction * (res.C - 1.0))  # A_eq limit for A -> infinity
        if A_eq_t > 0.95 * A_max:
            raise ValueError(f"Target spray angle not reachable with friction (A_eq target {A_eq_t:.1f}, "
                             f"max {A_max:.1f}). Reduce spray angle or swirl chamber ratio.")
        A_new = A * A_eq_t / res.A_eq
        mu_new = res.mu
        if abs(A_new / A - 1) < 1e-7 and abs(mu_new / mu - 1) < 1e-7:
            break
        A = A + 0.7 * (A_new - A)
        mu = mu + 0.7 * (mu_new - mu)
    return g


# ----------------------------------------------------------------------------
# Film thickness correlations [m]
# ----------------------------------------------------------------------------
FILM_CORRELATIONS = {"Suyari-Lefebvre": 2.7, "Fu": 3.1, "Rizk-Lefebvre": 3.66}


def film_correlation(C: float, d_out: float, mdot: float, visc: float, rho: float, dp: float) -> float:
    return C * (d_out * mdot * visc / (rho * dp)) ** 0.25


def axial_film_velocity(mdot: float, rho: float, r_n: float, t: float) -> float:
    t = min(t, r_n)
    return mdot / (rho * math.pi * (r_n ** 2 - (r_n - t) ** 2))


# ----------------------------------------------------------------------------
# Element sizing
# ----------------------------------------------------------------------------
def per_element_flows(cfg: dict, fraction: float = 1.0, of: float | None = None,
                      m_total: float | None = None) -> tuple[float, float]:
    """(ox, fuel) flow per element. The head total is m_total if given, else
    fraction x the nominal total flow."""
    op = cfg["operating_point"]
    of = op["of_ratio"] if of is None else of
    m_tot = (op["mdot_total_kg_s"] * fraction if m_total is None else m_total) / cfg["elements"]
    return m_tot * of / (1 + of), m_tot / (1 + of)  # ox, fuel


def gas_exit_state(cfg: dict, pc: float, h0: float, mdot_ox: float, area: float | None = None,
                   v: float | None = None) -> tuple[float, float]:
    """Annulus exit: p = pc, h = h0 - v^2/2. Give either area (-> solve v) or v."""
    if v is not None:
        return ox_state_ph(cfg, pc, h0 - 0.5 * v * v), v
    v = 100.0
    for _ in range(50):
        rho = ox_state_ph(cfg, pc, h0 - 0.5 * v * v)
        v_new = mdot_ox / (rho * area)
        if abs(v_new - v) < 1e-6:
            break
        v = v_new
    return rho, v


def recess_angle(cfg: dict, predicted_alpha: float) -> float:
    cal, ga = cfg["calibration"], cfg["gas_annulus"]
    if cal["spray_half_angle_measured_deg"] is not None:
        return cal["spray_half_angle_measured_deg"]
    if ga["recess_angle_source"] == "design":
        return cfg["swirl"]["spray_half_angle_deg"]
    return predicted_alpha


def round_geometry(g: dict, cfg: dict) -> dict:
    step = cfg["manufacturing"]["round_to_mm"]
    if not step:
        return dict(g)
    out = dict(g)
    for k in GEOMETRY_KEYS_MM:
        if k in out:
            out[k] = round(out[k] / MM / step) * step * MM
    return out


def size_element(cfg: dict) -> tuple[dict, dict]:
    """Size one element at the nominal operating point. Returns (geometry [m], info)."""
    op, fc, oc, ga, oh = cfg["operating_point"], cfg["fuel"], cfg["oxidizer"], cfg["gas_annulus"], cfg["ox_holes"]
    pc = op["chamber_pressure_bar"] * BAR
    dp_f, dp_o = fc["pressure_drop_bar"] * BAR, oc["pressure_drop_bar"] * BAR
    m_ox, m_f = per_element_flows(cfg)
    rho_f, visc_f = fuel_props(cfg, pc + dp_f)

    method = cfg["swirl"]["method"].lower()
    if method == "nardi":
        g = size_swirl_nardi(m_f, rho_f, dp_f, cfg)
    elif method == "bazarov":
        g = size_swirl_bazarov(m_f, rho_f, visc_f, dp_f, cfg)
    else:
        raise ValueError("swirl.method must be 'bazarov' or 'nardi'")
    sw_res = swirl_analyse(g, m_f, rho_f, visc_f, cfg)

    # oxidizer inlet state and annulus
    inlet = ox_inlet_state(cfg, pc + dp_o)
    rho_c, v = gas_exit_state(cfg, pc, inlet.h, m_ox, v=ga["exit_velocity_m_s"])
    A_gas = m_ox / (rho_c * v)
    r_post = g["d_out"] / 2 + ga["post_thickness_mm"] * MM
    r_gas = math.sqrt(A_gas / math.pi + r_post ** 2)
    g["post_thickness"] = ga["post_thickness_mm"] * MM
    g["d_gas_out"] = 2 * r_gas

    alpha_r = recess_angle(cfg, sw_res.alpha_deg)
    if ga["recess"] == "critical":
        g["recess"] = (r_gas - g["d_out"] / 2) / math.tan(math.radians(alpha_r))
    elif ga["recess"] == "ratio":
        g["recess"] = ga["recess_ratio"] * g["d_out"]
    else:
        raise ValueError("gas_annulus.recess must be 'critical' or 'ratio'")

    # metering holes, downstream pressure = chamber + annulus dynamic head
    p_down = pc + 0.5 * rho_c * v * v
    G, choked = ox_mass_flux(cfg, inlet, p_down)
    A_holes = m_ox / (oh["discharge_coefficient"] * G)
    g["n_ox_holes"] = oh["n_holes"]
    g["d_ox_hole"] = math.sqrt(4 * A_holes / (math.pi * oh["n_holes"]))

    info = {"rho_fuel": rho_f, "visc_fuel": visc_f, "ox_inlet": asdict(inlet), "rho_ox_exit": rho_c,
            "ox_hole_mass_flux": G, "ox_hole_choked_at_design": choked, "recess_angle_used_deg": alpha_r,
            "swirl_at_sizing": asdict(sw_res)}
    return g, info


def apply_as_built(g: dict, cfg: dict) -> dict:
    out = dict(g)
    for k, v in cfg["as_built"].items():
        out[k] = int(v) if k in GEOMETRY_KEYS_INT else float(v) * MM
    return out


# ----------------------------------------------------------------------------
# Analysis of a geometry at an operating point
# ----------------------------------------------------------------------------
def recess_regime(g: dict, alpha_deg: float, band_deg: float = 3.0) -> tuple[float, str]:
    phi_r = 2 * math.degrees(math.atan2(g["d_gas_out"] / 2 - g["d_out"] / 2, g["recess"]))
    if abs(phi_r - 2 * alpha_deg) <= band_deg:
        return phi_r, "critical"
    return phi_r, ("inner mixing" if phi_r < 2 * alpha_deg else "outer mixing")


def ox_feed_pressure_for_flow(cfg: dict, g: dict, m_ox: float, p_down: float,
                              T: float | None, quality: float | None) -> tuple[float, OxInlet, bool]:
    A = g["n_ox_holes"] * math.pi * g["d_ox_hole"] ** 2 / 4
    cd = cfg["ox_holes"]["discharge_coefficient"]

    def resid(p0):
        inl = ox_inlet_state(cfg, p0, T=T, quality=quality)
        G, _ = ox_mass_flux(cfg, inl, p_down)
        return cd * A * G - m_ox

    p_hi = 500 * BAR
    if T is None:  # saturated feed: stay below the critical pressure
        p_hi = 0.995 * state(cfg["oxidizer"]["fluid"]).p_critical()
    if resid(p_hi) < 0:
        raise ValueError(f"oxidizer flow {m_ox * 1e3:.0f} g/s not reachable with feed pressure up to "
                         f"{p_hi / BAR:.0f} bar through the given holes")
    p0 = brentq(resid, p_down * (1 + 1e-6), p_hi, xtol=1.0)
    inl = ox_inlet_state(cfg, p0, T=T, quality=quality)
    _, choked = ox_mass_flux(cfg, inl, p_down)
    return p0, inl, choked


def analyse_point(cfg: dict, g: dict, point: dict) -> dict:
    op, oc, lim = cfg["operating_point"], cfg["oxidizer"], cfg["limits"]
    frac = point.get("fraction", 1.0)
    pc = point.get("chamber_pressure_bar", op["chamber_pressure_bar"] * frac) * BAR
    of = point.get("of_ratio", op["of_ratio"])
    if "ox_quality" in point:            # saturated feed at the given quality
        T_ox, q_ox = None, point["ox_quality"]
    elif "ox_temperature_K" in point:
        T_ox, q_ox = point["ox_temperature_K"], None
    else:
        T_ox, q_ox = oc["temperature_K"], (oc["quality"] if oc["temperature_K"] is None else None)
    m_ox, m_f = per_element_flows(cfg, frac, of, point.get("mdot_total_kg_s"))
    warn = []

    # fuel side (properties at an estimated feed pressure, one refinement)
    rho_f, visc_f = fuel_props(cfg, pc + cfg["fuel"]["pressure_drop_bar"] * BAR * frac ** 2)
    sw = swirl_analyse(g, m_f, rho_f, visc_f, cfg)
    rho_f, visc_f = fuel_props(cfg, pc + sw.dp)
    sw = swirl_analyse(g, m_f, rho_f, visc_f, cfg)

    # oxidizer side: fixed point on annulus density <-> feed pressure
    A_gas = math.pi * ((g["d_gas_out"] / 2) ** 2 - (g["d_out"] / 2 + g["post_thickness"]) ** 2)
    p_down, p0 = pc, pc + cfg["oxidizer"]["pressure_drop_bar"] * BAR * frac ** 2
    for _ in range(6):
        inl_guess = ox_inlet_state(cfg, p0, T=T_ox, quality=q_ox)
        rho_c, v_ox = gas_exit_state(cfg, pc, inl_guess.h, m_ox, area=A_gas)
        p_down = pc + 0.5 * rho_c * v_ox ** 2
        p0_new, inl, choked = ox_feed_pressure_for_flow(cfg, g, m_ox, p_down, T_ox, q_ox)
        if abs(p0_new - p0) < 100.0:
            p0 = p0_new
            break
        p0 = p0_new

    # films and momentum flux ratio
    r_n = g["d_out"] / 2
    films = {"theory": sw.film_theory}
    for name, C in FILM_CORRELATIONS.items():
        films[name] = film_correlation(C, g["d_out"], m_f, visc_f, rho_f, sw.dp)
    J = {}
    for name, t in films.items():
        vf = axial_film_velocity(m_f, rho_f, r_n, t)
        J[name] = rho_c * v_ox ** 2 / (rho_f * vf ** 2)
    v_f_sl = axial_film_velocity(m_f, rho_f, r_n, films["Suyari-Lefebvre"])
    alpha_ref = cfg["calibration"]["spray_half_angle_measured_deg"] or sw.alpha_deg
    phi_r, regime = recess_regime(g, alpha_ref)

    stiff_f, stiff_o = sw.dp / pc, (p0 - pc) / pc
    if stiff_f < lim["min_stiffness"]:
        warn.append(f"fuel stiffness {stiff_f:.2f} below {lim['min_stiffness']}")
    if stiff_o < lim["min_stiffness"]:
        warn.append(f"ox stiffness {stiff_o:.2f} below {lim['min_stiffness']}")
    if min(J.values()) < lim["min_J"]:
        warn.append(f"J down to {min(J.values()):.2f} (below {lim['min_J']}): coarser atomization expected")
    if regime == "critical":
        warn.append("critical recess flow: finest atomization but highest self-pulsation risk")
    if inl.phase in ("two-phase",) or (0 <= inl.quality <= 1):
        warn.append(f"oxidizer two-phase at hole inlet (x = {inl.quality:.2f})")
    if T_ox is not None:
        p_sat = ox_saturation_pressure(cfg, inl.T)
        if p_sat and abs(p0 / p_sat - 1) < 2e-3:
            warn.append(f"oxidizer feed pressure sits on the saturation line ({p_sat / BAR:.1f} bar at {inl.T:.0f} K): "
                        "the required flow lies between vapour and liquid feed; define the inlet by quality instead")
    x_exit = ox_quality_ph(cfg, pc, inl.h - 0.5 * v_ox ** 2)
    if 0 <= x_exit <= 1:
        warn.append(f"oxidizer two-phase at annulus exit (x = {x_exit:.2f}): gas-side J and exit velocity are "
                    "homogeneous-mixture values")
    if choked:
        warn.append("oxidizer holes choked")

    return {
        "fraction": frac, "pc_bar": pc / BAR, "of_ratio": of,
        "mdot_fuel_g_s": m_f * 1e3, "mdot_ox_g_s": m_ox * 1e3,
        "fuel_dp_bar": sw.dp / BAR, "fuel_feed_bar": (pc + sw.dp) / BAR, "fuel_stiffness": stiff_f,
        "fuel_cd": sw.mu, "fuel_Re_inlet": sw.Re_in, "spray_half_angle_pred_deg": sw.alpha_deg,
        "ox_feed_bar": p0 / BAR, "ox_dp_bar": (p0 - pc) / BAR, "ox_stiffness": stiff_o,
        "ox_inlet_T_K": inl.T, "ox_inlet_phase": inl.phase, "ox_holes_choked": choked,
        "ox_exit_density": rho_c, "ox_exit_velocity_m_s": v_ox, "ox_exit_quality": x_exit,
        "film_mm": {k: v / MM for k, v in films.items()},
        "fuel_axial_velocity_m_s_SL": v_f_sl, "G_SL": v_ox / v_f_sl,
        "J": J, "J_min": min(J.values()), "J_max": max(J.values()), "J_SL": J["Suyari-Lefebvre"],
        "recess_angle_deg": phi_r, "recess_regime": regime, "warnings": warn,
    }


def evaluate_test_point(cfg: dict, g: dict, pc: float, m_f: float, p_fuel: float, T_fuel: float,
                        m_ox: float, p_ox: float, T_ox: float | None = None, x_ox: float | None = None) -> dict:
    """Compare measured values of one element with the model (all SI, per element).

    p_fuel / p_ox are the manifold pressures directly upstream of the swirl inlets / metering holes.
    Returns measured flow coefficients, the suggested calibration and model predictions at the
    measured flows.
    """
    c = deep_merge(cfg, {"fuel": {"temperature_K": T_fuel}})
    c_uncal = deep_merge(c, {"calibration": {"swirl_cd_factor": 1.0}})
    out = {}

    # fuel side
    rho_f, visc_f = fuel_props(c, p_fuel)
    dp_f = p_fuel - pc
    A_out = math.pi * g["d_out"] ** 2 / 4
    sw0 = swirl_analyse(g, m_f, rho_f, visc_f, c_uncal)
    sw = swirl_analyse(g, m_f, rho_f, visc_f, c)
    cd_meas = m_f / (A_out * math.sqrt(2 * rho_f * dp_f)) if dp_f > 0 else float("nan")
    out.update(fuel_dp_meas_bar=dp_f / BAR, fuel_dp_pred_bar=sw.dp / BAR, fuel_cd_meas=cd_meas,
               fuel_cd_model=sw0.mu_model, swirl_cd_factor_suggested=cd_meas / sw0.mu_model,
               fuel_stiffness_meas=dp_f / pc, rho_fuel=rho_f)

    # oxidizer side
    inl = ox_inlet_state(c, p_ox, T=T_ox, quality=x_ox)
    A_gas = math.pi * ((g["d_gas_out"] / 2) ** 2 - (g["d_out"] / 2 + g["post_thickness"]) ** 2)
    rho_c, v_ox = gas_exit_state(c, pc, inl.h, m_ox, area=A_gas)
    p_down = pc + 0.5 * rho_c * v_ox ** 2
    G, choked = ox_mass_flux(c, inl, p_down)
    A_h = g["n_ox_holes"] * math.pi * g["d_ox_hole"] ** 2 / 4
    out.update(ox_dp_meas_bar=(p_ox - pc) / BAR, ox_hole_cd_meas=m_ox / (A_h * G) if G > 0 else float("nan"),
               ox_holes_choked=choked, ox_inlet_T_K=inl.T, ox_inlet_phase=inl.phase, ox_inlet_quality=inl.quality,
               ox_exit_density=rho_c, ox_exit_velocity_m_s=v_ox, ox_stiffness_meas=(p_ox - pc) / pc)
    try:
        p0_pred, _, _ = ox_feed_pressure_for_flow(c, g, m_ox, p_down, T_ox if x_ox is None else None,
                                                  x_ox if x_ox is not None else None)
        out["ox_feed_pred_bar"] = p0_pred / BAR
    except ValueError:
        out["ox_feed_pred_bar"] = float("nan")

    # momentum flux ratio at the measured point (Suyari-Lefebvre film, measured fuel dp)
    t = film_correlation(FILM_CORRELATIONS["Suyari-Lefebvre"], g["d_out"], m_f, visc_f, rho_f, max(dp_f, 1.0))
    v_f = axial_film_velocity(m_f, rho_f, g["d_out"] / 2, t)
    out.update(J_SL=rho_c * v_ox ** 2 / (rho_f * v_f ** 2), of_ratio=m_ox / m_f,
               spray_half_angle_pred_deg=sw.alpha_deg)
    return out


def check_manufacturing(g: dict, cfg: dict) -> list:
    dmin = cfg["manufacturing"]["min_hole_diameter_mm"] * MM
    w = []
    for k in ("d_inlet", "d_ox_hole"):
        if g[k] < dmin:
            w.append(f"{k} = {g[k] / MM:.2f} mm below minimum {dmin / MM:.2f} mm")
    return w


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
GEOM_LABELS = [
    ("n_inlets", "Tangential fuel inlets", "-"), ("d_inlet", "Fuel inlet diameter", "mm"),
    ("l_inlet", "Fuel inlet length", "mm"), ("d_swirl_chamber", "Swirl chamber diameter", "mm"),
    ("l_swirl_chamber", "Swirl chamber length", "mm"), ("d_out", "Fuel outlet diameter", "mm"),
    ("l_out", "Fuel outlet length", "mm"), ("post_thickness", "Post (lip) thickness", "mm"),
    ("d_gas_out", "Gas annulus outer diameter", "mm"), ("recess", "Recess length", "mm"),
    ("n_ox_holes", "Oxidizer metering holes", "-"), ("d_ox_hole", "Oxidizer hole diameter", "mm"),
]


def fmt_geom(g: dict, k: str) -> str:
    return f"{g[k]:d}" if k in GEOMETRY_KEYS_INT else f"{g[k] / MM:.3f}"


def build_report(cfg, g_sized, g_used, info, points, ref) -> str:
    L = []
    op = cfg["operating_point"]
    m_ox, m_f = per_element_flows(cfg)
    L.append(f"# LCSC element: {cfg['name']}\n")
    L.append(f"Swirl method: {cfg['swirl']['method']}, oxidizer hole model: {cfg['ox_holes']['model']}, "
             f"elements: {cfg['elements']}\n")
    L.append("## Design point (per element)\n")
    L.append(f"- Chamber pressure {op['chamber_pressure_bar']:.1f} bar, O/F {op['of_ratio']:.2f}")
    L.append(f"- Fuel {m_f * 1e3:.1f} g/s ({cfg['fuel']['fluid']}, {cfg['fuel']['temperature_K']:.0f} K, "
             f"rho {info['rho_fuel']:.0f} kg/m3, mu {info['visc_fuel'] * 1e3:.3f} mPa s), "
             f"dp {cfg['fuel']['pressure_drop_bar']:.1f} bar")
    oi = info["ox_inlet"]
    L.append(f"- Oxidizer {m_ox * 1e3:.1f} g/s, hole inlet {oi['p'] / BAR:.1f} bar / {oi['T']:.0f} K "
             f"({oi['phase']}, rho {oi['rho']:.1f} kg/m3), dp {cfg['oxidizer']['pressure_drop_bar']:.1f} bar")
    L.append(f"- Annulus exit density {info['rho_ox_exit']:.1f} kg/m3 at "
             f"{cfg['gas_annulus']['exit_velocity_m_s']:.0f} m/s; recess angle basis "
             f"{info['recess_angle_used_deg']:.1f} deg (half-angle)\n")

    L.append("## Geometry\n")
    has_ab = bool(cfg["as_built"])
    hdr = "| Feature | Sized | Rounded" + (" | As-built (analysed)" if has_ab else "") + " | Unit |"
    L.append(hdr)
    L.append("|---|---|---" + ("|---" if has_ab else "") + "|---|")
    g_round = round_geometry(g_sized, cfg)
    for k, lab, u in GEOM_LABELS:
        row = f"| {lab} | {fmt_geom(g_sized, k)} | {fmt_geom(g_round, k)}"
        if has_ab:
            mark = " *" if k in cfg["as_built"] else ""
            row += f" | {fmt_geom(g_used, k)}{mark}"
        L.append(row + f" | {u} |")
    if has_ab:
        L.append("\n\\* value taken from as_built\n")

    s = info["swirl_used"]
    L.append("\n## Swirl element (analysed geometry, design flow)\n")
    L.append(f"- Geometric characteristic A = {s['A']:.2f}, with friction A_eq = {s['A_eq']:.2f}, "
             f"R_in/r_n = {s['C']:.2f}, inlet Re = {s['Re_in']:.0f}, friction factor {s['friction']:.4f}")
    L.append(f"- Flow coefficient (outlet area): inviscid {s['mu_ideal']:.4f}, viscous model "
             f"{s['mu_model']:.4f}, used {s['mu']:.4f} (calibration factor "
             f"{cfg['calibration']['swirl_cd_factor']:.2f})")
    L.append(f"- Predicted spray half-angle {s['alpha_deg']:.1f} deg (swirl element alone, J = 0), "
             f"filling coefficient {s['phi']:.3f}\n")

    L.append("## Operating envelope\n")
    L.append("| Thrust frac. | pc bar | Fuel dp bar | Fuel stiff. | Ox feed bar | Ox stiff. | Ox choked | "
             "v_ox m/s | rho_ox kg/m3 | Film S-L mm | J (S-L) | J range | Recess |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in points:
        L.append(f"| {p['fraction']:.2f} | {p['pc_bar']:.1f} | {p['fuel_dp_bar']:.2f} | {p['fuel_stiffness']:.2f} | "
                 f"{p['ox_feed_bar']:.1f} | {p['ox_stiffness']:.2f} | {'yes' if p['ox_holes_choked'] else 'no'} | "
                 f"{p['ox_exit_velocity_m_s']:.0f} | {p['ox_exit_density']:.1f} | "
                 f"{p['film_mm']['Suyari-Lefebvre']:.3f} | {p['J_SL']:.2f} | {p['J_min']:.2f}-{p['J_max']:.2f} | "
                 f"{p['recess_regime']} ({p['recess_angle_deg']:.0f} deg) |")

    L.append("\n## Reference: thesis (Nardi) sizing at the same design point\n")
    L.append("| Feature | Thesis method | Unit |")
    L.append("|---|---|---|")
    for k in ("d_inlet", "d_swirl_chamber", "d_out"):
        lab = dict((a, b) for a, b, _ in GEOM_LABELS)[k]
        L.append(f"| {lab} | {ref['geometry'][k] / MM:.3f} | mm |")
    L.append(f"| Flow coefficient assumed by thesis method | {ref['cd_thesis']:.4f} | - |")
    L.append(f"| Flow coefficient predicted by viscous model | {ref['swirl'].mu:.4f} | - |")
    L.append(f"| Fuel dp at design flow, viscous model | {ref['swirl'].dp / BAR:.1f} | bar |")

    warn = info["manufacturing_warnings"] + [f"[{p['fraction']:.2f}] {w}" for p in points for w in p["warnings"]]
    L.append("\n## Warnings\n")
    L.extend([f"- {w}" for w in warn] or ["- none"])
    L.append("\n## Notes\n")
    L.append("- J uses the axial film velocity from each film model; S-L = Suyari-Lefebvre (thesis convention). "
             "Thesis cold-flow data: SMD falls steeply up to J ~ 0.5 and levels out from J ~ 1.")
    L.append("- The viscous swirl model over-predicted the thesis SLA injector flow by ~25 %; set "
             "calibration.swirl_cd_factor from measured data.")
    if cfg.get("baseline_info"):
        b = cfg["baseline_info"]
        L.append(f"- Operating point and throttle points from the engine table of {b['file']} "
                 f"(revision {b['revision']}): fraction = fraction of rated thrust, fuel flow fixed, "
                 "oxidizer throttled.")
    else:
        L.append("- Throttle points scale both flows at the nominal O/F; chamber pressure at part "
                 "thrust defaults to fraction x nominal unless given per point.")
    return "\n".join(L) + "\n"


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(cfg: dict, out_dir: str | None = None, quiet: bool = False) -> dict:
    g_sized, info = size_element(cfg)
    g_used = apply_as_built(round_geometry(g_sized, cfg), cfg)

    op, fc = cfg["operating_point"], cfg["fuel"]
    m_ox, m_f = per_element_flows(cfg)
    pc = op["chamber_pressure_bar"] * BAR
    rho_f, visc_f = fuel_props(cfg, pc + fc["pressure_drop_bar"] * BAR)
    info["swirl_used"] = asdict(swirl_analyse(g_used, m_f, rho_f, visc_f, cfg))
    info["manufacturing_warnings"] = check_manufacturing(g_used, cfg)

    ref_cfg = deep_merge(cfg, {"calibration": {"swirl_cd_factor": 1.0}})
    g_ref = size_swirl_nardi(m_f, rho_f, fc["pressure_drop_bar"] * BAR, ref_cfg)
    X = 0.0042 * cfg["swirl"]["spray_half_angle_deg"] ** 1.2714
    ref = {"geometry": g_ref, "cd_thesis": math.sqrt((1 - X) ** 3 / (1 + X)),
           "swirl": swirl_analyse(g_ref, m_f, rho_f, visc_f, ref_cfg)}

    points = [analyse_point(cfg, g_used, p) for p in cfg["throttle_points"]]
    report = build_report(cfg, g_sized, g_used, info, points, ref)

    result = {"name": cfg["name"], "config": cfg,
              "geometry_sized_mm": {k: (v if k in GEOMETRY_KEYS_INT else v / MM) for k, v in g_sized.items()},
              "geometry_analysed_mm": {k: (v if k in GEOMETRY_KEYS_INT else v / MM) for k, v in g_used.items()},
              "design_info": info, "operating_points": points,
              "thesis_reference": {"geometry_mm": {k: (v if k in GEOMETRY_KEYS_INT else v / MM)
                                                   for k, v in g_ref.items()},
                                   "cd_thesis": ref["cd_thesis"], "swirl": asdict(ref["swirl"])}}
    if not quiet:
        print(report)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, cfg["name"])
        with open(base + "_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        with open(base + "_results.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(result), f, indent=2)
        with open(base + "_sweep.csv", "w", newline="", encoding="utf-8") as f:
            cols = ["fraction", "pc_bar", "mdot_fuel_g_s", "mdot_ox_g_s", "fuel_dp_bar", "fuel_feed_bar",
                    "fuel_stiffness", "fuel_cd", "spray_half_angle_pred_deg", "ox_feed_bar", "ox_dp_bar",
                    "ox_stiffness", "ox_inlet_T_K", "ox_inlet_phase", "ox_holes_choked", "ox_exit_density",
                    "ox_exit_velocity_m_s", "fuel_axial_velocity_m_s_SL", "J_SL", "J_min", "J_max",
                    "recess_angle_deg", "recess_regime"]
            w = csv.writer(f)
            w.writerow(cols)
            for p in points:
                w.writerow([p[c] for c in cols])
        if not quiet:
            print(f"Written: {base}_report.md, _results.json, _sweep.csv")
    return result


# ----------------------------------------------------------------------------
# Built-in checks
# ----------------------------------------------------------------------------
def self_check() -> bool:
    ok = True

    def check(label, value, target, tol):
        nonlocal ok
        good = abs(value - target) <= tol
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {label}: {value:.3f} (expected {target} +/- {tol})")

    print("1) Nardi relation reproduces the Abramovich spray-angle relation")
    for a in (40, 50, 60):
        X_pl = 0.0042 * a ** 1.2714
        check(f"X at {a} deg (theory vs power law {X_pl:.3f})", 1 - phi_of_angle(a), X_pl, 0.01)

    base = deep_merge(DEFAULTS, {
        "operating_point": {"chamber_pressure_bar": 25, "mdot_total_kg_s": 1.0, "of_ratio": 4.0},
        "fuel": {"pressure_drop_bar": 15, "temperature_K": 300},
        "oxidizer": {"temperature_K": 500, "pressure_drop_bar": 15},
        "swirl": {"method": "nardi"}, "gas_annulus": {"recess_angle_source": "design"},
        "manufacturing": {"round_to_mm": 0}})
    print("2) Thesis method reproduces P03 (3 elements) and P04 (6 elements)")
    # Gas side: this script evaluates the annulus exit density at pc and the throttled stagnation
    # enthalpy (slight Joule-Thomson cooling) instead of pc and 500 K, so d_gas_out / recess differ
    # by about 0.1 mm (2 % denser exit gas) from the original sizing.
    for n, tgt in ((3, {"d_out": 4.52, "d_swirl_chamber": 14.92, "d_inlet": 0.98, "d_gas_out": 12.49, "recess": 2.30}),
                   (6, {"d_out": 3.20, "d_swirl_chamber": 10.56, "d_inlet": 0.70, "d_gas_out": 8.97, "recess": 1.67})):
        g, _ = size_element(deep_merge(base, {"elements": n}))
        for k, v in tgt.items():
            check(f"{n} el. {k} [mm]", g[k] / MM, v, 0.02 if k in ("d_out", "d_swirl_chamber", "d_inlet") else 0.12)

    print("3) Viscous model vs thesis LCSC T3 cold flow (water 20 g/s, measured C_D ~0.057)")
    g_t = {"n_inlets": 3, "d_inlet": 0.74 * MM, "d_out": 3.4 * MM, "d_swirl_chamber": 11.2 * MM,
           "l_inlet": 3 * 0.74 * MM}
    r = swirl_analyse(g_t, 0.020, 998.0, 1.0e-3, base)
    print(f"  predicted C_D {r.mu:.4f} (thesis method assumed 0.0854, measured ~0.057), "
          f"predicted half-angle {r.alpha_deg:.1f} deg (measured 50-55)")
    check("predicted / measured C_D (within 30 %)", r.mu / 0.057, 1.0, 0.3)

    print("4) HEM: gaseous N2O 40 bar / 500 K to 25 bar, real-gas isentropic flux")
    inl = ox_inlet_state(base, 40 * BAR, T=500)
    G, choked, _ = hem_mass_flux("N2O", inl, 25 * BAR)
    check("mass flux [kg/m2s]", G, 8573, 60)
    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="LCSC injector element sizing and analysis")
    ap.add_argument("config", nargs="?", help="YAML input file")
    ap.add_argument("-o", "--out", help="output folder (default: results/ next to this script)")
    ap.add_argument("--check", action="store_true", help="run built-in validation cases")
    a = ap.parse_args(argv)
    if a.check:
        sys.exit(0 if self_check() else 1)
    if not a.config:
        ap.error("config file required (or --check)")
    cfg = load_config(a.config)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    run(cfg, out)


if __name__ == "__main__":
    main()