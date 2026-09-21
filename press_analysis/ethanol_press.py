"""
ethanol_press.py — Liquid-propellant tank pressurization analysis (single-phase).

Physical model
--------------
Constant-pressure expulsion of a liquid (negligible-vapor-pressure) propellant
from a capsule tank (cylinder + 2 hemispherical domes), pressurant fed by a
regulator holding total tank pressure P_set.

Control volumes:
  * Ullage gas (real-gas pressurant via CoolProp), open system at constant P.
  * Dry tank wall (lumped node; grows as liquid level drops, newly exposed
    wall enters at liquid temperature).
  * Liquid: treated as isothermal reservoir at T_liq (large thermal mass,
    short burn; propellant vapor contributes only its partial pressure).

Ullage energy balance (open system, constant pressure), exact differential form:

    d(m u)/dt = mdot_in * h_in + Qdot - P * dV/dt
    m = rho(P_g, T) * V                      (real-gas constraint)

Eliminating mdot_in gives a single ODE in ullage temperature T:

    dT/dt = [ rho * Vdot * (h_in - h_g) + Qdot ]
            / [ V * (d rho/dT)|P * (u_g - h_in) + m * (du/dT)|P ]

(check: adiabatic with h_in == h_g  ->  dT/dt = 0, as expected for ideal gas
inflow at ullage temperature.)

Conventions: SI everywhere. Pure stateless functions; state passed explicitly.
Geometry helpers are intentionally duplicated in n2o_press.py to keep the
solvers independent (project convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

from CoolProp.CoolProp import PropsSI

# =============================================================================
# Cached property access (CoolProp calls dominate runtime; key on rounded P, T)
# =============================================================================

@lru_cache(maxsize=200_000)
def _props(out: str, p_pa_r: int, t_mk: int, fluid: str) -> float:
    return PropsSI(out, "P", float(p_pa_r), "T", t_mk / 1000.0, fluid)


def props(out: str, p_pa: float, t_k: float, fluid: str) -> float:
    """PropsSI('X','P',p,'T',T,fluid) cached on (1 Pa, 1 mK) grid."""
    return _props(out, int(round(p_pa)), int(round(t_k * 1000.0)), fluid)


@lru_cache(maxsize=10_000)
def psat(t_k: float, fluid: str) -> float:
    return PropsSI("P", "T", round(t_k, 4), "Q", 0, fluid)


# =============================================================================
# Capsule tank geometry: cylinder (length L) + two hemispherical domes (radius R)
# =============================================================================

@dataclass(frozen=True)
class TankGeom:
    radius_m: float
    cyl_length_m: float

    @property
    def volume_m3(self) -> float:
        R, L = self.radius_m, self.cyl_length_m
        return (4.0 / 3.0) * math.pi * R**3 + math.pi * R**2 * L

    def ullage_geometry(self, v_ullage_m3: float) -> tuple[float, float]:
        """For ullage volume measured from the tank top, return
        (interface_area_m2, dry_wall_area_m2). Vertical tank, domes up/down."""
        R, L = self.radius_m, self.cyl_length_m
        v_dome = (2.0 / 3.0) * math.pi * R**3
        v_cyl = math.pi * R**2 * L
        v = min(max(v_ullage_m3, 0.0), self.volume_m3)

        if v <= v_dome:  # interface inside top dome: spherical cap of height x
            x = _cap_height_from_volume(v, R)
            a_int = math.pi * x * (2.0 * R - x)          # chord area = pi*a^2
            a_dry = 2.0 * math.pi * R * x                # spherical cap area
        elif v <= v_dome + v_cyl:  # interface in cylinder
            h = (v - v_dome) / (math.pi * R**2)
            a_int = math.pi * R**2
            a_dry = 2.0 * math.pi * R**2 + 2.0 * math.pi * R * h
        else:  # interface in bottom dome (cap of liquid remaining)
            v_liq = self.volume_m3 - v
            x = _cap_height_from_volume(v_liq, R)        # liquid cap height
            a_int = math.pi * x * (2.0 * R - x)
            a_dry = (4.0 * math.pi * R**2 - 2.0 * math.pi * R * x
                     + 2.0 * math.pi * R * L)
        return a_int, a_dry

    def ullage_height(self, v_ullage_m3: float) -> float:
        """Vertical extent of the ullage (characteristic length for wall convection)."""
        R, L = self.radius_m, self.cyl_length_m
        v_dome = (2.0 / 3.0) * math.pi * R**3
        v_cyl = math.pi * R**2 * L
        v = min(max(v_ullage_m3, 1e-9), self.volume_m3)
        if v <= v_dome:
            return _cap_height_from_volume(v, R)
        if v <= v_dome + v_cyl:
            return R + (v - v_dome) / (math.pi * R**2)
        v_liq = self.volume_m3 - v
        return 2.0 * R + L - _cap_height_from_volume(v_liq, R)


def _cap_height_from_volume(v: float, R: float) -> float:
    """Invert V = pi*x^2*(3R - x)/3 for cap height x in [0, R] (Newton)."""
    if v <= 0.0:
        return 0.0
    x = R * (3.0 * v / (2.0 * math.pi * R**3)) ** (1.0 / 3.0)  # small-cap seed
    for _ in range(40):
        f = math.pi * x * x * (3.0 * R - x) / 3.0 - v
        df = math.pi * x * (2.0 * R - x)
        if df <= 0.0:
            break
        x_new = min(max(x - f / df, 0.0), R)
        if abs(x_new - x) < 1e-12:
            return x_new
        x = x_new
    return x


# =============================================================================
# Natural convection (gas side). Transport properties: pressurant gas.
# =============================================================================

_G = 9.80665


def _h_natural_vertical(p_pa, t_gas, t_surf, length, fluid) -> float:
    """Churchill–Chu, vertical surface (dry wall <-> ullage gas)."""
    tf = 0.5 * (t_gas + t_surf)
    dT = abs(t_gas - t_surf)
    if dT < 1e-6 or length <= 0.0:
        return 0.0
    rho = props("D", p_pa, tf, fluid)
    mu = props("V", p_pa, tf, fluid)
    k = props("L", p_pa, tf, fluid)
    cp = props("C", p_pa, tf, fluid)
    beta = props("isobaric_expansion_coefficient", p_pa, tf, fluid)
    nu = mu / rho
    pr = cp * mu / k
    ra = _G * beta * dT * length**3 / (nu * (k / (rho * cp)))
    nu_n = (0.825 + 0.387 * ra ** (1.0 / 6.0)
            / (1.0 + (0.492 / pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    return nu_n * k / length


def _h_natural_horizontal(p_pa, t_gas, t_surf, area, perimeter, fluid) -> float:
    """Horizontal plate (liquid surface <-> ullage gas above it).
    Gas hotter than surface -> stable stratification (Nu = 0.27 Ra^1/4);
    gas colder -> unstable (Nu = 0.15 Ra^1/3)."""
    dT = abs(t_gas - t_surf)
    if dT < 1e-6 or area <= 0.0:
        return 0.0
    L = area / max(perimeter, 1e-9)
    tf = 0.5 * (t_gas + t_surf)
    rho = props("D", p_pa, tf, fluid)
    mu = props("V", p_pa, tf, fluid)
    k = props("L", p_pa, tf, fluid)
    cp = props("C", p_pa, tf, fluid)
    beta = props("isobaric_expansion_coefficient", p_pa, tf, fluid)
    nu = mu / rho
    ra = _G * beta * dT * L**3 / (nu * (k / (rho * cp)))
    ra = max(ra, 1.0)
    if t_gas > t_surf:          # warm gas above cold surface: stable
        nu_n = 0.27 * ra ** 0.25
    else:                        # cold gas above warm surface: unstable
        nu_n = 0.15 * ra ** (1.0 / 3.0)
    return nu_n * k / L


# =============================================================================
# Configuration / state containers
# =============================================================================

@dataclass(frozen=True)
class LiquidTankConfig:
    geom: TankGeom
    fluid_liq: str            # CoolProp name, e.g. 'Ethanol'
    fluid_gas: str            # pressurant, e.g. 'Nitrogen'
    p_set_pa: float
    t_prop_k: float           # initial propellant temperature
    t_amb_k: float
    mdot_out_kg_s: float
    m_liq0_kg: float
    wall_thickness_m: float
    wall_rho_cp: float        # rho * cp of wall material [J/(m^3 K)]
    ht_mode: str = "auto"     # 'auto' | 'fixed' | 'off'
    h_gw_fixed: float = 15.0
    h_gl_fixed: float = 10.0
    h_ext: float = 5.0


@dataclass
class LiquidTankState:
    t_s: float
    m_liq: float
    t_liq: float
    t_gas: float
    t_wall: float
    m_gas: float              # pressurant mass in ullage
    a_dry_prev: float
    m_gas_cum: float = 0.0    # cumulative pressurant fed during expulsion


# =============================================================================
# Model core
# =============================================================================

def p_gas_partial(cfg: LiquidTankConfig, t_liq: float) -> float:
    """Pressurant partial pressure: total minus propellant vapor pressure
    (Dalton; ethanol Psat < 0.1 bar — a <0.1 % correction, kept for rigor)."""
    return cfg.p_set_pa - psat(t_liq, cfg.fluid_liq)


def init_state(cfg: LiquidTankConfig) -> tuple[LiquidTankState, float]:
    """Initial state after isothermal pre-pressurization to P_set.
    Returns (state, prepress_pressurant_mass). Isothermal charge is the
    conservative assumption (maximum gas density -> maximum mass)."""
    rho_l = props("D", cfg.p_set_pa, cfg.t_prop_k, cfg.fluid_liq)
    v_u0 = cfg.geom.volume_m3 - cfg.m_liq0_kg / rho_l
    if v_u0 <= 0.0:
        raise ValueError("Liquid load exceeds tank volume.")
    p_g = p_gas_partial(cfg, cfg.t_prop_k)
    m_g0 = props("D", p_g, cfg.t_prop_k, cfg.fluid_gas) * v_u0
    _, a_dry = cfg.geom.ullage_geometry(v_u0)
    st = LiquidTankState(t_s=0.0, m_liq=cfg.m_liq0_kg, t_liq=cfg.t_prop_k,
                         t_gas=cfg.t_prop_k, t_wall=cfg.t_prop_k,
                         m_gas=m_g0, a_dry_prev=a_dry)
    return st, m_g0


def _derivs(cfg: LiquidTankConfig, st: LiquidTankState, h_in: float,
            outflow: bool = False):
    """Time derivatives (dT_gas/dt, mdot_in, dT_wall/dt, Vdot_u, diagnostics).
    `outflow=True` uses ullage enthalpy for mass leaving (regulator cannot
    extract; the first-law boundary enthalpy is h_g, not the bottle h_in)."""
    g = cfg.fluid_gas
    p_g = p_gas_partial(cfg, st.t_liq)

    rho_l = props("D", cfg.p_set_pa, st.t_liq, cfg.fluid_liq)
    v_u = cfg.geom.volume_m3 - st.m_liq / rho_l
    vdot = cfg.mdot_out_kg_s / rho_l                      # ullage growth rate

    rho_g = props("D", p_g, st.t_gas, g)
    u_g = props("U", p_g, st.t_gas, g)
    h_g = props("H", p_g, st.t_gas, g)
    drho_dT = props("d(D)/d(T)|P", p_g, st.t_gas, g)
    du_dT = props("d(U)/d(T)|P", p_g, st.t_gas, g)
    h_xfer = h_g if outflow else h_in

    a_int, a_dry = cfg.geom.ullage_geometry(v_u)
    perim = 2.0 * math.pi * math.sqrt(max(a_int / math.pi, 1e-12))
    h_ull = cfg.geom.ullage_height(v_u)

    if cfg.ht_mode == "off":
        h_gw = h_gl = 0.0
    elif cfg.ht_mode == "fixed":
        h_gw, h_gl = cfg.h_gw_fixed, cfg.h_gl_fixed
    else:
        h_gw = _h_natural_vertical(p_g, st.t_gas, st.t_wall, h_ull, g)
        h_gl = _h_natural_horizontal(p_g, st.t_gas, st.t_liq, a_int, perim, g)

    q_wall = h_gw * a_dry * (st.t_wall - st.t_gas)        # -> gas
    q_int = h_gl * a_int * (st.t_liq - st.t_gas)          # -> gas
    q_dot = q_wall + q_int

    denom = v_u * drho_dT * (u_g - h_xfer) + (rho_g * v_u) * du_dT
    dTg = (rho_g * vdot * (h_xfer - h_g) + q_dot) / denom
    mdot_in = v_u * drho_dT * dTg + rho_g * vdot

    # Dry-wall node: convection both sides + advection of newly exposed wall
    c_wall_area = cfg.wall_rho_cp * cfg.wall_thickness_m  # J/(m^2 K)
    # rate of newly exposed wall area: dA/dt = (dA/dV) * Vdot via finite step
    _, a_dry2 = cfg.geom.ullage_geometry(v_u + vdot * 1e-2)
    dA_dt = max((a_dry2 - a_dry) / 1e-2, 0.0)
    m_w = c_wall_area * max(a_dry, 1e-6)                  # J/K
    dTw = (h_gw * a_dry * (st.t_gas - st.t_wall)
           + cfg.h_ext * a_dry * (cfg.t_amb_k - st.t_wall)
           + dA_dt * c_wall_area * (st.t_liq - st.t_wall)) / m_w

    diag = {"p_gas_pa": p_g, "v_ullage_m3": v_u, "h_gw": h_gw, "h_gl": h_gl,
            "q_dot_w": q_dot, "a_dry_m2": a_dry, "rho_liq": rho_l}
    return dTg, mdot_in, dTw, vdot, diag


def _derivs_consistent(cfg: LiquidTankConfig, st: LiquidTankState, h_in: float):
    """Evaluate derivatives; if the inflow assumption yields ṁ < 0, redo
    the energy balance with ullage enthalpy as the leaving-stream enthalpy."""
    dTg, mdot, dTw, vdot, diag = _derivs(cfg, st, h_in, outflow=False)
    if mdot < 0.0:
        dTg, mdot, dTw, vdot, diag = _derivs(cfg, st, h_in, outflow=True)
    return dTg, mdot, dTw, vdot, diag


def step(cfg: LiquidTankConfig, st: LiquidTankState, dt: float,
         h_in: float) -> tuple[LiquidTankState, float, dict]:
    """Advance one step (Heun / explicit trapezoidal).
    Returns (new_state, pressurant_mass_flow_kg_s, diagnostics)."""
    dTg1, mdot1, dTw1, _, diag = _derivs_consistent(cfg, st, h_in)
    pred = replace(st,
                   t_s=st.t_s + dt,
                   m_liq=st.m_liq - cfg.mdot_out_kg_s * dt,
                   t_gas=st.t_gas + dTg1 * dt,
                   t_wall=st.t_wall + dTw1 * dt,
                   m_gas=st.m_gas + mdot1 * dt)
    dTg2, mdot2, dTw2, _, _ = _derivs_consistent(cfg, pred, h_in)

    mdot = 0.5 * (mdot1 + mdot2)
    new = replace(st,
                  t_s=st.t_s + dt,
                  m_liq=st.m_liq - cfg.mdot_out_kg_s * dt,
                  t_gas=st.t_gas + 0.5 * (dTg1 + dTg2) * dt,
                  t_wall=st.t_wall + 0.5 * (dTw1 + dTw2) * dt,
                  m_gas=st.m_gas + mdot * dt,
                  m_gas_cum=st.m_gas_cum + max(mdot, 0.0) * dt,
                  a_dry_prev=diag["a_dry_m2"])
    return new, mdot, diag


def run_standalone(cfg: LiquidTankConfig, t_burn: float, dt: float,
                   h_in: float | None = None) -> dict:
    """Convenience runner with constant inlet enthalpy (default: pressurant
    at ambient temperature, tank pressure)."""
    if h_in is None:
        h_in = props("H", cfg.p_set_pa, cfg.t_amb_k, cfg.fluid_gas)
    st, m_pre = init_state(cfg)
    ts = {k: [] for k in ("t", "t_gas", "t_wall", "mdot_n2", "m_n2_cum",
                          "v_ullage", "p_gas")}
    n = int(round(t_burn / dt))
    for _ in range(n):
        st, mdot, d = step(cfg, st, dt, h_in)
        ts["t"].append(st.t_s); ts["t_gas"].append(st.t_gas)
        ts["t_wall"].append(st.t_wall); ts["mdot_n2"].append(mdot)
        ts["m_n2_cum"].append(st.m_gas_cum)
        ts["v_ullage"].append(d["v_ullage_m3"]); ts["p_gas"].append(d["p_gas_pa"])
    return {"prepress_mass_kg": m_pre,
            "expulsion_mass_kg": st.m_gas_cum,
            "total_mass_kg": m_pre + st.m_gas_cum,
            "final_state": st, "timeseries": ts}