"""
n2o_press.py — N2O tank pressurization analysis (two-phase, two-node).

Physical situation
------------------
N2O at 5–25 degC has Psat = 35–57 bar. Externally pressurized to 100 bar, the
liquid is *subcooled*, but the ullage is a real-gas mixture of pressurant (N2)
and N2O vapor whose partial pressure is pinned to saturation at the liquid
surface. As liquid drains, the ullage grows and N2O continuously evaporates
into it — with sat-vapor densities of 100–190 kg/m3 this is a first-order mass
sink AND a strong evaporative-cooling term on the liquid. Cooling lowers
Psat(T_liq), which raises the required N2 partial pressure: the pressurant
budget and the liquid temperature are tightly coupled. This solver resolves
that coupling.

Model (two nodes + dry-wall node, constant total pressure P_set)
----------------------------------------------------------------
Ullage (well mixed, Dalton partial-pressure mixing of real components):
    P_vap = min( Psat(T_liq), Psat(T_ull) )   # cannot supersaturate the bulk
    P_N2  = P_set - P_vap
    m_vap = rho_N2O(P_vap, T_ull) * V_ull     # sat vapor at the colder T if T_ull < T_liq
    m_N2  = rho_N2 (P_N2,  T_ull) * V_ull

    When T_ull >= T_liq the interface pins P_vap = Psat(T_liq) (superheated bulk).
    When T_ull < T_liq the bulk cannot hold that vapor: P_vap = Psat(T_ull) and
    excess N2O rains back as sat liquid at T_ull (condensation). Using Psat(T_liq)
    in that regime is thermodynamically invalid and understates N2 demand.

Ullage energy (open system, constant P, enthalpy-correct form):
    dU_ull = dm_N2*h_N2,xfer + dm_vap*h_v,xfer + (Q_wall - Q_int)*dt - P*dV_ull
    h_N2,xfer = h_in if dm_N2>=0 else h_N2(P_N2, T_ull)
    h_v,xfer  = h_v,sat(T_liq) if dm_vap>=0 else h_l,sat(T_ull)

Liquid energy (constant P; subcooled liquid enthalpy, not sat h_fg):
    evap:  m*cp*dT = -dm_vap*(h_v,sat(T_l) - h_l(P_set,T_l)) + Q
    cond:  m*cp*dT =  |dm_vap|*(h_l,sat(T_ull) - h_l(P_set,T_l)) + Q

Numerics: staggered per-step update with fixed-point iteration (default 3),
which converges fast because the coupling is smooth; the ullage energy update
solves U(T) = U_target with Newton using CoolProp (dU/dT)|P derivatives.

Known model limitations (documented, not modeled):
  * N2 dissolution into liquid N2O ("supercharging") — applied in the main
    analysis as a configurable mass-fraction bottle draw, not a flash.
  * Interface kinetics: instantaneous equilibrium / bulk condensation. When
    T_ull < T_liq the bulk is held at Psat(T_ull) (not metastable Psat(T_liq)).
  * Liquid node is bulk-mixed (no surface stratification node).
  * N2O transport properties unavailable in CoolProp -> ullage convection
    uses N2 transport properties with mixture mass density in Ra.

Geometry/property helpers duplicated from ethanol_press.py by design — the
solvers stay independent (project convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

from CoolProp.CoolProp import PropsSI

# =============================================================================
# Cached properties
# =============================================================================

@lru_cache(maxsize=200_000)
def _props(out: str, p_pa_r: int, t_mk: int, fluid: str) -> float:
    return PropsSI(out, "P", float(p_pa_r), "T", t_mk / 1000.0, fluid)


def props(out: str, p_pa: float, t_k: float, fluid: str) -> float:
    return _props(out, int(round(p_pa)), int(round(t_k * 1000.0)), fluid)


@lru_cache(maxsize=20_000)
def _sat(out: str, t_mk: int, q: int, fluid: str) -> float:
    return PropsSI(out, "T", t_mk / 1000.0, "Q", q, fluid)


def sat(out: str, t_k: float, q: int, fluid: str) -> float:
    return _sat(out, int(round(t_k * 1000.0)), q, fluid)


# =============================================================================
# Capsule geometry (duplicate of ethanol_press.TankGeom — intentional)
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
        R, L = self.radius_m, self.cyl_length_m
        v_dome = (2.0 / 3.0) * math.pi * R**3
        v_cyl = math.pi * R**2 * L
        v = min(max(v_ullage_m3, 0.0), self.volume_m3)
        if v <= v_dome:
            x = _cap_height_from_volume(v, R)
            return math.pi * x * (2.0 * R - x), 2.0 * math.pi * R * x
        if v <= v_dome + v_cyl:
            h = (v - v_dome) / (math.pi * R**2)
            return math.pi * R**2, 2.0 * math.pi * R**2 + 2.0 * math.pi * R * h
        v_liq = self.volume_m3 - v
        x = _cap_height_from_volume(v_liq, R)
        a_dry = (4.0 * math.pi * R**2 - 2.0 * math.pi * R * x
                 + 2.0 * math.pi * R * L)
        return math.pi * x * (2.0 * R - x), a_dry

    def ullage_height(self, v_ullage_m3: float) -> float:
        R, L = self.radius_m, self.cyl_length_m
        v_dome = (2.0 / 3.0) * math.pi * R**3
        v_cyl = math.pi * R**2 * L
        v = min(max(v_ullage_m3, 1e-9), self.volume_m3)
        if v <= v_dome:
            return _cap_height_from_volume(v, R)
        if v <= v_dome + v_cyl:
            return R + (v - v_dome) / (math.pi * R**2)
        return 2.0 * R + L - _cap_height_from_volume(self.volume_m3 - v, R)


def _cap_height_from_volume(v: float, R: float) -> float:
    if v <= 0.0:
        return 0.0
    x = R * (3.0 * v / (2.0 * math.pi * R**3)) ** (1.0 / 3.0)
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
# Natural convection — transport props from the pressurant gas (see header)
# =============================================================================

_G = 9.80665


def _h_natural_vertical(p_pa, t_gas, t_surf, length, fluid,
                        rho_mix: float | None = None) -> float:
    tf, dT = 0.5 * (t_gas + t_surf), abs(t_gas - t_surf)
    if dT < 1e-6 or length <= 0.0:
        return 0.0
    rho = rho_mix if rho_mix is not None else props("D", p_pa, tf, fluid)
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


def _h_natural_horizontal(p_pa, t_gas, t_surf, area, perimeter, fluid,
                          rho_mix: float | None = None) -> float:
    dT = abs(t_gas - t_surf)
    if dT < 1e-6 or area <= 0.0:
        return 0.0
    L = area / max(perimeter, 1e-9)
    tf = 0.5 * (t_gas + t_surf)
    rho = rho_mix if rho_mix is not None else props("D", p_pa, tf, fluid)
    mu = props("V", p_pa, tf, fluid)
    k = props("L", p_pa, tf, fluid)
    cp = props("C", p_pa, tf, fluid)
    beta = props("isobaric_expansion_coefficient", p_pa, tf, fluid)
    nu = mu / rho
    ra = max(_G * beta * dT * L**3 / (nu * (k / (rho * cp))), 1.0)
    nu_n = 0.27 * ra ** 0.25 if t_gas > t_surf else 0.15 * ra ** (1.0 / 3.0)
    return nu_n * k / L


# =============================================================================
# Config / state
# =============================================================================

@dataclass(frozen=True)
class N2OTankConfig:
    geom: TankGeom
    fluid_liq: str            # 'NitrousOxide'
    fluid_gas: str            # 'Nitrogen'
    p_set_pa: float
    t_prop_k: float
    t_amb_k: float
    mdot_out_kg_s: float
    m_liq0_kg: float          # total N2O loaded (liquid + ullage vapor)
    wall_thickness_m: float
    wall_rho_cp: float
    fixed_point_iters: int = 3
    evap_model: str = "equilibrium"   # 'equilibrium' | 'frozen' (bracketing)
    ht_mode: str = "auto"
    h_gw_fixed: float = 15.0
    h_gl_fixed: float = 10.0
    h_ext: float = 5.0


@dataclass
class N2OTankState:
    t_s: float
    m_liq: float
    t_liq: float
    t_ull: float
    t_wall: float
    m_n2: float
    m_vap: float
    m_n2_cum: float = 0.0
    m_evap_cum: float = 0.0


# =============================================================================
# Helpers
# =============================================================================

_VAP_SUPERHEAT_EPS = 0.1  # K of superheat below which sat-vapor props are used
_T_N2O_MIN = 185.0        # K; just above the N2O triple point for sat queries
_T_N2O_TCRIT = 309.40     # K; just below CoolProp's numerical critical point


def _t_sat_ok(t_k: float) -> float:
    return min(max(t_k, _T_N2O_MIN), _T_N2O_TCRIT)


def _p_vap_eq(t_liq: float, t_ull: float, fluid: str) -> float:
    """Bulk vapor pressure: interface wants Psat(T_liq), but a subcritical
    well-mixed ullage cannot exceed Psat(T_ull). Above Tcrit there is no
    saturation cap — P_vap stays at the interface value."""
    p_if = sat("P", _t_sat_ok(t_liq), 0, fluid)
    if t_ull < t_liq and t_ull < _T_N2O_TCRIT:
        return min(p_if, sat("P", _t_sat_ok(t_ull), 0, fluid))
    return p_if


def _rho_vap(p_vap: float, t_ull: float, t_liq: float, fluid: str) -> float:
    """N2O vapor density at its partial pressure. Superheated evaluation when
    the ullage is warmer than the liquid surface; otherwise sat vapor at the
    colder of (T_ull, T_liq) — the temperature that actually sets P_vap."""
    if t_ull > t_liq + _VAP_SUPERHEAT_EPS:
        try:
            return props("D", p_vap, t_ull, fluid)
        except ValueError:
            pass
    return sat("D", _t_sat_ok(min(t_ull, t_liq)), 1, fluid)


def _u_vap(p_vap: float, t_ull: float, t_liq: float, fluid: str) -> float:
    if t_ull > t_liq + _VAP_SUPERHEAT_EPS:
        try:
            return props("U", p_vap, t_ull, fluid)
        except ValueError:
            pass
    return sat("U", _t_sat_ok(min(t_ull, t_liq)), 1, fluid)


def _cv_vap(p_vap: float, t_ull: float, t_liq: float, fluid: str) -> float:
    """(dU/dT)|P of the vapor; near saturation fall back to a slightly
    superheated state — only used as a Newton slope, accuracy uncritical."""
    try:
        t_eval = max(t_ull, t_liq + 0.5, _T_N2O_MIN)
        return props("d(U)/d(T)|P", p_vap * 0.995, t_eval, fluid)
    except ValueError:
        return sat("C", _t_sat_ok(min(t_ull, t_liq)), 1, fluid)


def _h_fg_subcooled(p_set: float, t_liq: float, fluid: str) -> float:
    """Enthalpy to take subcooled liquid at (P_set, T_liq) to sat vapor at T_liq.
    Larger than sat h_fg because the compressed liquid has lower enthalpy."""
    h_v = sat("H", _t_sat_ok(t_liq), 1, fluid)
    try:
        h_l = props("H", p_set, t_liq, fluid)
    except ValueError:
        h_l = sat("H", _t_sat_ok(t_liq), 0, fluid)
    return h_v - h_l


def _ullage_internal_energy(p_set, t_liq, t_ull, m_n2, m_vap,
                            f_gas, f_liq, p_vap: float | None = None) -> float:
    if p_vap is None:
        p_vap = _p_vap_eq(t_liq, t_ull, f_liq)
    p_n2 = max(p_set - p_vap, 1e4)
    u_n2 = props("U", p_n2, t_ull, f_gas)
    return m_n2 * u_n2 + m_vap * _u_vap(p_vap, t_ull, t_liq, f_liq)


def _ullage_du_dT(p_set, t_liq, t_ull, m_n2, m_vap, f_gas, f_liq,
                  p_vap: float | None = None) -> float:
    if p_vap is None:
        p_vap = _p_vap_eq(t_liq, t_ull, f_liq)
    p_n2 = max(p_set - p_vap, 1e4)
    c_n2 = props("d(U)/d(T)|P", p_n2, t_ull, f_gas)
    c_eff = m_n2 * c_n2 + m_vap * _cv_vap(p_vap, t_ull, t_liq, f_liq)
    return max(c_eff, (m_n2 + m_vap) * 300.0)


# =============================================================================
# Init / step
# =============================================================================

def init_state(cfg: N2OTankConfig) -> tuple[N2OTankState, float]:
    """State after isothermal pre-pressurization: tank is self-pressurized at
    Psat(T0) (ullage = saturated N2O vapor), then N2 is added up to P_set.
    `cfg.m_liq0_kg` is the total N2O loaded; it is split so
    m_liq + m_vap = m_liq0 (vapor comes from the loaded mass).
    Conservative (max-density) N2 charge. Returns (state, prepress_N2_mass)."""
    fl, T0, P, V = cfg.fluid_liq, cfg.t_prop_k, cfg.p_set_pa, cfg.geom.volume_m3
    p_vap0 = sat("P", T0, 0, fl)
    if p_vap0 >= P:
        raise ValueError("Psat exceeds tank set pressure — not a pressurized-"
                         "subcooled configuration at this temperature.")
    rho_l = props("D", P, T0, fl)
    rho_v = sat("D", T0, 1, fl)
    if rho_v >= rho_l:
        raise ValueError("N2O vapor density >= liquid density at init.")
    # m_liq + rho_v*(V - m_liq/rho_l) = M  =>  m_liq = (M - rho_v*V)/(1 - rho_v/rho_l)
    m_liq = (cfg.m_liq0_kg - rho_v * V) / (1.0 - rho_v / rho_l)
    v_u0 = V - m_liq / rho_l
    if m_liq <= 0.0 or v_u0 <= 0.0:
        raise ValueError("Liquid load exceeds tank volume (or vapor fill "
                         "consumes the entire load).")
    m_vap0 = rho_v * v_u0
    m_n20 = props("D", P - p_vap0, T0, cfg.fluid_gas) * v_u0
    st = N2OTankState(t_s=0.0, m_liq=m_liq, t_liq=T0,
                      t_ull=T0, t_wall=T0,
                      m_n2=m_n20, m_vap=m_vap0)
    return st, m_n20


def _dmv_from_constraint(rho_v: float, rho_l: float, V: float,
                         m_liq_drained: float, m_vap_old: float) -> tuple[float, float]:
    """Exact ullage-volume / vapor-mass constraint (linear in dmv).
    v_u = V - (m_liq_drained - dmv)/rho_l ;  dmv = rho_v*v_u - m_vap_old."""
    v_u0 = V - m_liq_drained / rho_l
    gain = rho_v / max(rho_l, 1e-12)
    if gain < 0.95:
        dmv = (rho_v * v_u0 - m_vap_old) / (1.0 - gain)
    else:
        dmv = 0.0
        for _ in range(8):
            v_u0 = V - (m_liq_drained - dmv) / rho_l
            dmv = rho_v * v_u0 - m_vap_old
    v_u = V - (m_liq_drained - dmv) / rho_l
    return dmv, v_u


def _solve_liquid(cfg: N2OTankConfig, st: N2OTankState, dt: float,
                  m_liq_drained: float, t_ull: float, q_ext_w: float):
    """Implicit liquid-temperature update with equilibrium evaporation /
    bulk condensation. Safeguarded Newton on

        F(T_l) = m_avg*cp_l*(T_l - T_l_old) + dmv*h_xfer(T_l) - Q_ext*dt

    Returns (t_liq, d_m_vap, m_liq_new, v_u, p_vap, h_fg)."""
    fl, P, V = cfg.fluid_liq, cfg.p_set_pa, cfg.geom.volume_m3

    def evaluate(tl: float):
        rho_l = props("D", P, tl, fl)
        p_vap = _p_vap_eq(tl, t_ull, fl)
        rho_v = _rho_vap(p_vap, t_ull, tl, fl)
        dmv, v_u = _dmv_from_constraint(rho_v, rho_l, V, m_liq_drained, st.m_vap)
        h_l = props("H", P, tl, fl)
        if dmv >= 0.0:
            h_xfer = _h_fg_subcooled(P, tl, fl)
            h_fg = h_xfer
        elif t_ull >= tl:
            # Interface condensation: latent heat is released to the liquid.
            try:
                h_v_bulk = props("H", p_vap, t_ull, fl)
            except ValueError:
                h_v_bulk = sat("H", _t_sat_ok(tl), 1, fl)
            h_xfer = h_v_bulk - h_l
            h_fg = h_xfer
        else:
            # Bulk/fog condensation: rain-back is sat liquid at T_ull.
            h_drop = sat("H", _t_sat_ok(t_ull), 0, fl)
            h_xfer = h_drop - h_l
            h_fg = sat("H", _t_sat_ok(tl), 1, fl) - sat("H", _t_sat_ok(tl), 0, fl)
        cp_l = props("C", P, tl, fl)
        m_avg = max(0.5 * (st.m_liq + m_liq_drained - dmv), 0.05)
        f = m_avg * cp_l * (tl - st.t_liq) + dmv * h_xfer - q_ext_w * dt
        return f, dmv, v_u, p_vap, h_fg

    tl = st.t_liq
    f, dmv, v_u, p_vap, h_fg = evaluate(tl)
    for _ in range(15):
        f2 = evaluate(tl + 0.01)[0]
        df = (f2 - f) / 0.01
        d_t = -f / df if abs(df) > 1e-9 else 0.0
        d_t = max(min(d_t, 2.0), -2.0)
        tl += d_t
        f, dmv, v_u, p_vap, h_fg = evaluate(tl)
        if abs(d_t) < 1e-6:
            break
    m_liq_new = m_liq_drained - dmv
    if m_liq_new <= 0.0:
        raise RuntimeError(
            f"n2o_press: liquid depleted at t={st.t_s:.2f}s "
            f"(load too small for burn + evaporation losses).")
    return tl, dmv, m_liq_new, v_u, p_vap, h_fg


def step(cfg: N2OTankConfig, st: N2OTankState, dt: float,
         h_in: float) -> tuple[N2OTankState, float, dict]:
    """Advance one step. Outer fixed-point loop over (heat transfer, ullage
    temperature, wall); inner implicit Newton solve for the liquid node with
    equilibrium evaporation / bulk condensation. Ullage energy Newton includes
    m_N2(T_ull) so the committed state satisfies rho_N2(P_N2, T)*V = m_N2.
    Returns (state, mdot_N2, diagnostics)."""
    fl, fg = cfg.fluid_liq, cfg.fluid_gas
    P = cfg.p_set_pa
    frozen = cfg.evap_model == "frozen"

    p_vap_old = _p_vap_eq(st.t_liq, st.t_ull, fl)
    u_old = _ullage_internal_energy(P, st.t_liq, st.t_ull, st.m_n2, st.m_vap,
                                    fg, fl, p_vap_old)
    rho_l_old = props("D", P, st.t_liq, fl)
    v_u_old = cfg.geom.volume_m3 - st.m_liq / rho_l_old

    t_liq, t_ull, t_wall = st.t_liq, st.t_ull, st.t_wall
    m_liq_drained = st.m_liq - cfg.mdot_out_kg_s * dt
    if m_liq_drained <= 0.0:
        raise RuntimeError(f"n2o_press: liquid depleted at t={st.t_s:.2f}s "
                           f"(load too small for burn + evaporation losses).")
    m_liq_new, d_m_vap, d_m_n2 = m_liq_drained, 0.0, 0.0
    m_n2_new = st.m_n2
    m_vap_new = st.m_vap
    p_vap = p_vap_old
    p_n2 = P - p_vap
    q_int = q_amb_liq = 0.0
    v_u = v_u_old
    rho_l = rho_l_old
    h_fg = 0.0
    h_gw = h_gl = 0.0
    diag: dict = {}

    for _ in range(max(cfg.fixed_point_iters, 1)):
        # --- liquid node (implicit, with evaporation / condensation) ---------
        if frozen:
            d_m_vap = 0.0
            m_liq_new = m_liq_drained
            rho_l = props("D", P, t_liq, fl)
            v_u = cfg.geom.volume_m3 - m_liq_new / rho_l
            p_vap = _p_vap_eq(st.t_liq, t_ull, fl)
            h_fg = 0.0
            cp_l = props("C", P, t_liq, fl)
            t_liq = st.t_liq + (q_int + q_amb_liq) * dt / (st.m_liq * cp_l)
        else:
            t_liq, d_m_vap, m_liq_new, v_u, p_vap, h_fg = _solve_liquid(
                cfg, st, dt, m_liq_drained, t_ull, q_int + q_amb_liq)
            rho_l = props("D", P, t_liq, fl)
        p_n2 = max(P - p_vap, 1e4)
        m_vap_new = st.m_vap + d_m_vap

        # --- heat transfer (mixture density in Ra) ---------------------------
        a_int, a_dry = cfg.geom.ullage_geometry(v_u)
        perim = 2.0 * math.pi * math.sqrt(max(a_int / math.pi, 1e-12))
        h_ull_len = cfg.geom.ullage_height(v_u)
        rho_mix = (st.m_n2 + m_vap_new) / max(v_u, 1e-12)
        if cfg.ht_mode == "off":
            h_gw = h_gl = 0.0
        elif cfg.ht_mode == "fixed":
            h_gw, h_gl = cfg.h_gw_fixed, cfg.h_gl_fixed
        else:
            h_gw = _h_natural_vertical(p_n2, t_ull, t_wall, h_ull_len, fg,
                                       rho_mix)
            h_gl = _h_natural_horizontal(p_n2, t_ull, t_liq, a_int, perim, fg,
                                         rho_mix)
        q_int = h_gl * a_int * (t_ull - t_liq)        # ullage -> liquid
        q_wall = h_gw * a_dry * (t_wall - t_ull)      # wall  -> ullage
        a_wet = (4.0 * math.pi * cfg.geom.radius_m**2
                 + 2.0 * math.pi * cfg.geom.radius_m * cfg.geom.cyl_length_m
                 - a_dry)
        q_amb_liq = cfg.h_ext * a_wet * (cfg.t_amb_k - t_liq)

        # --- ullage energy: U(T) = U_target with composition lagged, then
        #     refresh m_N2 at the new T (outer fixed-point closes the lag).
        p_vap = _p_vap_eq(t_liq, t_ull, fl)
        p_n2 = max(P - p_vap, 1e4)
        m_n2_new = props("D", p_n2, t_ull, fg) * v_u
        d_m_n2 = m_n2_new - st.m_n2
        if d_m_vap >= 0.0:
            h_v_xfer = sat("H", _t_sat_ok(t_liq), 1, fl)
        elif t_ull >= t_liq:
            try:
                h_v_xfer = props("H", p_vap, t_ull, fl)
            except ValueError:
                h_v_xfer = sat("H", _t_sat_ok(t_liq), 1, fl)
        else:
            h_v_xfer = sat("H", _t_sat_ok(t_ull), 0, fl)
        h_n2_xfer = h_in if d_m_n2 >= 0.0 else props("H", p_n2, t_ull, fg)
        dV = v_u - v_u_old
        u_target = (u_old + d_m_n2 * h_n2_xfer + d_m_vap * h_v_xfer
                    + (q_wall - q_int) * dt - P * dV)
        for _ in range(4):
            u_cur = _ullage_internal_energy(P, t_liq, t_ull, m_n2_new,
                                            m_vap_new, fg, fl, p_vap)
            c_eff = _ullage_du_dT(P, t_liq, t_ull, m_n2_new, m_vap_new,
                                  fg, fl, p_vap)
            d_t = (u_target - u_cur) / c_eff
            t_ull = max(t_ull + max(min(d_t, 3.0), -3.0), _T_N2O_MIN)

        p_vap = _p_vap_eq(t_liq, t_ull, fl)
        p_n2 = max(P - p_vap, 1e4)
        m_n2_new = props("D", p_n2, t_ull, fg) * v_u
        d_m_n2 = m_n2_new - st.m_n2

        if abs(t_ull - st.t_ull) > 30.0:
            raise RuntimeError(
                f"n2o_press: ullage energy solve diverging at t={st.t_s:.2f}s "
                f"(T_ull {st.t_ull:.1f} -> {t_ull:.1f} K). Reduce dt.")

        # --- dry-wall node ----------------------------------------------------
        c_area = cfg.wall_rho_cp * cfg.wall_thickness_m
        _, a_dry2 = cfg.geom.ullage_geometry(v_u + 1e-5)
        dA_dt = max((a_dry2 - a_dry) / 1e-5, 0.0) * (dV / dt if dt > 0 else 0.0)
        m_w = c_area * max(a_dry, 1e-6)
        t_wall = st.t_wall + dt * (h_gw * a_dry * (t_ull - st.t_wall)
                                   + cfg.h_ext * a_dry * (cfg.t_amb_k - st.t_wall)
                                   + dA_dt * c_area * (t_liq - st.t_wall)) / m_w

        diag = {"p_vap_pa": p_vap, "p_n2_pa": p_n2, "v_ullage_m3": v_u,
                "h_gw": h_gw, "h_gl": h_gl, "q_int_W": q_int,
                "h_fg_J_kg": h_fg, "rho_liq": rho_l,
                "mdot_evap_kg_s": d_m_vap / dt, "m_vap_kg": m_vap_new}

    mdot_n2 = d_m_n2 / dt
    if mdot_n2 < -1e-9:
        diag["warning_overpressure_tendency"] = True

    new = replace(st,
                  t_s=st.t_s + dt,
                  m_liq=m_liq_new,
                  t_liq=t_liq, t_ull=t_ull, t_wall=t_wall,
                  m_n2=m_n2_new,
                  m_vap=m_vap_new,
                  m_n2_cum=st.m_n2_cum + max(d_m_n2, 0.0),
                  m_evap_cum=st.m_evap_cum + max(d_m_vap, 0.0))
    return new, mdot_n2, diag


def run_standalone(cfg: N2OTankConfig, t_burn: float, dt: float,
                   h_in: float | None = None) -> dict:
    if h_in is None:
        h_in = props("H", cfg.p_set_pa, cfg.t_amb_k, cfg.fluid_gas)
    st, m_pre = init_state(cfg)
    keys = ("t", "t_liq", "t_ull", "t_wall", "mdot_n2", "m_n2_cum",
            "m_evap_cum", "p_vap", "p_n2", "v_ullage", "rho_liq", "mdot_evap")
    ts = {k: [] for k in keys}
    for _ in range(int(round(t_burn / dt))):
        st, mdot, d = step(cfg, st, dt, h_in)
        ts["t"].append(st.t_s); ts["t_liq"].append(st.t_liq)
        ts["t_ull"].append(st.t_ull); ts["t_wall"].append(st.t_wall)
        ts["mdot_n2"].append(mdot); ts["m_n2_cum"].append(st.m_n2_cum)
        ts["m_evap_cum"].append(st.m_evap_cum)
        ts["p_vap"].append(d["p_vap_pa"]); ts["p_n2"].append(d["p_n2_pa"])
        ts["v_ullage"].append(d["v_ullage_m3"])
        ts["rho_liq"].append(d["rho_liq"])
        ts["mdot_evap"].append(d["mdot_evap_kg_s"])
    return {"prepress_mass_kg": m_pre,
            "expulsion_mass_kg": st.m_n2_cum,
            "total_mass_kg": m_pre + st.m_n2_cum,
            "evaporated_n2o_kg": st.m_evap_cum,
            "final_state": st, "timeseries": ts}