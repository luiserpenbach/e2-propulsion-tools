"""
Thermal analysis for the E2 torch igniter.
Operating point and geometry come from configs/*.yaml.
Pc is solved from the fixed throat, mass flow and η_c*.

═══════════════════════════════════════════════════════════════════════════════
THEORY SUMMARY
═══════════════════════════════════════════════════════════════════════════════

1. CHAMBER PRESSURE — solved from fixed throat area
   ─────────────────────────────────────────────────
   For a given throat area At, mass flow Ṁ, and combustion efficiency η_c*:

       Pc = Ṁ · c*(OF, Pc) · η_c* / At

   c* is weakly dependent on Pc, so this is solved iteratively (~5 iterations)
   using a damped fixed-point update.  This ensures Pc is thermodynamically
   consistent with the CEA c* at every O/F point.

2. BARTZ CONVECTIVE HEAT TRANSFER (Bartz 1957, Sutton & Biblarz RPE 8th ed.)
   ──────────────────────────────────────────────────────────────────────────
   h_g(x) = (0.026 / D_t^0.2) × (μ^0.2 × Cp / Pr^0.6) × (Pc/c*)^0.8
            × (At/A(x))^0.9 × (Dt/Rc)^0.1 × σ(x)

   Variable-property correction (Bartz σ):
       σ = [0.5·(Tw/Tc)·(1+(γ−1)/2·M²) + 0.5]^(−0.68)
           × [1 + (γ−1)/2·M²]^(−0.12)

   Frozen Cp and chamber viscosity/Prandtl are used throughout — physically
   correct because heat transfer in the boundary layer occurs at approximately
   frozen composition.  Equilibrium Cp diverges at fuel-rich exit conditions
   and must not be used in Bartz or the energy budget.

3. ADIABATIC WALL TEMPERATURE
   ──────────────────────────
   T_aw(x) = Tc · [1 + r·(γ−1)/2·M²] / [1 + (γ−1)/2·M²]

   r = Pr^(1/3)  — turbulent recovery factor.
   Driving heat flux: q(x) = h_g(x) × (T_aw(x) − Tw(x))

4. LOCAL MACH NUMBER — isentropic area–Mach relation (solved via brentq)
   A/At = (1/M)·[(2/(γ+1))·(1+(γ−1)/2·M²)]^((γ+1)/(2(γ−1)))

5. ENERGY BUDGET
   ──────────────
   The wall fraction uses the sensible gas power (frozen Cp, 298 K reference);
   the CEA enthalpy drop P_noz is reported for reference only:

       P_noz  = Ṁ·(H_chamber − H_exit)
       P_wall = Bartz integral over the cold wall
       η_wall = P_wall / P_sens,   P_sens = Ṁ·Cp_frz·(Tc − 298)

   A constant frozen-Cp estimate, Ṁ·Cp·(Tc − 298), is printed beside it.
   Its kinetic / exhaust split is an algebraic identity of that estimate.

6. HEAT SOAK — semi-infinite slab model
   ─────────────────────────────────────
   ΔT_surface(t) = 2·q · √(t / (π·k·ρ·Cp_wall))

   Valid while thermal penetration depth √(α·t) is below the wall thickness δ:
       t_valid = δ² / α

   Time to reach temperature limit:
       t_limit = (π·k·ρ·Cp_wall) · (ΔT_limit / (2·q))²

Dependencies:
    pip install rocketcea plotly kaleido==0.2.1 scipy numpy
"""

import argparse
from itertools import cycle
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rocketcea.cea_obj import CEA_Obj as CEA_base
from rocketcea.cea_obj_w_units import CEA_Obj as CEA_units
from scipy.optimize import brentq

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from igniter_config import DEFAULT_CONFIG, load, write_figure  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS AND PROGRAMME PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

BAR_TO_PSIA = 14.5038
R_UNIV      = 8314.46    # J/(kmol·K)
G0          = 9.80665    # m/s²

# ── Torch igniter operating parameters (replaced by apply_case) ─────────────
OX_NAME   = "N2O"
FUEL_NAME = "C2H5OH"
MDOT      = 0.050        # total mass flow [kg/s]
ETA_CSTAR = 0.92         # combustion efficiency — used to compute Pc from At
EPS       = 1.5          # nozzle area ratio
PA_BAR    = 1.01325      # ambient pressure [bar]
PC_GUESS  = 40.0         # bar — iteration start only

# O/F design points — Pc is computed from fixed At + MDOT + ETA_CSTAR
OF_POINTS = [1.6, 2.0, 2.2]
OF_NOM    = 2.2

# ── Baseline chamber geometry (from Section 3.2.3) ───────────────────────────
DT_M    = 4.40e-3        # throat diameter [m]   — input
AT_M2   = math.pi / 4 * DT_M**2   # throat area [m²]  — derived
DC_M    = 24.9e-3        # chamber diameter [m]
LCYL_M  = 27.1e-3        # cylindrical chamber length [m]
LCONV_M = 10.2e-3        # convergent length [m]
THETA_C = 45.0           # convergent half-angle [deg]
THETA_D = 15.0           # divergent half-angle [deg]
RT_M    = DT_M / 2
RC_M    = DC_M / 2
RC_CURV = DT_M           # throat radius of curvature [m]  (≈ 1×D_t)


def apply_case(cfg) -> None:
    """Copy a YAML case onto the module globals the thermal model reads."""
    global OX_NAME, FUEL_NAME, MDOT, ETA_CSTAR, EPS, PA_BAR, PC_GUESS, OF_POINTS, OF_NOM
    global DT_M, AT_M2, DC_M, LCYL_M, LCONV_M, THETA_C, THETA_D, RT_M, RC_M, RC_CURV
    global WALL_T0, WALL_THICKNESS_M
    OX_NAME = cfg.ox
    FUEL_NAME = cfg.fuel
    MDOT = cfg.mdot_kg_s
    ETA_CSTAR = cfg.eta_cstar
    EPS = cfg.eps
    PA_BAR = cfg.pa_bar
    PC_GUESS = cfg.pc_guess_bar
    OF_POINTS = list(cfg.of_points)
    OF_NOM = cfg.of_nominal
    DT_M = cfg.dt_mm * 1e-3
    AT_M2 = cfg.at_m2
    DC_M = cfg.dc_mm * 1e-3
    LCYL_M = cfg.lcyl_mm * 1e-3
    LCONV_M = cfg.lconv_mm * 1e-3
    THETA_C = cfg.theta_conv_deg
    THETA_D = cfg.theta_div_deg
    RT_M = DT_M / 2.0
    RC_M = DC_M / 2.0
    RC_CURV = cfg.rc_curv_m
    WALL_T0 = cfg.wall_t0_k
    WALL_THICKNESS_M = cfg.wall_thickness_m


# ── Wall material properties ──────────────────────────────────────────────────
WALL_T0          = 293.0    # initial wall temperature [K]
WALL_THICKNESS_M = 2.0e-3   # assumed wall thickness for semi-infinite validity [m]

MATERIALS = {
    "SS316":    dict(k=16.0,  rho=8000, Cp=500, T_limit=1073, label="SS 316"),
    "Copper":   dict(k=400.0, rho=8960, Cp=385, T_limit=1357, label="Copper"),
    "Graphite": dict(k=150.0, rho=2200, Cp=709, T_limit=3000, label="Graphite"),
}

# ── Colour palette (dark theme) ───────────────────────────────────────────────
C = dict(
    bg="#0f1117", surf="#1a1d2e", grid="#2a2d3e", text="#e2e8f0", sub="#94a3b8",
    blue="#4a9eff", green="#4ade80", amber="#fbbf24", red="#f87171",
    pur="#c084fc", ora="#fb923c", teal="#2dd4bf",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CEA INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GasProps:
    """SI combustion gas properties at a given O/F, Pc."""
    of: float;  pc_bar: float
    Tc: float        # K    — chamber stagnation temperature
    Tt: float        # K    — throat static temperature
    Te: float        # K    — nozzle exit static temperature
    cstar: float     # m/s  — characteristic velocity
    Isp_vac: float   # s    — vacuum Isp (ε = EPS)
    gam_c: float     # γ    — chamber
    gam_t: float     # γ    — throat
    Cp_c: float      # J/(kg·K) — equilibrium Cp (diagnostics only)
    Cp_c_frz: float  # J/(kg·K) — frozen Cp, chamber
    Cp_t_frz: float  # J/(kg·K) — frozen Cp, throat
    Cp_e_frz: float  # J/(kg·K) — frozen Cp, exit
    mu_c: float      # Pa·s — chamber viscosity
    mu_t: float      # Pa·s — throat viscosity
    Pr_c: float      # −    — chamber Prandtl
    Pr_t: float      # −    — throat Prandtl
    k_c:  float      # W/(m·K) — chamber conductivity
    k_t:  float      # W/(m·K) — throat conductivity
    H_c:  float      # J/kg — CEA chamber enthalpy
    H_e:  float      # J/kg — CEA exit enthalpy


def build_cea():
    cea_b = CEA_base(oxName=OX_NAME, fuelName=FUEL_NAME)
    cea_u = CEA_units(oxName=OX_NAME, fuelName=FUEL_NAME,
                      pressure_units='Bar', cstar_units='m/s',
                      temperature_units='K', isp_units='sec',
                      enthalpy_units='kJ/kg')
    return cea_b, cea_u


def calc_pc(cea_u, of: float, eta: float = None,
            pc_init: float = None, tol: float = 1e-4, maxiter: int = 40) -> float:
    """
    Iteratively solve Pc [bar] for a fixed throat area, mass flow and η_c*:

        Pc = Ṁ · c*(OF, Pc) · η / At

    c*(Pc) is weakly dependent on Pc so the fixed-point iteration
    converges in ~5 iterations with a damped update.
    """
    if eta is None:
        eta = ETA_CSTAR
    if pc_init is None:
        pc_init = PC_GUESS
    pc = pc_init
    for _ in range(maxiter):
        cstar  = cea_u.get_Cstar(Pc=pc, MR=of)
        pc_new = MDOT * cstar * eta / AT_M2 / 1e5   # bar
        if abs(pc_new - pc) < tol:
            return pc_new
        pc = 0.6 * pc + 0.4 * pc_new
    return pc


def _transport_si(tup):
    """Convert CEA_base transport tuple (Cp, visc, cond, Pr) from CGS to SI."""
    Cp_cgs, visc_mP, cond_mcal, Pr = tup
    return (
        Cp_cgs   * 4184.0,   # cal/(g·K)       → J/(kg·K)
        visc_mP  * 1e-4,     # milliPoise       → Pa·s
        cond_mcal * 0.4184,  # mcal/(cm·s·K)   → W/(m·K)
        Pr,
    )


def get_gas_props(cea_b, cea_u, of: float, pc_bar: float) -> GasProps:
    pc_psia = pc_bar * BAR_TO_PSIA

    # Temperatures (CEA_base returns Rankine → convert to K)
    Tc_R, Tt_R, Te_R = cea_b.get_Temperatures(Pc=pc_psia, MR=of, eps=EPS)
    Tc, Tt, Te = Tc_R / 1.8, Tt_R / 1.8, Te_R / 1.8

    cstar   = cea_u.get_Cstar(Pc=pc_bar, MR=of)
    Isp_vac = cea_u.get_Isp(Pc=pc_bar, MR=of, eps=EPS)
    _, gam_c = cea_u.get_Chamber_MolWt_gamma(Pc=pc_bar, MR=of)
    _, gam_t = cea_u.get_Throat_MolWt_gamma(Pc=pc_bar, MR=of)

    # Equilibrium transport (viscosity and Pr for Bartz)
    Cp_c_eql, mu_c, k_c, Pr_c = _transport_si(
        cea_b.get_Chamber_Transport(Pc=pc_psia, MR=of, eps=EPS, frozen=0))
    _,         mu_t, k_t, Pr_t = _transport_si(
        cea_b.get_Throat_Transport(Pc=pc_psia, MR=of, eps=EPS, frozen=0))

    # Frozen Cp — correct for Bartz and energy budget.
    # Equilibrium Cp diverges at fuel-rich conditions (e.g. 5796 vs 1828
    # J/(kg·K) at O/F=1.6); frozen Cp must be used instead.
    Cp_c_frz, *_ = _transport_si(
        cea_b.get_Chamber_Transport(Pc=pc_psia, MR=of, eps=EPS, frozen=1))
    Cp_t_frz, *_ = _transport_si(
        cea_b.get_Throat_Transport(Pc=pc_psia, MR=of, eps=EPS, frozen=1))
    Cp_e_frz, *_ = _transport_si(
        cea_b.get_Exit_Transport(Pc=pc_psia, MR=of, eps=EPS, frozen=1))

    H_c, _H_t, H_e = cea_u.get_Enthalpies(Pc=pc_bar, MR=of, eps=EPS)

    return GasProps(
        of=of, pc_bar=pc_bar,
        Tc=Tc, Tt=Tt, Te=Te, cstar=cstar, Isp_vac=Isp_vac,
        gam_c=gam_c, gam_t=gam_t,
        Cp_c=Cp_c_eql,
        Cp_c_frz=Cp_c_frz, Cp_t_frz=Cp_t_frz, Cp_e_frz=Cp_e_frz,
        mu_c=mu_c, mu_t=mu_t, Pr_c=Pr_c, Pr_t=Pr_t,
        k_c=k_c, k_t=k_t,
        H_c=float(H_c) * 1e3, H_e=float(H_e) * 1e3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ISENTROPIC FLOW FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def area_mach_residual(M: float, gam: float, AR: float) -> float:
    """Isentropic area–Mach residual  (A/A* − AR)."""
    return (1.0 / M) * ((2 / (gam + 1)) * (1 + (gam - 1) / 2 * M**2))**((gam + 1) / (2 * (gam - 1))) - AR


def mach_from_AR(AR: float, gam: float, subsonic: bool) -> float:
    """Invert the area–Mach relation numerically."""
    if abs(AR - 1.0) < 1e-9:
        return 1.0
    if subsonic:
        return brentq(area_mach_residual, 1e-9, 1.0 - 1e-9, args=(gam, AR))
    else:
        return brentq(area_mach_residual, 1.0 + 1e-9, 20.0, args=(gam, AR))


def T_adiabatic_wall(Tc: float, gam: float, M: float, Pr: float) -> float:
    """
    Adiabatic (recovery) wall temperature.
        T_aw = Tc · (1 + r·(γ−1)/2·M²) / (1 + (γ−1)/2·M²)
    r = Pr^(1/3) is the turbulent recovery factor.
    """
    r = Pr ** (1.0 / 3.0)
    return Tc * (1.0 + r * (gam - 1) / 2 * M**2) / (1.0 + (gam - 1) / 2 * M**2)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BARTZ HEAT TRANSFER COEFFICIENT
# ─────────────────────────────────────────────────────────────────────────────

def bartz_sigma(Tw: float, Tc: float, gam: float, M: float) -> float:
    """
    Variable-property correction factor σ (Bartz 1957):
        σ = [0.5·(Tw/Tc)·(1+(γ−1)/2·M²) + 0.5]^(−0.68)
            × [1 + (γ−1)/2·M²]^(−0.12)
    """
    factor = 0.5 * (Tw / Tc) * (1.0 + (gam - 1) / 2 * M**2) + 0.5
    return factor**(-0.68) * (1.0 + (gam - 1) / 2 * M**2)**(-0.12)


def bartz_hg(
    Dt: float, mu: float, Cp: float, Pr: float,
    Pc_Pa: float, cstar: float,
    AR: float,              # At / A_local  (= 1 at throat, < 1 elsewhere)
    sigma: float,
    Rc_curv: float = None,  # throat radius of curvature [m]; None → Dt
) -> float:
    """
    Bartz (1957) local heat transfer coefficient h_g  [W/(m²·K)]:

        h_g = (0.026/Dt^0.2) × (μ^0.2·Cp/Pr^0.6) × (Pc/c*)^0.8
              × (At/A)^0.9 × (Dt/Rc)^0.1 × σ

    All inputs in SI.  Empirical constant 0.026 valid for 0.1–200 bar,
    Dt up to ~0.5 m.  Expected accuracy ±30–50 % for small chambers.
    """
    if Rc_curv is None:
        Rc_curv = Dt
    Rc_term = (Dt / Rc_curv) ** 0.1
    h0 = (0.026 / Dt**0.2) * (mu**0.2 * Cp / Pr**0.6) * (Pc_Pa / cstar)**0.8
    return h0 * AR**0.9 * Rc_term * sigma


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NOZZLE GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def build_nozzle(n_pts: int = 400) -> dict:
    """
    Build axial nozzle profile arrays.

    AR_inv = At / A(x)  — equals 1 at throat, < 1 away from throat.
    Wall surface area element: dA/dx = 2πr·√(1 + (dr/dx)²)
    """
    At  = AT_M2
    Re_M  = RT_M * math.sqrt(EPS)
    L_div = (Re_M - RT_M) / math.tan(math.radians(THETA_D))

    x_cyl_end = LCYL_M
    x_throat  = LCYL_M + LCONV_M
    x_exit    = x_throat + L_div

    n_c    = n_pts // 4
    n_conv = n_pts // 4
    n_div  = n_pts // 2

    x_cyl  = np.linspace(0,         x_cyl_end, n_c,    endpoint=False)
    x_conv = np.linspace(x_cyl_end, x_throat,  n_conv, endpoint=False)
    x_divg = np.linspace(x_throat,  x_exit,    n_div)
    x_all  = np.concatenate([x_cyl, x_conv, x_divg])

    r   = np.zeros_like(x_all)
    sub = np.ones(len(x_all), dtype=bool)
    sec = np.empty(len(x_all), dtype=object)

    for i, xi in enumerate(x_all):
        if xi < x_cyl_end:
            r[i]   = RC_M
            sec[i] = 'chamber'
        elif xi < x_throat:
            frac   = (xi - x_cyl_end) / LCONV_M
            r[i]   = RC_M + frac * (RT_M - RC_M)
            sec[i] = 'convergent'
        else:
            frac   = (xi - x_throat) / L_div
            r[i]   = RT_M + frac * (Re_M - RT_M)
            sec[i] = 'divergent'
            sub[i] = False

    A      = math.pi * r**2
    AR_inv = At / A   # = 1 at throat, < 1 elsewhere

    dr           = np.gradient(r, x_all)
    dA_per_dx    = 2 * math.pi * r * np.sqrt(1.0 + dr**2)

    return dict(
        x=x_all, r=r, A=A, AR_inv=AR_inv,
        sec=sec, sub=sub, dA_per_dx=dA_per_dx,
        x_throat=x_throat, x_cyl_end=x_cyl_end, x_exit=x_exit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HEAT FLUX DISTRIBUTION AND INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_heat_flux(gp: GasProps, nozzle: dict, Tw: float = None) -> dict:
    """
    Compute Bartz h_g, T_aw, q at every axial station and integrate P_wall.

    Uses frozen Cp and chamber μ/Pr throughout (correct for boundary-layer
    heat transfer at frozen composition).
    """
    if Tw is None:
        Tw = WALL_T0
    Pc_Pa = gp.pc_bar * 1e5
    gam   = gp.gam_c
    Tc    = gp.Tc
    mu    = gp.mu_c
    Cp    = gp.Cp_c_frz
    Pr    = gp.Pr_c

    n   = len(nozzle['x'])
    hg  = np.zeros(n)
    Taw = np.zeros(n)
    q   = np.zeros(n)
    M   = np.zeros(n)

    for i in range(n):
        AR  = nozzle['AR_inv'][i]
        sub = nozzle['sub'][i]
        Mi  = 1.0 if abs(AR - 1.0) < 1e-4 else mach_from_AR(1.0 / AR, gam, subsonic=sub)
        M[i]   = Mi
        Taw[i] = T_adiabatic_wall(Tc, gam, Mi, Pr)
        sig    = bartz_sigma(Tw, Tc, gam, Mi)
        # Pc/c* in Bartz is the throat mass flux. Use the delivered c*
        # (Pc*At/mdot), which is eta times the ideal CEA c*.
        cstar_del = gp.pc_bar * 1e5 * AT_M2 / MDOT
        hg[i]  = bartz_hg(DT_M, mu, Cp, Pr, Pc_Pa, cstar_del,
                           AR, sig, Rc_curv=RC_CURV)
        q[i]   = hg[i] * (Taw[i] - Tw)

    trap = getattr(np, "trapezoid", None) or np.trapz
    P_wall = trap(q * nozzle['dA_per_dx'], nozzle['x'])

    return dict(M=M, hg=hg, Taw=Taw, q=q, P_wall=P_wall)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENERGY BUDGET
# ─────────────────────────────────────────────────────────────────────────────

def energy_budget(gp: GasProps, P_wall: float) -> dict:
    """
    Two views of the gas power, plus the Bartz wall loss.

    Constant frozen-Cp estimate, referenced to 298 K. The split below is an
    algebraic identity of that estimate, not a CEA energy balance:

        P_sens = Ṁ·Cp·(Tc − 298)
        P_KE   = Ṁ·Cp·(Tc − Te)
        P_exh  = Ṁ·Cp·(Te − 298)

    The wall fraction is referenced to the sensible gas power, because the
    Bartz integral covers the whole chamber and nozzle wall, not only the
    expansion from chamber to exit:

        η_wall = P_wall / P_sens
        P_jet  = P_sens − P_wall

    The CEA equilibrium enthalpy drop from chamber to exit,
    P_noz = Ṁ·(H_c − H_e), is reported for reference only. Delivered vacuum effective exhaust speed uses
    η_c* · Isp_CEA · g0 (Cf taken as ideal, so pressure thrust is included).
    """
    T_ref = 298.0
    Cp = gp.Cp_c_frz

    P_sens = MDOT * Cp * (gp.Tc - T_ref)
    P_KE = MDOT * Cp * (gp.Tc - gp.Te)
    P_exh = MDOT * Cp * (gp.Te - T_ref)
    sens_closure = (P_KE + P_exh) / P_sens

    P_noz = MDOT * (gp.H_c - gp.H_e)
    eta_wall = P_wall / P_sens
    P_jet = P_sens - P_wall

    ve_vac = ETA_CSTAR * gp.Isp_vac * G0
    P_ke_vac = 0.5 * MDOT * ve_vac ** 2
    v_exit = math.sqrt(max(2.0 * Cp * (gp.Tc - gp.Te), 0.0))

    return dict(
        P_chem=P_noz, P_sens=P_sens, P_KE=P_KE, P_exh=P_exh,
        P_wall=P_wall, P_jet_real=P_jet,
        eta_wall=eta_wall,
        isentropic_closure=sens_closure,
        device_closure=(P_jet + P_wall) / P_sens,
        v_exit=v_exit, ve_vac=ve_vac, P_ke_vac=P_ke_vac,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  HEAT SOAK — semi-infinite slab
# ─────────────────────────────────────────────────────────────────────────────

def heat_soak_surface_temp(q: float, mat: dict, t_arr: np.ndarray,
                            T0: float = None) -> np.ndarray:
    """
    Inner-wall surface temperature rise: semi-infinite slab under constant q.

        ΔT(t) = 2·q · √(t / (π·k·ρ·Cp_wall))

    Valid while √(α·t) ≪ wall thickness  (i.e. t ≪ t_valid).
    """
    if T0 is None:
        T0 = WALL_T0
    return T0 + 2.0 * q * np.sqrt(t_arr / (math.pi * mat['k'] * mat['rho'] * mat['Cp']))


def t_to_limit(q: float, mat: dict, T0: float = None) -> float:
    """Time [s] for inner wall to reach T_limit under constant heat flux q."""
    if T0 is None:
        T0 = WALL_T0
    dT    = mat['T_limit'] - T0
    coeff = 2.0 * q / math.sqrt(math.pi * mat['k'] * mat['rho'] * mat['Cp'])
    return (dT / coeff) ** 2


def t_slab_valid(mat: dict, delta_m: float = None) -> float:
    """
    Maximum time [s] for which the semi-infinite slab approximation holds.
    Criterion: penetration depth √(α·t) = wall thickness δ  →  t = δ²/α.
    """
    if delta_m is None:
        delta_m = WALL_THICKNESS_M
    alpha = mat['k'] / (mat['rho'] * mat['Cp'])
    return delta_m**2 / alpha


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list) -> None:
    W = 72
    print("=" * W)
    print("  TORCH IGNITER — THERMAL ANALYSIS REPORT")
    print(f"  N₂O / Ethanol  |  Ṁ = {MDOT} kg/s  |  η_c* = {ETA_CSTAR}")
    print(f"  Dt = {DT_M*1e3:.2f} mm  |  At = {AT_M2*1e6:.3f} mm²")
    print("=" * W)

    for res in results:
        gp, hf, eb = res['gp'], res['hf'], res['eb']
        i_t = int(np.argmax(hf['hg']))   # throat: peak h_g

        print(f"\n{'─'*W}")
        print(f"  O/F = {gp.of}   →  Pc = {gp.pc_bar:.2f} bar  (solved from At + η_c*)")
        print(f"{'─'*W}")

        print(f"\n  GAS PROPERTIES (CEA, frozen Cp for Bartz + budget)")
        print(f"    Tc  = {gp.Tc:.0f} K     Tt = {gp.Tt:.0f} K     Te = {gp.Te:.0f} K")
        print(f"    c* CEA / delivered = {gp.cstar:.1f} / {ETA_CSTAR * gp.cstar:.1f} m/s")
        print(f"    Isp_vac CEA / delivered (ε={EPS}) = "
              f"{gp.Isp_vac:.1f} / {ETA_CSTAR * gp.Isp_vac:.1f} s")
        print(f"    γ_c = {gp.gam_c:.4f}   γ_t = {gp.gam_t:.4f}")
        print(f"    Cp_c (equil)  = {gp.Cp_c:.0f} J/(kg·K)  [diagnostics only]")
        print(f"    Cp_c (frozen) = {gp.Cp_c_frz:.0f} J/(kg·K)  [Bartz + energy budget]")
        print(f"    μ_c = {gp.mu_c:.2e} Pa·s   Pr_c = {gp.Pr_c:.4f}   k_c = {gp.k_c:.4f} W/(m·K)")

        print(f"\n  BARTZ HEAT TRANSFER")
        print(f"    h_g,throat  = {hf['hg'][i_t]/1e3:.2f} kW/(m²·K)")
        print(f"    q_throat    = {hf['q'][i_t]/1e6:.3f} MW/m²")
        print(f"    T_aw,throat = {hf['Taw'][i_t]:.0f} K   (Tw = {WALL_T0:.0f} K cold wall)")
        print(f"    σ_throat    = {bartz_sigma(WALL_T0, gp.Tc, gp.gam_c, 1.0):.4f}")
        print(f"    h_g,chamber = {hf['hg'][0]/1e3:.3f} kW/(m²·K)   "
              f"q_chamber = {hf['q'][0]/1e6:.4f} MW/m²")

        print(f"\n  TOTAL WALL THERMAL POWER (cold wall, Tw = {WALL_T0:.0f} K)")
        print(f"    P_wall = {hf['P_wall']/1e3:.3f} kW")

        print(f"\n  ENERGY BUDGET")
        print(f"  ── Constant frozen-Cp estimate (algebraic split) ──")
        print(f"    P_sensible           = {eb['P_sens']/1e3:.2f} kW  (Ṁ·Cp_frz·(Tc−298))")
        print(f"    ├─ P_kinetic         = {eb['P_KE']/1e3:.2f} kW  ({eb['P_KE']/eb['P_sens']*100:.1f}%)")
        print(f"    └─ P_exhaust         = {eb['P_exh']/1e3:.2f} kW  ({eb['P_exh']/eb['P_sens']*100:.1f}%)")
        print(f"    Split check          = {eb['isentropic_closure']*100:.4f}%")
        print(f"  ── CEA enthalpy drop, chamber to exit ──")
        print(f"    P_nozzle             = {eb['P_chem']/1e3:.2f} kW  (Ṁ·(Hc−He))")
        print(f"  ── Wall loss ──")
        print(f"    P_wall  (Bartz)      = {eb['P_wall']/1e3:.3f} kW   η_wall = {eb['eta_wall']*100:.2f}%  (of P_sensible)")
        print(f"    P_jet                = {eb['P_jet_real']/1e3:.2f} kW   (P_sensible − P_wall)")
        print(f"    v_e from frozen Cp   = {eb['v_exit']:.0f} m/s")
        print(f"    v_e,vac delivered    = {eb['ve_vac']:.0f} m/s"
              f"   (½ṁv² = {eb['P_ke_vac']/1e3:.2f} kW, includes pressure thrust)")

        print(f"\n  HEAT SOAK — throat heat flux  "
              f"(semi-infinite slab, δ_wall = {WALL_THICKNESS_M*1e3:.1f} mm)")
        print(f"  {'Material':<12}  {'q [MW/m²]':<12}  {'T_limit [K]':<14}  "
              f"{'t_limit [s]':<14}  t_valid [s]")
        q_t = hf['q'][i_t]
        for mat_name, mat in MATERIALS.items():
            t_lim   = t_to_limit(q_t, mat)
            t_valid = t_slab_valid(mat)
            flag    = "  ⚠ exceeds t_valid" if t_lim > t_valid else ""
            print(f"    {mat['label']:<10}  {q_t/1e6:<12.3f}  {mat['T_limit']:<14}  "
                  f"{t_lim:<14.4f}  {t_valid:.4f}{flag}")

    print(f"\n{'='*W}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PLOTLY VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(results: list, nozzle: dict) -> go.Figure:
    of_colors  = [C['blue'], C['green'], C['amber']]
    mat_colors = [C['red'], C['ora'], C['teal']]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Heat Flux Distribution  q(x)  [MW/m²]',
            'Bartz Coefficient  h<sub>g</sub>(x)  [kW/(m²·K)]',
            'Adiabatic Wall Temperature  T<sub>aw</sub>(x)  [K]',
            'Energy Budget  [kW]',
            f'Heat Soak  T<sub>wall</sub>(t) at Throat  (δ={WALL_THICKNESS_M*1e3:.0f} mm)',
            'Throat Heat Flux vs Wall Temperature  T<sub>w</sub>',
        ],
        vertical_spacing=0.11, horizontal_spacing=0.12,
    )

    x_mm  = nozzle['x'] * 1e3
    x_thr = nozzle['x_throat'] * 1e3
    x_cyl = nozzle['x_cyl_end'] * 1e3

    # ── Panels 1–3: axial distributions ──────────────────────────────────────
    for res, col, ls in zip(results, cycle(of_colors), cycle(['solid', 'dash', 'dot'])):
        gp, hf = res['gp'], res['hf']
        lbl = f"O/F={gp.of}  Pc={gp.pc_bar:.1f} bar"

        fig.add_trace(go.Scatter(x=x_mm, y=hf['q'] / 1e6, name=lbl,
            mode='lines', line=dict(color=col, width=2, dash=ls)),       row=1, col=1)

        fig.add_trace(go.Scatter(x=x_mm, y=hf['hg'] / 1e3, name=lbl,
            mode='lines', line=dict(color=col, width=2, dash=ls),
            showlegend=False),                                             row=1, col=2)

        fig.add_trace(go.Scatter(x=x_mm, y=hf['Taw'], name=lbl,
            mode='lines', line=dict(color=col, width=2, dash=ls),
            showlegend=False),                                             row=2, col=1)

    for r, c_ in [(1, 1), (1, 2), (2, 1)]:
        fig.add_vline(x=x_thr, line=dict(color=C['sub'], dash='dot', width=1),
                      annotation_text=' throat',
                      annotation_font_color=C['sub'], row=r, col=c_)
        fig.add_vline(x=x_cyl, line=dict(color=C['sub'], dash='dash', width=0.8),
                      annotation_text=' L_cyl',
                      annotation_font_color=C['sub'], row=r, col=c_)

    fig.add_hline(y=WALL_T0,
                  line=dict(color=C['red'], dash='dash', width=1),
                  annotation_text=f' Tw = {WALL_T0:.0f} K',
                  annotation_font_color=C['red'], row=2, col=1)

    # ── Panel 4: energy budget ────────────────────────────────────────────────
    # Stacked bar: P_jet_real + P_wall = P_sens (device closure).
    # P_KE and P_exh are isentropic sub-components of P_jet_real and are
    # shown as annotations — NOT stacked on top of P_wall.
    of_lbls     = [f"O/F={r['gp'].of}" for r in results]
    p_jet_real  = [r['eb']['P_jet_real'] / 1e3 for r in results]
    p_wall      = [r['eb']['P_wall']     / 1e3 for r in results]
    eta_wall_pct = [r['eb']['eta_wall'] * 100  for r in results]

    fig.add_trace(go.Bar(name='Jet power  (P<sub>jet,real</sub>)',
        x=of_lbls, y=p_jet_real,
        marker_color=C['blue'], opacity=0.88,
        text=[f"{v:.1f} kW" for v in p_jet_real],
        textposition='inside', textfont=dict(color=C['text'], size=10)),  row=2, col=2)

    fig.add_trace(go.Bar(name='Wall heat loss  (P<sub>wall</sub>)',
        x=of_lbls, y=p_wall,
        marker_color=C['red'], opacity=0.88,
        text=[f"η={e:.2f}%" for e in eta_wall_pct],
        textposition='inside', textfont=dict(color=C['text'], size=10)),  row=2, col=2)

    # Annotate isentropic sub-breakdown above each bar
    p_chem_vals = [r['eb']['P_sens'] / 1e3 for r in results]
    p_ke_pct    = [r['eb']['P_KE']   / r['eb']['P_sens'] * 100 for r in results]
    p_exh_pct   = [r['eb']['P_exh']  / r['eb']['P_sens'] * 100 for r in results]
    for i, (lbl, pc, ke_p, ex_p) in enumerate(zip(of_lbls, p_chem_vals, p_ke_pct, p_exh_pct)):
        fig.add_annotation(
            x=lbl, y=pc + 0.5,
            text=f"Cp split: KE {ke_p:.0f}% | exh {ex_p:.0f}%",
            showarrow=False, row=2, col=2,
            font=dict(size=8.5, color=C['sub']),
            xanchor='center', yanchor='bottom',
        )

    fig.update_layout(barmode='stack')

    # ── Panel 5: heat soak ────────────────────────────────────────────────────
    # Use the throat heat flux at the nominal O/F (closest point if not listed)
    res_nom = min(results, key=lambda r: abs(r['gp'].of - OF_NOM))
    i_t     = int(np.argmax(res_nom['hf']['q']))
    q_thr   = res_nom['hf']['q'][i_t]

    # Extend time axis to just past the longest material t_limit
    t_max   = max(t_to_limit(q_thr, mat) for mat in MATERIALS.values()) * 1.25
    t_arr   = np.linspace(0, t_max, 400)

    for (mat_name, mat), col_m in zip(MATERIALS.items(), mat_colors):
        T_surf  = heat_soak_surface_temp(q_thr, mat, t_arr)
        t_valid = t_slab_valid(mat)
        t_lim   = t_to_limit(q_thr, mat)

        fig.add_trace(go.Scatter(x=t_arr, y=T_surf, name=mat['label'],
            mode='lines', line=dict(color=col_m, width=2)),               row=3, col=1)

        # Material temperature limit line
        fig.add_hline(y=mat['T_limit'],
            line=dict(color=col_m, dash='dot', width=1.2),
            row=3, col=1)

        # Semi-infinite validity boundary (vertical)
        fig.add_vline(x=t_valid,
            line=dict(color=col_m, dash='dash', width=1.0),
            row=3, col=1)

    # Shade the invalid region (beyond strictest validity limit)
    t_valid_min = min(t_slab_valid(mat) for mat in MATERIALS.values())
    fig.add_vrect(x0=t_valid_min, x1=t_max,
        fillcolor="rgba(148,163,184,0.07)", line_width=0,
        annotation_text="semi-∞ invalid →",
        annotation_font_color=C['sub'], annotation_position="top left",
        row=3, col=1)

    # ── Panel 6: q_throat vs T_w ──────────────────────────────────────────────
    Tw_range = np.linspace(WALL_T0, 1800, 200)
    for res, col, ls in zip(results, cycle(of_colors), cycle(['solid', 'dash', 'dot'])):
        gp, hf = res['gp'], res['hf']
        i_t    = int(np.argmax(hf['hg']))
        hg_t   = hf['hg'][i_t]
        Taw_t  = hf['Taw'][i_t]
        # Cold-wall h_g already contains σ(Tw0). σ changes as the wall warms.
        sig0   = bartz_sigma(WALL_T0, gp.Tc, gp.gam_c, 1.0)
        sig_Tw = np.array([bartz_sigma(float(Tw), gp.Tc, gp.gam_c, 1.0) for Tw in Tw_range])
        q_Tw   = np.maximum(hg_t * (sig_Tw / sig0) * (Taw_t - Tw_range), 0)
        fig.add_trace(go.Scatter(x=Tw_range, y=q_Tw / 1e6,
            name=f"O/F={gp.of}",
            mode='lines', line=dict(color=col, width=2, dash=ls),
            showlegend=False),                                             row=3, col=2)

    fig.add_vline(x=WALL_T0,
                  line=dict(color=C['sub'], dash='dot', width=1),
                  annotation_text=' cold wall',
                  annotation_font_color=C['sub'], row=3, col=2)

    # ── Axis labels ───────────────────────────────────────────────────────────
    axis_cfg = [
        (1, 1, 'x', 'x  [mm]'),
        (1, 1, 'y', 'q  [MW/m²]'),
        (1, 2, 'x', 'x  [mm]'),
        (1, 2, 'y', 'h<sub>g</sub>  [kW/(m²·K)]'),
        (2, 1, 'x', 'x  [mm]'),
        (2, 1, 'y', 'T<sub>aw</sub>  [K]'),
        (2, 2, 'x', 'O/F'),
        (2, 2, 'y', 'Power  [kW]'),
        (3, 1, 'x', 't  [s]'),
        (3, 1, 'y', 'T<sub>wall</sub>  [K]'),
        (3, 2, 'x', 'T<sub>w</sub>  [K]'),
        (3, 2, 'y', 'q<sub>throat</sub>  [MW/m²]'),
    ]
    for (r, c_, axis, title) in axis_cfg:
        upd = dict(title_text=title, gridcolor=C['grid'], linecolor=C['grid'])
        if axis == 'x':
            fig.update_xaxes(**upd, row=r, col=c_)
        else:
            fig.update_yaxes(**upd, zerolinecolor=C['grid'], row=r, col=c_)

    for ann in fig.layout.annotations:
        ann.font.color = C['text']

    fig.update_layout(
        title=dict(
            text=(
                f"N₂O / Ethanol Torch Igniter — Thermal Analysis  "
                f"(Bartz, cold wall Tw={WALL_T0:.0f} K, η_c*={ETA_CSTAR})"
            ),
            font=dict(size=14, color=C['text']), x=0.5,
        ),
        paper_bgcolor=C['bg'], plot_bgcolor=C['surf'],
        font=dict(color=C['text'], family='Arial, sans-serif', size=11),
        legend=dict(bgcolor='rgba(26,29,46,0.85)', bordercolor=C['grid'],
                    borderwidth=1, font=dict(size=10)),
        height=1100, width=1150,
        margin=dict(t=70, b=50, l=60, r=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Torch igniter Bartz thermal analysis")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load(args.config)
    apply_case(cfg)
    print(f"Case: {cfg.name}  ({args.config})")
    print(f"  Dc = {cfg.dc_mm:.2f} mm   Lcyl = {cfg.lcyl_mm:.1f} mm   "
          f"L* = {cfg.lstar_m:.3f} m   Lconv = {cfg.lconv_mm:.2f} mm")

    cea_b, cea_u = build_cea()
    nozzle       = build_nozzle(n_pts=500)
    results      = []

    print("Computing Pc from fixed throat area and η_c* ...")
    for of in OF_POINTS:
        pc_bar = calc_pc(cea_u, of, eta=ETA_CSTAR)
        print(f"  O/F={of:.1f}  →  Pc = {pc_bar:.2f} bar")
        gp  = get_gas_props(cea_b, cea_u, of, pc_bar)
        hf  = compute_heat_flux(gp, nozzle, Tw=WALL_T0)
        eb  = energy_budget(gp, hf['P_wall'])
        results.append(dict(gp=gp, hf=hf, eb=eb, nozzle=nozzle))

    print()
    print_report(results)

    if args.no_plot:
        return

    html_path, image_path = write_figure(build_figure(results, nozzle),
                                         f"{cfg.name}_thermal.html", f"{cfg.name}_thermal.png")
    print(f"\nPlot written: {html_path}")
    if image_path:
        print(f"              {image_path}")


if __name__ == "__main__":
    main()