"""
script is referenced in IGN-DOC-001 — Section 3 Chamber Sizing
CEA Combustion Analysis: N2O / Ethanol Torch Igniter

Dependencies:
    pip install rocketcea plotly kaleido==0.2.1

"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rocketcea.cea_obj_w_units import CEA_Obj

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from igniter_config import DEFAULT_CONFIG, load, write_figure  # noqa: E402


# ── Propellant / operating conditions ────────────────────────────────────────
OX_NAME   = "N2O"
FUEL_NAME = "C2H5OH"

OF_NOM    = 2.2        # nominal O/F ratio
PC_NOM    = 30.0       # bar  — reference Pc (initial guess only; actual Pc is solved)
MDOT      = 0.015      # kg/s — total mass flow
ETA_CSTAR = 0.8       # combustion efficiency (user-specified for nominal point)
PA_BAR    = 1.01325    # bar  — ambient pressure for Isp calculation
EPS_NOZ   = 1.5        # nozzle area ratio

# O/F design points — Pc is computed from fixed At + mdot for each
OF_POINTS = [1.6, 2.0, 2.2]

# η_c* values for Pc sensitivity panel (ETA_CSTAR is inserted automatically)
_ETA_BASE = [1.00, 0.9, 0.8, 0.7]
ETA_VALS  = sorted(set(_ETA_BASE + [ETA_CSTAR]), reverse=True)

# ── Chamber geometry ──────────────────────────────────────────────────────────
DT_MM      = 4.4                             # mm  — throat diameter (input)
AT_MM2     = math.pi / 4 * DT_MM**2            # mm² — throat area (derived)
AT_M2      = AT_MM2 * 1e-6                      # m²
EPS_C      = (12**2)/(DT_MM**2)                             # contraction ratio
AC_MM2     = EPS_C * AT_MM2                     # mm² — chamber area
DC_MM      = 2 * math.sqrt(AC_MM2 / math.pi)   # mm  — chamber diameter
THETA_CONV = 45.0                               # deg — convergent half-angle
THETA_DIV  = 15.0                               # deg — divergent half-angle
L_STAR     = 1.2                             # m   — characteristic chamber length
G0         = 9.80665


def apply_case(cfg) -> None:
    """Copy a YAML case onto the module globals both the report and the plots use."""
    global OX_NAME, FUEL_NAME, OF_NOM, PC_NOM, MDOT, ETA_CSTAR, PA_BAR, EPS_NOZ
    global OF_POINTS, ETA_VALS, DT_MM, AT_MM2, AT_M2, EPS_C, AC_MM2, DC_MM
    global THETA_CONV, THETA_DIV, L_STAR
    OX_NAME = cfg.ox
    FUEL_NAME = cfg.fuel
    OF_NOM = cfg.of_nominal
    PC_NOM = cfg.pc_guess_bar
    MDOT = cfg.mdot_kg_s
    ETA_CSTAR = cfg.eta_cstar
    PA_BAR = cfg.pa_bar
    EPS_NOZ = cfg.eps
    OF_POINTS = list(cfg.of_points)
    ETA_VALS = sorted(set([1.00, 0.9, 0.8, 0.7, ETA_CSTAR]), reverse=True)
    DT_MM = cfg.dt_mm
    AT_MM2 = cfg.at_mm2
    AT_M2 = cfg.at_m2
    EPS_C = cfg.eps_c
    AC_MM2 = cfg.ac_mm2
    DC_MM = cfg.dc_mm
    THETA_CONV = cfg.theta_conv_deg
    THETA_DIV = cfg.theta_div_deg
    L_STAR = cfg.lstar_m


# ── Colour palette (light theme) ─────────────────────────────────────────────
C = {
    "bg":    "#f8f9fc",
    "surf":  "#ffffff",
    "grid":  "#d1d5e0",
    "text":  "#1a1d2e",
    "sub":   "#6b7280",
    "blue":  "#1d6fcc",
    "green": "#1a7a4a",
    "amber": "#b45309",
    "ora":   "#c2410c",
    "red":   "#b91c1c",
    "pur":   "#6d28d9",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CEA wrapper
# ─────────────────────────────────────────────────────────────────────────────
def build_cea() -> CEA_Obj:
    return CEA_Obj(
        oxName=OX_NAME, fuelName=FUEL_NAME,
        pressure_units="Bar", cstar_units="m/s",
        temperature_units="K",  isp_units="sec",
        density_units="kg/m^3",
    )


def calc_pc(cea: CEA_Obj, of: float, eta: float,
            pc_init: float = None, tol: float = 1e-4, maxiter: int = 40) -> float:
    """
    Iteratively solve Pc [bar] for a fixed throat, mass flow and efficiency:
        Pc = mdot · c*(OF, Pc) · η / At
    c*(Pc) is weak, so convergence is fast (~5 iterations).
    """
    pc = PC_NOM if pc_init is None else pc_init
    for _ in range(maxiter):
        cstar  = cea.get_Cstar(Pc=pc, MR=of)       # m/s
        pc_new = MDOT * cstar * eta / AT_M2 / 1e5  # bar
        if abs(pc_new - pc) < tol:
            return pc_new
        pc = 0.6 * pc + 0.4 * pc_new               # damped update
    return pc


def isp_delivered(cea: CEA_Obj, of: float, pc: float, eta: float) -> dict:
    """
    Ideal CEA vacuum Isp, then c* efficiency and the ambient correction.

    eta_c* scales c* and vacuum Isp. Nozzle Cf is taken as ideal.
        Isp_vac = eta * Isp_CEA
        Isp_amb = Isp_vac - Pa * Ae / (mdot * g0)
    The ambient correction assumes attached flow.
    """
    isp_vac_cea = cea.get_Isp(Pc=pc, MR=of, eps=EPS_NOZ)
    isp_vac = eta * isp_vac_cea
    ae_m2 = EPS_NOZ * AT_M2
    isp_amb = isp_vac - PA_BAR * 1e5 * ae_m2 / (MDOT * G0)
    return dict(isp_vac_cea=isp_vac_cea, isp_vac=isp_vac, isp_amb=isp_amb)


def cea_point(cea: CEA_Obj, of: float, eta: float = None) -> dict:
    """Compute Pc from fixed throat, then extract CEA and delivered performance."""
    if eta is None:
        eta = ETA_CSTAR
    pc        = calc_pc(cea, of, eta)
    Tc        = cea.get_Tcomb(Pc=pc, MR=of)
    cstar_cea = cea.get_Cstar(Pc=pc, MR=of)
    isp       = isp_delivered(cea, of, pc, eta)
    _, gamma  = cea.get_Chamber_MolWt_gamma(Pc=pc, MR=of)
    return dict(
        of=of, pc=pc, Tc=Tc, gamma=gamma, eta=eta,
        cstar_cea=cstar_cea, cstar=eta * cstar_cea,
        isp_amb=isp["isp_amb"], isp_vac=isp["isp_vac"], isp_vac_cea=isp["isp_vac_cea"],
        mdot_ox=MDOT * of / (1 + of),
        mdot_fuel=MDOT / (1 + of),
    )


def cea_sweep(cea: CEA_Obj, of_range, eta: float = None) -> dict:
    """O/F sweep with Pc solved from the fixed throat at each point."""
    if eta is None:
        eta = ETA_CSTAR
    out = {k: [] for k in ["OF", "Tc", "cstar", "isp_amb", "pc", "gamma"]}
    for of in of_range:
        try:
            point = cea_point(cea, of, eta)
            out["OF"].append(float(of))
            out["Tc"].append(point["Tc"])
            out["cstar"].append(point["cstar"])
            out["isp_amb"].append(point["isp_amb"])
            out["pc"].append(point["pc"])
            out["gamma"].append(point["gamma"])
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def convergent_geometry(Rc: float, Rt: float, theta_deg: float) -> tuple:
    """Cone frustum: returns (L_conv [mm], V_conv [cm³])."""
    theta  = math.radians(theta_deg)
    L_conv = (Rc - Rt) / math.tan(theta)
    V_conv = (math.pi / 3) * (L_conv / 10) * (
        (Rc / 10)**2 + (Rc / 10) * (Rt / 10) + (Rt / 10)**2
    )
    return L_conv, V_conv


def divergent_geometry(Rt: float, eps: float, theta_deg: float) -> tuple:
    """Conical nozzle: returns (L_div [mm], Re [mm])."""
    Re    = Rt * math.sqrt(eps)
    L_div = (Re - Rt) / math.tan(math.radians(theta_deg))
    return L_div, Re


def chamber_lengths(L_star: float, At_mm2: float, Ac_mm2: float,
                    V_conv_cm3: float, Dc_mm: float) -> dict:
    """Cylindrical chamber length for a given L* [m]."""
    Vc_cm3  = L_star * At_mm2 * 1e-6 * 1e6   # L*[m] × At[m²] → [m³] → [cm³]
    Ac_cm2  = Ac_mm2 * 1e-2                   # mm² → cm²
    Lcyl_mm = (Vc_cm3 - V_conv_cm3) / Ac_cm2 * 10   # cm → mm
    return dict(Vc_cm3=Vc_cm3, Lcyl_mm=Lcyl_mm, LD=Lcyl_mm / Dc_mm)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Console report
# ─────────────────────────────────────────────────────────────────────────────
def print_report(cea: CEA_Obj) -> None:
    SEP = "=" * 70
    print(SEP)
    print("CEA RESULTS — N2O / Ethanol Torch Igniter")
    print(f"  At = {AT_MM2:.2f} mm²  (Dt = {DT_MM} mm)   "
          f"ṁ = {MDOT:.3f} kg/s   η_c* = {ETA_CSTAR}")
    print(SEP)

    for of in OF_POINTS:
        p   = cea_point(cea, of)
        tag = "  [NOMINAL]" if of == OF_NOM else ""
        print(f"\nO/F = {of:.1f}   →  Pc = {p['pc']:.2f} bar{tag}")
        print(f"  T_c                   = {p['Tc']:.0f} K")
        print(f"  c* (CEA ideal)        = {p['cstar_cea']:.1f} m/s")
        print(f"  c* (delivered)        = {p['cstar']:.1f} m/s")
        print(f"  γ                     = {p['gamma']:.4f}")
        print(f"  Isp vac (CEA ideal)   = {p['isp_vac_cea']:.1f} s")
        print(f"  Isp vac (delivered)   = {p['isp_vac']:.1f} s")
        print(f"  Isp amb (delivered)   = {p['isp_amb']:.1f} s  "
              f"(ε={EPS_NOZ}, Pa={PA_BAR:.4f} bar)")
        print(f"  ṁ_ox                  = {p['mdot_ox']:.4f} kg/s")
        print(f"  ṁ_fuel                = {p['mdot_fuel']:.4f} kg/s")

    # ── Chamber geometry ──────────────────────────────────────────────────────
    Rc_mm = DC_MM / 2
    Rt_mm = DT_MM / 2
    Lconv, Vconv = convergent_geometry(Rc_mm, Rt_mm, THETA_CONV)
    Ldiv, Re_mm  = divergent_geometry(Rt_mm, EPS_NOZ, THETA_DIV)
    g = chamber_lengths(L_STAR, AT_MM2, AC_MM2, Vconv, DC_MM)

    print(f"\n{SEP}")
    print(f"CHAMBER GEOMETRY  (L* = {L_STAR} m, θ_c = {THETA_CONV}°, θ_e = {THETA_DIV}°)")
    print(SEP)
    print(f"  Dt = {DT_MM:.2f} mm   At = {AT_MM2:.2f} mm²")
    print(f"  Dc = {DC_MM:.2f} mm   Ac = {AC_MM2:.1f} mm²   ε_c = {EPS_C:.2f}")
    print(f"  De = {2*Re_mm:.2f} mm   Ae = {math.pi*Re_mm**2:.2f} mm²   ε = {EPS_NOZ}")
    print(f"  L_conv = {Lconv:.2f} mm   L_div = {Ldiv:.2f} mm")
    print(f"  V_conv = {Vconv:.3f} cm³   V_c = {g['Vc_cm3']:.2f} cm³")
    print(f"  L_cyl  = {g['Lcyl_mm']:.1f} mm   L/D = {g['LD']:.2f}")
    print(f"  L_total = {g['Lcyl_mm'] + Lconv + Ldiv:.1f} mm")

    print("\nL* SENSITIVITY:")
    print(f"  {'L* [m]':<10}{'Vc [cm³]':<12}{'L_cyl [mm]':<14}{'L/D'}")
    for ls in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        g2   = chamber_lengths(ls, AT_MM2, AC_MM2, Vconv, DC_MM)
        flag = "  ← baseline" if ls == L_STAR else ""
        print(f"  {ls:<10.1f}{g2['Vc_cm3']:<12.2f}{g2['Lcyl_mm']:<14.1f}{g2['LD']:.2f}{flag}")



# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plotly figure
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 4a.  Standalone document-quality chamber contour
# ─────────────────────────────────────────────────────────────────────────────

def build_contour_figure(cea: CEA_Obj) -> go.Figure:
    """
    Professional engineering-drawing-quality axisymmetric cross-section.
    1:1 aspect ratio. Export as SVG for document inclusion.
    """
    # ── Geometry ──────────────────────────────────────────────────────────
    Rc_mm  = DC_MM / 2
    Rt_mm  = DT_MM / 2
    Lconv, Vconv = convergent_geometry(Rc_mm, Rt_mm, THETA_CONV)
    Ldiv, Re_mm  = divergent_geometry(Rt_mm, EPS_NOZ, THETA_DIV)
    Lcyl   = chamber_lengths(L_STAR, AT_MM2, AC_MM2, Vconv, DC_MM)["Lcyl_mm"]
    x_thr  = Lcyl + Lconv
    x_exit = x_thr + Ldiv
    Pc_nom = calc_pc(cea, OF_NOM, ETA_CSTAR)

    # Dimension line x-positions (diameter arrows) — staggered to avoid overlap
    EXT   = 1.5     # mm: extension line overshoots arrow by this much
    x_Dc  = -Rc_mm * 0.55                # ≈ −6.8 mm  (left of injector face)
    x_De  = x_exit + Rc_mm * 0.45       # ≈ +44 mm   (right of exit, close)
    x_Dt  = x_exit + Rc_mm * 1.65       # ≈ +60 mm   (right of exit, far)

    fig = go.Figure()

    # ── Gas-path fill + wall outline ──────────────────────────────────────
    xw = [0, Lcyl, x_thr,  x_exit, x_exit,  x_thr,  Lcyl,   0,     0    ]
    yw = [Rc_mm, Rc_mm, Rt_mm, Re_mm, -Re_mm, -Rt_mm, -Rc_mm, -Rc_mm, Rc_mm]
    fig.add_trace(go.Scatter(
        x=xw, y=yw, mode="lines", fill="toself",
        fillcolor="rgba(29,111,204,0.09)",
        line=dict(color=C["blue"], width=2.5),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Centreline (dash-dot, standard engineering) ───────────────────────
    CL_EXT = Rc_mm * 0.4
    fig.add_trace(go.Scatter(
        x=[-CL_EXT, x_exit + CL_EXT], y=[0, 0], mode="lines",
        line=dict(color="#94a3b8", width=1.0, dash="dashdot"),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Section dividers ───────────────────────────────────────────────────
    for xv in [Lcyl, x_thr]:
        fig.add_shape(type="line",
            x0=xv, y0=-Rc_mm * 1.04, x1=xv, y1=Rc_mm * 1.04,
            line=dict(color=C["sub"], width=1.0, dash="dot"),
        )

    # ── Throat marker ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_thr], y=[0], mode="markers",
        marker=dict(color=C["red"], size=7, symbol="diamond"),
        showlegend=False, hoverinfo="skip",
    ))

    # ────────────────────────────────────────────────────────────────────────
    # Dimension-line primitives
    # ────────────────────────────────────────────────────────────────────────
    def dim_arrow(x0, y0, x1, y1):
        """Double-headed dimension arrow in data coordinates."""
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=0.9, arrowwidth=1.3,
            arrowcolor=C["text"], arrowside="end+start",
            showarrow=True, text="",
        )

    def ext_line(x0, y0, x1, y1):
        """Extension line (thin solid)."""
        fig.add_shape(type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=C["text"], width=0.8),
        )

    def dim_label(x, y, txt, size=9.5, anchor="center", valign="middle"):
        fig.add_annotation(
            x=x, y=y, text=txt, showarrow=False,
            xref="x", yref="y",
            font=dict(size=size, color=C["text"], family="Arial"),
            xanchor=anchor, yanchor=valign,
            bgcolor=C["bg"], borderpad=2,
        )

    def leader(x_tip, y_tip, x_text, y_text, txt, size=8.5):
        """Leader line from a feature to a text label."""
        fig.add_annotation(
            x=x_tip, y=y_tip, ax=x_text, ay=y_text,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=0.8, arrowwidth=1.0,
            arrowcolor=C["sub"], showarrow=True,
            text=txt,
            font=dict(size=size, color=C["sub"], family="Arial"),
            xanchor="center", yanchor="bottom",
        )

    # ── Diameter dimensions ────────────────────────────────────────────────
    # Dc — left of injector face
    ext_line(0,     Rc_mm,  x_Dc - EXT, Rc_mm)
    ext_line(0,    -Rc_mm,  x_Dc - EXT, -Rc_mm)
    dim_arrow(x_Dc, -Rc_mm, x_Dc, Rc_mm)
    dim_label(x_Dc - EXT - 0.5, 0,
              f"Ø {DC_MM:.1f}<br><sub>D<sub>c</sub></sub>", anchor="right")

    # De — right of exit (closer column)
    ext_line(x_exit,  Re_mm,  x_De - EXT, Re_mm)
    ext_line(x_exit, -Re_mm,  x_De - EXT, -Re_mm)
    dim_arrow(x_De, -Re_mm, x_De, Re_mm)
    dim_label(x_De + EXT + 0.5, 0,
              f"Ø {2*Re_mm:.1f}<br><sub>D<sub>e</sub></sub>", anchor="left")

    # Dt — right of exit (far column, no overlap with De)
    ext_line(x_thr,   Rt_mm,  x_Dt - EXT, Rt_mm)
    ext_line(x_thr,  -Rt_mm,  x_Dt - EXT, -Rt_mm)
    dim_arrow(x_Dt, -Rt_mm, x_Dt, Rt_mm)
    dim_label(x_Dt + EXT + 0.5, 0,
              f"Ø {DT_MM:.1f}<br><sub>D<sub>t</sub></sub>", anchor="left")

    # ── Axial dimension lines ──────────────────────────────────────────────
    yd1 = -Rc_mm * 1.30     # individual sections
    yd2 = -Rc_mm * 1.82     # total span
    TICK = 0.6              # stub past dimension line

    for xv in [0, Lcyl, x_thr, x_exit]:
        ext_line(xv, -Rc_mm, xv, yd2 - TICK)

    dim_arrow(0,     yd1, Lcyl,    yd1)
    dim_arrow(Lcyl,  yd1, x_thr,  yd1)
    dim_arrow(x_thr, yd1, x_exit, yd1)
    dim_arrow(0,     yd2, x_exit, yd2)

    for xc, val in [
        (Lcyl * 0.5,               Lcyl),
        ((Lcyl + x_thr) * 0.5,    Lconv),
        ((x_thr + x_exit) * 0.5,  Ldiv),
    ]:
        dim_label(xc, yd1, f"{val:.1f}", size=9)

    dim_label(x_exit * 0.5, yd2,
              f"L<sub>total</sub> = {x_exit:.1f} mm", size=9.5)

    # Sub-labels below dimension values
    for xc, lbl in [
        (Lcyl * 0.5,               "L<sub>cyl</sub>"),
        ((Lcyl + x_thr) * 0.5,    "L<sub>conv</sub>"),
        ((x_thr + x_exit) * 0.5,  "L<sub>div</sub>"),
    ]:
        fig.add_annotation(
            x=xc, y=yd1 - Rc_mm * 0.17, text=lbl,
            showarrow=False, xref="x", yref="y",
            font=dict(size=7.5, color=C["sub"], family="Arial"),
            xanchor="center", yanchor="top",
        )

    # ── Section labels (inside gas path) ──────────────────────────────────
    fig.add_annotation(
        x=Lcyl * 0.5, y=Rc_mm * 0.28, text="Combustion Chamber",
        showarrow=False, xref="x", yref="y",
        font=dict(size=9, color=C["blue"], family="Arial"),
        xanchor="center",
    )
    fig.add_annotation(
        x=(Lcyl + x_thr) * 0.5, y=(Rc_mm + Rt_mm) * 0.22, text="Conv.",
        showarrow=False, xref="x", yref="y",
        font=dict(size=8.5, color=C["blue"], family="Arial"),
        xanchor="center",
    )
    # Divergent section is tiny — use a leader line from above
    leader(
        x_tip=(x_thr + x_exit) * 0.5, y_tip=Re_mm * 1.1,
        x_text=(x_thr + x_exit) * 0.5, y_text=Rc_mm * 0.75,
        txt="Div.",
    )

    # ── Angle annotations ─────────────────────────────────────────────────
    # θ_conv — inside convergent section
    fig.add_annotation(
        x=(Lcyl + x_thr) * 0.5 - 1.5, y=-(Rc_mm + Rt_mm) * 0.16,
        text=f"θ = {THETA_CONV:.0f}°",
        showarrow=False, xref="x", yref="y",
        font=dict(size=8, color=C["sub"], family="Arial"), xanchor="center",
    )
    # θ_div — below the tiny divergent section (outside wall)
    fig.add_annotation(
        x=(x_thr + x_exit) * 0.5, y=-Rt_mm * 3.2, text=f"θ = {THETA_DIV:.0f}°",
        showarrow=False, xref="x", yref="y",
        font=dict(size=8, color=C["sub"], family="Arial"), xanchor="center",
    )

    # ── Injector-face leader ───────────────────────────────────────────────
    fig.add_annotation(
        x=0, y=Rc_mm * 1.10,
        ax=0, ay=Rc_mm * 1.55,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowwidth=1.0, arrowcolor=C["sub"],
        showarrow=True, text="Injector face",
        font=dict(size=8.5, color=C["sub"], family="Arial"),
        xanchor="center", yanchor="bottom",
    )

    # ── Design parameters box ──────────────────────────────────────────────
    fig.add_annotation(
        x=0.988, y=0.975, xref="paper", yref="paper",
        text=(
            f"<b>Design Parameters</b><br>"
            f"Propellants: N₂O / C₂H₅OH<br>"
            f"ṁ = {MDOT:.3f} kg/s  ·  O/F = {OF_NOM:.1f}<br>"
            f"P<sub>c</sub> = {Pc_nom:.1f} bar  ·  η<sub>c*</sub> = {ETA_CSTAR:.2f}<br>"
            f"ε<sub>c</sub> = {EPS_C:.2f}  ·  ε = {EPS_NOZ:.1f}<br>"
            f"L* = {L_STAR:.2f} m  ·  θ = {THETA_CONV:.0f}° / {THETA_DIV:.0f}°"
        ),
        showarrow=False,
        xanchor="right", yanchor="top", align="left",
        font=dict(size=9, color=C["text"], family="Arial"),
        bgcolor=C["bg"], bordercolor=C["grid"],
        borderwidth=1, borderpad=8,
    )

    # ── Footer note ────────────────────────────────────────────────────────
    fig.add_annotation(
        x=0.01, y=0.01, xref="paper", yref="paper",
        text="Axisymmetric cross-section  ·  all dimensions in mm",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=8, color=C["sub"], family="Arial"),
    )

    # ── Axes (1:1 enforced via scaleanchor) ───────────────────────────────
    x_range = [x_Dc - Rc_mm * 0.45,  x_Dt + Rc_mm * 0.9]
    y_range = [yd2  - Rc_mm * 0.40,  Rc_mm * 1.90]

    fig.update_xaxes(
        range=x_range, title_text="Axial Position  [mm]",
        gridcolor=C["grid"], linecolor=C["grid"],
        showgrid=True, zeroline=False, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        range=y_range, title_text="Radial Position  [mm]",
        gridcolor=C["grid"], linecolor=C["grid"],
        showgrid=True, zeroline=False,
        scaleanchor="x", scaleratio=1,
        tickfont=dict(size=10),
    )

    fig.update_layout(
        title=dict(
            text=(
                "N₂O / Ethanol Torch Igniter — Combustion Chamber Cross-Section<br>"
                f"<sup>D<sub>t</sub> = {DT_MM} mm  ·  D<sub>c</sub> = {DC_MM:.1f} mm  ·  "
                f"ṁ = {MDOT} kg/s  ·  P<sub>c,nom</sub> = {Pc_nom:.1f} bar</sup>"
            ),
            font=dict(size=12, color=C["text"]), x=0.5,
        ),
        paper_bgcolor=C["bg"], plot_bgcolor=C["surf"],
        font=dict(color=C["text"], family="Arial, sans-serif"),
        showlegend=False,
        width=1050, height=560,
        margin=dict(t=72, b=65, l=68, r=28),
    )

    return fig



def build_figure(cea: CEA_Obj) -> go.Figure:

    # ── Sweeps ───────────────────────────────────────────────────────────────
    OF_range = np.linspace(1.0, 4.0, 80)
    sweep    = cea_sweep(cea, OF_range)
    OF_v, Tc_v, cs_v, isp_v, pc_v = (
        sweep["OF"], sweep["Tc"], sweep["cstar"], sweep["isp_amb"], sweep["pc"]
    )

    # Design points
    dp     = [cea_point(cea, of) for of in OF_POINTS]
    dp_of  = [p["of"]      for p in dp]
    dp_Tc  = [p["Tc"]      for p in dp]
    dp_cs  = [p["cstar"]   for p in dp]
    dp_isp = [p["isp_amb"] for p in dp]
    dp_pc  = [p["pc"]      for p in dp]

    # Pc sweeps for each η_c* value
    eta_pc_sweeps = {
        eta: [calc_pc(cea, of, eta) for of in OF_range]
        for eta in ETA_VALS
    }

    # ── Chamber geometry for contour ──────────────────────────────────────────
    Rc_mm = DC_MM / 2
    Rt_mm = DT_MM / 2
    Lconv, Vconv = convergent_geometry(Rc_mm, Rt_mm, THETA_CONV)
    Ldiv, Re_mm  = divergent_geometry(Rt_mm, EPS_NOZ, THETA_DIV)
    Lcyl         = chamber_lengths(L_STAR, AT_MM2, AC_MM2, Vconv, DC_MM)["Lcyl_mm"]

    x_throat = Lcyl + Lconv
    x_exit   = x_throat + Ldiv

    # ── Layout: 2×2 performance + full-width chamber contour ─────────────────
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
        ],
        subplot_titles=[
            "Combustion Temperature  T<sub>c</sub>",
            "Delivered c*  (η<sub>c*</sub> · c*<sub>CEA</sub>)",
            f"Delivered ambient I<sub>sp</sub>  (ε={EPS_NOZ}, P<sub>a</sub>={PA_BAR:.2f} bar)",
            "Chamber Pressure  P<sub>c</sub> — η<sub>c*</sub> sensitivity",
            "Chamber Contour  (1:1 scale)",
        ],
        vertical_spacing=0.09, horizontal_spacing=0.12,
        row_heights=[0.27, 0.27, 0.46],
    )

    # ── Scatter helper ────────────────────────────────────────────────────────
    def S(x, y, name, col, dash="solid", w=2, mode="lines", sym=None, sz=9, show=True):
        mk = dict(color=col, size=sz, symbol=sym) if sym else None
        ln = dict(color=col, width=w, dash=dash) if "lines" in mode else None
        return go.Scatter(x=x, y=y, name=name, mode=mode,
                          line=ln, marker=mk, showlegend=show)

    # ── Panel 1: T_c ─────────────────────────────────────────────────────────
    fig.add_trace(S(OF_v, Tc_v, "T<sub>c</sub>", C["blue"]),                     row=1, col=1)
    fig.add_trace(S(dp_of, dp_Tc, "Design pts", C["amber"],
                    mode="markers", sym="diamond"),                                row=1, col=1)

    # ── Panel 2: c* ──────────────────────────────────────────────────────────
    fig.add_trace(S(OF_v, cs_v, "c*", C["green"]),                               row=1, col=2)
    fig.add_trace(S(dp_of, dp_cs, "Design pts", C["amber"],
                    mode="markers", sym="diamond", show=False),                   row=1, col=2)

    # ── Panel 3: Ambient Isp ─────────────────────────────────────────────────
    fig.add_trace(S(OF_v, isp_v, "I<sub>sp,amb</sub>", C["blue"]),               row=2, col=1)
    fig.add_trace(S(dp_of, dp_isp, "Design pts", C["amber"],
                    mode="markers", sym="diamond", show=False),                   row=2, col=1)

    # ── Panel 4: Pc sensitivity over η_c* ────────────────────────────────────
    eta_colours = [C["green"], C["amber"], C["blue"], C["ora"], C["red"]]
    for eta, col in zip(ETA_VALS, eta_colours):
        lbl = (f"η<sub>c*</sub>={eta:.2f}"
               + ("  ← nominal" if eta == ETA_CSTAR else ""))
        fig.add_trace(
            S(OF_v, eta_pc_sweeps[eta], lbl, col,
              w=2.5 if eta == ETA_CSTAR else 1.5),                                row=2, col=2
        )
    fig.add_trace(S(dp_of, dp_pc, "Design pts", C["amber"],
                    mode="markers", sym="diamond", show=False),                   row=2, col=2)

    # Panel 5
    Rc_mm = DC_MM / 2
    Rt_mm = DT_MM / 2
    Lconv_p, Vconv_p = convergent_geometry(Rc_mm, Rt_mm, THETA_CONV)
    Ldiv_p, Re_mm_p = divergent_geometry(Rt_mm, EPS_NOZ, THETA_DIV)
    Lcyl_p = chamber_lengths(L_STAR, AT_MM2, AC_MM2, Vconv_p, DC_MM)["Lcyl_mm"]
    x_thr_p = Lcyl_p + Lconv_p
    x_ex_p = x_thr_p + Ldiv_p

    xw_p = [0, Lcyl_p, x_thr_p, x_ex_p,
            x_ex_p, x_thr_p, Lcyl_p, 0, 0]
    yw_p = [Rc_mm, Rc_mm, Rt_mm, Re_mm_p,
            -Re_mm_p, -Rt_mm, -Rc_mm, -Rc_mm, Rc_mm]

    fig.add_trace(go.Scatter(
        x=xw_p, y=yw_p, mode="lines", fill="toself",
        fillcolor="rgba(29,111,204,0.09)",
        line=dict(color=C["blue"], width=2.0),
        showlegend=False,
    ), row=3, col=1)

    # Centreline
    fig.add_trace(go.Scatter(
        x=[-Rc_mm * 0.3, x_ex_p + Rc_mm * 0.3], y=[0, 0], mode="lines",
        line=dict(color="#94a3b8", width=1.0, dash="dashdot"),
        showlegend=False,
    ), row=3, col=1)

    # Throat marker
    fig.add_trace(go.Scatter(
        x=[x_thr_p], y=[0], mode="markers",
        marker=dict(color=C["red"], size=6, symbol="diamond"),
        showlegend=False,
    ), row=3, col=1)

    # Section dividers
    for xv in [Lcyl_p, x_thr_p]:
        fig.add_shape(type="line",
                      x0=xv, y0=-Rc_mm * 1.02, x1=xv, y1=Rc_mm * 1.02,
                      line=dict(color=C["sub"], width=1.0, dash="dot"),
                      xref="x5", yref="y5",
                      )

    # Diameter labels — staggered vertically to avoid overlap
    # Dc: centre of chamber at upper wall
    # Dt: above throat (positive y leader)
    # De: to the right of exit with offset
    for x_pos, y_pos, txt in [
        (Lcyl_p * 0.35, Rc_mm * 0.55, f"D<sub>c</sub>={DC_MM:.1f} mm"),
        (x_thr_p, Rt_mm * 5.5, f"D<sub>t</sub>={DT_MM:.1f} mm"),
        (x_ex_p + Rc_mm * 0.15, Re_mm_p * 5.5, f"D<sub>e</sub>={2 * Re_mm_p:.1f} mm"),
    ]:
        fig.add_annotation(
            x=x_pos, y=y_pos, text=txt,
            showarrow=False, row=3, col=1,
            font=dict(size=8.5, color=C["text"], family="Arial"),
            xanchor="center",
            bgcolor=C["bg"], borderpad=1,
        )

    # Axial length annotations below profile
    yd_sub = -Rc_mm * 1.25
    for xc, lbl in [
        (Lcyl_p * 0.5, f"L<sub>cyl</sub>={Lcyl_p:.1f}"),
        ((Lcyl_p + x_thr_p) * 0.5, f"L<sub>conv</sub>={Lconv_p:.1f}"),
        ((x_thr_p + x_ex_p) * 0.5, f"L<sub>div</sub>={Ldiv_p:.1f}"),
    ]:
        fig.add_annotation(
            x=xc, y=yd_sub, text=lbl,
            showarrow=False, row=3, col=1,
            font=dict(size=8, color=C["sub"], family="Arial"),
            xanchor="center",
        )

    # Section name labels inside gas path
    for x_pos, y_pos, txt in [
        (Lcyl_p * 0.5, Rc_mm * 0.22, "Combustion Chamber"),
        ((Lcyl_p + x_thr_p) * 0.5, Rc_mm * 0.10, "Conv."),
    ]:
        fig.add_annotation(
            x=x_pos, y=y_pos, text=txt,
            showarrow=False, row=3, col=1,
            font=dict(size=8.5, color=C["blue"], family="Arial"),
            xanchor="center",
        )

    # Contour axes
    fig.update_xaxes(
        title_text="Axial Position  [mm]",
        gridcolor=C["grid"], linecolor=C["grid"],
        range=[-Rc_mm * 0.5, x_ex_p + Rc_mm * 1.8],
        row=3, col=1,
    )
    fig.update_yaxes(
        title_text="Radius  [mm]",
        gridcolor=C["grid"], linecolor=C["grid"], zerolinecolor=C["grid"],
        scaleanchor="x5", scaleratio=1,
        row=3, col=1,
    )

    for ann in fig.layout.annotations:
        ann.font.color = C["text"]

    fig.update_layout(
        title=dict(
            text=(
                "N₂O / Ethanol Torch Igniter — CEA Combustion Analysis"
                f"  (D<sub>t</sub>={DT_MM} mm, ṁ={MDOT} kg/s,"
                f" η<sub>c*</sub>={ETA_CSTAR})"
            ),
            font=dict(size=14, color=C["text"]), x=0.5,
        ),
        paper_bgcolor=C["bg"], plot_bgcolor=C["surf"],
        font=dict(color=C["text"], family="Arial, sans-serif", size=11),
        legend=dict(bgcolor="rgba(248,249,252,0.90)", bordercolor=C["grid"],
                    borderwidth=1, font=dict(size=10)),
        height=1100, width=1100,
        margin=dict(t=80, b=55, l=65, r=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Torch igniter CEA chamber sizing")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load(args.config)
    apply_case(cfg)
    print(f"Case: {cfg.name}  ({args.config})")

    cea = build_cea()
    print_report(cea)

    if args.no_plot:
        return

    dash_html, dash_img = write_figure(
        build_figure(cea), f"{cfg.name}_combustion.html", f"{cfg.name}_combustion.svg"
    )
    contour_html, contour_img = write_figure(
        build_contour_figure(cea), f"{cfg.name}_contour.html", f"{cfg.name}_contour.svg"
    )
    print("\nOutputs written:")
    print(f"  {dash_html}")
    print(f"  {contour_html}")
    if dash_img:
        print(f"  {dash_img}")
    if contour_img:
        print(f"  {contour_img}")


if __name__ == "__main__":
    main()