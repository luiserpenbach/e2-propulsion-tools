"""Incompressible liquid orifice.

Mass flow is the stagnation form of Bernoulli's equation, with losses and
contraction collected into a discharge coefficient on the geometric area:

    mdot = Cd * A * sqrt(2 * rho * (P_up - P_down))

Density is the upstream liquid density. Upstream velocity is neglected.
Flashing and cavitation are not modeled; the result says so when the
downstream pressure is below the vapor pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from common import (
    BAR,
    PHASE_SUPERCRITICAL_LIQUID,
    Fluid,
    diameter_for_mass_flow,
    is_liquid,
    orifice_area,
    phase_name,
    require_drop,
    row,
)


@dataclass
class LiquidResult:
    fluid: str
    backend: str
    T_K: float
    P_up_Pa: float
    P_down_Pa: float
    d_m: float
    Cd: float
    area_m2: float
    phase: int
    rho: float
    mu: float
    P_vap_Pa: float | None
    mdot_kg_s: float
    velocity_m_s: float
    Re: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": "liquid",
            "fluid": self.fluid,
            "backend": self.backend,
            "T_C": self.T_K - 273.15,
            "P_up_bar": self.P_up_Pa / BAR,
            "P_down_bar": self.P_down_Pa / BAR,
            "dP_bar": (self.P_up_Pa - self.P_down_Pa) / BAR,
            "phase": phase_name(self.phase),
            "d_mm": self.d_m * 1e3,
            "area_mm2": self.area_m2 * 1e6,
            "Cd": self.Cd,
            "rho_kg_m3": self.rho,
            "mu_Pa_s": self.mu,
            "P_vap_bar": None if self.P_vap_Pa is None else self.P_vap_Pa / BAR,
            "mdot_kg_s": self.mdot_kg_s,
            "mdot_g_s": self.mdot_kg_s * 1e3,
            "velocity_m_s": self.velocity_m_s,
            "Re": self.Re,
            "notes": list(self.notes),
        }


def liquid_orifice(
    fluid: str,
    T_K: float,
    P_down_Pa: float,
    d_m: float,
    Cd: float,
    P_up_Pa: float | None = None,
    saturated_liquid: bool = False,
    backend: str = "HEOS",
) -> LiquidResult:
    if not (0.0 < Cd <= 1.2):
        raise ValueError("Cd must be in (0, 1.2]")
    medium = Fluid(fluid, backend)
    if saturated_liquid:
        up = medium.at_quality(T_K, 0.0, viscosity=True)
    else:
        if P_up_Pa is None:
            raise ValueError("P_up_Pa is required unless saturated_liquid is set")
        up = medium.at_TP(T_K, P_up_Pa, viscosity=True)
    if not is_liquid(up.phase, up.Q):
        raise ValueError(
            f"liquid model does not apply to a {phase_name(up.phase)} upstream state; "
            "use model: gas, or model: two_phase if the liquid can flash"
        )

    dP = require_drop(up.P, P_down_Pa)
    area = orifice_area(d_m)
    mdot = Cd * area * math.sqrt(2.0 * up.rho * dP)
    velocity = mdot / (up.rho * area)  # mean velocity on the geometric area
    Re = math.nan
    if math.isfinite(up.mu) and up.mu > 0.0:
        Re = up.rho * velocity * d_m / up.mu

    P_vap = medium.vapor_pressure(up.T)
    notes: list[str] = []
    if medium.backend != medium.backend_requested:
        notes.append(f"Backend {medium.backend_requested} was unavailable; used {medium.backend}.")
    if Cd > 1.0:
        notes.append("Cd > 1 is unusual when the area is the geometric orifice area.")
    if P_vap is not None and P_down_Pa < P_vap:
        notes.append(
            f"Downstream pressure is below the vapor pressure ({P_vap / BAR:.3f} bar). "
            "This model ignores flashing and will over-predict the mass flow. Use model: two_phase."
        )
    if up.phase == PHASE_SUPERCRITICAL_LIQUID:
        notes.append(
            "Upstream pressure is above the critical pressure and the temperature is below "
            "the critical temperature. The fluid is still a compressed liquid."
        )
    if math.isfinite(Re) and Re < 1.0e4:
        notes.append("Reynolds number is below 10⁴. Cd is often still changing with Re in this range.")

    return LiquidResult(
        fluid=medium.name,
        backend=medium.backend,
        T_K=up.T,
        P_up_Pa=up.P if saturated_liquid else P_up_Pa,
        P_down_Pa=P_down_Pa,
        d_m=d_m,
        Cd=Cd,
        area_m2=area,
        phase=up.phase,
        rho=up.rho,
        mu=up.mu,
        P_vap_Pa=P_vap,
        mdot_kg_s=mdot,
        velocity_m_s=velocity,
        Re=Re,
        notes=notes,
    )


def format_report(result: LiquidResult, mdot_target_kg_s: float | None = None) -> str:
    Re = f"{result.Re:.3e}" if math.isfinite(result.Re) else "n/a"
    mu = f"{result.mu * 1e3:.4f} mPa·s" if math.isfinite(result.mu) else "n/a"
    pv = f"{result.P_vap_Pa / BAR:.4f} bar" if result.P_vap_Pa is not None else "n/a (above critical T)"
    lines = [
        f"Liquid orifice  —  {result.fluid} ({result.backend})",
        f"  {phase_name(result.phase)}, incompressible, upstream velocity neglected",
        row("Upstream temperature", f"{result.T_K - 273.15:.2f} °C"),
        row("Upstream pressure", f"{result.P_up_Pa / BAR:.3f} bar"),
        row("Downstream pressure", f"{result.P_down_Pa / BAR:.3f} bar"),
        row("Pressure drop", f"{(result.P_up_Pa - result.P_down_Pa) / BAR:.3f} bar"),
        row("Density", f"{result.rho:.2f} kg/m³"),
        row("Viscosity", mu),
        row("Vapor pressure", pv),
        row("Diameter", f"{result.d_m * 1e3:.3f} mm"),
        row("Area", f"{result.area_m2 * 1e6:.4f} mm²"),
        row("Discharge coefficient", f"{result.Cd:.3f}"),
        row("Mass flow", f"{result.mdot_kg_s * 1e3:.3f} g/s   ({result.mdot_kg_s:.6f} kg/s)"),
        row("Mean velocity", f"{result.velocity_m_s:.2f} m/s"),
        row("Reynolds number", Re),
    ]
    if mdot_target_kg_s is not None:
        d = diameter_for_mass_flow(result.d_m, result.mdot_kg_s, mdot_target_kg_s)
        lines.append(
            row(
                f"Diameter for {mdot_target_kg_s * 1e3:.3f} g/s",
                f"{d * 1e3:.3f} mm",
            )
        )
    if result.notes:
        lines.append("  Notes")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)
