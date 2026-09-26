"""Compressible gas orifice.

Two mass flows are reported from the same stagnation state.

Real gas. Isentropic expansion along s = s0 with CoolProp. The throat is
choked where the velocity equals the local speed of sound; the mass flux
there is rho* * c*. Below that critical pressure the throat is not sonic
and the fluid expands to the downstream pressure:

    mdot = Cd * A * rho * v,    v = sqrt(2 * (h0 - h))

Constant-k. The closed-form isentropic orifice equations, using the real
upstream density and the upstream isentropic exponent k. This is the usual
algebraic formula. It is not a pure ideal gas: density is the real value,
and k is held constant at its upstream value, so the two results diverge
when k changes along the expansion.

    choked:     mdot = Cd * A * rho0 * sqrt( k * (P0/rho0) * (2/(k+1))^((k+1)/(k-1)) )
    subsonic:   mdot = Cd * A * sqrt( 2*rho0*P0*k/(k-1) * (r^(2/k) - r^((k+1)/k)) )

with r = P_down / P_up. Upstream velocity is neglected. Cd multiplies the
geometric area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from common import (
    BAR,
    PHASE_SUPERCRITICAL,
    PHASE_TWOPHASE,
    Fluid,
    State,
    diameter_for_mass_flow,
    is_gas,
    orifice_area,
    phase_name,
    require_drop,
    row,
)


@dataclass
class GasResult:
    fluid: str
    backend: str
    T_K: float
    P_up_Pa: float
    P_down_Pa: float
    d_m: float
    Cd: float
    area_m2: float
    phase: int
    rho0: float
    h0: float
    s0: float
    c0: float
    Z: float
    k: float
    gamma: float
    P_star_Pa: float | None
    T_star_K: float | None
    rho_star: float | None
    c_star: float | None
    P_twophase_Pa: float | None
    choked_real: bool
    choked_k: bool
    pr_crit_k: float | None
    mdot_real_kg_s: float | None
    mdot_k_kg_s: float | None
    velocity_m_s: float | None
    mach: float | None
    T_norm_K: float
    P_norm_Pa: float
    Q_up_m3_h: float | None
    Q_norm_m3_h: float | None
    valid: bool
    medium: Fluid = field(repr=False)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": "gas",
            "fluid": self.fluid,
            "backend": self.backend,
            "valid": self.valid,
            "T_C": self.T_K - 273.15,
            "P_up_bar": self.P_up_Pa / BAR,
            "P_down_bar": self.P_down_Pa / BAR,
            "pressure_ratio": self.P_down_Pa / self.P_up_Pa,
            "phase": phase_name(self.phase),
            "d_mm": self.d_m * 1e3,
            "area_mm2": self.area_m2 * 1e6,
            "Cd": self.Cd,
            "rho0_kg_m3": self.rho0,
            "Z": self.Z,
            "k_isentropic": self.k,
            "gamma_cp_cv": self.gamma,
            "P_star_bar": None if self.P_star_Pa is None else self.P_star_Pa / BAR,
            "T_star_C": None if self.T_star_K is None else self.T_star_K - 273.15,
            "choked_real": self.choked_real,
            "choked_constant_k": self.choked_k,
            "pr_crit_constant_k": self.pr_crit_k,
            "P_condensation_bar": None if self.P_twophase_Pa is None else self.P_twophase_Pa / BAR,
            "mdot_real_kg_s": self.mdot_real_kg_s,
            "mdot_real_g_s": None if self.mdot_real_kg_s is None else self.mdot_real_kg_s * 1e3,
            "mdot_constant_k_kg_s": self.mdot_k_kg_s,
            "mdot_constant_k_g_s": None if self.mdot_k_kg_s is None else self.mdot_k_kg_s * 1e3,
            "Q_upstream_m3_h": self.Q_up_m3_h,
            "Q_normal_m3_h": self.Q_norm_m3_h,
            "T_norm_C": self.T_norm_K - 273.15,
            "P_norm_bar": self.P_norm_Pa / BAR,
            "notes": list(self.notes),
        }

    def curve(self, P_down_Pa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Real-gas and constant-k mass flow at each downstream pressure.

        Real-gas values are NaN once the isentrope is two-phase.
        """
        real = np.empty(len(P_down_Pa))
        const = np.empty(len(P_down_Pa))
        for i, P in enumerate(P_down_Pa):
            const[i] = constant_k_mass_flow(self.Cd, self.area_m2, self.rho0, self.P_up_Pa, float(P), self.k)
            real[i] = self._real_mass_flow(float(P))
        return real, const

    def _real_mass_flow(self, P_down: float) -> float:
        if self.P_star_Pa is not None and P_down <= self.P_star_Pa:
            return self.Cd * self.area_m2 * self.rho_star * self.c_star
        if self.P_twophase_Pa is not None and P_down <= self.P_twophase_Pa:
            return math.nan
        try:
            st = self.medium.at_PS(P_down, self.s0, sound=True)
        except ValueError:
            return math.nan
        if st.phase == PHASE_TWOPHASE:
            return math.nan
        v = math.sqrt(max(2.0 * (self.h0 - st.h), 0.0))
        return self.Cd * self.area_m2 * st.rho * v


def constant_k_mass_flow(Cd: float, area: float, rho0: float, P_up: float, P_down: float, k: float) -> float:
    """Closed-form isentropic mass flow with constant upstream k and real rho0."""
    if not (k > 1.01):
        return math.nan
    pr_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
    if P_down / P_up <= pr_crit:
        return Cd * area * rho0 * math.sqrt(
            k * (P_up / rho0) * (2.0 / (k + 1.0)) ** ((k + 1.0) / (k - 1.0))
        )
    r = P_down / P_up
    inner = (r ** (2.0 / k)) - (r ** ((k + 1.0) / k))
    if inner <= 0.0:
        return 0.0
    return Cd * area * math.sqrt(2.0 * rho0 * P_up * (k / (k - 1.0)) * inner)


def _sonic_state(medium: Fluid, h0: float, s0: float, P_up: float) -> tuple[State | None, float | None]:
    """Sonic state on s = s0, and the pressure where the isentrope first goes two-phase.

    The residual is v - c. It is negative near stagnation and positive once the
    expansion is supersonic, so a sign change brackets the sonic pressure.
    The scan stops at the saturation curve: speed of sound in an equilibrium
    two-phase mixture is not the gas-orifice throat condition.
    """
    grid = np.linspace(0.999 * P_up, max(0.02 * P_up, 5.0e4), 160)
    last_P = None
    last_f = None
    bracket = None
    P_two = None
    for P in grid:
        try:
            st = medium.at_PS(float(P), s0, sound=True)
        except ValueError:
            break
        if st.phase == PHASE_TWOPHASE:
            P_two = float(P)
            break
        if not math.isfinite(st.c) or st.c <= 0.0 or st.h >= h0:
            last_P = last_f = None
            continue
        f = math.sqrt(2.0 * (h0 - st.h)) - st.c
        if last_f is not None and last_f < 0.0 <= f:
            bracket = (float(P), last_P)
            break
        last_P, last_f = float(P), f

    if bracket is None:
        return None, P_two

    def residual(P: float) -> float:
        st = medium.at_PS(P, s0, sound=True)
        if st.phase == PHASE_TWOPHASE or not math.isfinite(st.c) or st.c <= 0.0:
            raise ValueError("sonic search left the gas region")
        return math.sqrt(max(2.0 * (h0 - st.h), 0.0)) - st.c

    try:
        P_star = brentq(residual, bracket[0], bracket[1], xtol=1.0, rtol=1e-8)
        return medium.at_PS(P_star, s0, sound=True), P_two
    except ValueError:
        return None, P_two


def gas_orifice(
    fluid: str,
    T_K: float,
    P_up_Pa: float,
    P_down_Pa: float,
    d_m: float,
    Cd: float,
    backend: str = "HEOS",
    T_norm_K: float = 273.15,
    P_norm_Pa: float = 101325.0,
) -> GasResult:
    if not (0.0 < Cd <= 1.2):
        raise ValueError("Cd must be in (0, 1.2]")
    medium = Fluid(fluid, backend)
    up = medium.at_TP(T_K, P_up_Pa, sound=True, kappa=True)
    if not is_gas(up.phase, up.Q):
        raise ValueError(
            f"gas model does not apply to a {phase_name(up.phase)} upstream state; "
            "use model: liquid, or model: two_phase if the liquid can flash"
        )
    require_drop(up.P, P_down_Pa)
    area = orifice_area(d_m)

    sonic, P_two = _sonic_state(medium, up.h, up.s, up.P)
    choked_real = sonic is not None and P_down_Pa <= sonic.P
    hits_dome = P_two is not None and P_down_Pa <= P_two and not choked_real

    notes: list[str] = []
    if medium.backend != medium.backend_requested:
        notes.append(f"Backend {medium.backend_requested} was unavailable; used {medium.backend}.")
    if up.phase == PHASE_SUPERCRITICAL:
        notes.append(
            "Upstream temperature and pressure are both above the critical point. "
            "The real-gas integration still applies; it does not assume an ideal gas."
        )
    if Cd > 1.0:
        notes.append("Cd > 1 is unusual when the area is the geometric orifice area.")

    velocity = mach = None
    valid = True
    mdot_real = None
    if choked_real:
        velocity = sonic.c
        mach = 1.0
        mdot_real = Cd * area * sonic.rho * sonic.c
    elif hits_dome:
        valid = False
        notes.append(
            f"The isentrope reaches the saturation curve at {P_two / BAR:.2f} bar, "
            "before a sonic point in the gas. The single-phase gas model does not "
            "apply at this downstream pressure. Use model: two_phase."
        )
    else:
        throat = medium.at_PS(P_down_Pa, up.s, sound=True)
        if throat.phase == PHASE_TWOPHASE:
            valid = False
            notes.append(
                "The isentropic state at the downstream pressure is two-phase. "
                "Use model: two_phase."
            )
        else:
            velocity = math.sqrt(max(2.0 * (up.h - throat.h), 0.0))
            mdot_real = Cd * area * throat.rho * velocity
            if math.isfinite(throat.c) and throat.c > 0.0:
                mach = velocity / throat.c

    pr_crit_k = None
    mdot_k = None
    choked_k = False
    if math.isfinite(up.k) and up.k > 1.01:
        pr_crit_k = (2.0 / (up.k + 1.0)) ** (up.k / (up.k - 1.0))
        choked_k = (P_down_Pa / up.P) <= pr_crit_k
        mdot_k = constant_k_mass_flow(Cd, area, up.rho, up.P, P_down_Pa, up.k)
        if (
            valid
            and mdot_real is not None
            and mdot_real > 0.0
            and abs(mdot_k - mdot_real) / mdot_real > 0.02
        ):
            notes.append(
                "Constant-k and real-gas mass flows differ by more than 2%. "
                "Use the real-gas value; k is not constant along this expansion."
            )
    else:
        notes.append("Upstream isentropic exponent was not available; constant-k mass flow was skipped.")

    if P_two is not None and choked_real:
        notes.append(
            f"Saturation is reached at {P_two / BAR:.2f} bar on this isentrope, "
            "downstream of a gas-phase sonic throat. Mass flow is set by the throat."
        )

    Q_up = Q_norm = None
    if mdot_real is not None:
        Q_up = mdot_real / up.rho * 3600.0
        ref = medium.at_TP(T_norm_K, P_norm_Pa)
        if is_gas(ref.phase, ref.Q):
            Q_norm = mdot_real / ref.rho * 3600.0
        else:
            notes.append(
                "The normal-volume reference state is not a gas, so Nm³/h was not computed."
            )

    return GasResult(
        fluid=medium.name,
        backend=medium.backend,
        T_K=up.T,
        P_up_Pa=P_up_Pa,
        P_down_Pa=P_down_Pa,
        d_m=d_m,
        Cd=Cd,
        area_m2=area,
        phase=up.phase,
        rho0=up.rho,
        h0=up.h,
        s0=up.s,
        c0=up.c,
        Z=up.Z,
        k=up.k,
        gamma=up.gamma,
        P_star_Pa=None if sonic is None else sonic.P,
        T_star_K=None if sonic is None else sonic.T,
        rho_star=None if sonic is None else sonic.rho,
        c_star=None if sonic is None else sonic.c,
        P_twophase_Pa=P_two,
        choked_real=choked_real,
        choked_k=choked_k,
        pr_crit_k=pr_crit_k,
        mdot_real_kg_s=mdot_real,
        mdot_k_kg_s=mdot_k,
        velocity_m_s=velocity,
        mach=mach,
        T_norm_K=T_norm_K,
        P_norm_Pa=P_norm_Pa,
        Q_up_m3_h=Q_up,
        Q_norm_m3_h=Q_norm,
        valid=valid,
        medium=medium,
        notes=notes,
    )


def format_report(result: GasResult, mdot_target_kg_s: float | None = None) -> str:
    status = "real-gas isentropic" if result.valid else "REAL-GAS MODEL DOES NOT APPLY"
    lines = [
        f"Gas orifice  —  {result.fluid} ({result.backend})",
        f"  {phase_name(result.phase)}, {status}",
        row("Upstream temperature", f"{result.T_K - 273.15:.2f} °C"),
        row("Upstream pressure", f"{result.P_up_Pa / BAR:.3f} bar"),
        row("Downstream pressure", f"{result.P_down_Pa / BAR:.3f} bar"),
        row("Pressure ratio", f"{result.P_down_Pa / result.P_up_Pa:.4f}"),
        row("Density", f"{result.rho0:.3f} kg/m³"),
        row("Compressibility Z", f"{result.Z:.4f}" if math.isfinite(result.Z) else "n/a"),
        row("Isentropic exponent k", f"{result.k:.4f}" if math.isfinite(result.k) else "n/a"),
        row("cp/cv", f"{result.gamma:.4f}" if math.isfinite(result.gamma) else "n/a"),
        row("Upstream sound speed", f"{result.c0:.1f} m/s" if math.isfinite(result.c0) else "n/a"),
    ]
    if result.pr_crit_k is not None:
        lines.append(row("Critical pressure ratio, constant k", f"{result.pr_crit_k:.4f}"))
    if result.P_star_Pa is not None:
        lines.append(row("Sonic pressure, real gas", f"{result.P_star_Pa / BAR:.3f} bar"))
        lines.append(row("Sonic temperature", f"{result.T_star_K - 273.15:.2f} °C"))
    if result.P_twophase_Pa is not None:
        lines.append(row("Saturation on the isentrope", f"{result.P_twophase_Pa / BAR:.3f} bar"))
    lines.extend(
        [
            row("Diameter", f"{result.d_m * 1e3:.3f} mm"),
            row("Area", f"{result.area_m2 * 1e6:.4f} mm²"),
            row("Discharge coefficient", f"{result.Cd:.3f}"),
        ]
    )
    if result.mdot_real_kg_s is None:
        lines.append(row("Real-gas mass flow", "not applicable"))
    else:
        regime = "choked" if result.choked_real else "subsonic"
        lines.append(row("Real-gas mass flow", f"{result.mdot_real_kg_s * 1e3:.3f} g/s   ({regime})"))
    if result.mdot_k_kg_s is None or not math.isfinite(result.mdot_k_kg_s):
        lines.append(row("Constant-k mass flow", "n/a"))
    else:
        regime = "choked" if result.choked_k else "subsonic"
        lines.append(row("Constant-k mass flow", f"{result.mdot_k_kg_s * 1e3:.3f} g/s   ({regime})"))
    if (
        result.mdot_real_kg_s not in (None, 0.0)
        and result.mdot_k_kg_s is not None
        and math.isfinite(result.mdot_k_kg_s)
    ):
        delta = (result.mdot_k_kg_s - result.mdot_real_kg_s) / result.mdot_real_kg_s * 100.0
        lines.append(row("Constant-k relative to real gas", f"{delta:+.2f} %"))
    if result.mach is not None:
        lines.append(row("Throat Mach", f"{result.mach:.3f}"))
    if result.velocity_m_s is not None:
        lines.append(row("Throat velocity", f"{result.velocity_m_s:.1f} m/s"))
    if result.Q_up_m3_h is not None:
        lines.append(row("Volume flow at upstream state", f"{result.Q_up_m3_h:.3f} m³/h"))
    if result.Q_norm_m3_h is not None:
        lines.append(
            row(
                f"Normal volume ({result.T_norm_K - 273.15:.0f} °C, {result.P_norm_Pa / BAR:.5f} bar)",
                f"{result.Q_norm_m3_h:.3f} Nm³/h",
            )
        )
    if mdot_target_kg_s is not None and result.mdot_real_kg_s:
        d = diameter_for_mass_flow(result.d_m, result.mdot_real_kg_s, mdot_target_kg_s)
        lines.append(row(f"Diameter for {mdot_target_kg_s * 1e3:.3f} g/s", f"{d * 1e3:.3f} mm"))
    if result.notes:
        lines.append("  Notes")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)
