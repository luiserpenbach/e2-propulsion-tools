"""Flashing-liquid orifice: SPI, HEM, and the Dyer/Solomon NHNE blend.

SPI, single-phase incompressible, is the non-flashing upper estimate:

    mdot_SPI = Cd * A * sqrt(2 * rho_L * (P_up - P_down))

HEM, homogeneous equilibrium, expands isentropically and lets the phases
stay in equilibrium. The mass flux at a throat pressure P is

    G(P) = rho(P, s0) * sqrt(2 * (h0 - h(P, s0)))

The critical flux is the maximum of G. If that maximum sits above the
downstream pressure, the orifice is choked at G*. Otherwise the throat
is at the downstream pressure.

NHNE blends the two with the non-equilibrium parameter of Dyer et al.
(AIAA 2007-5702), in the form corrected by Solomon (USU, 2011) and used
by Waxman et al. (AIAA 2013-3636):

    kappa = sqrt( (P_up - P_down) / (P_sat(T_up) - P_down) )
    mdot  = (kappa * mdot_SPI + mdot_HEM) / (1 + kappa)

kappa is the bubble-growth time divided by the residence time, with the
proportionality constant taken as 1. Large kappa (subcooled liquid, slow
bubble growth) approaches SPI. kappa = 1 for saturated liquid. When the
downstream pressure is at or above the vapor pressure there is no flashing
and the result is pure SPI. Orifice L/D is reported as a check on that
assumption; it does not enter the mass-flow calculation. Cd should be
anchored to a cold-flow test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from common import (
    BAR,
    PHASE_TWOPHASE,
    Fluid,
    State,
    diameter_for_mass_flow,
    is_gas,
    is_liquid,
    orifice_area,
    phase_name,
    require_drop,
    row,
)

# Bottom of the HEM pressure scan. Low enough to see a flashing peak,
# high enough to stay near the region CoolProp can still evaluate.
_P_FLOOR = 1.0e5  # Pa


@dataclass
class TwoPhaseResult:
    fluid: str
    backend: str
    T_K: float
    P_up_Pa: float
    P_down_Pa: float
    d_m: float
    L_m: float | None
    Cd: float
    area_m2: float
    phase: int
    rho0: float
    rho_L: float
    h0: float
    s0: float
    P_sat_Pa: float | None
    P_star_Pa: float | None
    G_star: float | None
    hem_peak_resolved: bool
    choked: bool
    throat_phase: int
    throat_Q: float
    throat_rho: float
    kappa: float
    w_spi: float
    w_hem: float
    mdot_spi_kg_s: float
    mdot_hem_kg_s: float
    mdot_nhne_kg_s: float
    medium: Fluid = field(repr=False)
    scan_P: np.ndarray = field(repr=False)
    scan_G: np.ndarray = field(repr=False)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        LD = None if self.L_m is None else self.L_m / self.d_m
        return {
            "model": "two_phase",
            "fluid": self.fluid,
            "backend": self.backend,
            "T_C": self.T_K - 273.15,
            "P_up_bar": self.P_up_Pa / BAR,
            "P_down_bar": self.P_down_Pa / BAR,
            "dP_bar": (self.P_up_Pa - self.P_down_Pa) / BAR,
            "phase": phase_name(self.phase),
            "d_mm": self.d_m * 1e3,
            "L_mm": None if self.L_m is None else self.L_m * 1e3,
            "L_over_D": LD,
            "area_mm2": self.area_m2 * 1e6,
            "Cd": self.Cd,
            "rho0_kg_m3": self.rho0,
            "rho_liquid_kg_m3": self.rho_L,
            "P_sat_bar": None if self.P_sat_Pa is None else self.P_sat_Pa / BAR,
            "subcooling_bar": None
            if self.P_sat_Pa is None
            else (self.P_up_Pa - self.P_sat_Pa) / BAR,
            "P_star_hem_bar": None if self.P_star_Pa is None else self.P_star_Pa / BAR,
            "G_star_kg_m2_s": self.G_star,
            "hem_choked": self.choked,
            "throat_phase": phase_name(self.throat_phase),
            "throat_quality": self.throat_Q,
            "throat_rho_kg_m3": self.throat_rho,
            "kappa": None if not math.isfinite(self.kappa) else self.kappa,
            "weight_spi": self.w_spi,
            "weight_hem": self.w_hem,
            "mdot_spi_kg_s": self.mdot_spi_kg_s,
            "mdot_hem_kg_s": self.mdot_hem_kg_s,
            "mdot_nhne_kg_s": self.mdot_nhne_kg_s,
            "mdot_spi_g_s": self.mdot_spi_kg_s * 1e3,
            "mdot_hem_g_s": self.mdot_hem_kg_s * 1e3,
            "mdot_nhne_g_s": self.mdot_nhne_kg_s * 1e3,
            "notes": list(self.notes),
        }

    def fluxes(self, P_down: float) -> tuple[float, float, float]:
        """SPI, HEM, and NHNE mass flow at one downstream pressure."""
        dP = self.P_up_Pa - P_down
        if dP <= 0.0:
            return 0.0, 0.0, 0.0
        m_spi = self.Cd * self.area_m2 * math.sqrt(2.0 * self.rho_L * dP)
        if self.hem_peak_resolved and self.P_star_Pa is not None and P_down <= self.P_star_Pa:
            G = self.G_star
        else:
            G, _ = _mass_flux(self.medium, self.h0, self.s0, P_down)
        m_hem = self.Cd * self.area_m2 * G
        kap = non_equilibrium_kappa(self.P_up_Pa, P_down, self.P_sat_Pa)
        m_nhne, _, _ = _blend(m_spi, m_hem, kap)
        return m_spi, m_hem, m_nhne


def non_equilibrium_kappa(P_up: float, P_down: float, P_sat: float | None) -> float:
    """Solomon/Waxman kappa. Infinity means no flashing (pure SPI). NaN if undefined."""
    if P_sat is None or not math.isfinite(P_sat):
        return math.nan
    if P_down >= P_up:
        return math.nan
    if P_down >= P_sat:
        return math.inf
    return math.sqrt((P_up - P_down) / (P_sat - P_down))


def _blend(m_spi: float, m_hem: float, kappa: float) -> tuple[float, float, float]:
    """Return NHNE mass flow and the SPI and HEM weights."""
    if math.isnan(kappa):
        return math.nan, math.nan, math.nan
    if math.isinf(kappa):
        return m_spi, 1.0, 0.0
    w_hem = 1.0 / (1.0 + kappa)
    w_spi = kappa * w_hem
    return w_spi * m_spi + w_hem * m_hem, w_spi, w_hem


def _mass_flux(medium: Fluid, h0: float, s0: float, P: float) -> tuple[float, State | None]:
    try:
        st = medium.at_PS(P, s0)
    except ValueError:
        return 0.0, None
    dh = h0 - st.h
    if dh <= 0.0 or not math.isfinite(st.rho) or st.rho <= 0.0:
        return 0.0, st
    return st.rho * math.sqrt(2.0 * dh), st


def _hem_critical(medium: Fluid, h0: float, s0: float, P_up: float):
    """Maximize G along the isentrope. Returns P*, G*, resolved flag, and the scan."""
    n = 320
    P_grid = np.linspace(0.995 * P_up, _P_FLOOR, n)
    G = np.empty(n)
    for i, P in enumerate(P_grid):
        G[i], _ = _mass_flux(medium, h0, s0, float(P))
    if not np.any(G > 0.0):
        raise ValueError("HEM mass flux could not be evaluated along the isentrope")
    i_max = int(np.argmax(G))
    # An endpoint maximum, or a cliff to a failed property call (G = 0), means
    # the unconstrained peak was not found. The flow is then unchoked at any
    # downstream pressure above the scan floor.
    resolved = (
        1 <= i_max <= n - 3
        and G[i_max] >= G[i_max - 1]
        and G[i_max + 1] > 0.0
        and G[i_max] >= G[i_max + 1]
        and G[i_max + 2] > 0.0
    )
    P_star = float(P_grid[i_max])
    G_star = float(G[i_max])
    if resolved:
        lo = float(P_grid[min(i_max + 5, n - 1)])
        hi = float(P_grid[max(i_max - 5, 0)])
        if lo < hi:
            opt = minimize_scalar(
                lambda P: -_mass_flux(medium, h0, s0, float(P))[0],
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 20.0},
            )
            if math.isfinite(opt.fun) and -opt.fun > G_star:
                P_star = float(opt.x)
                G_star = float(-opt.fun)
    return P_star, G_star, resolved, P_grid, G


def _ld_note(L_m: float | None, d_m: float) -> str | None:
    if L_m is None:
        return None
    LD = L_m / d_m
    if LD < 1.0:
        regime = "thin plate; SPI is usually the better estimate"
    elif LD < 5.0:
        regime = "short orifice; NHNE is the appropriate estimate"
    elif LD < 10.0:
        regime = "intermediate length; NHNE, with HEM gaining weight"
    else:
        regime = "long orifice; HEM becomes increasingly appropriate"
    return f"L/D = {LD:.2f}: {regime}. L/D is not inside the mass-flow equations."


def two_phase_orifice(
    fluid: str,
    T_K: float,
    P_down_Pa: float,
    d_m: float,
    Cd: float,
    P_up_Pa: float | None = None,
    L_m: float | None = None,
    saturated_liquid: bool = False,
    backend: str = "HEOS",
) -> TwoPhaseResult:
    if not (0.0 < Cd <= 1.2):
        raise ValueError("Cd must be in (0, 1.2]")
    if L_m is not None and L_m <= 0.0:
        raise ValueError("orifice length must be positive")
    medium = Fluid(fluid, backend)
    if saturated_liquid:
        up = medium.at_quality(T_K, 0.0)
    else:
        if P_up_Pa is None:
            raise ValueError("P_up_Pa is required unless saturated_liquid is set")
        up = medium.at_TP(T_K, P_up_Pa)
    if is_gas(up.phase, up.Q):
        raise ValueError(
            f"two-phase model does not apply to a {phase_name(up.phase)} upstream state; use model: gas"
        )
    if not is_liquid(up.phase, up.Q) and up.phase != PHASE_TWOPHASE:
        raise ValueError(
            f"two-phase model does not apply to a {phase_name(up.phase)} upstream state"
        )

    require_drop(up.P, P_down_Pa)
    area = orifice_area(d_m)
    P_sat = up.P if saturated_liquid else medium.vapor_pressure(up.T)

    if is_liquid(up.phase, up.Q):
        rho_L = up.rho
    else:
        rho_L = medium.at_PQ(up.P, 0.0).rho

    P_star, G_star, resolved, scan_P, scan_G = _hem_critical(medium, up.h, up.s, up.P)
    choked = resolved and P_down_Pa <= P_star
    if choked:
        G_eff = G_star
        throat = medium.at_PS(P_star, up.s)
    else:
        G_eff, throat = _mass_flux(medium, up.h, up.s, P_down_Pa)
        if throat is None:
            raise ValueError("could not evaluate the downstream isentropic state")

    m_spi = Cd * area * math.sqrt(2.0 * rho_L * (up.P - P_down_Pa))
    m_hem = Cd * area * G_eff
    kappa = non_equilibrium_kappa(up.P, P_down_Pa, P_sat)
    m_nhne, w_spi, w_hem = _blend(m_spi, m_hem, kappa)
    throat_Q = throat.Q
    if not math.isfinite(throat_Q):
        if is_liquid(throat.phase, 0.0):
            throat_Q = 0.0
        elif is_gas(throat.phase, 1.0):
            throat_Q = 1.0

    notes: list[str] = []
    if medium.backend != medium.backend_requested:
        notes.append(f"Backend {medium.backend_requested} was unavailable; used {medium.backend}.")
    if Cd > 1.0:
        notes.append("Cd > 1 is unusual when the area is the geometric orifice area.")
    if not resolved:
        notes.append(
            f"HEM mass flux was still increasing at { _P_FLOOR / BAR:.2f} bar. "
            "No choked HEM solution was found above that pressure, so HEM is evaluated "
            "at the downstream pressure."
        )
    ld = _ld_note(L_m, d_m)
    if ld:
        notes.append(ld)
    if choked and throat_Q < 0.02:
        notes.append(
            "The HEM mass-flux peak is at the onset of flashing. "
            "Upstream of that pressure the isentrope is still liquid."
        )
    if P_sat is not None and P_down_Pa >= P_sat:
        notes.append("Downstream pressure is at or above the vapor pressure, so there is no bulk flashing. NHNE reduces to SPI.")
    if math.isnan(kappa):
        notes.append("Vapor pressure is not defined above the critical temperature. NHNE was not evaluated; use SPI and HEM directly.")
    notes.append("Cd is an input. Anchor it with a cold-flow test; it dominates the uncertainty.")

    return TwoPhaseResult(
        fluid=medium.name,
        backend=medium.backend,
        T_K=up.T,
        P_up_Pa=up.P if saturated_liquid else P_up_Pa,
        P_down_Pa=P_down_Pa,
        d_m=d_m,
        L_m=L_m,
        Cd=Cd,
        area_m2=area,
        phase=up.phase,
        rho0=up.rho,
        rho_L=rho_L,
        h0=up.h,
        s0=up.s,
        P_sat_Pa=P_sat,
        P_star_Pa=P_star if resolved else None,
        G_star=G_star if resolved else None,
        hem_peak_resolved=resolved,
        choked=choked,
        throat_phase=throat.phase,
        throat_Q=throat_Q,
        throat_rho=throat.rho,
        kappa=kappa,
        w_spi=w_spi,
        w_hem=w_hem,
        mdot_spi_kg_s=m_spi,
        mdot_hem_kg_s=m_hem,
        mdot_nhne_kg_s=m_nhne,
        medium=medium,
        scan_P=scan_P,
        scan_G=scan_G,
        notes=notes,
    )


def format_report(result: TwoPhaseResult, mdot_target_kg_s: float | None = None) -> str:
    if result.P_sat_Pa is None:
        sat = "n/a (above critical temperature)"
        sub = "n/a"
    else:
        sat = f"{result.P_sat_Pa / BAR:.3f} bar"
        sub = f"{(result.P_up_Pa - result.P_sat_Pa) / BAR:.3f} bar"
    if math.isinf(result.kappa):
        kap = "no flashing (pure SPI)"
    elif math.isnan(result.kappa):
        kap = "undefined"
    else:
        kap = f"{result.kappa:.3f}   ({result.w_spi * 100:.1f}% SPI, {result.w_hem * 100:.1f}% HEM)"
    Q = f"{result.throat_Q:.4f}" if math.isfinite(result.throat_Q) else "n/a"
    lines = [
        f"Two-phase orifice  —  {result.fluid} ({result.backend})",
        f"  {phase_name(result.phase)} upstream, SPI / HEM / NHNE",
        row("Upstream temperature", f"{result.T_K - 273.15:.2f} °C"),
        row("Upstream pressure", f"{result.P_up_Pa / BAR:.3f} bar"),
        row("Downstream pressure", f"{result.P_down_Pa / BAR:.3f} bar"),
        row("Vapor pressure at T", sat),
        row("Subcooling (P_up - P_sat)", sub),
        row("Upstream density", f"{result.rho0:.2f} kg/m³"),
        row("Liquid density used in SPI", f"{result.rho_L:.2f} kg/m³"),
        row("Diameter", f"{result.d_m * 1e3:.3f} mm"),
        row("Length", "n/a" if result.L_m is None else f"{result.L_m * 1e3:.3f} mm"),
        row("Area", f"{result.area_m2 * 1e6:.4f} mm²"),
        row("Discharge coefficient", f"{result.Cd:.3f}"),
        row(
            "HEM critical pressure",
            "not found above 1 bar" if result.P_star_Pa is None else f"{result.P_star_Pa / BAR:.3f} bar",
        ),
        row("HEM critical mass flux", "n/a" if result.G_star is None else f"{result.G_star:.1f} kg/(m²·s)"),
        row("HEM choked", "yes" if result.choked else "no"),
        row("Throat phase", phase_name(result.throat_phase)),
        row("Throat quality", Q),
        row("Throat density", f"{result.throat_rho:.2f} kg/m³"),
        row("kappa", kap),
        row("SPI mass flow", f"{result.mdot_spi_kg_s * 1e3:.3f} g/s"),
        row("NHNE mass flow", "n/a" if math.isnan(result.mdot_nhne_kg_s) else f"{result.mdot_nhne_kg_s * 1e3:.3f} g/s"),
        row("HEM mass flow", f"{result.mdot_hem_kg_s * 1e3:.3f} g/s"),
    ]
    if mdot_target_kg_s is not None and mdot_target_kg_s > 0.0:
        lines.append(f"  Diameter for {mdot_target_kg_s * 1e3:.3f} g/s, Cd fixed")
        for label, mdot in (
            ("SPI", result.mdot_spi_kg_s),
            ("NHNE", result.mdot_nhne_kg_s),
            ("HEM", result.mdot_hem_kg_s),
        ):
            if mdot is None or not math.isfinite(mdot) or mdot <= 0.0:
                continue
            d = diameter_for_mass_flow(result.d_m, mdot, mdot_target_kg_s)
            lines.append(row(label, f"{d * 1e3:.3f} mm"))
        lines.append(
            "  NHNE is the estimate. Use the HEM diameter when the target is a "
            "minimum flow with equilibrium flashing."
        )
    if result.notes:
        lines.append("  Notes")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)
