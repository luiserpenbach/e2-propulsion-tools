"""Shared state, geometry, and unit helpers for the orifice models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import CoolProp.CoolProp as CP

BAR = 1.0e5  # Pa

# CoolProp iphase indices.
PHASE_LIQUID = 0
PHASE_SUPERCRITICAL = 1
PHASE_SUPERCRITICAL_GAS = 2
PHASE_SUPERCRITICAL_LIQUID = 3
PHASE_GAS = 5
PHASE_TWOPHASE = 6

PHASE_NAME = {
    0: "liquid",
    1: "supercritical",
    2: "supercritical gas",
    3: "supercritical liquid",
    4: "critical point",
    5: "gas",
    6: "two-phase",
    7: "unknown",
    8: "not imposed",
}


def phase_name(phase: int) -> str:
    return PHASE_NAME.get(phase, f"phase {phase}")


def orifice_area(d_m: float) -> float:
    """Geometric area from diameter, m²."""
    if d_m <= 0.0:
        raise ValueError("orifice diameter must be positive")
    return math.pi / 4.0 * d_m * d_m


def diameter_for_mass_flow(d_ref_m: float, mdot_ref: float, mdot_target: float) -> float:
    """Diameter that delivers mdot_target, scaling from a solved reference point.

    Every model in this library is linear in area, and the discharge coefficient
    is held constant, so mass flow scales with d².
    """
    if mdot_ref <= 0.0:
        raise ValueError("reference mass flow is zero; check the pressures")
    if mdot_target <= 0.0:
        raise ValueError("target mass flow must be positive")
    return d_ref_m * math.sqrt(mdot_target / mdot_ref)


def require_drop(P_up: float, P_down: float) -> float:
    if P_down >= P_up:
        raise ValueError(
            f"downstream pressure ({P_down / BAR:.3f} bar) must be below "
            f"upstream pressure ({P_up / BAR:.3f} bar)"
        )
    return P_up - P_down


def is_liquid(phase: int, quality: float) -> bool:
    """Compressed liquid, supercritical liquid, or saturated liquid (Q = 0)."""
    if phase in (PHASE_LIQUID, PHASE_SUPERCRITICAL_LIQUID):
        return True
    return phase == PHASE_TWOPHASE and math.isfinite(quality) and quality <= 1e-6


def is_gas(phase: int, quality: float) -> bool:
    """Gas, supercritical fluid, or saturated vapor (Q = 1)."""
    if phase in (PHASE_GAS, PHASE_SUPERCRITICAL, PHASE_SUPERCRITICAL_GAS):
        return True
    return phase == PHASE_TWOPHASE and math.isfinite(quality) and quality >= 1.0 - 1e-6


def row(label: str, value: str) -> str:
    return f"  {label:<32} {value}"


@dataclass
class State:
    T: float
    P: float
    rho: float
    h: float
    s: float
    phase: int
    Q: float
    c: float = math.nan
    mu: float = math.nan
    k: float = math.nan
    gamma: float = math.nan
    Z: float = math.nan


class Fluid:
    """One CoolProp state. Each call overwrites it and returns a snapshot."""

    def __init__(self, name: str, backend: str = "HEOS"):
        self.name = name
        self.backend_requested = backend
        try:
            self._as = CP.AbstractState(backend, name)
            self._as.T_critical()
            self.backend = backend
        except Exception:
            if backend.upper() == "HEOS":
                raise
            self._as = CP.AbstractState("HEOS", name)
            self.backend = "HEOS"
        self.T_crit = float(self._as.T_critical())
        self.P_crit = float(self._as.p_critical())

    def at_TP(self, T: float, P: float, **flags) -> State:
        try:
            self._as.update(CP.PT_INPUTS, P, T)
        except ValueError as exc:
            if "Saturation pressure" in str(exc):
                raise ValueError(
                    f"{self.name} at {T - 273.15:.2f} °C is on the saturation curve. "
                    "Move the pressure off the curve, or set saturated_liquid: true."
                ) from exc
            raise ValueError(
                f"Could not set {self.name} at T={T:.2f} K, P={P:.4e} Pa: {exc}"
            ) from exc
        return self._read(**flags)

    def at_quality(self, T: float, Q: float, **flags) -> State:
        """Saturation state at temperature T. Q = 0 is saturated liquid."""
        if T >= self.T_crit:
            raise ValueError(
                f"{self.name} at {T - 273.15:.2f} °C is above the critical "
                f"temperature ({self.T_crit - 273.15:.2f} °C); there is no saturation state"
            )
        try:
            self._as.update(CP.QT_INPUTS, Q, T)
        except ValueError as exc:
            raise ValueError(f"Could not set {self.name} at T={T:.2f} K, Q={Q}: {exc}") from exc
        return self._read(**flags)

    def at_PQ(self, P: float, Q: float, **flags) -> State:
        try:
            self._as.update(CP.PQ_INPUTS, P, Q)
        except ValueError as exc:
            raise ValueError(f"Could not set {self.name} at P={P:.4e} Pa, Q={Q}: {exc}") from exc
        return self._read(**flags)

    def at_PS(self, P: float, s: float, **flags) -> State:
        try:
            self._as.update(CP.PSmass_INPUTS, P, s)
        except ValueError as exc:
            raise ValueError(f"Isentropic state failed at P={P:.4e} Pa: {exc}") from exc
        return self._read(**flags)

    def vapor_pressure(self, T: float) -> float | None:
        if T >= self.T_crit:
            return None
        return self.at_quality(T, 0.0).P

    def _read(self, sound: bool = False, viscosity: bool = False, kappa: bool = False) -> State:
        a = self._as
        phase = int(a.phase())
        Q = float(a.Q())
        if Q < 0.0:
            Q = math.nan
        c = math.nan
        if sound and phase != PHASE_TWOPHASE:
            try:
                c = float(a.speed_sound())
            except Exception:
                c = math.nan
        mu = math.nan
        if viscosity:
            try:
                mu = float(a.viscosity())
            except Exception:
                mu = math.nan
        k = gamma = Z = math.nan
        try:
            Z = float(a.keyed_output(CP.iZ))
        except Exception:
            Z = math.nan
        if kappa and phase != PHASE_TWOPHASE:
            try:
                k = float(a.keyed_output(CP.iisentropic_expansion_coefficient))
            except Exception:
                k = math.nan
            try:
                cp = float(a.cpmass())
                cv = float(a.cvmass())
                if cv > 0.0:
                    gamma = cp / cv
            except Exception:
                gamma = math.nan
        return State(
            T=float(a.T()),
            P=float(a.p()),
            rho=float(a.rhomass()),
            h=float(a.hmass()),
            s=float(a.smass()),
            phase=phase,
            Q=Q,
            c=c,
            mu=mu,
            k=k,
            gamma=gamma,
            Z=Z,
        )
