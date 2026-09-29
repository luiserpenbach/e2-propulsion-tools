"""Coolant properties for the channel march (handbook sections 10.1 to 10.3).

`Coolant(name).state(h, p)` returns everything a channel station needs from the
specific enthalpy and pressure, which is what the march carries (enthalpy stays
valid through the two-phase dome and the pseudo-critical region).

Water: CoolProp (IAPWS-95 equation of state, IAPWS transport).

Nitrous oxide: CoolProp equation of state (Span-Wagner). CoolProp has no transport
model for N2O, so viscosity and conductivity come from extended corresponding
states with CO2 as the reference fluid: the two molecules are isoelectronic, have
the same molar mass and nearly the same critical point. The N2O state is mapped to
CO2 at the same reduced temperature and reduced density, and the CO2 values are
scaled with the usual ECS factors

    mu_N2O = mu_CO2 * (M_N2O/M_CO2)^1/2 (Tc_N2O/Tc_CO2)^1/2 (Vc_CO2/Vc_N2O)^2/3
    k_N2O  = k_CO2  * (M_CO2/M_N2O)^1/2 (Tc_N2O/Tc_CO2)^1/2 (Vc_CO2/Vc_N2O)^2/3

Expected accuracy is 10 to 15 % away from the critical point, worse close to it.
In the two-phase region the saturated-liquid transport properties are returned,
because the march uses a liquid-only heat transfer coefficient there.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from CoolProp.CoolProp import PropsSI

_N = "NitrousOxide"
_C = "CarbonDioxide"


@dataclass
class State:
    T: float          # K
    p: float          # Pa
    h: float          # J/kg
    rho: float        # kg/m3 (homogeneous in the two-phase region)
    cp: float         # J/(kg K) (liquid value in the two-phase region)
    mu: float         # Pa s
    k: float          # W/(m K)
    Pr: float
    x: float          # vapour quality; <0 subcooled, >1 superheated, nan if supercritical
    Tsat: float       # K, nan if supercritical
    regime: str       # 'liquid', 'two-phase', 'vapour', 'supercritical'
    mu_l: float = float("nan")   # saturated liquid/vapour transport for two-phase
    mu_v: float = float("nan")


class Coolant:
    def __init__(self, name):
        self.name = name
        self.fluid = {"water": "Water", "n2o": _N, "nitrousoxide": _N}.get(name.lower(), name)
        self.pcrit = PropsSI("pcrit", self.fluid)
        self.Tcrit = PropsSI("Tcrit", self.fluid)
        self.is_n2o = self.fluid == _N
        if self.is_n2o:
            self._ecs_init()

    # ------------------------------------------------------------ N2O transport
    def _ecs_init(self):
        Tn, Tc = PropsSI("Tcrit", _N), PropsSI("Tcrit", _C)
        rn, rc = PropsSI("rhocrit", _N), PropsSI("rhocrit", _C)
        Mn, Mc = PropsSI("M", _N), PropsSI("M", _C)
        self._fT, self._frho = Tc / Tn, rc / rn
        vr = ((Mc / rc) / (Mn / rn)) ** (2.0 / 3.0)            # (Vc_CO2/Vc_N2O)^2/3
        self._fmu = (Mn / Mc) ** 0.5 * (Tn / Tc) ** 0.5 * vr
        self._fk = (Mc / Mn) ** 0.5 * (Tn / Tc) ** 0.5 * vr
        self._Tt_co2 = PropsSI("Ttriple", _C)

    def _ecs(self, T, rho, liquid_side):
        T0 = max(T * self._fT, self._Tt_co2 + 0.5)
        r0 = rho * self._frho
        if T0 < PropsSI("Tcrit", _C):
            rl = PropsSI("D", "T", T0, "Q", 0, _C)
            rv = PropsSI("D", "T", T0, "Q", 1, _C)
            if rv < r0 < rl:                                       # mapped into the dome
                r0 = rl * 1.0005 if liquid_side else rv * 0.9995
        mu = PropsSI("V", "T", T0, "D", r0, _C) * self._fmu
        k = PropsSI("L", "T", T0, "D", r0, _C) * self._fk
        return mu, k

    # ------------------------------------------------------------------ state
    def state(self, h, p):
        return _state_cached(self.fluid, round(h, 1), round(p, 0))

    def sat(self, p):
        """Saturation properties at p (Pa); None above the critical pressure."""
        if p >= self.pcrit:
            return None
        return _sat_cached(self.fluid, round(p, 0))


@lru_cache(maxsize=400_000)
def _sat_cached(fluid, p):
    return {"T": PropsSI("T", "P", p, "Q", 0, fluid),
            "rho_l": PropsSI("D", "P", p, "Q", 0, fluid),
            "rho_v": PropsSI("D", "P", p, "Q", 1, fluid),
            "h_l": PropsSI("H", "P", p, "Q", 0, fluid),
            "h_v": PropsSI("H", "P", p, "Q", 1, fluid),
            "sigma": PropsSI("I", "P", p, "Q", 0, fluid)}


_COOLANTS = {}


def _get(fluid):
    if fluid not in _COOLANTS:
        _COOLANTS[fluid] = Coolant(fluid)
    return _COOLANTS[fluid]


@lru_cache(maxsize=400_000)
def _state_cached(fluid, h, p):
    c = _get(fluid)
    T = PropsSI("T", "H", h, "P", p, fluid)
    rho = PropsSI("D", "H", h, "P", p, fluid)
    if p < c.pcrit:
        s = _sat_cached(fluid, p)
        x = (h - s["h_l"]) / (s["h_v"] - s["h_l"])
        Tsat = s["T"]
        regime = "liquid" if x <= 0 else ("vapour" if x >= 1 else "two-phase")
    else:
        x, Tsat = float("nan"), float("nan")
        regime = "supercritical"
    mu_l = mu_v = float("nan")
    if regime == "two-phase":
        cp = PropsSI("C", "P", p, "Q", 0, fluid)
        if c.is_n2o:
            mu_l, k = c._ecs(s["T"], s["rho_l"], True)
            mu_v, _ = c._ecs(s["T"], s["rho_v"], False)
        else:
            mu_l = PropsSI("V", "P", p, "Q", 0, fluid)
            mu_v = PropsSI("V", "P", p, "Q", 1, fluid)
            k = PropsSI("L", "P", p, "Q", 0, fluid)
        mu = mu_l
    else:
        cp = PropsSI("C", "H", h, "P", p, fluid)
        if c.is_n2o:
            liquid_side = regime == "liquid" or (regime == "supercritical" and rho > PropsSI("rhocrit", fluid))
            mu, k = c._ecs(T, rho, liquid_side)
        else:
            mu = PropsSI("V", "H", h, "P", p, fluid)
            k = PropsSI("L", "H", h, "P", p, fluid)
    return State(T=T, p=p, h=h, rho=rho, cp=cp, mu=mu, k=k, Pr=cp * mu / k, x=x,
                 Tsat=Tsat, regime=regime, mu_l=mu_l, mu_v=mu_v)


def h_from_Tp(name, T, p):
    """Specific enthalpy (J/kg) of coolant `name` at T (K) and p (Pa)."""
    return PropsSI("H", "T", T, "P", p, _get(name).fluid)
