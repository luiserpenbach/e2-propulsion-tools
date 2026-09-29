"""Propellant cards for RocketCEA built from real-fluid states.

Why this matters: CEA solves an adiabatic, constant-pressure equilibrium, so the
answer depends on the enthalpy the reactants bring in. RocketCEA's built-in "N2O"
card is ideal GAS at 298.15 K. Our oxidiser leaves the tank as a compressed LIQUID,
about 260 kJ/kg lower in enthalpy. That difference is worth ~70 K of flame
temperature and ~1.5 % of c*, which is more than most of the losses we try to
estimate. See handbook section 5.4.
"""
from functools import lru_cache

from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import add_new_fuel, add_new_oxidizer, oxCards, fuelCards

CAL = 4.184                     # J/cal (thermochemical calorie, as used by CEA)
M_N2O = 44.0128e-3              # kg/mol
# Formation enthalpy of the built-in RocketCEA "N2O" card (ideal gas, 298.15 K):
# h,cal=19467.0  ->  81.45 kJ/mol. Using the same base keeps our liquid card
# consistent with the built-in gas card: the two differ only by the real-fluid
# enthalpy change between the tank state and the ideal-gas reference state.
HF_N2O_GAS_298 = 19467.0 * CAL  # J/mol


def n2o_enthalpy_below_ideal_gas(T, p_bar):
    """h_ig(298.15 K) - h(T, p) in J/kg from the CoolProp reference EOS."""
    h_ig = PropsSI("H", "T", 298.15, "P", 100.0, "NitrousOxide")   # ~ideal gas
    h = PropsSI("H", "T", T, "P", p_bar * 1e5, "NitrousOxide")
    return h_ig - h


@lru_cache(maxsize=None)
def n2o_card(T=293.15, p_bar=70.0, extra_h_kJkg=0.0):
    """Register an N2O oxidiser card at (T, p) and return its name.

    extra_h_kJkg adds enthalpy per kg of N2O, e.g. to study jacket heat that is
    returned to the chamber (handbook 5.4) or a hot tank.
    """
    dh = n2o_enthalpy_below_ideal_gas(T, p_bar) - extra_h_kJkg * 1e3   # J/kg
    hf = HF_N2O_GAS_298 - dh * M_N2O                                   # J/mol
    name = f"N2O_T{T*10:.0f}_P{p_bar*10:.0f}_H{extra_h_kJkg*10+100000:.0f}"   # alphanumeric only
    card = (f"oxid N2O  N 2 O 1  wt%=100.0\n"
            f"h,cal={hf / CAL:.1f}  t(k)={T:.2f}\n")
    if name not in oxCards:          # re-adding a name clears RocketCEA's run cache
        add_new_oxidizer(name, card)
    return name


@lru_cache(maxsize=None)
def ethanol_card(wt_pct=100.0):
    """Ethanol/water blend as two fuel species (liquid, 298.15 K)."""
    if wt_pct >= 100.0:
        return "Ethanol"
    name = f"Ethanol{wt_pct:.0f}"
    card = (f"fuel C2H5OH(L)  C 2 H 6 O 1  wt%={wt_pct:.2f}\n"
            f"h,cal=-66370.0  t(k)=298.15\n"
            f"fuel H2O(L)  H 2 O 1  wt%={100.0 - wt_pct:.2f}\n"
            f"h,cal=-68315.0  t(k)=298.15\n")
    if name not in fuelCards:
        add_new_fuel(name, card)
    return name
