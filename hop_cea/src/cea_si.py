"""SI front end for RocketCEA.

One function, `point()`, returns everything a thrust chamber analysis needs at one
(pc, O/F, eps, pamb) in SI units, cached. Nothing downstream should call RocketCEA
directly: the base class works in psia/degR/ft/s, and several return values (the
`mode` string of estimate_Ambient_Isp, for example) stay in psia even in the
unit-aware class.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache

import rocketcea
from rocketcea.cea_obj_w_units import CEA_Obj

from .config import G0, P_AMB
from .propellants import n2o_card, ethanol_card

R_UNIV = 8314.462618  # J/(kmol K)


_OBJ = {}


def cea_obj(ox, fuel, fac_CR=None):
    """Cached unit-aware CEA object.

    RocketCEA clears its internal result cache whenever a new propellant card is
    added, which invalidates older objects (KeyError in get_Cstar). Re-create the
    object if its entry has gone.
    """
    import rocketcea.cea_obj as base
    key = (ox, fuel, fac_CR)
    o = _OBJ.get(key)
    if o is not None and o.cea_obj.desc in base._CacheObjDict:
        return o
    o = _make(ox, fuel, fac_CR)
    _OBJ[key] = o
    return o


def _make(ox, fuel, fac_CR=None):
    return CEA_Obj(oxName=ox, fuelName=fuel, fac_CR=fac_CR,
                   pressure_units="bar", temperature_units="K",
                   cstar_units="m/s", sonic_velocity_units="m/s",
                   isp_units="sec", enthalpy_units="kJ/kg",
                   density_units="kg/m^3", specific_heat_units="J/kg-K",
                   viscosity_units="poise",          # x0.1 -> Pa s
                   thermal_cond_units="W/cm-degC")   # x100 -> W/(m K)


@dataclass
class Point:
    ox: str
    fuel: str
    pc: float          # bar
    mr: float
    eps: float
    pamb: float        # bar
    Tc: float          # K, chamber (nozzle inlet stagnation)
    Tt: float          # K, throat static, shifting equilibrium
    Te: float          # K, exit static
    cstar: float       # m/s, ideal
    isp_vac: float     # s, shifting equilibrium
    isp_vac_frozen: float  # s, frozen at throat
    cf_vac: float
    cf_amb: float      # ideal, attached flow: cf_vac - pamb/pc*eps
    isp_amb: float     # s, ideal, attached flow
    pe: float          # bar
    Me: float
    mw_c: float        # kg/kmol
    gam_eq_c: float    # CEA isentropic exponent, chamber (equilibrium)
    gam_fr_c: float    # frozen cp/cv, chamber
    mw_t: float
    gam_eq_t: float
    cp_fr_c: float     # J/(kg K) frozen
    mu_c: float        # Pa s
    k_fr_c: float      # W/(m K) frozen
    pr_fr_c: float
    cp_eq_c: float     # J/(kg K) equilibrium (includes reaction)
    a_c: float         # m/s, sonic velocity in chamber
    rocketcea_version: str = rocketcea.__version__

    def as_dict(self):
        return asdict(self)


@lru_cache(maxsize=200_000)
def point(pc, mr, eps, pamb=P_AMB, ox=None, fuel=None, fac_CR=None):
    ox = ox or n2o_card()
    fuel = fuel or ethanol_card()
    c = cea_obj(ox, fuel, fac_CR)
    Tc, Tt, Te = c.get_Temperatures(Pc=pc, MR=mr, eps=eps)
    cstar = c.get_Cstar(Pc=pc, MR=mr)
    isp_v = c.get_Isp(Pc=pc, MR=mr, eps=eps)
    isp_vf = c.get_Isp(Pc=pc, MR=mr, eps=eps, frozen=1, frozenAtThroat=1)
    mw_c, g_c = c.get_Chamber_MolWt_gamma(Pc=pc, MR=mr, eps=eps)
    mw_t, g_t = c.get_Throat_MolWt_gamma(Pc=pc, MR=mr, eps=eps)
    cp_fr, mu, k_fr, pr_fr = c.get_Chamber_Transport(Pc=pc, MR=mr, eps=eps, frozen=1)
    cp_eq = c.get_Chamber_Cp(Pc=pc, MR=mr, eps=eps, frozen=0)
    pc_pe = c.get_PcOvPe(Pc=pc, MR=mr, eps=eps)
    Me = c.get_MachNumber(Pc=pc, MR=mr, eps=eps)
    a_c = c.get_Chamber_SonicVel(Pc=pc, MR=mr, eps=eps)
    R = R_UNIV / mw_c
    cf_vac = isp_v * G0 / cstar
    cf_amb = cf_vac - pamb / pc * eps
    return Point(ox, fuel, pc, mr, eps, pamb, float(Tc), float(Tt), float(Te),
                 float(cstar), float(isp_v), float(isp_vf), float(cf_vac),
                 float(cf_amb), float(cstar * cf_amb / G0), float(pc / pc_pe),
                 float(Me), float(mw_c), float(g_c),
                 float(cp_fr / (cp_fr - R)), float(mw_t), float(g_t),
                 float(cp_fr), float(mu) * 0.1, float(k_fr) * 100.0, float(pr_fr),
                 float(cp_eq), float(a_c))


def full_output(pc, mr, eps, ox=None, fuel=None):
    """Full CEA printout in SI units, for archiving next to a design document.

    Note: get_full_cea_output lives on the base object (`.cea_obj`) and takes the
    pressure in the units named by pc_units.
    """
    c = cea_obj(ox or n2o_card(), fuel or ethanol_card())
    return c.cea_obj.get_full_cea_output(Pc=pc, MR=mr, eps=eps, pc_units="bar",
                                         output="siunits", short_output=0,
                                         show_transport=1)


def species(pc, mr, eps, ox=None, fuel=None, where=0, min_fraction=1e-4):
    """Mole fractions; where = 0 chamber, 1 throat, 2 exit."""
    c = cea_obj(ox or n2o_card(), fuel or ethanol_card())
    _, d = c.get_SpeciesMoleFractions(Pc=pc, MR=mr, eps=eps, min_fraction=min_fraction)
    return {k.strip('*'): v[where] for k, v in d.items()}


def pinj_over_pcomb(pc, mr, fac_CR, ox=None, fuel=None):
    c = cea_obj(ox or n2o_card(), fuel or ethanol_card(), fac_CR)
    return float(c.get_Pinj_over_Pcomb(Pc=pc, MR=mr, fac_CR=fac_CR))
