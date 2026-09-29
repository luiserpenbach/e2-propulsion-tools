"""Throttle operating line with the fuel flow fixed (handbook sections 3.5 and 6).

The E2 feed system fixes the fuel flow with a cavitating venturi (0.20 kg/s) and
throttles the oxidiser only, so the mixture ratio moves with thrust. At each
oxidiser flow:

    mdot = mf (1 + MR)                              MR = mox / mf
    pc   = mdot * eta_cstar * cstar(pc, MR) / At    (solved by fixed point)
    F    = eta_CF * CF_vac(pc, MR) * pc * At - pa * Ae

The throat area is fixed by requiring F_MAX at PC_MAX with the same fuel flow.
"""
import numpy as np
from scipy.optimize import brentq

from . import config as C
from .cea_si import point


def pc_at(mdot, mr, At, eps=C.EPS, eta_c=C.ETA_CSTAR, ox=None, fuel=None):
    pc = 20.0
    for _ in range(30):
        cs = point(round(pc, 4), round(mr, 5), eps, C.P_AMB, ox, fuel).cstar
        new = mdot * eta_c * cs / At / 1e5
        if abs(new - pc) < 1e-5:
            break
        pc = new
    return pc


def state(mox, At, eps=C.EPS, mf=C.MDOT_FUEL, eta_c=C.ETA_CSTAR, eta_f=C.ETA_CF_VAC,
          ox=None, fuel=None, flat_eta_isp=None):
    """Delivered state at one oxidiser flow.

    flat_eta_isp: if given, reproduce the architecture-document method
    (delivered Isp = flat_eta_isp * ideal sea-level Isp, pc from ideal c*).
    """
    mr = mox / mf
    mdot = mox + mf
    if flat_eta_isp is not None:
        pc = pc_at(mdot, mr, At, eps, 1.0, ox, fuel)
        p = point(round(pc, 4), round(mr, 5), eps, C.P_AMB, ox, fuel)
        isp = flat_eta_isp * p.isp_amb
        F = isp * C.G0 * mdot
    else:
        pc = pc_at(mdot, mr, At, eps, eta_c, ox, fuel)
        p = point(round(pc, 4), round(mr, 5), eps, C.P_AMB, ox, fuel)
        F = eta_f * p.cf_vac * pc * 1e5 * At - C.P_AMB * 1e5 * eps * At
        isp = F / (mdot * C.G0)
    return {"mox": mox, "mf": mf, "mdot": mdot, "MR": mr, "pc": pc, "F": F,
            "isp": isp, "isp_ideal": p.isp_amb, "eta_isp": isp / p.isp_amb,
            "Tc": p.Tc, "cstar_ideal": p.cstar, "cf_vac": p.cf_vac,
            "pe": p.pe * pc / p.pc, "Me": p.Me, "gam_fr": p.gam_fr_c, "mw": p.mw_c,
            "pt": p}


def size_throat(F=C.F_MAX, pc=C.PC_MAX, eps=C.EPS, mf=C.MDOT_FUEL, eta_c=C.ETA_CSTAR,
                eta_f=C.ETA_CF_VAC, ox=None, fuel=None, flat_eta_isp=None):
    """Find the mixture ratio and throat area that give F at pc with fuel flow mf."""
    def resid(mr):
        p = point(pc, round(mr, 6), eps, C.P_AMB, ox, fuel)
        mdot = mf * (1 + mr)
        if flat_eta_isp is not None:
            At = mdot * p.cstar / (pc * 1e5)
            return flat_eta_isp * p.isp_amb * C.G0 * mdot - F
        At = mdot * eta_c * p.cstar / (pc * 1e5)
        return eta_f * p.cf_vac * pc * 1e5 * At - C.P_AMB * 1e5 * eps * At - F
    mr = brentq(resid, 2.5, 6.0, xtol=1e-6)
    p = point(pc, round(mr, 6), eps, C.P_AMB, ox, fuel)
    eta = 1.0 if flat_eta_isp is not None else eta_c
    At = mf * (1 + mr) * eta * p.cstar / (pc * 1e5)
    return mr, At


def thrust_to_mox(F_target, At, **kw):
    return brentq(lambda m: state(m, At, **kw)["F"] - F_target, 0.15, 1.2, xtol=1e-5)
