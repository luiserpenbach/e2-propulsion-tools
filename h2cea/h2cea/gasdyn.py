"""Ideal one-dimensional gas dynamics (handbook section 3)."""
import numpy as np
from scipy.optimize import brentq


def area_ratio(M, g):
    t = 1 + 0.5 * (g - 1) * M * M
    return (1.0 / M) * ((2.0 / (g + 1)) * t) ** ((g + 1) / (2 * (g - 1)))


def mach_from_area(eps, g, supersonic):
    """Invert A/A* with a bracketed root finder (never plain Newton near M = 1)."""
    if abs(eps - 1.0) < 1e-12:
        return 1.0
    f = lambda M: area_ratio(M, g) - eps
    return brentq(f, 1.0 + 1e-9, 20.0) if supersonic else brentq(f, 1e-6, 1.0 - 1e-9)


def p_ratio(M, g):
    """p/p0."""
    return (1 + 0.5 * (g - 1) * M * M) ** (-g / (g - 1))


def T_ratio(M, g):
    """T/T0."""
    return 1.0 / (1 + 0.5 * (g - 1) * M * M)


def big_gamma(g):
    return np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))


def cf_ideal(g, eps, pc, pa):
    """Ideal thrust coefficient for a calorically perfect gas (constant gamma)."""
    Me = mach_from_area(eps, g, True)
    pe_pc = p_ratio(Me, g)
    mom = np.sqrt(2 * g * g / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1))
                  * (1 - pe_pc ** ((g - 1) / g)))
    return mom + (pe_pc - pa / pc) * eps


def schmucker_psep_over_pa(M_sep):
    """Schmucker (1984) separation criterion: p_sep/p_a = (1.88 M - 1)^-0.64."""
    return (1.88 * M_sep - 1.0) ** -0.64


def summerfield_psep_over_pa():
    return 0.37   # middle of the 0.35-0.40 band


def rayleigh_p0_loss(eps_c, g):
    """Stagnation pressure loss of heat addition in a constant-area chamber.

    Exact Rayleigh result for heating from M ~ 0 to the chamber exit Mach number:
    p0_exit/p0_inj = (1+g)/(1+g M^2) * ((1 + (g-1)/2 M^2)/((g+1)/2))^(g/(g-1)) scaled
    to the M -> 0 inlet. Returns (M_c, 1 - p0_noz/p0_inj).
    """
    Mc = mach_from_area(eps_c, g, False)
    # Rayleigh line: p0/p0* as a function of M
    def p0_ratio(M):
        return (1 + g) / (1 + g * M * M) * ((2 + (g - 1) * M * M) / (g + 1)) ** (g / (g - 1))
    loss = 1.0 - p0_ratio(Mc) / p0_ratio(1e-6)
    return Mc, loss
