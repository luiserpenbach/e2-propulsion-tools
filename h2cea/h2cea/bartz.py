"""Gas-side convective heat flux along the contour (handbook section 9).

Bartz (1957) in SI units with frozen CEA transport properties at chamber
stagnation conditions, and a prescribed gas-side wall temperature. This is the
first-cut load for hand checks; the coupled wall solution is regen.march().
"""
import numpy as np

from .gasdyn import mach_from_area


def bartz_profile(contour, pt, pc_bar, Twg=1000.0, factor=1.0, gamma=None,
                  rc_throat=None):
    """Return dict of arrays: x, r, M, Taw, hg, q (W/m2) and integrals.

    pt: cea_si.Point at the operating point (supplies Tc, c*, cp, mu, Pr, gamma).
    gamma: exponent for the local Mach number; default frozen chamber value
           (handbook 4.4: constant gamma is fine for M and T_aw, not for c*).
    """
    g = gamma or pt.gam_fr_c
    x, r = contour["x"], contour["r"]
    Rt, xt = contour["Rt"], contour["xt"]
    At = np.pi * Rt ** 2
    Dt = 2 * Rt
    rc = rc_throat or 0.5 * (1.5 * Rt + 0.382 * Rt)     # mean throat curvature radius
    pc = pc_bar * 1e5
    Tc = pt.Tc
    mu, cp, Pr = pt.mu_c, pt.cp_fr_c, pt.pr_fr_c
    eps_loc = (np.pi * r ** 2) / At
    M = np.array([mach_from_area(max(e, 1.0), g, xi > xt) for e, xi in zip(eps_loc, x)])
    t = 1 + 0.5 * (g - 1) * M ** 2
    rec = Pr ** (1 / 3)
    Taw = Tc * (1 + rec * 0.5 * (g - 1) * M ** 2) / t
    sigma = (0.5 * Twg / Tc * t + 0.5) ** -0.68 * t ** -0.12
    h0 = 0.026 / Dt ** 0.2 * (mu ** 0.2 * cp / Pr ** 0.6) * (pc / pt.cstar) ** 0.8 \
        * (Dt / rc) ** 0.1
    hg = factor * h0 * (At / (np.pi * r ** 2)) ** 0.9 * sigma
    q = hg * (Taw - Twg)
    # wetted area element along the wall
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(r) ** 2)
    rm = 0.5 * (r[1:] + r[:-1])
    qm = 0.5 * (q[1:] + q[:-1])
    dQ = qm * 2 * np.pi * rm * ds
    Q = float(dQ.sum())
    up = x[1:] <= xt
    Q_up = float(dQ[up].sum())
    cyl = x[1:] <= contour.get("x_cyl_end", 0.0) if "x_cyl_end" in contour else None
    return {"x": x, "r": r, "M": M, "Taw": Taw, "hg": hg, "q": q, "Q": Q,
            "Q_upstream": Q_up, "Q_downstream": Q - Q_up,
            "q_max": float(q.max()), "x_qmax": float(x[np.argmax(q)])}
