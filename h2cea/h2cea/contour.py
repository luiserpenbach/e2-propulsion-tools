"""Chamber and nozzle contour (handbook sections 7 and 8).

Coordinates: x along the axis from the injector face, r radius, both in metres.
Default parameters are the E2-REG-1 as-built values with the baseline area ratio 4.
"""
import numpy as np

from . import config as C


def e2_contour(eps=C.EPS, n=1500, Rt=C.R_T, Rc=C.R_C, L_cyl=C.L_CYL, R1=C.R_CONV1,
               th_conv=C.THETA_CONV, Ru=None, Rd=None, K=C.BELL_FRACTION,
               th_n=C.THETA_N, th_e=C.THETA_E, L_bell=None):
    Ru = Ru if Ru is not None else 1.5 * Rt
    Rd = Rd if Rd is not None else 0.382 * Rt
    a = np.radians(th_conv)
    # --- convergent: cylinder, blend arc R1, cone, throat arc Ru -----------------
    x1 = L_cyl
    xa_end = x1 + R1 * np.sin(a)
    ra_end = Rc - R1 * (1 - np.cos(a))
    rb_start = Rt + Ru * (1 - np.cos(a))
    dx_cone = (ra_end - rb_start) / np.tan(a)
    if dx_cone < 0:
        raise ValueError("arcs too large for this contraction ratio")
    xt = xa_end + dx_cone + Ru * np.sin(a)

    def r_of_x(x):
        if x <= x1:
            return Rc
        if x <= xa_end:
            th = np.arcsin((x - x1) / R1)
            return Rc - R1 * (1 - np.cos(th))
        if x <= xa_end + dx_cone:
            return ra_end - (x - xa_end) * np.tan(a)
        if x <= xt:
            th = np.arcsin((xt - x) / Ru)
            return Rt + Ru * (1 - np.cos(th))
        return None

    xs_conv = np.linspace(0, xt, n)
    rs_conv = np.array([r_of_x(x) for x in xs_conv])

    # --- divergent: arc Rd to inflection N, then Rao parabola (quadratic Bezier) ---
    tn, te = np.radians(th_n), np.radians(th_e)
    xN, rN = Rd * np.sin(tn), Rt + Rd * (1 - np.cos(tn))
    xE = L_bell if L_bell is not None else K * Rt * (np.sqrt(eps) - 1) / np.tan(np.radians(15))
    rE = np.sqrt(eps) * Rt
    xQ = (rE - rN + xN * np.tan(tn) - xE * np.tan(te)) / (np.tan(tn) - np.tan(te))
    rQ = rN + (xQ - xN) * np.tan(tn)
    th = np.linspace(0, tn, 60)[1:]
    xs_arc = Rd * np.sin(th)
    rs_arc = Rt + Rd * (1 - np.cos(th))
    s = np.linspace(0, 1, n)[1:]
    xs_bell = (1 - s) ** 2 * xN + 2 * s * (1 - s) * xQ + s * s * xE
    rs_bell = (1 - s) ** 2 * rN + 2 * s * (1 - s) * rQ + s * s * rE
    x = np.concatenate([xs_conv, xt + xs_arc, xt + xs_bell])
    r = np.concatenate([rs_conv, rs_arc, rs_bell])
    return {"x": x, "r": r, "xt": xt, "Rt": Rt, "Rc": Rc, "eps": eps,
            "L_chamber": xt, "L_nozzle": xE, "Ru": Ru, "Rd": Rd, "x_cyl_end": x1}


def e2_reg1_asbuilt(n=1500):
    """E2-REG-1 as built: same convergent, 43.6 mm bell to eps 2.99 (18.8 / 8 deg)."""
    return e2_contour(eps=C.EPS_E2, n=n, th_n=C.THETA_N_E2, th_e=C.THETA_E_E2,
                      L_bell=C.L_BELL_E2)


def wall_stations(c, n=None):
    """Resample the contour at n stations equally spaced along the wall.

    Returns x, r, s (arc length from the injector face), and the outward unit
    normal (nx, nr) of the gas-side wall, all in metres.
    """
    n = n or C.N_STATIONS
    x, r = c["x"], c["r"]
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(r)) > 1e-12]
    x, r = x[keep], r[keep]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(r)))]
    S = np.linspace(0.0, s[-1], n)
    X, Rr = np.interp(S, s, x), np.interp(S, s, r)
    dx, dr = np.gradient(X, S), np.gradient(Rr, S)
    L = np.hypot(dx, dr)
    return {"x": X, "r": Rr, "s": S, "nx": -dr / L, "nr": dx / L, "xt": c["xt"],
            "Rt": c["Rt"], "Ru": c["Ru"], "Rd": c["Rd"]}


def chamber_volume(c):
    """Volume from injector face to throat (for L*)."""
    m = c["x"] <= c["xt"] + 1e-12
    x, r = c["x"][m], c["r"][m]
    return float(np.trapezoid(np.pi * r * r, x))


def L_star(c):
    return chamber_volume(c) / (np.pi * c["Rt"] ** 2)


def cone_length(Rt, eps, alpha_deg=15.0, Rd_frac=0.382):
    a = np.radians(alpha_deg)
    return (Rt * (np.sqrt(eps) - 1) + Rd_frac * Rt * (1 / np.cos(a) - 1)) / np.tan(a)
