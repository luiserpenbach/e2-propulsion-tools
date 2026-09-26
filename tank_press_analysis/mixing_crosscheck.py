"""
mixing_crosscheck.py — Cross-check of the Dalton (additive-pressure) mixing rule
used for the N2 / N2O ullage in n2o_press.py against a cubic mixture EOS.

Question answered
-----------------
The sizing model fills the oxidizer ullage with
    rho_N2 (P_set - Psat(T), T)  +  rho_N2O,sat,v(T)
i.e. each component at its own partial pressure (Dalton). How far is that from a
mixture equation of state that includes N2–N2O interactions, and how much N2
dissolves into the liquid at P_set?

Method
------
Peng–Robinson EOS with van der Waals one-fluid mixing and an interaction
parameter k12 swept over a plausible range (N2–CO2 analogy: k12 ≈ -0.02 … 0).
Three quantities are compared at the same (T_liq, P_set):

  (a) Dalton with reference (CoolProp HEOS) pure-fluid properties  -> the model
  (b) Dalton with PR pure-fluid properties                         -> same rule, same EOS as (c)
  (c) PR mixture at the real-gas Dalton composition (y_i ~ rho_i/M_i) -> isolates the mixing effect
  (d) PR mixture at the full two-phase equilibrium composition     -> adds the Poynting /
      fugacity shift of y_N2O and gives dissolved N2 in the liquid

The ratio (c)/(b) and (d)/(b) is the correction the additive-pressure rule
misses, independent of PR's own pure-fluid error. Applying it to (a) gives the
best estimate of the true ullage inventory.

Run:  python mixing_crosscheck.py [--p-bar 100] [--t-c 0 10 20 30]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from CoolProp.CoolProp import PropsSI

R = 8.314462618
C0 = 273.15
FLUIDS = ("Nitrogen", "NitrousOxide")
M = np.array([PropsSI("M", f) for f in FLUIDS])                  # kg/mol
TC = np.array([PropsSI("Tcrit", f) for f in FLUIDS])
PC = np.array([PropsSI("pcrit", f) for f in FLUIDS])
W = np.array([PropsSI("acentric", f) for f in FLUIDS])


# ----------------------------------------------------------------------------
# Peng–Robinson
# ----------------------------------------------------------------------------

def _ab(T: float) -> tuple[np.ndarray, np.ndarray]:
    kappa = 0.37464 + 1.54226 * W - 0.26992 * W**2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(T / TC)))**2
    a = 0.45724 * R**2 * TC**2 / PC * alpha
    b = 0.07780 * R * TC / PC
    return a, b


def _z_roots(A: float, B: float) -> np.ndarray:
    c = [1.0, -(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)]
    r = np.roots(c)
    r = np.real(r[np.abs(np.imag(r)) < 1e-9])
    return np.sort(r[r > B])


def pr_state(T: float, P: float, z: np.ndarray, k12: float, phase: str):
    """Return (Z, rho_kg_m3, ln_phi[2]) for composition z at (T, P)."""
    a, b = _ab(T)
    K = np.array([[0.0, k12], [k12, 0.0]])
    aij = np.sqrt(np.outer(a, a)) * (1.0 - K)
    am = z @ aij @ z
    bm = z @ b
    A = am * P / (R * T)**2
    B = bm * P / (R * T)
    roots = _z_roots(A, B)
    if len(roots) == 0:
        raise ValueError("no PR root")
    Z = roots[-1] if phase == "v" else roots[0]
    s2 = math.sqrt(2.0)
    ln_phi = (b / bm * (Z - 1.0) - math.log(Z - B)
              - A / (2.0 * s2 * B) * (2.0 * (aij @ z) / am - b / bm)
              * math.log((Z + (1.0 + s2) * B) / (Z + (1.0 - s2) * B)))
    v_molar = Z * R * T / P
    rho = (z @ M) / v_molar
    return Z, rho, ln_phi


def pr_pure_density(T: float, P: float, i: int, phase: str) -> float:
    z = np.zeros(2); z[i] = 1.0
    return pr_state(T, P, z, 0.0, phase)[1]


def pr_vle(T: float, P: float, k12: float, y1_guess: float, x1_guess: float = 0.03):
    """Two-phase equilibrium of the binary at fixed (T, P) by successive
    substitution on K-values (K_i = phi_i^L / phi_i^V). For a binary at fixed
    T, P the compositions follow directly from the two K-values:
        x1 = (1 - K2) / (K1 - K2),   y1 = K1 * x1.
    Raises if the phases collapse onto the trivial solution (x == y)."""
    y1, x1 = y1_guess, x1_guess
    for it in range(500):
        y = np.array([y1, 1.0 - y1]); x = np.array([x1, 1.0 - x1])
        Zv, rho_v, lnphi_v = pr_state(T, P, y, k12, "v")
        Zl, rho_l, lnphi_l = pr_state(T, P, x, k12, "l")
        K = np.exp(lnphi_l - lnphi_v)
        if not np.all(np.isfinite(K)) or abs(K[0] - K[1]) < 1e-12:
            raise RuntimeError("K-values degenerate — no distinct two-phase state")
        x1_new = (1.0 - K[1]) / (K[0] - K[1])
        x1_new = min(max(x1_new, 1e-6), 1.0 - 1e-6)
        y1_new = min(max(K[0] * x1_new, 1e-6), 1.0 - 1e-6)
        # damped update for robustness near the critical locus
        x1 = 0.5 * x1 + 0.5 * x1_new
        y1 = 0.5 * y1 + 0.5 * y1_new
        if abs(x1_new - x1) < 1e-10 and abs(y1_new - y1) < 1e-10:
            break
    y = np.array([y1, 1.0 - y1]); x = np.array([x1, 1.0 - x1])
    Zv, rho_v, _ = pr_state(T, P, y, k12, "v")
    Zl, rho_l, _ = pr_state(T, P, x, k12, "l")
    if abs(y1 - x1) < 1e-3 or rho_v > 0.7 * rho_l:
        raise RuntimeError("trivial / near-critical solution — no distinct two-phase state")
    return y, x, rho_v, rho_l, Zv, Zl


# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------

def compare(T: float, P: float, k12_list) -> dict:
    psat = PropsSI("P", "T", T, "Q", 0, "NitrousOxide")
    p_n2 = P - psat
    y_pp = np.array([p_n2 / P, psat / P])          # ideal-gas partial-pressure fractions (not used for real gas)

    # (a) model: Dalton with reference properties
    rho_n2_ref = PropsSI("D", "P", p_n2, "T", T, "Nitrogen")
    rho_vap_ref = PropsSI("D", "T", T, "Q", 1, "NitrousOxide")
    a_n2, a_n2o = rho_n2_ref, rho_vap_ref

    # (b) Dalton with PR pure properties
    b_n2 = pr_pure_density(T, p_n2, 0, "v")
    b_n2o = pr_pure_density(T, psat, 1, "v")

    # Real-gas Dalton composition: n_i proportional to rho_i / M_i
    n = np.array([b_n2 / M[0], b_n2o / M[1]])
    y_dalton = n / n.sum()

    out = {"T_K": T, "psat_bar": psat / 1e5, "y_n2o_dalton": y_dalton[1],
           "a_n2": a_n2, "a_n2o": a_n2o, "b_n2": b_n2, "b_n2o": b_n2o, "k": {}}

    for k12 in k12_list:
        # (c) PR mixture at Dalton composition
        _, rho_c, _ = pr_state(T, P, y_dalton, k12, "v")
        w_n2o_c = y_dalton[1] * M[1] / (y_dalton @ M)
        c_n2, c_n2o = rho_c * (1 - w_n2o_c), rho_c * w_n2o_c
        # (d) PR full VLE
        try:
            y, x, rho_d, rho_l, Zv, Zl = pr_vle(T, P, k12, y_dalton[0])
            w_n2o_d = y[1] * M[1] / (y @ M)
            d_n2, d_n2o = rho_d * (1 - w_n2o_d), rho_d * w_n2o_d
            w_n2_liq = x[0] * M[0] / (x @ M)
            vle_ok = True
        except (RuntimeError, ValueError):
            y = np.array([np.nan, np.nan]); x = y.copy()
            d_n2 = d_n2o = w_n2_liq = rho_l = Zv = Zl = float("nan")
            vle_ok = False
        out["k"][k12] = {"vle_ok": vle_ok, "Zv": Zv, "Zl": Zl, "rho_liq_vle": rho_l,
            "c_n2": c_n2, "c_n2o": c_n2o,
            "d_n2": d_n2, "d_n2o": d_n2o, "y_n2o_vle": y[1],
            "x_n2_liq_mol": x[0], "w_n2_liq": w_n2_liq,
            # corrections relative to Dalton with the SAME (PR) pure properties
            "f_n2_mix": c_n2 / b_n2, "f_n2o_mix": c_n2o / b_n2o,
            "f_n2_vle": d_n2 / b_n2, "f_n2o_vle": d_n2o / b_n2o,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-bar", type=float, default=100.0)
    ap.add_argument("--t-c", type=float, nargs="+", default=[0.0, 10.0, 20.0, 30.0])
    ap.add_argument("--k12", type=float, nargs="+", default=[-0.02, 0.0, 0.03, 0.06])
    args = ap.parse_args(argv)
    P = args.p_bar * 1e5

    print(f"N2 / N2O ullage mixing cross-check at P_set = {args.p_bar:.0f} bar")
    print(f"PR inputs: Tc={TC.round(2)} K, Pc={(PC/1e5).round(2)} bar, w={W.round(4)}\n")

    for tc in args.t_c:
        r = compare(tc + C0, P, args.k12)
        print(f"--- T_liq = {tc:5.1f} C | Psat = {r['psat_bar']:.2f} bar | "
              f"Psat/P = {r['psat_bar']*1e5/P:.3f}")
        print(f"  (a) model  Dalton/ref :  N2 {r['a_n2']:7.2f}  N2O {r['a_n2o']:7.2f}  kg/m3")
        print(f"  (b) Dalton/PR         :  N2 {r['b_n2']:7.2f}  N2O {r['b_n2o']:7.2f}  kg/m3  "
              f"(y_N2O real-gas Dalton = {r['y_n2o_dalton']:.3f})"
              f"   (PR pure-fluid error: N2 {100*(r['b_n2']/r['a_n2']-1):+.1f} %,"
              f" N2O {100*(r['b_n2o']/r['a_n2o']-1):+.1f} %)")
        for k12, k in r["k"].items():
            print(f"  k12={k12:+.2f}  (c) PR mix @Dalton y :  N2 {k['c_n2']:7.2f}  N2O {k['c_n2o']:7.2f}"
                  f"   -> mixing corr. N2 x{k['f_n2_mix']:.3f}  N2O x{k['f_n2o_mix']:.3f}")
            print(f"            (d) PR full VLE       :  N2 {k['d_n2']:7.2f}  N2O {k['d_n2o']:7.2f}"
                  f"   -> total corr.  N2 x{k['f_n2_vle']:.3f}  N2O x{k['f_n2o_vle']:.3f}"
                  f"   | y_N2O = {k['y_n2o_vle']:.3f}  Zv={k['Zv']:.3f} Zl={k['Zl']:.3f}"
                  f"  | dissolved N2 in liquid: {100*k['x_n2_liq_mol']:.2f} mol% = {100*k['w_n2_liq']:.2f} wt%"
                  f"  | rho_liq {k['rho_liq_vle']:.0f} kg/m3")
        print()


if __name__ == "__main__":
    main()