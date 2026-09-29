"""Regenerative cooling: one-dimensional channel march (handbook sections 9, 10.3, 10.8).

At every wall station the gas side (Bartz with CEA transport, sigma at the actual
wall temperature), the hot wall (conductivity at the mean wall temperature) and the
coolant side (Gnielinski with rib fin efficiency) are solved as a series circuit;
the coolant enthalpy and pressure are then advanced to the next station. The
coolant state is carried as (h, p), so the same march works for water, subcooled
or boiling nitrous oxide and supercritical nitrous oxide.

Coolant-side heat transfer by regime
    liquid, vapour, supercritical   Gnielinski with bulk properties
    two-phase (0 < x < 1)           tp_model="liquid_only" (default): Gnielinski,
                                    whole flow as liquid, no boiling credit.
                                    This is NOT conservative for N2O: the
                                    regime-switching model (tp_model="regime",
                                    see twophase.py) gives a much lower coolant-side
                                    coefficient and a hotter wall. Use "regime" for
                                    the N2O design case.
The critical heat flux is checked with Hall-Mudawar (2000), outlet form, wherever
the pressure is subcritical; onset of nucleate boiling is flagged when the
coolant-side wall exceeds the local saturation temperature.

Pressure: Darcy-Weisbach (Haaland friction factor, homogeneous density, McAdams
two-phase viscosity) plus the acceleration term G^2 (1/rho_out - 1/rho_in).
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import materials as mat
from .coolant import Coolant
from .gasdyn import mach_from_area
from .jacket import Jacket, channels
from . import twophase as TP


# --------------------------------------------------------------------------- correlations
def friction_haaland(Re, rel_rough):
    if Re < 2300.0:
        return 64.0 / Re
    return (-1.8 * np.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def nusselt_gnielinski(Re, Pr, f):
    """Gnielinski (1976); laminar fully developed value 4.36 below Re 2300, linear
    blend 2300-4000 (Gnielinski's own recommendation for the transition)."""
    def turb(Re_):
        return (f / 8) * (Re_ - 1000.0) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))
    if Re < 2300.0:
        return 4.36
    if Re < 4000.0:
        g = (Re - 2300.0) / 1700.0
        return (1 - g) * 4.36 + g * turb(4000.0)
    return turb(Re)


def fin_efficiency(hc, k_wall, t_rib, h_ch):
    m = np.sqrt(2 * hc / (k_wall * t_rib))
    return np.tanh(m * h_ch) / (m * h_ch)


def chf_hall_mudawar(G, D, sat, x):
    """Hall and Mudawar (2000) CHF, outlet-conditions form (see n2o.py)."""
    hfg = sat["h_v"] - sat["h_l"]
    We = G * G * D / (sat["rho_l"] * sat["sigma"])
    rr = sat["rho_l"] / sat["rho_v"]
    Bo = 0.0722 * We ** -0.312 * rr ** -0.644 * (1 - 0.900 * rr ** 0.724 * x)
    return max(Bo, 0.0) * G * hfg


X_CHF_MAX = 0.05   # highest quality at which the Hall-Mudawar CHF estimate is used


def bartz_sigma(Twg, Tc, X):
    return (0.5 * Twg / Tc * X + 0.5) ** -0.68 * X ** -0.12


# --------------------------------------------------------------------------- result
@dataclass
class RegenResult:
    case: str
    coolant: str
    mdot: float
    arrays: dict
    scalars: dict
    jacket: Jacket
    gas: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.arrays[k] if k in self.arrays else self.scalars[k]

    def station(self, i):
        return {k: (v[i] if hasattr(v, "__len__") and len(v) == len(self.arrays["x"]) else v)
                for k, v in self.arrays.items()}

    def at_x(self, x):
        return int(np.argmin(np.abs(self.arrays["x"] - x)))

    def dataframe(self):
        import pandas as pd
        return pd.DataFrame({k: v for k, v in self.arrays.items() if np.ndim(v) == 1})


# --------------------------------------------------------------------------- march
def march(wall, pt, pc_bar, jacket: Jacket, coolant, mdot, p_in_bar, h_in, flow="counter",
          bartz_factor=C.BARTZ_FACTOR, rc_throat=C.RC_THROAT_BARTZ, case="",
          tp_model="liquid_only", film_subcooled=False):
    """March the coolant along the wall stations.

    wall     contour.wall_stations(...) (x, r, s, nx, nr, xt, Rt, Ru, Rd)
    pt       cea_si.Point at the operating point (Tc, c*, frozen gamma, CEA transport)
    pc_bar   delivered chamber pressure
    coolant  'water', 'n2o' or a Coolant
    mdot     total coolant flow (kg/s); blocked channels in jacket.blocked carry none
    h_in     coolant specific enthalpy at the jacket inlet (J/kg)
    flow     'counter' (enters at the nozzle exit) or 'co' (enters at the injector end)
    tp_model coolant-side model for boiling coolant, see twophase.MODELS
    film_subcooled  also apply the film/near-critical model to subcooled liquid
             when the coolant-side wall is above the limiting liquid superheat
    """
    cool = coolant if isinstance(coolant, Coolant) else Coolant(coolant)
    m = mat.get(jacket.material)
    x, r, s = wall["x"], wall["r"], wall["s"]
    N = len(x)
    Rt, xt = wall["Rt"], wall["xt"]
    At, Dt = np.pi * Rt ** 2, 2 * Rt
    rc = rc_throat or 0.5 * (wall["Ru"] + wall["Rd"])
    g, Tc = pt.gam_fr_c, pt.Tc
    pc = pc_bar * 1e5
    h0 = bartz_factor * 0.026 / Dt ** 0.2 * (pt.mu_c ** 0.2 * pt.cp_fr_c / pt.pr_fr_c ** 0.6) \
        * (pc / pt.cstar) ** 0.8 * (Dt / rc) ** 0.1
    rec = pt.pr_fr_c ** (1 / 3)
    geo = channels(r, jacket)
    n_act = jacket.n - len(jacket.blocked)
    G_all = mdot / (n_act * geo["A_one"])
    n_unconverged = 0

    names = ["M", "X", "Taw", "hg0", "sigma", "hg", "q", "Twg", "Twc", "Tb", "p", "h", "rho", "v",
             "Re", "Pr", "f", "hc", "eta_fin", "hce", "kw", "Tsat", "xq", "chf", "chf_ratio",
             "Rg", "Rw", "Rc", "G", "hc_lo", "q_c", "T_sl", "q_nb_max"]
    A = {k: np.full(N, np.nan) for k in names}
    regime = [""] * N
    order = range(N - 1, -1, -1) if flow == "counter" else range(N)
    h, p = h_in, p_in_bar * 1e5
    props = TP.Props(cool) if tp_model != "liquid_only" else None
    for i in order:
        AR = np.pi * r[i] ** 2 / At
        M = 1.0 if AR < 1.0 + 1e-9 else mach_from_area(AR, g, x[i] > xt)
        X = 1 + 0.5 * (g - 1) * M * M
        Taw = Tc * (1 + rec * 0.5 * (g - 1) * M * M) / X
        hg0 = h0 * AR ** -0.9
        st = cool.state(h, p)
        G = G_all[i]
        D = geo["Dh"][i]
        two_phase = st.regime == "two-phase"
        mu_ht = st.mu_l if two_phase else st.mu
        Re = G * D / mu_ht
        f = friction_haaland(Re, jacket.roughness / D)
        hc = nusselt_gnielinski(Re, st.Pr, f) * st.k / D
        h_lo = hc
        w = geo["w"][i]
        sat = cool.sat(p)
        tsat = props.sat(p) if (props is not None and sat is not None) else None
        use_tp = tsat is not None and (two_phase or (film_subcooled and st.regime == "liquid"))
        Twg, Twc = 800.0, st.T + 50.0
        q = None
        for it in range(400):
            sig = bartz_sigma(Twg, Tc, X)
            hg = hg0 * sig
            kw = float(mat.k(m, 0.5 * (Twg + Twc)))
            eta = fin_efficiency(hc, kw, jacket.t_rib, jacket.h)
            if use_tp and q is not None:
                q_c = q * (w + jacket.t_rib) / (w + 2 * eta * jacket.h)
                active = two_phase or Twc > tsat["T_sl"]
                ph_pf = (w + 2 * jacket.h) / (2 * w + 2 * jacket.h)
                hc_new = (TP.coolant_htc(tp_model, G, D, st, tsat, props, Twc, q_c, h_lo, ph_pf)
                          if active else h_lo)
                hc = 0.5 * hc + 0.5 * hc_new
                eta = fin_efficiency(hc, kw, jacket.t_rib, jacket.h)
            hce = hc * (w + 2 * eta * jacket.h) / (w + jacket.t_rib)
            q_old = q
            q = (Taw - st.T) / (1 / hg + jacket.t_wall / kw + 1 / hce)
            Twg_new, Twc = Taw - q / hg, st.T + q / hce
            done = abs(Twg_new - Twg) < 0.02 and (q_old is None or abs(q - q_old) < 1e-5 * q)
            Twg = Twg_new if done else 0.5 * (Twg + Twg_new)
            if done and (not use_tp or it > 3):
                break
        else:
            n_unconverged += 1
        # Hall-Mudawar is a subcooled / low-quality correlation; beyond x ~ 0.05 the
        # limit is dryout, which it does not describe, so it is not evaluated there.
        chf = chf_hall_mudawar(G, D, sat, st.x) if sat is not None and st.x <= X_CHF_MAX else np.nan
        vals = dict(M=M, X=X, Taw=Taw, hg0=hg0, sigma=sig, hg=hg, q=q, Twg=Twg, Twc=Twc,
                    Tb=st.T, p=p, h=h, rho=st.rho, v=G / st.rho, Re=Re, Pr=st.Pr, f=f, hc=hc,
                    eta_fin=eta, hce=hce, kw=kw, Tsat=st.Tsat, xq=st.x, chf=chf,
                    chf_ratio=q / chf if chf and np.isfinite(chf) and chf > 0 else np.nan,
                    Rg=1 / hg, Rw=jacket.t_wall / kw, Rc=1 / hce, G=G, hc_lo=h_lo,
                    q_c=q * (w + jacket.t_rib) / (w + 2 * eta * jacket.h),
                    T_sl=tsat["T_sl"] if tsat else (TP.T_superheat_limit(sat["T"], cool.Tcrit) if sat else np.nan),
                    q_nb_max=np.nan)
        for k_, v_ in vals.items():
            A[k_][i] = v_
        regime[i] = st.regime
        # ---- advance to the next station in flow direction
        nxt = i - 1 if flow == "counter" else i + 1
        if 0 <= nxt < N:
            ds = abs(s[nxt] - s[i])
            rm = 0.5 * (r[i] + r[nxt])
            h = h + q * 2 * np.pi * rm * ds / mdot
            # friction with homogeneous density; McAdams viscosity for the Reynolds number
            if two_phase:
                xx = min(max(st.x, 0.0), 1.0)
                mu_tp = 1.0 / (xx / st.mu_v + (1 - xx) / st.mu_l)
                f = friction_haaland(G * D / mu_tp, jacket.roughness / D)
            dp_f = f * ds / D * G * G / (2 * st.rho)
            st_new = cool.state(h, p - dp_f)
            dp_a = G * G * (1 / st_new.rho - 1 / st.rho)
            p = p - dp_f - dp_a
            if p <= 1e5:
                raise RuntimeError(f"coolant pressure fell below 1 bar at x = {x[nxt]*1e3:.1f} mm")
    A.update(x=x, r=r, s=s, nx=wall["nx"], nr=wall["nr"], w=geo["w"], Dh=geo["Dh"],
             pitch=geo["pitch"], A_flow=geo["A_flow"])
    A["regime"] = np.array(regime)
    pr = A["p"] / cool.pcrit
    ok = np.isfinite(A["T_sl"]) & (pr < 1)
    A["q_nb_max"][ok] = [TP.q_nb_max(a, b) for a, b in zip(pr[ok], (A["T_sl"] - A["Tsat"])[ok])]
    A["onb"] = (A["Twc"] > A["Tsat"]) & (A["xq"] < 0)
    # boiling margin is meaningful only while the coolant is still subcooled liquid
    A["margin_sat"] = np.where(A["regime"] == "liquid", A["Tsat"] - A["Twc"], np.nan)

    # ---- scalars
    ds = np.diff(s)
    rm = 0.5 * (r[1:] + r[:-1])
    qm = 0.5 * (A["q"][1:] + A["q"][:-1])
    Q = float(np.sum(qm * 2 * np.pi * rm * ds))
    i_out = 0 if flow == "counter" else N - 1
    i_in = N - 1 - i_out
    st_out = cool.state(A["h"][i_out], A["p"][i_out])
    dh = A["h"][i_out] - h_in
    it = int(np.nanargmax(A["Twg"]))
    ic = int(np.nanargmax(A["Twc"]))
    if n_unconverged:
        warnings.warn(f"regen.march ({case}): wall temperature iteration not converged at "
                      f"{n_unconverged} of {N} stations", RuntimeWarning, stacklevel=2)
    sc = dict(case=case, coolant=cool.name, mdot=mdot, pc_bar=pc_bar, MR=pt.mr, Tc=Tc,
              Q=Q, energy_balance_err=abs(mdot * dh - Q) / Q,
              p_in_bar=p_in_bar, p_out_bar=A["p"][i_out] / 1e5,
              dp_bar=(p_in_bar * 1e5 - A["p"][i_out]) / 1e5,
              T_in=A["Tb"][i_in], T_out=st_out.T, h_in=h_in, h_out=A["h"][i_out],
              x_out=st_out.x, regime_out=st_out.regime,
              Twg_max=float(A["Twg"][it]), x_Twg_max=float(x[it]),
              Twc_max=float(A["Twc"][ic]), x_Twc_max=float(x[ic]),
              q_max=float(np.nanmax(A["q"])), v_min=float(np.nanmin(A["v"])),
              chf_ratio_max=float(np.nanmax(A["chf_ratio"])) if np.any(np.isfinite(A["chf_ratio"])) else np.nan,
              margin_sat_min=float(np.nanmin(A["margin_sat"])) if np.any(np.isfinite(A["margin_sat"])) else np.nan,
              onb_any=bool(A["onb"].any()), two_phase_any=bool((A["regime"] == "two-phase").any()),
              x_onb=_first_x(x, A["onb"], flow), x_sat=_first_x(x, A["regime"] == "two-phase", flow),
              x_dry=_first_x(x, A["regime"] == "vapour", flow),
              frac_two_phase=float(np.mean(A["regime"] == "two-phase")),
              frac_vapour=float(np.mean(A["regime"] == "vapour")),
              bartz_factor=bartz_factor, flow=flow, n_stations=N, tp_model=tp_model)
    return RegenResult(case=case, coolant=cool.name, mdot=mdot, arrays=A, scalars=sc,
                       jacket=jacket, gas=dict(Tc=Tc, gamma=g, cstar=pt.cstar, pc_bar=pc_bar,
                                               MR=pt.mr, h0=h0, rc_throat=rc, xt=xt))


def _first_x(x, mask, flow):
    """Axial position where mask first becomes true in the flow direction (nan if never)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return float("nan")
    return float(x[idx.max()] if flow == "counter" else x[idx.min()])


def march_to_outlet_pressure(wall, pt, pc_bar, jacket, coolant, mdot, p_out_bar, h_in,
                             p_guess_bar=None, tol_bar=0.02, **kw):
    """Find the jacket inlet pressure that delivers p_out_bar at the jacket outlet
    (for example the oxidiser injector inlet pressure). Secant iteration."""
    p0 = p_guess_bar or p_out_bar + 10.0
    r0 = march(wall, pt, pc_bar, jacket, coolant, mdot, p0, h_in, **kw)
    e0 = r0["p_out_bar"] - p_out_bar
    p1 = p0 - e0
    for _ in range(20):
        r1 = march(wall, pt, pc_bar, jacket, coolant, mdot, p1, h_in, **kw)
        e1 = r1["p_out_bar"] - p_out_bar
        if abs(e1) < tol_bar:
            return r1
        p0, p1, e0 = p1, p1 - e1 * (p1 - p0) / (e1 - e0), e1
    warnings.warn(f"march_to_outlet_pressure ({kw.get('case', '')}): outlet pressure off by "
                  f"{e1:.3f} bar after 20 iterations", RuntimeWarning, stacklevel=2)
    return r1


def checks(res: RegenResult):
    """Handbook limits (10.3, 11.1) evaluated on a march result."""
    sc = res.scalars
    out = {"Twg_max_K": (sc["Twg_max"], C.T_WG_LIMIT, sc["Twg_max"] <= C.T_WG_LIMIT)}
    if res.coolant.lower() in ("n2o", "nitrousoxide"):
        out["Twc_max_K"] = (sc["Twc_max"], C.T_WC_LIMIT_N2O, sc["Twc_max"] <= C.T_WC_LIMIT_N2O)
        out["v_min_ms"] = (sc["v_min"], C.V_MIN_N2O, sc["v_min"] >= C.V_MIN_N2O)
    if np.isfinite(sc["chf_ratio_max"]):
        out["chf_ratio_max"] = (sc["chf_ratio_max"], C.CHF_RATIO_LIMIT,
                                sc["chf_ratio_max"] <= C.CHF_RATIO_LIMIT)
    if np.isfinite(sc["margin_sat_min"]) and not sc["two_phase_any"]:
        out["boiling_margin_K"] = (sc["margin_sat_min"], 0.0, sc["margin_sat_min"] > 0.0)
    if sc["two_phase_any"]:
        # flow boiling in the jacket: dryout CHF is not evaluated by this tool
        out["saturated_boiling"] = (sc["frac_two_phase"], 0.0, False)
    return out


def verify(wall_fn, *args, n=C.N_STATIONS, **kw):
    """Handbook 10.8 checks: energy balance and station-count convergence.

    wall_fn(n) must return wall stations; remaining arguments go to march().
    """
    r1 = march(wall_fn(n), *args, **kw)
    r2 = march(wall_fn(2 * n), *args, **kw)
    return {"energy_balance_err": r1["energy_balance_err"],
            "dTwg_max_K_halving": abs(r2["Twg_max"] - r1["Twg_max"]),
            "dQ_rel_halving": abs(r2["Q"] - r1["Q"]) / r1["Q"],
            "ddp_rel_halving": abs(r2["dp_bar"] - r1["dp_bar"]) / max(r1["dp_bar"], 1e-9)}
