#!/usr/bin/env python3
"""
flight_config_sizing.py — Flight-config tank sizing & pressurization analysis.

Reads a YAML mission/config file (single source of truth) and performs:

  1. Tank geometry sizing (fixed D=300 mm, hemispherical domes):
     liquid load from burn time / mass flow / O/F, sized at worst-case
     (warmest -> least dense) fill density with the minimum-ullage rule.

  2. Coupled pressurization simulation per temperature case:
     ethanol tank (ethanol_press) and N2O tank (n2o_press) are co-stepped
     against ONE shared pressurant bottle (rigid vessel, adiabatic blowdown,
     isenthalpic regulator: tank inlet enthalpy = bottle enthalpy h(P_b, T_b)).

  3. Pressurant bottle sizing: bisection on bottle volume against the
     recorded demand profile, requiring P_bottle(end) >= regulator minimum
     inlet pressure; one outer refinement pass; margin factor applied.

  4. Plotly HTML report + results YAML, named after the config file
     (<config>_report.html, <config>_results.yaml).

Usage:  python flight_config_sizing.py [config.yaml] [--outdir DIR]
        (defaults: configs/h2_tank_config_mission1.yaml, results/)
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
from pathlib import Path

import yaml
from CoolProp.CoolProp import PropsSI

import ethanol_press as ep
import n2o_press as np2

C0 = 273.15


# =============================================================================
# Config loading
# =============================================================================

def load_config(path: str | Path) -> dict:
    """Config file, completed from the shared baseline if it names one
    (`baseline:` key). Values set in the config win and are reported."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("baseline"):            # configs without it are independent of the baseline
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
        import e2_baseline
        bl = e2_baseline.load(e2_baseline.resolve(cfg["baseline"], path))
        pr = bl["propellants"]
        e2_baseline.fill(cfg, {
            "propellants.oxidizer.coolprop": pr["oxidizer"]["coolprop"],
            "propellants.fuel.coolprop": pr["fuel"]["coolprop"],
            "operating.pressure_set_bar": bl["feed"]["tank_pressure_bar"],
            "operating.temp_min_C": pr["oxidizer"]["temperature_min_C"],
            "operating.temp_max_C": pr["oxidizer"]["temperature_max_C"],
            "tanks.residual_ox_kg": bl["residuals"]["oxidizer_kg"],
            "tanks.residual_fu_kg": bl["residuals"]["fuel_kg"],
        }, label=Path(path).name)
        cfg.setdefault("cases", [
            {"name": "cold", "prop_temp_C": cfg["operating"]["temp_min_C"]},
            {"name": "hot", "prop_temp_C": cfg["operating"]["temp_max_C"]}])
        cfg["_baseline"] = e2_baseline.describe(bl)
    if (cfg.get("mission") or {}).get("from_mission_results"):
        _mission_from_handover(cfg, path)
    return cfg


def _mission_from_handover(cfg: dict, cfg_path) -> None:
    """Fill the `mission` block from the handover section of the mission
    tool's results: usable N2O and ethanol, delivered at the mission's mean
    flow over the equivalent constant-flow duration."""
    mi = cfg["mission"]
    manual = [k for k in ("burn_time_s", "mdot_total_kg_s", "of_ratio") if k in mi]
    if manual:
        raise SystemExit(f"{Path(cfg_path).name}: mission.from_mission_results replaces "
                         f"{', '.join(manual)}; remove them or remove from_mission_results.")
    src = Path(mi["from_mission_results"])
    if not src.is_absolute():
        src = (Path(cfg_path).resolve().parent / src).resolve()
    if not src.exists():
        raise SystemExit(f"{src} not found. Run the mission tool first:  "
                         f"cd mission_analysis && python mission_sizing.py")
    res = yaml.safe_load(src.read_text(encoding="utf-8"))
    ho = res["handover"]
    m_ox, m_fu, t = ho["usable_n2o_kg"], ho["usable_ethanol_kg"], ho["duration_mean_flow_s"]
    mi.update(burn_time_s=t, mdot_total_kg_s=(m_ox + m_fu) / t, of_ratio=m_ox / m_fu)
    meta = res.get("meta", {})
    cfg["_mission_handover"] = {
        "file": src.name, "mission_date": meta.get("date"), "sizing_case": ho["sizing_case"],
        "usable_n2o_kg": m_ox, "usable_ethanol_kg": m_fu, "duration_s": t,
        "mission_overrides": meta.get("overrides") or []}
    print(f"Mission from {src.name} ({ho['sizing_case']}): N2O {m_ox:.2f} kg, ethanol {m_fu:.2f} kg "
          f"over {t:.2f} s")
    if meta.get("overrides"):
        print(f"NOTE: those mission results were run with --set {meta['overrides']}")


# =============================================================================
# 1) Tank geometry sizing
# =============================================================================

def size_tanks(cfg: dict, m_ox_load: float | None = None,
               m_fu_load: float | None = None) -> dict:
    """Tank geometry from the sizing rules.

    `m_ox_load` is the TOTAL oxidizer loaded (liquid + ullage vapor after
    pre-press). The oxidizer tank is sized for a two-phase hot fill:
    V = m_total / (rho_l,sat*(1-u) + rho_v,sat*u) so the ullage vapor is
    not double-counted as liquid volume.

    `m_fu_load` is the fuel liquid mass (delivered + residual).
    """
    mi, tk, op = cfg["mission"], cfg["tanks"], cfg["operating"]
    of = mi["of_ratio"]
    mdot = mi["mdot_total_kg_s"]
    t_b = mi["burn_time_s"]
    m_ox_del = mdot * t_b * of / (1.0 + of)
    m_fu_del = mdot * t_b * 1.0 / (1.0 + of)
    m_res_ox = float(tk.get("residual_ox_kg", 0.0))
    m_res_fu = float(tk.get("residual_fu_kg", 0.0))
    m_ox = m_ox_load if m_ox_load is not None else (m_ox_del + m_res_ox)
    m_fu = m_fu_load if m_fu_load is not None else (m_fu_del + m_res_fu)

    T_hot = op["temp_max_C"] + C0
    T_cold = op["temp_min_C"] + C0
    fl_ox = cfg["propellants"]["oxidizer"]["coolprop"]
    fl_fu = cfg["propellants"]["fuel"]["coolprop"]
    u_min = tk["ullage_min_frac"]

    # Sizing densities = lowest plausible (warmest, unpressurized fill).
    rho_ox_l = PropsSI("D", "T", T_hot, "Q", 0, fl_ox)
    rho_ox_v = PropsSI("D", "T", T_hot, "Q", 1, fl_ox)
    rho_fu = PropsSI("D", "T", T_hot, "P", 101325.0, fl_fu)

    R = tk["diameter_inner_m"] / 2.0
    v_sphere = (4.0 / 3.0) * math.pi * R**3
    a_cyl = math.pi * R**2
    grid = tk["cyl_length_round_up_mm"] / 1000.0
    l_min = tk["min_cyl_length_mm"] / 1000.0

    def _capsule(v_req: float) -> tuple[float, ep.TankGeom]:
        l_cyl = max((v_req - v_sphere) / a_cyl, l_min)
        l_cyl = max(math.ceil(l_cyl / grid) * grid if l_cyl > 0 else 0.0, 0.0)
        return l_cyl, ep.TankGeom(radius_m=R, cyl_length_m=l_cyl)

    # Oxidizer: two-phase fill volume for total mass m_ox.
    v_req_ox = m_ox / (rho_ox_l * (1.0 - u_min) + rho_ox_v * u_min)
    l_ox, geom_ox = _capsule(v_req_ox)
    V_ox = geom_ox.volume_m3
    m_liq_ox_fill = (m_ox - rho_ox_v * V_ox) / (1.0 - rho_ox_v / rho_ox_l)
    v_liq_ox = m_liq_ox_fill / rho_ox_l

    # Fuel: single-phase liquid + ullage fraction (vapor mass negligible).
    v_liq_fu = m_fu / rho_fu
    v_req_fu = v_liq_fu / (1.0 - u_min)
    l_fu, geom_fu = _capsule(v_req_fu)
    V_fu = geom_fu.volume_m3

    def _pack(l_cyl, geom, v_liq, V):
        return {
            "cyl_length_mm": l_cyl * 1000.0,
            "total_length_mm": (l_cyl + 2.0 * geom.radius_m) * 1000.0,
            "volume_L": V * 1000.0,
            "liquid_volume_fill_L": v_liq * 1000.0,
            "ullage_frac_fill_hot": 1.0 - v_liq / V,
            "geom": geom,
        }

    out = {
        "m_ox_kg": m_ox, "m_ox_delivered_kg": m_ox_del,
        "m_ox_liquid_fill_kg": m_liq_ox_fill,
        "m_fu_kg": m_fu, "m_fu_delivered_kg": m_fu_del,
        "rho_sizing_ox": rho_ox_l, "rho_sizing_fu": rho_fu,
        "residual_ox_kg": m_res_ox, "residual_fu_kg": m_res_fu,
        "ox": _pack(l_ox, geom_ox, v_liq_ox, V_ox),
        "fu": _pack(l_fu, geom_fu, v_liq_fu, V_fu),
    }

    # Envelope ullage: cold unpressurized fill of the same total masses.
    rho_ox_l_c = PropsSI("D", "T", T_cold, "Q", 0, fl_ox)
    rho_ox_v_c = PropsSI("D", "T", T_cold, "Q", 1, fl_ox)
    m_liq_ox_cold = (m_ox - rho_ox_v_c * V_ox) / (1.0 - rho_ox_v_c / rho_ox_l_c)
    rho_fu_c = PropsSI("D", "T", T_cold, "P", 101325.0, fl_fu)
    out["ox"]["ullage_frac_fill_cold"] = 1.0 - m_liq_ox_cold / (rho_ox_l_c * V_ox)
    out["fu"]["ullage_frac_fill_cold"] = 1.0 - m_fu / (rho_fu_c * V_fu)
    out["ox"]["overfill_hot"] = v_liq_ox > V_ox
    out["fu"]["overfill_hot"] = v_liq_fu > V_fu
    return out


# =============================================================================
# 2) Shared pressurant bottle (rigid vessel)
# =============================================================================

class Bottle:
    """Rigid pressurant vessel. Adiabatic discharge:
    d(m u)/dt = -mdot * h(P_b, T_b); states recovered via (rho, u) flash."""

    def __init__(self, fluid: str, volume_m3: float, p0_pa: float, t0_k: float,
                 mode: str = "adiabatic"):
        self.fluid, self.V, self.mode = fluid, volume_m3, mode
        self.T = t0_k
        self.t0 = t0_k
        rho0 = PropsSI("D", "P", p0_pa, "T", t0_k, fluid)
        self.m = rho0 * volume_m3
        self.m0 = self.m

    @property
    def rho(self) -> float:
        return self.m / self.V

    @property
    def P(self) -> float:
        return PropsSI("P", "D", self.rho, "T", self.T, self.fluid)

    @property
    def h(self) -> float:
        return PropsSI("H", "D", self.rho, "T", self.T, self.fluid)

    def draw_isothermal(self, dm: float) -> None:
        """Slow draw (pre-pressurization): vessel stays at its initial temp."""
        self.m -= dm
        if self.mode == "adiabatic":
            self.T = self.t0  # re-equilibrated before the burn

    def draw(self, mdot: float, dt: float) -> None:
        if self.mode == "isothermal":
            self.m -= mdot * dt
            return
        h_b = self.h
        u_old = PropsSI("U", "D", self.rho, "T", self.T, self.fluid)
        U_new = self.m * u_old - mdot * h_b * dt
        self.m -= mdot * dt
        u_new = U_new / self.m
        self.T = PropsSI("T", "D", self.m / self.V, "U", u_new, self.fluid)


def size_bottle_volume(fluid: str, p0: float, t0: float, p_min: float,
                       mode: str, m_prepress: float, mdot_series: list[float],
                       dt: float) -> float:
    """Smallest bottle volume that keeps P >= p_min through the demand
    profile (prepress isothermal draw + burn-time flow series). Bisection."""

    def p_end(vol: float) -> float:
        b = Bottle(fluid, vol, p0, t0, mode)
        if b.m <= m_prepress + sum(mdot_series) * dt:
            return -1.0
        b.draw_isothermal(m_prepress)
        for md in mdot_series:
            b.draw(md, dt)
            if b.m <= 0:
                return -1.0
        return b.P

    lo, hi = 1e-3, 0.5
    if p_end(hi) < p_min:
        raise RuntimeError("Bottle sizing: 500 L insufficient — check inputs.")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if p_end(mid) >= p_min:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return hi


# =============================================================================
# 3) Coupled case simulation
# =============================================================================

def run_case(cfg: dict, sizing: dict, t_prop_c: float,
             bottle_volume_m3: float) -> dict:
    op, pr, mo, mi = (cfg["operating"], cfg["pressurant"],
                      cfg["model"], cfg["mission"])
    wall = cfg["tanks"]["wall"]
    ht = mo["heat_transfer"]
    of = mi["of_ratio"]
    P = op["pressure_set_bar"] * 1e5
    T0 = t_prop_c + C0
    Tamb = op["temp_ambient_C"] + C0
    dt = mo["dt_s"]
    t_burn = mi["burn_time_s"]

    common = dict(p_set_pa=P, t_prop_k=T0, t_amb_k=Tamb,
                  wall_thickness_m=wall["thickness_m"],
                  wall_rho_cp=wall["density_kg_m3"] * wall["cp_J_kgK"],
                  ht_mode=ht["mode"],
                  h_gw_fixed=ht["h_gas_wall_W_m2K"],
                  h_gl_fixed=ht["h_gas_liquid_W_m2K"],
                  h_ext=ht["h_external_W_m2K"])

    cfg_fu = ep.LiquidTankConfig(
        geom=sizing["fu"]["geom"],
        fluid_liq=cfg["propellants"]["fuel"]["coolprop"],
        fluid_gas=pr["fluid"],
        mdot_out_kg_s=mi["mdot_total_kg_s"] / (1.0 + of),
        m_liq0_kg=sizing["m_fu_kg"], **common)
    cfg_ox = np2.N2OTankConfig(
        geom=np2.TankGeom(sizing["ox"]["geom"].radius_m,
                          sizing["ox"]["geom"].cyl_length_m),
        fluid_liq=cfg["propellants"]["oxidizer"]["coolprop"],
        fluid_gas=pr["fluid"],
        mdot_out_kg_s=mi["mdot_total_kg_s"] * of / (1.0 + of),
        m_liq0_kg=sizing["m_ox_kg"],
        fixed_point_iters=mo["fixed_point_iters"],
        evap_model=mo.get("evap_model", "equilibrium"), **common)

    st_fu, mpre_fu = ep.init_state(cfg_fu)
    st_ox, mpre_ox = np2.init_state(cfg_ox)
    m_ullage_pre = mpre_fu + mpre_ox
    m_diss = (float(pr.get("dissolution_n2_mass_frac_ox", 0.0))
              * sizing.get("m_ox_liquid_fill_kg", sizing["m_ox_kg"])
              + float(pr.get("dissolution_n2_mass_frac_fu", 0.0))
              * sizing["m_fu_kg"])
    m_prepress = m_ullage_pre + m_diss

    bottle = Bottle(pr["fluid"], bottle_volume_m3,
                    pr["bottle_pressure_bar"] * 1e5, pr["bottle_temp_C"] + C0,
                    pr["bottle_blowdown"])
    if mo.get("prepress", "isothermal") == "adiabatic":
        if m_prepress > 0.0:
            bottle.draw(m_prepress, 1.0)
    else:
        bottle.draw_isothermal(m_prepress)

    n = int(round(t_burn / dt))
    depleted_at = None
    ts = {k: [] for k in
          ("t", "mdot_total", "mdot_fu", "mdot_ox", "p_bottle", "t_bottle",
           "t_gas_fu", "t_ull_ox", "t_liq_ox", "p_vap_ox", "p_n2_ox",
           "m_evap_ox", "m_vap_ox", "m_n2_cum", "rho_liq_ox",
           "v_ull_fu", "v_ull_ox")}
    for _ in range(n):
        h_in = bottle.h  # isenthalpic regulator: inlet enthalpy = bottle enthalpy
        st_fu, md_fu, d_fu = ep.step(cfg_fu, st_fu, dt, h_in)
        try:
            st_ox, md_ox, d_ox = np2.step(cfg_ox, st_ox, dt, h_in)
        except RuntimeError as exc:
            if "liquid depleted" in str(exc):
                depleted_at = st_ox.t_s
                break
            raise
        md_tot = max(md_fu, 0.0) + max(md_ox, 0.0)
        bottle.draw(md_tot, dt)

        ts["t"].append(st_fu.t_s)
        ts["mdot_total"].append(md_tot)
        ts["mdot_fu"].append(md_fu)
        ts["mdot_ox"].append(md_ox)
        ts["p_bottle"].append(bottle.P)
        ts["t_bottle"].append(bottle.T)
        ts["t_gas_fu"].append(st_fu.t_gas)
        ts["t_ull_ox"].append(st_ox.t_ull)
        ts["t_liq_ox"].append(st_ox.t_liq)
        ts["p_vap_ox"].append(d_ox["p_vap_pa"])
        ts["p_n2_ox"].append(d_ox["p_n2_pa"])
        ts["m_evap_ox"].append(st_ox.m_evap_cum)
        ts["m_vap_ox"].append(st_ox.m_vap)
        ts["m_n2_cum"].append(m_prepress + st_fu.m_gas_cum + st_ox.m_n2_cum)
        ts["rho_liq_ox"].append(d_ox["rho_liq"])
        ts["v_ull_fu"].append(d_fu["v_ullage_m3"])
        ts["v_ull_ox"].append(d_ox["v_ullage_m3"])

    # Pressurant flow statistics (regulator selection): peak & mean demand
    md = ts["mdot_total"]
    if md:
        i_pk = max(range(len(md)), key=md.__getitem__)
        flow = {
            "mdot_n2_peak_g_s": md[i_pk] * 1000.0,
            "mdot_n2_peak_at_s": ts["t"][i_pk],
            "mdot_n2_end_g_s": md[-1] * 1000.0,
            "mdot_n2_mean_g_s": sum(md) / len(md) * 1000.0,
            "mdot_n2_ox_peak_g_s": max(ts["mdot_ox"]) * 1000.0,
            "mdot_n2_fu_peak_g_s": max(ts["mdot_fu"]) * 1000.0,
        }
        rho_end = ts["rho_liq_ox"][-1]
    else:
        flow = {k: 0.0 for k in (
            "mdot_n2_peak_g_s", "mdot_n2_peak_at_s", "mdot_n2_end_g_s",
            "mdot_n2_mean_g_s", "mdot_n2_ox_peak_g_s", "mdot_n2_fu_peak_g_s")}
        rho_end = float("nan")

    return {
        "pressurant_flow": flow,
        "depleted_at_s": depleted_at,
        "m_liq_ox_end_kg": st_ox.m_liq,
        "m_vap_ox_end_kg": st_ox.m_vap,
        "m_n2o_ox_end_kg": st_ox.m_liq + st_ox.m_vap,
        "prepress_fu_kg": mpre_fu,
        "prepress_ox_kg": mpre_ox,
        "n2_dissolved_kg": m_diss,
        "expulsion_fu_kg": st_fu.m_gas_cum,
        "expulsion_ox_kg": st_ox.m_n2_cum,
        "n2_total_kg": m_prepress + st_fu.m_gas_cum + st_ox.m_n2_cum,
        "n2o_evaporated_kg": st_ox.m_evap_cum,
        "t_liq_ox_end_C": st_ox.t_liq - C0,
        "t_ull_ox_end_C": st_ox.t_ull - C0,
        "t_gas_fu_end_C": st_fu.t_gas - C0,
        "rho_liq_ox_end": rho_end,
        "p_bottle_end_bar": bottle.P / 1e5,
        "t_bottle_end_C": bottle.T - C0,
        "timeseries": ts,
        "m_prepress": m_prepress,
    }


# =============================================================================
# 4) Reporting
# =============================================================================

def build_report(cfg: dict, sizing: dict, cases: dict, bottle: dict,
                 outdir: Path, title: str = "tank_press") -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figs = []
    for name, r in cases.items():
        ts = r["timeseries"]
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=(
                "Temperatures", "Pressurant demand",
                "N2O ullage partial pressures", "Bottle state"))
        t = ts["t"]
        fig.add_trace(go.Scatter(x=t, y=[v - C0 for v in ts["t_liq_ox"]],
                                 name="T_liq N2O"), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=[v - C0 for v in ts["t_ull_ox"]],
                                 name="T_ull N2O"), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=[v - C0 for v in ts["t_gas_fu"]],
                                 name="T_ull EtOH"), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=ts["mdot_ox"], name="mdot N2 -> ox"), 1, 2)
        fig.add_trace(go.Scatter(x=t, y=ts["mdot_fu"], name="mdot N2 -> fu"), 1, 2)
        fig.add_trace(go.Scatter(x=t, y=ts["m_n2_cum"], name="m_N2 cum [kg]",
                                 yaxis="y6", line=dict(dash="dot")), 1, 2)
        fig.add_trace(go.Scatter(x=t, y=[v / 1e5 for v in ts["p_vap_ox"]],
                                 name="P_vap N2O [bar]"), 2, 1)
        fig.add_trace(go.Scatter(x=t, y=[v / 1e5 for v in ts["p_n2_ox"]],
                                 name="P_N2 [bar]"), 2, 1)
        fig.add_trace(go.Scatter(x=t, y=[v / 1e5 for v in ts["p_bottle"]],
                                 name="P_bottle [bar]"), 2, 2)
        fig.add_trace(go.Scatter(x=t, y=[v - C0 for v in ts["t_bottle"]],
                                 name="T_bottle [C]"), 2, 2)
        fig.update_layout(title=f"{title} — case '{name}' "
                                f"(N2 total {r['n2_total_kg']:.2f} kg)",
                          height=750, template="plotly_white")
        fig.update_xaxes(title_text="time [s]")
        figs.append(fig)

    html = [f"<html><head><meta charset='utf-8'><title>{title} pressurization "
            "report</title></head><body>",
            f"<h1>{title}: tank & pressurization sizing report</h1>",
            _summary_html(cfg, sizing, cases, bottle)]
    for i, fig in enumerate(figs):
        html.append(fig.to_html(full_html=False,
                                include_plotlyjs="cdn" if i == 0 else False))
    html.append("</body></html>")
    out = outdir / f"{title}_report.html"
    out.write_text("\n".join(html))
    return out


def _summary_html(cfg, sizing, cases, bottle) -> str:
    rows = []
    for tag, label in (("ox", "N2O tank"), ("fu", "Ethanol tank")):
        s = sizing[tag]
        rows.append(
            f"<tr><td>{label}</td><td>{s['cyl_length_mm']:.0f}</td>"
            f"<td>{s['total_length_mm']:.0f}</td><td>{s['volume_L']:.2f}</td>"
            f"<td>{s['liquid_volume_fill_L']:.2f}</td>"
            f"<td>{100 * s['ullage_frac_fill_hot']:.1f}% / "
            f"{100 * s.get('ullage_frac_fill_cold', 0):.1f}%</td></tr>")
    case_rows = "".join(
        f"<tr><td>{n}</td><td>{r['n2_total_kg']:.2f}</td>"
        f"<td>{r['pressurant_flow']['mdot_n2_peak_g_s']:.0f} g/s "
        f"@ {r['pressurant_flow']['mdot_n2_peak_at_s']:.1f} s</td>"
        f"<td>{r['n2o_evaporated_kg']:.2f}</td>"
        f"<td>{r.get('m_vap_ox_end_kg', 0):.2f}</td>"
        f"<td>{r['t_liq_ox_end_C']:.1f} / {r['t_ull_ox_end_C']:.1f}</td>"
        f"<td>{r['p_bottle_end_bar']:.0f}</td></tr>"
        for n, r in cases.items())
    return f"""
<h2>Tank geometry (D = {cfg['tanks']['diameter_inner_m'] * 1000:.0f} mm,
hemispherical domes)</h2>
<table border=1 cellpadding=4>
<tr><th>Tank</th><th>L_cyl [mm]</th><th>L_total [mm]</th><th>V [L]</th>
<th>V_liq fill [L]</th><th>Ullage hot / cold</th></tr>{''.join(rows)}</table>
<h2>Pressurization cases (P_set = {cfg['operating']['pressure_set_bar']:.0f} bar)</h2>
<table border=1 cellpadding=4>
<tr><th>Case</th><th>N2 total [kg]</th><th>N2 flow peak</th>
<th>N2O evap [kg]</th><th>N2O vapor end [kg]</th>
<th>T_liq / T_ull ox [C]</th><th>P_bottle end [bar]</th></tr>{case_rows}</table>
<h2>Bottle sizing (worst case: {bottle['worst_case']})</h2>
<p>Required: {bottle['v_required_L']:.1f} L &rarr; with margin
{bottle['margin_factor']:.2f}: <b>{bottle['v_margined_L']:.1f} L</b>,
loaded N2 mass {bottle['m_loaded_kg']:.2f} kg at
{cfg['pressurant']['bottle_pressure_bar']:.0f} bar.
Dissolved-N2 allowance {bottle.get('m_dissolved_kg', 0):.2f} kg.</p>
"""


# =============================================================================
# Main
# =============================================================================

def _vap_end_for_makeup(r: dict) -> float:
    """End-of-burn (or at-depletion) N2O vapor mass. Do not time-extrapolate:
    that overshoots a full-tank sat-vapor inventory and makes makeup oscillate."""
    m_vap = r.get("m_vap_ox_end_kg")
    if m_vap is None:
        m_vap = r["n2o_evaporated_kg"]
    return m_vap


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?",
                    default=str(here / "configs" / "h2_tank_config_mission1.yaml"))
    ap.add_argument("--outdir", default=str(here / "results"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg_name = Path(args.config).stem
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pr, mo, tk = cfg["pressurant"], cfg["model"], cfg["tanks"]
    mi = cfg["mission"]

    m_ox_del = (mi["mdot_total_kg_s"] * mi["burn_time_s"]
                * mi["of_ratio"] / (1.0 + mi["of_ratio"]))
    m_res_ox = float(tk.get("residual_ox_kg", 0.0))
    makeup_on = tk.get("oxidizer_evap_makeup", True)
    headroom = float(mo.get("makeup_headroom", 1.05))
    v_bottle = float(mo.get("bottle_volume_guess_m3", 0.060))
    m_load = float(mo.get("ox_load_guess_kg", m_ox_del + m_res_ox))

    print("=== 1) Coupled makeup + bottle-volume iteration ===")
    sizing, cases, bottle_req = None, {}, {}
    converged = False
    n_outer = int(mo.get("outer_iterations", 10))
    for it in range(n_outer):
        sizing = size_tanks(cfg, m_ox_load=m_load)
        cases = {}
        for c in cfg["cases"]:
            print(f"    run '{c['name']}' at V_b={v_bottle * 1000:.1f} L, "
                  f"load={m_load:.2f} kg", flush=True)
            cases[c["name"]] = run_case(
                cfg, sizing, c["prop_temp_C"], v_bottle)
        t_b = mi["burn_time_s"]
        vap_max = max(_vap_end_for_makeup(r) for r in cases.values())
        depleted = any(r["depleted_at_s"] for r in cases.values())
        m_need = m_ox_del + m_res_ox + headroom * vap_max
        if depleted:
            # Add the undelivered liquid from the worst early-empty case.
            missing = max(
                (m_ox_del * (1.0 - r["depleted_at_s"] / t_b)
                 if r["depleted_at_s"] else 0.0)
                for r in cases.values())
            m_need = max(m_need, m_load + missing)   # m_load already holds the residual

        bottle_req = {}
        for name, r in cases.items():
            md = r["timeseries"]["mdot_total"]
            if not md:
                bottle_req[name] = v_bottle
                continue
            bottle_req[name] = size_bottle_volume(
                pr["fluid"], pr["bottle_pressure_bar"] * 1e5,
                pr["bottle_temp_C"] + C0, pr["regulator_min_inlet_bar"] * 1e5,
                pr["bottle_blowdown"], r["m_prepress"], md, mo["dt_s"])
        v_new = max(bottle_req.values()) if bottle_req else v_bottle

        load_ok = (not makeup_on) or (
            abs(m_need - m_load) <= 0.005 * max(m_load, 1.0))
        vol_ok = abs(v_new - v_bottle) <= max(1e-4, 0.01 * v_bottle)
        print(f"  outer {it}: load {m_load:.2f} kg (need {m_need:.2f}, "
              f"vap_end {vap_max:.2f}) | V_b {v_bottle * 1000:.1f} L "
              f"(sized {v_new * 1000:.1f} L)", flush=True)
        if load_ok and vol_ok:
            converged = True
            break
        if makeup_on and not load_ok:
            m_load = 0.5 * (m_load + m_need)   # under-relax: avoid load/vapor hunt
        v_bottle = v_new

    if not converged:
        print(f"  WARNING: load / bottle iteration not converged after {n_outer} passes "
              f"(load {m_load:.2f} kg, need {m_need:.2f} kg). Check "
              f"model.ox_load_guess_kg, model.bottle_volume_guess_m3 or raise "
              f"model.outer_iterations.")
    if makeup_on:
        m_load = max(m_load, m_need)
        sizing = size_tanks(cfg, m_ox_load=m_load)

    # Final pass at the converged bottle volume so P_end matches the sizer.
    cases, bottle_req = {}, {}
    for case in cfg["cases"]:
        name, tc = case["name"], case["prop_temp_C"]
        r = run_case(cfg, sizing, tc, v_bottle)
        md = r["timeseries"]["mdot_total"]
        v_req = size_bottle_volume(
            pr["fluid"], pr["bottle_pressure_bar"] * 1e5,
            pr["bottle_temp_C"] + C0, pr["regulator_min_inlet_bar"] * 1e5,
            pr["bottle_blowdown"], r["m_prepress"], md, mo["dt_s"])
        cases[name], bottle_req[name] = r, v_req
        fl = r["pressurant_flow"]
        print(f"  case '{name}' ({tc:.0f} C): "
              f"N2 = {r['n2_total_kg']:.2f} kg "
              f"(pre {r['prepress_fu_kg'] + r['prepress_ox_kg']:.2f} | "
              f"diss {r['n2_dissolved_kg']:.2f} | "
              f"fu {r['expulsion_fu_kg']:.2f} | ox {r['expulsion_ox_kg']:.2f}) | "
              f"N2O vap end = {r['m_vap_ox_end_kg']:.2f} kg | "
              f"T_liq/ull = {r['t_liq_ox_end_C']:.1f}/{r['t_ull_ox_end_C']:.1f} C | "
              f"V_bottle req = {v_req * 1000:.1f} L")
        print(f"      N2 flow: peak {fl['mdot_n2_peak_g_s']:.0f} g/s "
              f"@ t={fl['mdot_n2_peak_at_s']:.1f} s "
              f"(ox {fl['mdot_n2_ox_peak_g_s']:.0f} | "
              f"fu {fl['mdot_n2_fu_peak_g_s']:.0f}) | "
              f"mean {fl['mdot_n2_mean_g_s']:.0f} g/s | "
              f"liq end {r['m_liq_ox_end_kg']:.2f} kg")

    print("\n=== 2) Tank geometry ===")
    for tag, label in (("ox", "N2O"), ("fu", "EtOH")):
        s = sizing[tag]
        print(f"  {label}: L_cyl = {s['cyl_length_mm']:6.0f} mm | "
              f"L_total = {s['total_length_mm']:5.0f} mm | "
              f"V = {s['volume_L']:6.2f} L | "
              f"ullage hot/cold = {100 * s['ullage_frac_fill_hot']:.1f}/"
              f"{100 * s['ullage_frac_fill_cold']:.1f} %")
        if s.get("overfill_hot"):
            print(f"  WARNING: {label} liquid volume exceeds tank at hot fill")
    print(f"  oxidizer load (total N2O): {sizing['m_ox_kg']:.2f} kg "
          f"(delivered {m_ox_del:.2f} + residual {m_res_ox:.2f} + "
          f"vapor/makeup {sizing['m_ox_kg'] - m_ox_del - m_res_ox:.2f})")
    print(f"  oxidizer liquid at hot fill: {sizing['m_ox_liquid_fill_kg']:.2f} kg")
    print(f"  fuel load: {sizing['m_fu_kg']:.2f} kg "
          f"(delivered {sizing['m_fu_delivered_kg']:.2f} + "
          f"residual {sizing['residual_fu_kg']:.2f})")

    print("\n=== 3) Bottle sizing ===")
    worst = max(bottle_req, key=bottle_req.get)
    v_req = bottle_req[worst]
    v_marg = v_req * pr["margin_factor"]
    rho0 = PropsSI("D", "P", pr["bottle_pressure_bar"] * 1e5,
                   "T", pr["bottle_temp_C"] + C0, pr["fluid"])
    m_diss_rep = next(iter(cases.values()))["n2_dissolved_kg"] if cases else 0.0
    bottle = {"worst_case": worst, "v_required_L": v_req * 1000.0,
              "margin_factor": pr["margin_factor"],
              "v_margined_L": v_marg * 1000.0,
              "m_loaded_kg": rho0 * v_marg,
              "m_dissolved_kg": m_diss_rep}
    print(f"  worst case '{worst}': V_req = {v_req * 1000:.1f} L "
          f"-> margined {v_marg * 1000:.1f} L "
          f"({bottle['m_loaded_kg']:.2f} kg N2 loaded, "
          f"{m_diss_rep:.2f} kg dissolution allowance)")

    results = {
        "meta": {"config": Path(args.config).name,
                 "baseline": cfg.get("_baseline"),
                 "date": str(datetime.date.today()),
                 "converged": converged},
        "tank_sizing": {t: {k: v for k, v in sizing[t].items() if k != "geom"}
                        for t in ("ox", "fu")},
        "propellant_loads_kg": {
            "oxidizer_loaded_total": sizing["m_ox_kg"],
            "oxidizer_liquid_fill": sizing["m_ox_liquid_fill_kg"],
            "oxidizer_delivered": sizing["m_ox_delivered_kg"],
            "oxidizer_residual": sizing["residual_ox_kg"],
            "oxidizer_vapor_allowance": (sizing["m_ox_kg"]
                                         - sizing["m_ox_delivered_kg"]
                                         - sizing["residual_ox_kg"]),
            "fuel": sizing["m_fu_kg"],
            "fuel_delivered": sizing["m_fu_delivered_kg"],
            "fuel_residual": sizing["residual_fu_kg"]},
        "cases": {n: {k: v for k, v in r.items() if k != "timeseries"}
                  for n, r in cases.items()},
        "bottle": bottle,
        # Input to mission_analysis (pressurization.from_tank_results). The
        # mission tool scales mass and vapour to its own tank volumes.
        "handback": {
            "pressurant_mass_kg": bottle["m_loaded_kg"],
            "pressurant_margin_factor": pr["margin_factor"],
            "n2o_vapour_makeup_kg": (sizing["m_ox_kg"] - sizing["m_ox_delivered_kg"]
                                     - sizing["residual_ox_kg"]),
            "oxidizer_volume_L": sizing["ox"]["volume_L"],
            "propellant_volume_L": sizing["ox"]["volume_L"] + sizing["fu"]["volume_L"],
            "sized_for": cfg.get("_mission_handover"),
        },
    }
    res_path = outdir / f"{cfg_name}_results.yaml"
    res_path.write_text(yaml.safe_dump(results, sort_keys=False))
    hb = results["handback"]
    print(f"  handback: {hb['pressurant_mass_kg']:.2f} kg N2 (x{hb['pressurant_margin_factor']:.2f}), "
          f"{hb['n2o_vapour_makeup_kg']:.2f} kg N2O vapour, tanks {hb['propellant_volume_L']:.1f} L "
          f"(N2O {hb['oxidizer_volume_L']:.1f} L)")
    rep = build_report(cfg, sizing, cases, bottle, outdir, cfg_name)
    print(f"\nReport: {rep}\nResults: {res_path}")
    return results


if __name__ == "__main__":
    main(sys.argv[1:])