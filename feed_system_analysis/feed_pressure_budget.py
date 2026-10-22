#!/usr/bin/env python3
"""
feed_pressure_budget.py — pressure budget of the H2 vehicle fluid system at full
thrust: regulator -> propellant tanks (nitrogen side) and tanks -> cavitating
venturi inlets (propellant side). Notion: Pressurant and Feed System Pressure
Drop Analysis (H2-FLD-ANA-003) and Feed Line Sizing (H2-FLD-ANA-002).

  1. Line losses: Darcy-Weisbach with the Swamee-Jain friction factor, fittings
     as K times the dynamic pressure. Nitrogen at the regulated (or storage)
     state, liquids at tank pressure (CoolProp; N2O viscosity is an input
     because CoolProp has none).
  2. Component losses: allowances, or the liquid Cv equation where a Cv is given
     (throttle valve; option valves).
  3. Venturis: mdot = Cd A sqrt(2 rho (p_in - p_v)), rho of the compressed
     liquid at the inlet. Flow at the available inlet pressure per N2O
     temperature, the temperature below which full flow is reached, and the
     choking margin (recovery limit x inlet - outlet pressure the engine needs).
  4. Required Cv of each component at its allowance: liquid Cv equation, and
     IEC 60534-2-1 for the gas valves and the regulator.
  5. Options: cheaper hardware changes on the oxidizer side, applied in order.

Usage, from this directory:
    python feed_pressure_budget.py                          # configs/h2_feed_v12.yaml
    python feed_pressure_budget.py configs/<case>.yaml --outdir results
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
from pathlib import Path

import yaml
from CoolProp.CoolProp import PropsSI

BAR = 1e5
C0 = 273.15
GPM_PER_M3S = 15850.32
PSI_PER_BAR = 14.5038


# =============================================================================
# Config
# =============================================================================

def load_config(path: str | Path) -> dict:
    """Config file, completed from the shared baseline if it names one: rated
    N2O and fuel flow (100 % row of the engine table), regulator set pressure
    (tank pressure) and ambient pressure. Values set in the config win."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("baseline"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
        import e2_baseline
        bl = e2_baseline.load(e2_baseline.resolve(cfg["baseline"], path))
        rated = e2_baseline.table_point(e2_baseline.engine_table(bl), 1.0)
        e2_baseline.fill(cfg, {
            "operating.n2o_flow_kg_s": rated["ox_flow_kg_s"],
            "operating.fuel_flow_kg_s": rated["fuel_flow_kg_s"],
            "operating.regulator_set_bar_g": bl["feed"]["tank_pressure_bar"],
            "operating.ambient_bar": bl["engine"]["ambient_pressure_bar"],
        }, label=path.name)
        cfg["_baseline"] = e2_baseline.describe(bl)
    return cfg


# =============================================================================
# Physics
# =============================================================================

def pipe_loss(mdot: float, rho: float, mu: float, line: dict, roughness_m: float,
              share: int = 1) -> tuple[float, float, float]:
    """(total loss bar, velocity m/s, friction bar/m) of one tube carrying
    mdot / share. line: tube_mm [OD, wall], length_m, K."""
    od, wall = line["tube_mm"]
    D = (od - 2.0 * wall) / 1e3
    v = mdot / share / rho / (math.pi * D**2 / 4.0)
    Re = rho * v * D / mu
    f = 0.25 / math.log10(roughness_m / (3.7 * D) + 5.74 / Re**0.9) ** 2
    q = 0.5 * rho * v**2
    return (f * line["length_m"] / D + line["K"]) * q / BAR, v, f / D * q / BAR


def n2_state(p_bar: float, T_C: float) -> tuple[float, float]:
    T = T_C + C0
    return (PropsSI("D", "P", p_bar * BAR, "T", T, "Nitrogen"),
            PropsSI("V", "P", p_bar * BAR, "T", T, "Nitrogen"))


def liquid_cv_loss(mdot: float, rho: float, cv: float) -> float:
    """Pressure loss in bar of a liquid valve with flow coefficient cv."""
    q_gpm = mdot / rho * GPM_PER_M3S
    return (q_gpm / cv) ** 2 * (rho / 1000.0) / PSI_PER_BAR


def liquid_cv_required(mdot: float, rho: float, dp_bar: float) -> float:
    q_gpm = mdot / rho * GPM_PER_M3S
    return q_gpm / math.sqrt(dp_bar * PSI_PER_BAR / (rho / 1000.0))


def gas_cv_required(mdot: float, p1_bar: float, p2_bar: float, T_C: float,
                    gamma: float = 1.4, x_T: float = 0.70) -> float:
    """IEC 60534-2-1 gas sizing (choked flow capped at F_gamma x_T)."""
    rho1, _ = n2_state(p1_bar, T_C)
    f_gamma = gamma / 1.4
    x = min((p1_bar - p2_bar) / p1_bar, f_gamma * x_T)
    Y = 1.0 - x / (3.0 * f_gamma * x_T)
    kv = mdot * 3600.0 / (31.6 * Y * math.sqrt(x * p1_bar * rho1))
    return kv * 1.156


def venturi_flow(p_in_bar: float, T_C: float, v: dict, fluid: str) -> float:
    T = T_C + C0
    rho = PropsSI("D", "P", p_in_bar * BAR, "T", T, fluid)
    p_v = PropsSI("P", "T", T, "Q", 0, fluid)
    area = math.pi * (v["throat_mm"] / 1e3) ** 2 / 4.0
    return v["cd"] * area * math.sqrt(2.0 * rho * max(p_in_bar * BAR - p_v, 0.0))


def venturi_inlet_required(mdot: float, T_C: float, v: dict, fluid: str) -> float:
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if venturi_flow(mid, T_C, v, fluid) >= mdot else (mid, hi)
    return hi


def full_flow_temperature(p_in_bar: float, mdot: float, v: dict) -> float:
    """Highest N2O temperature at which the venturi still passes mdot."""
    lo, hi = -20.0, 35.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if venturi_flow(p_in_bar, mid, v, "NitrousOxide") >= mdot else (lo, mid)
    return lo


# =============================================================================
# Budget
# =============================================================================

def oxidizer_path(cfg: dict, p_tank: float, rho: float, mu: float, eps: float,
                  overrides: dict | None = None) -> list[tuple[str, float]]:
    """Losses from the oxidizer tank to the venturi inlet, in flow order."""
    o = overrides or {}
    op, ln, al = cfg["operating"], cfg["lines"], cfg["allowances_bar"]
    m = op["n2o_flow_kg_s"]
    if "joined_legs_loss_bar" in o:
        lines = [("Tank legs joined at the main valve", o["joined_legs_loss_bar"])]
    else:
        lines = [(f"Tank legs {ln['ox_tank_leg']['tube_mm'][0]} × {ln['ox_tank_leg']['tube_mm'][1]:g}, "
                  f"1/{ln['ox_tank_leg'].get('count', 1)} flow each",
                  pipe_loss(m, rho, mu, ln["ox_tank_leg"], eps, ln["ox_tank_leg"].get("count", 1))[0]),
                 (f"Single line {ln['ox_main_line']['tube_mm'][0]} × {ln['ox_main_line']['tube_mm'][1]:g} "
                  f"to the main valve, full flow", pipe_loss(m, rho, mu, ln["ox_main_line"], eps)[0])]
    main_valve = (liquid_cv_loss(m, rho, o["ox_main_valve_cv"]) if "ox_main_valve_cv" in o
                  else al["ox_main_valve"])
    throttle_cv = o.get("throttle_valve_cv", cfg["throttle_valve_cv"])
    return lines + [("Filter (allowance)", al["ox_filter"]),
                    ("Main valve" + (f", Cv {o['ox_main_valve_cv']:g}" if "ox_main_valve_cv" in o
                                     else " (allowance)"), main_valve),
                    ("Spool to throttle valve", al["ox_spool"]),
                    (f"Throttle valve fully open, Cv {throttle_cv:g}", liquid_cv_loss(m, rho, throttle_cv))]


def venturi_summary(cfg: dict, p_in: float) -> dict:
    v, m = cfg["venturis"]["oxidizer"], cfg["operating"]["n2o_flow_kg_s"]
    flows = {T: venturi_flow(p_in, T, v, "NitrousOxide") for T in cfg["operating"]["n2o_temperatures_C"]}
    return {
        "inlet_bar_a": p_in,
        "flow_kg_s": flows,
        "flow_fraction": {T: f / m for T, f in flows.items()},
        "inlet_required_bar_a": {T: venturi_inlet_required(m, T, v, "NitrousOxide") for T in flows},
        "full_flow_below_C": full_flow_temperature(p_in, m, v),
        "choking_margin_bar": v["recovery_limit"] * p_in - v["outlet_needed_bar_a"],
    }


def run(cfg: dict) -> dict:
    op, ln, al, n2 = cfg["operating"], cfg["lines"], cfg["allowances_bar"], cfg["nitrogen"]
    eps = cfg["roughness_um"] * 1e-6
    p_set = op["regulator_set_bar_g"] + op["ambient_bar"]
    m_ox_n2, m_fu_n2 = n2["flow_ox_kg_s"], n2["flow_fuel_kg_s"]

    # --- nitrogen side -------------------------------------------------------
    rho_r, mu_r = n2_state(n2["regulated"]["pressure_bar"], n2["regulated"]["temperature_C"])
    n2_lines = {}
    for key, mdot in (("n2_reg_outlet", m_ox_n2 + m_fu_n2), ("n2_ox_branch", m_ox_n2),
                      ("n2_fuel_branch", m_fu_n2), ("n2_ox_standpipe", m_ox_n2),
                      ("n2_fuel_standpipe", m_fu_n2)):
        loss, v, fr = pipe_loss(mdot, rho_r, mu_r, ln[key], eps, ln[key].get("count", 1))
        n2_lines[key] = {"flow_each_kg_s": mdot / ln[key].get("count", 1), "velocity_m_s": v,
                         "friction_bar_m": fr, "loss_bar": loss}
    rho_s, mu_s = n2_state(n2["storage"]["pressure_bar"], n2["storage"]["temperature_C"])
    loss, v, fr = pipe_loss(m_ox_n2 + m_fu_n2, rho_s, mu_s, ln["n2_hp"], eps)
    n2_lines["n2_hp"] = {"flow_each_kg_s": m_ox_n2 + m_fu_n2, "velocity_m_s": v, "friction_bar_m": fr,
                         "loss_bar": loss}
    L = {k: d["loss_bar"] for k, d in n2_lines.items()}
    ox_press = [("Regulator outlet, set", p_set), ("Droop at full flow", -al["regulator_droop"]),
                ("Regulator outlet line", -L["n2_reg_outlet"]), ("Branch line", -L["n2_ox_branch"]),
                ("Pressurisation valve, check valves, hose", -al["ox_pressurisation_valves"]),
                ("Standpipe(s)", -L["n2_ox_standpipe"])]
    fu_press = [("Regulator outlet, set", p_set), ("Droop at full flow", -al["regulator_droop"]),
                ("Regulator outlet line", -L["n2_reg_outlet"]), ("Branch line", -L["n2_fuel_branch"]),
                ("Pressurisation valve, check valves, hose", -al["fuel_pressurisation_valves"]),
                ("Standpipe(s)", -L["n2_fuel_standpipe"])]
    p_ox_tank = sum(v for _, v in ox_press)
    p_fu_tank = sum(v for _, v in fu_press)

    # --- oxidizer side -------------------------------------------------------
    liq = cfg["n2o_liquid"]
    T_l = liq["temperature_C"] + C0
    rho_ox = PropsSI("D", "P", liq["pressure_bar"] * BAR, "T", T_l, "NitrousOxide")
    mu_ox = liq["viscosity_Pa_s"]
    ox_steps = oxidizer_path(cfg, p_ox_tank, rho_ox, mu_ox, eps)
    p_ox_venturi = p_ox_tank - sum(d for _, d in ox_steps)
    ox_venturi = venturi_summary(cfg, p_ox_venturi)

    # --- fuel side -----------------------------------------------------------
    fl = cfg["fuel_liquid"]
    T_f = fl["temperature_C"] + C0
    rho_fu = PropsSI("D", "P", fl["pressure_bar"] * BAR, "T", T_f, "Ethanol")
    mu_fu = PropsSI("V", "P", fl["pressure_bar"] * BAR, "T", T_f, "Ethanol")
    m_fu = op["fuel_flow_kg_s"]
    fu_steps = [("Line to the main valve", pipe_loss(m_fu, rho_fu, mu_fu, ln["fuel_line"], eps)[0]),
                ("Filter (allowance)", al["fuel_filter"]), ("Main valve (allowance)", al["fuel_main_valve"])]
    p_fu_venturi = p_fu_tank - sum(d for _, d in fu_steps)
    fu_flow = venturi_flow(p_fu_venturi, fl["temperature_C"], cfg["venturis"]["fuel"], "Ethanol")

    # --- regulator and required flow coefficients ----------------------------
    reg = cfg["regulator"]
    m_n2 = m_ox_n2 + m_fu_n2
    T_store = n2["storage"]["temperature_C"]
    m_ox = op["n2o_flow_kg_s"]
    cv = {
        "ox_filter": liquid_cv_required(m_ox, rho_ox, al["ox_filter"]),
        "ox_main_valve": liquid_cv_required(m_ox, rho_ox, al["ox_main_valve"]),
        "ox_throttle_valve": liquid_cv_required(m_ox, rho_ox, cfg["throttle_valve_allowance_bar"]),
        "fuel_filter": liquid_cv_required(m_fu, rho_fu, al["fuel_filter"]),
        "fuel_main_valve": liquid_cv_required(m_fu, rho_fu, al["fuel_main_valve"]),
        "regulator_at_sizing_limit": gas_cv_required(m_n2, reg["sizing_limit_bar_a"], p_set, T_store),
        "regulator_at_end_of_burn": gas_cv_required(m_n2, reg["end_of_burn_bar_a"], p_set, T_store),
        "ox_pressurisation_valves": gas_cv_required(m_ox_n2, p_set, p_set - al["ox_pressurisation_valves"],
                                                    n2["regulated"]["temperature_C"]),
        "fuel_pressurisation_valves": gas_cv_required(m_fu_n2, p_set, p_set - al["fuel_pressurisation_valves"],
                                                      n2["regulated"]["temperature_C"]),
    }
    dropout = p_set + reg["dropout_bar"] + L["n2_hp"]

    # --- options, applied in order ------------------------------------------
    options, acc = [], {}
    p_tank_opt = p_ox_tank
    for opt in cfg.get("options", []):
        acc.update({k: v for k, v in opt.items() if k not in ("name", "ox_standpipe_tube_mm")})
        if "ox_standpipe_tube_mm" in opt:
            sp = dict(ln["n2_ox_standpipe"], tube_mm=opt["ox_standpipe_tube_mm"])
            p_tank_opt = (p_ox_tank + L["n2_ox_standpipe"]
                          - pipe_loss(m_ox_n2, rho_r, mu_r, sp, eps, sp.get("count", 1))[0])
        steps = oxidizer_path(cfg, p_tank_opt, rho_ox, mu_ox, eps, acc)
        options.append({"name": opt["name"], "ox_tank_bar_a": p_tank_opt,
                        **venturi_summary(cfg, p_tank_opt - sum(d for _, d in steps))})

    return {
        "nitrogen_lines": n2_lines,
        "pressurant": {"oxidizer": ox_press, "fuel": fu_press,
                       "ox_tank_bar_a": p_ox_tank, "fuel_tank_bar_a": p_fu_tank},
        "oxidizer": {"steps": ox_steps, "tank_to_venturi_bar": p_ox_tank - p_ox_venturi, **ox_venturi},
        "fuel": {"steps": fu_steps, "venturi_inlet_bar_a": p_fu_venturi, "flow_kg_s": fu_flow,
                 "flow_fraction": fu_flow / m_fu},
        "regulator": {"dropout_storage_bar_a": dropout, "hp_line_loss_bar": L["n2_hp"]},
        "required_cv": cv,
        "options": options,
    }


# =============================================================================
# Reporting
# =============================================================================

def report(cfg: dict, r: dict) -> None:
    op, n2 = cfg["operating"], cfg["nitrogen"]
    print(f"=== {cfg.get('name', 'feed pressure budget')}")
    print(f"  N2O {op['n2o_flow_kg_s']:.3f} kg/s, ethanol {op['fuel_flow_kg_s']:.3f} kg/s; N2 "
          f"{(n2['flow_ox_kg_s'] + n2['flow_fuel_kg_s']) * 1e3:.0f} g/s (ox {n2['flow_ox_kg_s'] * 1e3:.0f}, "
          f"fuel {n2['flow_fuel_kg_s'] * 1e3:.0f})")
    print("\n  Nitrogen lines              flow/each  velocity  friction    loss")
    for k, d in r["nitrogen_lines"].items():
        print(f"  {k:<26}{d['flow_each_kg_s'] * 1e3:7.0f} g/s{d['velocity_m_s']:7.1f} m/s"
              f"{d['friction_bar_m']:7.2f} bar/m{d['loss_bar']:6.2f} bar")
    pr = r["pressurant"]
    print(f"\n  Tank ullage at full flow: oxidizer {pr['ox_tank_bar_a']:.1f} bar a, fuel {pr['fuel_tank_bar_a']:.1f} bar a")
    p = pr["ox_tank_bar_a"]
    print("\n  Oxidizer: tank to venturi")
    for name, dp in r["oxidizer"]["steps"]:
        p -= dp
        print(f"    {name:<48}{dp:5.2f}  -> {p:6.2f} bar a")
    ox = r["oxidizer"]
    for T, f in ox["flow_kg_s"].items():
        print(f"    {T:>4} °C: needs {ox['inlet_required_bar_a'][T]:5.1f} bar a, delivers {f:.3f} kg/s "
              f"({100 * ox['flow_fraction'][T]:.0f} %)")
    print(f"    full flow below {ox['full_flow_below_C']:.1f} °C; choking margin {ox['choking_margin_bar']:.1f} bar")
    fu = r["fuel"]
    print(f"\n  Fuel venturi inlet {fu['venturi_inlet_bar_a']:.1f} bar a, {fu['flow_kg_s']:.3f} kg/s "
          f"({100 * fu['flow_fraction']:.0f} %)")
    rg = r["regulator"]
    print(f"\n  Regulator drop-out at a storage pressure of about {rg['dropout_storage_bar_a']:.0f} bar a "
          f"(HP line {rg['hp_line_loss_bar']:.2f} bar)")
    print("\n  Required Cv at the allowances")
    for k, v in r["required_cv"].items():
        print(f"    {k:<30}{v:6.2f}")
    if r["options"]:
        print("\n  Options (cumulative)")
        for o in r["options"]:
            fr = o["flow_fraction"]
            print(f"    {o['name']:<36}{o['inlet_bar_a']:6.1f} bar a, "
                  + ", ".join(f"{T} °C {100 * f:.0f} %" for T, f in fr.items())
                  + f", full flow below {o['full_flow_below_C']:.1f} °C, margin {o['choking_margin_bar']:.1f} bar")


def _plain(x):
    """Rounded plain types for the results file."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, float):
        return round(x, 4)
    return x


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=str(here / "configs" / "h2_feed_v12.yaml"))
    ap.add_argument("--outdir", default=str(here / "results"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    r = run(cfg)
    report(cfg, r)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(a.config).stem}_results.yaml"
    meta = {"config": Path(a.config).name, "baseline": cfg.get("_baseline"), "date": str(datetime.date.today())}
    path.write_text(yaml.safe_dump(_plain({"meta": meta, **r}), sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    print(f"\nResults: {path}")
    return r


if __name__ == "__main__":
    main(sys.argv[1:])
