"""
mission_sizing.py - mission and propellant sizing for the H2 lander.

Two missions are flown with the same vehicle and engine:

  hover : Tethered Hover milestone. Climb to a 1.5 m setpoint, hover counted
          from 0.5 m height for 10 s (+1 s margin), descend and land.
  hop   : Bess Touchdown Award / 10 m hop. Climb to a 10 m apex, stop at zero
          velocity, descend, come to zero vertical velocity within 1 m of the
          ground, then land.

Each mission is a one-dimensional vertical point mass flown along a prescribed
feed-forward profile of trapezoidal velocity segments. The throttle follows
from T = m (g + a_cmd). The propellant flows come from the engine throttle table
of the shared baseline, baseline/h2_baseline_engine_table.yaml (fuel fixed by the venturi,
oxidizer throttled, O/F and chamber pressure falling with thrust). From the
burned propellant the script builds the budget per propellant (burned, 2 s hover reserve, control
allowance, trapped residual, N2O vapour make-up), the mass point at a fixed
liftoff thrust-to-weight, the ballast, the minimum tank volumes and the fill
of the present tanks, and the control authority along the flight. The hover is
the baseline mission; flying the hop with the same hardware is to be decided.

Files
  h2_mission_inputs.yaml    mission inputs (vehicle, tanks, pressurization,
                            allowances, missions) and the path of the baseline
  baseline/h2_baseline.yaml engine, propellant temperatures, residuals
  baseline/h2_baseline_engine_table.yaml  throttle table (h2cea/run_engine_table.py)
  mission_results.yaml      results and the hand-over to the tank sizing

Usage
  python mission_sizing.py                       # both missions + comparison
  python mission_sizing.py --mission hop
  python mission_sizing.py --max-apex            # highest hop the present tanks allow
  python mission_sizing.py --size-tanks          # tanks for both missions (first order)
  python mission_sizing.py --sweep               # liftoff T/W trade
  python mission_sizing.py --plot                # mission_profile_<mission>.png
  python mission_sizing.py --set missions.hop.apex_m=12 --set vehicle.dry_mass_kg=130
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import e2_baseline  # noqa: E402

G0 = 9.80665


# ============================================================================
# Inputs
# ============================================================================
def load_inputs(path, overrides=()):
    """Mission inputs (with --set overrides), the baseline and its engine table."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:                       # --set a.b.c=value
        key, val = item.split("=", 1)
        if key.split(".")[0] == "engine":
            raise SystemExit("The engine is defined in the baseline. Edit baseline/h2_baseline.yaml "
                             "and regenerate the table:  " + e2_baseline.REGENERATE)
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    bl = e2_baseline.load(e2_baseline.resolve(cfg["baseline"], path))
    return cfg, bl, e2_baseline.engine_table(bl)


class Engine:
    """Throttle line from the baseline engine table, interpolated in throttle."""

    def __init__(self, bl_engine, table):
        t = table["rows"]
        self.F_max = bl_engine["rated_thrust_N"]
        self.thr_min = bl_engine["throttle_min"]
        self.fuel_flow = bl_engine["fuel_flow_kg_s"]
        self.thr = np.array([r["throttle"] for r in t])
        self.ox = np.array([r["ox_flow_kg_s"] for r in t])
        self.mr = np.array([r["mixture_ratio"] for r in t])
        self.pc = np.array([r["chamber_pressure_bar"] for r in t])
        self.isp_tab = np.array([r["isp_s"] for r in t])
        self.perf_meta = table.get("meta", {})

    def flows(self, thr):
        """(m_dot_ox, m_dot_fuel) in kg/s at throttle thr."""
        return float(np.interp(thr, self.thr, self.ox)), self.fuel_flow

    def point(self, thr):
        return dict(mixture_ratio=float(np.interp(thr, self.thr, self.mr)),
                    chamber_pressure_bar=float(np.interp(thr, self.thr, self.pc)),
                    isp_s=float(np.interp(thr, self.thr, self.isp_tab)))


@dataclass
class Vehicle:
    dry_mass: float
    payload_min: float
    tw_liftoff: float
    V_ox_tank: float
    V_fuel_tank: float
    ullage_load: float
    T_load_C: float
    rho_fuel: float
    residual_ox: float
    residual_fuel: float
    m_pressurant: float
    m_n2o_vapour: float
    ref_V_prop: float
    ref_V_ox: float
    ref_m_pressurant: float
    ref_m_vapour: float

    @classmethod
    def from_cfg(cls, c, bl):
        t, p, v = c["propellant_tanks"], c["pressurization"], c["vehicle"]
        res = bl["residuals"]
        return cls(dry_mass=v["dry_mass_kg"], payload_min=v["ballast_min_kg"],
                   tw_liftoff=v["liftoff_thrust_to_weight"],
                   V_ox_tank=t["oxidizer"]["count"] * t["oxidizer"]["volume_each_L"],
                   V_fuel_tank=t["fuel"]["count"] * t["fuel"]["volume_each_L"],
                   ullage_load=t["initial_ullage_fraction"],
                   T_load_C=bl["propellants"]["oxidizer"]["temperature_max_C"],   # warm end of the window
                   rho_fuel=t["fuel"]["density_kg_m3"],
                   residual_ox=res["oxidizer_kg"], residual_fuel=res["fuel_kg"],
                   m_pressurant=p["pressurant_mass_kg"], m_n2o_vapour=p["n2o_vapour_makeup_kg"],
                   ref_V_prop=p["reference"]["propellant_volume_L"],
                   ref_V_ox=p["reference"]["oxidizer_volume_L"],
                   ref_m_pressurant=p["pressurant_mass_kg"], ref_m_vapour=p["n2o_vapour_makeup_kg"])


@dataclass
class Allowances:
    reserve_s: float
    control_frac: float


# ============================================================================
# Trajectory building blocks
# ============================================================================
def trapezoid(d, vmax, a_acc, a_dec, v_end=0.0):
    """Start from rest, accelerate, cruise, decelerate to v_end over distance d.
    Returns [(duration, acceleration)] in the direction of motion."""
    d1 = vmax**2 / (2 * a_acc)
    d3 = (vmax**2 - v_end**2) / (2 * a_dec)
    if d1 + d3 > d:  # triangular profile, peak speed below vmax
        vmax = np.sqrt((d + v_end**2 / (2 * a_dec)) / (1 / (2 * a_acc) + 1 / (2 * a_dec)))
        d1 = vmax**2 / (2 * a_acc)
        d3 = (vmax**2 - v_end**2) / (2 * a_dec)
    segs = [(vmax / a_acc, a_acc)]
    if d - d1 - d3 > 1e-9:
        segs.append(((d - d1 - d3) / vmax, 0.0))
    if vmax - v_end > 1e-9:
        segs.append(((vmax - v_end) / a_dec, -a_dec))
    return segs


def down(segs):
    return [(t, -a) for t, a in segs]


def time_above(segs, h0, h_c, dt=1e-3):
    """Kinematic time spent above h_c while flying segs from rest at h0."""
    h, v, t_ab = h0, 0.0, 0.0
    for dur, a in segs:
        n = max(1, int(round(dur / dt)))
        for _ in range(n):
            t_ab += dur / n if h >= h_c else 0.0
            v += a * dur / n
            h += v * dur / n
    return t_ab


# ============================================================================
# Missions
# ============================================================================
class HoverMission:
    name = "hover"

    def __init__(self, c):
        self.c = c
        self.t_startup, self.t_tailoff = c["t_startup_s"], c["t_tailoff_s"]

    def phases(self):
        c = self.c
        up = trapezoid(c["hover_altitude_m"], c["climb_rate_max_m_s"], c["climb_accel_m_s2"], c["climb_accel_m_s2"])
        dn = down(trapezoid(c["hover_altitude_m"], c["descent_rate_max_m_s"], c["descent_accel_m_s2"],
                            c["descent_accel_m_s2"], c["touchdown_speed_m_s"]))
        if c["count_altitude_m"] is None:
            hold = c["hover_hold_s"]
        else:
            hold = max(0.0, c["hover_required_s"] + c["hover_margin_s"]
                       - time_above(up, 0.0, c["count_altitude_m"])
                       - time_above(dn, c["hover_altitude_m"], c["count_altitude_m"]))
        return [("Start-up on pad", [(c["t_startup_s"], 0.0)], True),
                ("Climb", up, False),
                ("Hover hold", [(hold, 0.0)], False),
                ("Descent", dn, False),
                ("Tail-off", [(c["t_tailoff_s"], 0.0)], True)]

    def key_points(self):
        return ["Hover hold"]

    def checks(self, r):
        c, lg = self.c, r["log"]
        h_win = c["count_altitude_m"] if c["count_altitude_m"] is not None else c["hover_altitude_m"]
        dtl = np.diff(lg["t"], append=lg["t"][-1])
        above = lg["h"] >= h_win
        t_counted = float(np.sum(dtl * above))
        if not above.any():                      # never reaches the count altitude
            return dict(hover_counted_s=0.0, hover_window_after_liftoff_s=None,
                        ascent_time_ok=False, hover_time_ok=False)
        t_win = float(lg["t"][np.argmax(above)] - c["t_startup_s"])
        return dict(hover_counted_s=round(t_counted, 2), hover_window_after_liftoff_s=round(t_win, 2),
                    ascent_time_ok=bool(t_win <= c["ascent_time_limit_s"]),
                    hover_time_ok=bool(t_counted >= c["hover_required_s"] - 1e-6))


class HopMission:
    name = "hop"

    def __init__(self, c):
        self.c = c
        self.t_startup, self.t_tailoff = c["t_startup_s"], c["t_tailoff_s"]

    def phases(self):
        c = self.c
        up = trapezoid(c["apex_m"], c["climb_rate_max_m_s"], c["climb_accel_m_s2"], c["climb_accel_m_s2"])
        dn = down(trapezoid(c["apex_m"] - c["stop_altitude_m"], c["descent_rate_max_m_s"],
                            c["descent_accel_m_s2"], c["descent_accel_m_s2"], 0.0))
        land = down(trapezoid(c["stop_altitude_m"], c["touchdown_speed_m_s"], c["descent_accel_m_s2"],
                              c["descent_accel_m_s2"], c["touchdown_speed_m_s"]))
        return [("Start-up on pad", [(c["t_startup_s"], 0.0)], True),
                ("Climb to apex", up, False),
                ("Hold at apex", [(c["apex_hold_s"], 0.0)], False),
                ("Descent", dn, False),
                ("Stop near ground", [(c["stop_hold_s"], 0.0)], False),
                ("Final descent", land, False),
                ("Tail-off", [(c["t_tailoff_s"], 0.0)], True)]

    def key_points(self):
        return ["Hold at apex", "Stop near ground"]

    def checks(self, r):
        return dict(apex_m=self.c["apex_m"], stop_altitude_m=self.c["stop_altitude_m"])


MISSIONS = {"hover": HoverMission, "hop": HopMission}


# ============================================================================
# Flight integration, budget, mass point
# ============================================================================
def fly(m0, eng, mis, dt=0.005):
    log = {k: [] for k in ("t", "h", "v", "a", "thr", "m", "ox", "fu")}
    phases_out, violations = [], []
    t = h = v = ox = fu = 0.0
    m = m0
    for name, segs, ground in mis.phases():
        t0, ox0, fu0, m_start, thr_seen = t, ox, fu, m, []
        for dur, a in segs:
            n = max(1, int(round(dur / dt)))
            hs = dur / n
            for _ in range(n):
                if ground:
                    thr, a_eff, v = eng.thr_min, 0.0, 0.0
                else:
                    thr_req = m * (G0 + a) / eng.F_max
                    thr = float(np.clip(thr_req, eng.thr_min, 1.0))
                    if abs(thr - thr_req) > 1e-6:
                        violations.append((round(t, 2), name, round(thr_req, 3)))
                    a_eff = thr * eng.F_max / m - G0
                    thr_seen.append(thr)
                mo, mf = eng.flows(thr)
                for k, val in zip(log, (t, h, v, a_eff, thr, m, ox, fu)):
                    log[k].append(val)
                v += a_eff * hs
                h = max(0.0, h + v * hs)
                m -= (mo + mf) * hs
                ox += mo * hs
                fu += mf * hs
                t += hs
        rng = (min(thr_seen), max(thr_seen)) if thr_seen else (eng.thr_min, eng.thr_min)
        phases_out.append(dict(phase=name, start_s=t0, duration_s=t - t0, throttle_min=rng[0],
                               throttle_max=rng[1], n2o_kg=ox - ox0, ethanol_kg=fu - fu0,
                               mass_start_kg=m_start, mass_end_kg=m))
    log = {k: np.array(val) for k, val in log.items()}
    return dict(log=log, phases=phases_out, burned_ox=ox, burned_fuel=fu,
                t_on=t, m_touch=m, violations=violations)


def budget(m0, eng, veh, mis, allw):
    r = fly(m0, eng, mis)
    thr_td = r["m_touch"] * G0 / eng.F_max
    mo_td, mf_td = eng.flows(thr_td)
    b = dict(burned_ox=r["burned_ox"], burned_fuel=r["burned_fuel"],
             reserve_ox=mo_td * allw.reserve_s, reserve_fuel=mf_td * allw.reserve_s,
             control_ox=allw.control_frac * r["burned_ox"],
             control_fuel=allw.control_frac * r["burned_fuel"],
             residual_ox=veh.residual_ox, residual_fuel=veh.residual_fuel,
             vapour_ox=veh.m_n2o_vapour, thr_touchdown=thr_td)
    for p in ("ox", "fuel"):
        b[f"usable_{p}"] = b[f"burned_{p}"] + b[f"reserve_{p}"] + b[f"control_{p}"]
    b["loaded_ox"] = b["usable_ox"] + veh.residual_ox + veh.m_n2o_vapour
    b["loaded_fuel"] = b["usable_fuel"] + veh.residual_fuel
    b["fluids_total"] = b["loaded_ox"] + b["loaded_fuel"] + veh.m_pressurant
    return r, b


def solve(eng, veh, mis, allw, tw=None):
    """Mass point from the liftoff T/W target; ballast fills the rest."""
    tw = tw or veh.tw_liftoff
    m0 = eng.F_max / (tw * G0)
    r, b = budget(m0, eng, veh, mis, allw)
    return dict(m0=m0, tw=tw, payload=m0 - veh.dry_mass - b["fluids_total"], r=r, b=b)


def rho_n2o(T_C):
    """Saturated liquid N2O density, kg/m^3 (CoolProp values)."""
    T = [0, 5, 10, 15, 20, 25, 30, 35.0]
    rho = [907.1, 880.7, 851.5, 820.7, 785.1, 742.9, 688.1, 589.5]
    return float(np.interp(T_C, T, rho))


def tank_check(b, veh):
    rho = rho_n2o(veh.T_load_C)
    return dict(n2o_density_kg_m3=rho,
                n2o_volume_required_L=b["loaded_ox"] / rho * 1e3 / (1 - veh.ullage_load),
                ethanol_volume_required_L=b["loaded_fuel"] / veh.rho_fuel * 1e3 / (1 - veh.ullage_load),
                n2o_volume_available_L=veh.V_ox_tank, ethanol_volume_available_L=veh.V_fuel_tank)


def tank_capacity(veh):
    """Usable propellant the present tanks can deliver (kg)."""
    ox = veh.V_ox_tank * 1e-3 * (1 - veh.ullage_load) * rho_n2o(veh.T_load_C) \
        - veh.residual_ox - veh.m_n2o_vapour
    fu = veh.V_fuel_tank * 1e-3 * (1 - veh.ullage_load) * veh.rho_fuel - veh.residual_fuel
    return ox, fu


def fits(res, veh):
    tc = tank_check(res["b"], veh)
    return (tc["n2o_volume_required_L"] <= veh.V_ox_tank + 1e-9
            and tc["ethanol_volume_required_L"] <= veh.V_fuel_tank + 1e-9)


def resize(veh, V_ox, V_fu):
    veh.V_ox_tank, veh.V_fuel_tank = V_ox, V_fu
    veh.m_pressurant = veh.ref_m_pressurant * (V_ox + V_fu) / veh.ref_V_prop
    veh.m_n2o_vapour = veh.ref_m_vapour * V_ox / veh.ref_V_ox


def size_tanks(eng, veh, missions, allw):
    """Smallest tanks that cover every mission; pressurant and vapour make-up
    scaled with volume (first order, rerun the tank sizing toolchain)."""
    for _ in range(40):
        V_ox = V_fu = 0.0
        for mis in missions:
            tc = tank_check(solve(eng, veh, mis, allw)["b"], veh)
            V_ox = max(V_ox, tc["n2o_volume_required_L"])
            V_fu = max(V_fu, tc["ethanol_volume_required_L"])
        done = abs(V_ox - veh.V_ox_tank) < 1e-4 and abs(V_fu - veh.V_fuel_tank) < 1e-4
        resize(veh, V_ox, V_fu)
        if done:
            break


def max_apex(eng, veh, allw, hop_cfg):
    """Highest hop apex whose propellant fits the present tanks (bisection)."""
    c = copy.deepcopy(hop_cfg)
    lo, hi = 1.0, 60.0
    for _ in range(40):
        c["apex_m"] = 0.5 * (lo + hi)
        ok = fits(solve(eng, veh, HopMission(c), allw), veh)
        lo, hi = (c["apex_m"], hi) if ok else (lo, c["apex_m"])
    c["apex_m"] = lo
    return c


def authority(m, eng):
    return eng.F_max / m - G0, eng.thr_min * eng.F_max / m - G0


# ============================================================================
# Results and reporting
# ============================================================================
def mission_result(eng, veh, mis, res):
    r, b = res["r"], res["b"]
    rated_ox, rated_fu = eng.flows(1.0)
    usable = b["usable_ox"] + b["usable_fuel"]
    mean = (b["burned_ox"] + b["burned_fuel"]) / r["t_on"]
    pts = [("liftoff", res["m0"])] + [(p["phase"], p["mass_start_kg"]) for p in r["phases"]
                                       if p["phase"] in mis.key_points()] + [("touchdown", r["m_touch"])]
    fl = [p for p in r["phases"] if p["throttle_max"] > eng.thr_min + 1e-6]
    cap_ox, cap_fu = tank_capacity(veh)
    rnd = lambda x, n=3: round(float(x), n)
    return dict(
        engine_on_s=rnd(r["t_on"], 2),
        airborne_s=rnd(r["t_on"] - mis.t_startup - mis.t_tailoff, 2),
        checks=mis.checks(r),
        phases=[{k: (rnd(v) if isinstance(v, float) else v) for k, v in p.items()} for p in r["phases"]],
        propellant_kg={
            "n2o": dict(burned=rnd(b["burned_ox"]), reserve=rnd(b["reserve_ox"]), control=rnd(b["control_ox"]),
                        usable=rnd(b["usable_ox"]), residual=rnd(b["residual_ox"]),
                        vapour_makeup=rnd(veh.m_n2o_vapour), loaded=rnd(b["loaded_ox"])),
            "ethanol": dict(burned=rnd(b["burned_fuel"]), reserve=rnd(b["reserve_fuel"]),
                            control=rnd(b["control_fuel"]), usable=rnd(b["usable_fuel"]),
                            residual=rnd(b["residual_fuel"]), loaded=rnd(b["loaded_fuel"]))},
        flows_kg_s=dict(mean_total=rnd(mean), mean_n2o=rnd(b["burned_ox"] / r["t_on"]),
                        mean_ethanol=rnd(b["burned_fuel"] / r["t_on"]),
                        rated_n2o=rnd(rated_ox), rated_ethanol=rnd(rated_fu),
                        mean_mixture_ratio=rnd(b["burned_ox"] / b["burned_fuel"], 2)),
        mass_kg=dict(dry=rnd(veh.dry_mass, 1), ballast=rnd(res["payload"], 2),
                     pressurant=rnd(veh.m_pressurant, 2), fluids=rnd(b["fluids_total"], 2),
                     liftoff=rnd(res["m0"], 2), touchdown=rnd(r["m_touch"], 2),
                     ballast_ok=bool(res["payload"] >= veh.payload_min)),
        throttle=dict(in_flight_min=rnd(min(p["throttle_min"] for p in fl)),
                      in_flight_max=rnd(max(p["throttle_max"] for p in fl)),
                      touchdown_hover=rnd(b["thr_touchdown"]), limit_hits=len(r["violations"])),
        authority_m_s2=[dict(point=lab, hover_throttle=rnd(m * G0 / eng.F_max),
                             up=rnd(authority(m, eng)[0], 2), down=rnd(authority(m, eng)[1], 2))
                        for lab, m in pts],
        tanks={**{k: rnd(v, 2) for k, v in tank_check(b, veh).items()},
               "n2o_usable_margin_kg": rnd(cap_ox - b["usable_ox"]),
               "ethanol_usable_margin_kg": rnd(cap_fu - b["usable_fuel"]),
               "ethanol_margin_engine_s": rnd((cap_fu - b["usable_fuel"]) / eng.fuel_flow, 2)},
        handover=dict(usable_n2o_kg=rnd(b["usable_ox"]), usable_ethanol_kg=rnd(b["usable_fuel"]),
                      trapped_n2o_kg=rnd(veh.residual_ox), trapped_ethanol_kg=rnd(veh.residual_fuel),
                      mean_flow_n2o_kg_s=rnd(b["burned_ox"] / r["t_on"]),
                      mean_flow_ethanol_kg_s=rnd(eng.fuel_flow),
                      duration_mean_flow_s=rnd(usable / mean, 2),
                      rated_flow_n2o_kg_s=rnd(rated_ox), rated_flow_ethanol_kg_s=rnd(rated_fu),
                      duration_rated_flow_s=rnd(usable / (rated_ox + rated_fu), 2)))


def print_mission(name, x):
    print(f"=== {name}: engine-on {x['engine_on_s']:.1f} s, airborne {x['airborne_s']:.1f} s  {x['checks']}")
    print(f"{'Phase':<20}{'start s':>8}{'dur s':>7}{'throttle':>13}{'N2O kg':>8}{'EtOH kg':>9}")
    for p in x["phases"]:
        thr = (f"{p['throttle_min']:.2f}" if abs(p["throttle_max"] - p["throttle_min"]) < 5e-3
               else f"{p['throttle_min']:.2f}-{p['throttle_max']:.2f}")
        print(f"{p['phase']:<20}{p['start_s']:8.2f}{p['duration_s']:7.2f}{thr:>13}{p['n2o_kg']:8.2f}{p['ethanol_kg']:9.2f}")
    po, pf = x["propellant_kg"]["n2o"], x["propellant_kg"]["ethanol"]
    print(f"{'Budget, kg':<22}{'N2O':>8}{'EtOH':>8}")
    for k in ("burned", "reserve", "control", "usable", "residual", "loaded"):
        print(f"  {k:<20}{po[k]:8.2f}{pf[k]:8.2f}")
    print(f"  {'vapour make-up':<20}{po['vapour_makeup']:8.2f}")
    m, f, t, tk = x["mass_kg"], x["flows_kg_s"], x["throttle"], x["tanks"]
    print(f"Mass: liftoff {m['liftoff']:.1f}, touchdown {m['touchdown']:.1f}, ballast {m['ballast']:.1f} "
          f"({'OK' if m['ballast_ok'] else 'BELOW MINIMUM'}); mean flow {f['mean_total']:.3f} kg/s, "
          f"mean O/F {f['mean_mixture_ratio']:.2f}")
    print(f"Throttle in flight {t['in_flight_min']:.2f}-{t['in_flight_max']:.2f}, touchdown {t['touchdown_hover']:.2f}"
          + ("  WARNING < 10 % headroom" if t["in_flight_max"] > 0.9 else ""))
    for a in x["authority_m_s2"]:
        print(f"  {a['point']:<18} hover throttle {a['hover_throttle']:.2f}  up {a['up']:+.2f}  down {a['down']:+.2f}")
    print(f"Tanks: N2O {tk['n2o_volume_required_L']:.1f} of {tk['n2o_volume_available_L']:.1f} L "
          f"(margin {tk['n2o_usable_margin_kg']:+.2f} kg), ethanol {tk['ethanol_volume_required_L']:.2f} of "
          f"{tk['ethanol_volume_available_L']:.2f} L (margin {tk['ethanol_usable_margin_kg']:+.2f} kg = "
          f"{tk['ethanol_margin_engine_s']:+.1f} s)\n")


def sweep(eng, veh, missions, allw, tws=(1.15, 1.20, 1.25, 1.30, 1.35)):
    rows = []
    for tw in tws:
        m0 = eng.F_max / (tw * G0)
        row = dict(tw=tw, liftoff_kg=round(m0, 1), up_authority_liftoff_m_s2=round(authority(m0, eng)[0], 2))
        for mis in missions:
            res = solve(eng, veh, mis, allw, tw)
            row[f"ballast_{mis.name}_kg"] = round(res["payload"], 1)
            row[f"peak_throttle_{mis.name}"] = round(max(p["throttle_max"] for p in res["r"]["phases"]), 3)
            row[f"touchdown_throttle_{mis.name}"] = round(res["b"]["thr_touchdown"], 3)
        rows.append(row)
    return rows


def clean(x):
    """Convert numpy scalars and tuples so that yaml.safe_dump can write them."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def plot(res, path, title, thr_min=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lg = res["r"]["log"]
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for a, key, lab in zip(ax, ("h", "v", "thr", "m"),
                           ("Height, m", "Vertical velocity, m/s", "Throttle", "Mass, kg")):
        a.plot(lg["t"], lg[key], lw=1.6, color="#2a6fdb")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
        for ph in res["r"]["phases"]:
            a.axvline(ph["start_s"], color="0.85", lw=0.8)
    ax[2].axhline(thr_min, color="#c0392b", ls="--", lw=1)
    ax[2].axhline(1.0, color="#c0392b", ls="--", lw=1)
    ax[0].set_title(title)
    ax[-1].set_xlabel("Time from engine start, s")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="h2_mission_inputs.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an input, e.g. vehicle.dry_mass_kg=130")
    ap.add_argument("--mission", choices=["hover", "hop", "both"], default="both")
    ap.add_argument("--max-apex", action="store_true", help="highest hop the present tanks allow")
    ap.add_argument("--size-tanks", action="store_true", help="right-size tanks for the selected missions")
    ap.add_argument("--sweep", action="store_true", help="liftoff T/W trade")
    ap.add_argument("--plot", action="store_true", help="write mission_profile_<mission>.png")
    ap.add_argument("--out", default="mission_results.yaml")
    a = ap.parse_args()

    cfg, bl, table = load_inputs(a.inputs, a.set)
    eng = Engine(bl["engine"], table)
    veh = Vehicle.from_cfg(cfg, bl)
    allw = Allowances(cfg["allowances"]["reserve_hover_s"], cfg["allowances"]["control_fraction"])
    names = ["hover", "hop"] if a.mission == "both" else [a.mission]
    missions = [MISSIONS[n](cfg["missions"][n]) for n in names]

    out = dict(meta=dict(generated_by="mission_sizing.py", date=str(dt.date.today()),
                         inputs=Path(a.inputs).name, overrides=a.set,
                         baseline=e2_baseline.describe(bl), engine_table=table.get("meta", {})))
    veh_present = copy.deepcopy(veh)             # --max-apex always uses the present tanks
    if a.size_tanks:
        size_tanks(eng, veh, missions, allw)
        out["sized_tanks"] = dict(n2o_volume_L=round(float(veh.V_ox_tank), 2),
                                  ethanol_volume_L=round(float(veh.V_fuel_tank), 2),
                                  pressurant_mass_kg=round(float(veh.m_pressurant), 2),
                                  n2o_vapour_makeup_kg=round(float(veh.m_n2o_vapour), 2),
                                  note="first-order scaling from the tank sizing reference point")
        print(f"Tanks sized for {', '.join(names)}: {out['sized_tanks']}\n")

    out["missions"] = {}
    for mis in missions:
        res = solve(eng, veh, mis, allw)
        x = mission_result(eng, veh, mis, res)
        out["missions"][mis.name] = x
        print_mission(mis.name, x)
        if a.plot:
            plot(res, Path(a.inputs).parent / f"mission_profile_{mis.name}.png", f"H2 lander, {mis.name} mission",
                 eng.thr_min)

    # Hand-over: sizing case from the inputs, else the mission with the largest usable load
    sizing = (cfg.get("analysis") or {}).get("sizing_case")
    if sizing not in out["missions"]:
        sizing = max(out["missions"], key=lambda n: out["missions"][n]["handover"]["usable_n2o_kg"]
                     + out["missions"][n]["handover"]["usable_ethanol_kg"])
    out["handover"] = dict(sizing_case=sizing,
                           check_cases=[n for n in out["missions"] if n != sizing],
                           **out["missions"][sizing]["handover"])
    if a.sweep:
        out["tw_sweep"] = sweep(eng, veh, missions, allw)
        print("Liftoff T/W trade")
        for row in out["tw_sweep"]:
            print("  ", clean(row))
    if a.max_apex and "hop" in cfg["missions"]:
        c = max_apex(eng, veh_present, allw, cfg["missions"]["hop"])
        out["max_apex_present_tanks_m"] = round(c["apex_m"], 2)
        print(f"Highest hop with the present tanks: {c['apex_m']:.1f} m")

    out_path = Path(a.inputs).parent / a.out
    with open(out_path, "w") as fh:
        fh.write("# H2 mission analysis results. Generated by mission_sizing.py, do not edit by hand.\n"
                 "# The 'handover' section is the input to the tank and pressurization sizing.\n")
        yaml.safe_dump(clean(out), fh, sort_keys=False)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
