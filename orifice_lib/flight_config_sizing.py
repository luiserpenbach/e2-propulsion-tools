#!/usr/bin/env python3
"""
flight_config_sizing.py — Orifice mass flow from a YAML case list.

Usage, from this directory:
    python flight_config_sizing.py configs/orifice_cases.yaml --outdir results
    python flight_config_sizing.py configs/orifice_cases.yaml --case n2o_injector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

import gas
import liquid
import two_phase
from common import diameter_for_mass_flow

try:
    import plots
except ImportError:  # plotly is optional until a figure is written
    plots = None

MODELS = ("liquid", "gas", "two_phase")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("config must contain a 'cases' list")
    return data


def _target(case: dict) -> float | None:
    raw = case.get("mdot_target_g_s")
    if raw in (None, ""):
        return None
    target = float(raw) / 1e3
    if target <= 0.0:
        raise ValueError(f"{case.get('name')}: mdot_target_g_s must be positive")
    return target


def _require(case: dict, *keys: str) -> None:
    missing = [key for key in keys if case.get(key) in (None, "")]
    if missing:
        name = case.get("name", "(unnamed)")
        raise ValueError(f"{name}: missing {', '.join(missing)}")


def run_case(case: dict):
    _require(case, "name", "model", "fluid", "T_C", "P_down_bar", "d_mm", "Cd")
    model = case["model"]
    if model not in MODELS:
        raise ValueError(f"{case['name']}: model must be one of {', '.join(MODELS)}")
    if not case.get("saturated_liquid") and case.get("P_up_bar") in (None, ""):
        raise ValueError(f"{case['name']}: set P_up_bar, or saturated_liquid: true")

    T_K = float(case["T_C"]) + 273.15
    P_down = float(case["P_down_bar"]) * 1e5
    d_m = float(case["d_mm"]) * 1e-3
    Cd = float(case["Cd"])
    backend = case.get("backend", "HEOS")
    saturated = bool(case.get("saturated_liquid", False))
    P_up = None if saturated else float(case["P_up_bar"]) * 1e5
    target = _target(case)

    if model == "liquid":
        result = liquid.liquid_orifice(
            case["fluid"], T_K, P_down, d_m, Cd, P_up, saturated, backend
        )
        text = liquid.format_report(result, target)
    elif model == "gas":
        if saturated:
            raise ValueError(f"{case['name']}: saturated_liquid applies to liquid and two_phase only")
        result = gas.gas_orifice(
            case["fluid"], T_K, P_up, P_down, d_m, Cd, backend,
            T_norm_K=float(case.get("T_norm_C", 0.0)) + 273.15,
            P_norm_Pa=float(case.get("P_norm_bar", 1.01325)) * 1e5,
        )
        text = gas.format_report(result, target)
    else:
        length = case.get("L_mm")
        L_m = None if length in (None, "") else float(length) * 1e-3
        result = two_phase.two_phase_orifice(
            case["fluid"], T_K, P_down, d_m, Cd, P_up, L_m, saturated, backend
        )
        text = two_phase.format_report(result, target)

    data = {"name": case["name"], **result.to_dict()}
    data.update(_sizing(result, target))
    return result, text, data


def _sizing(result, target: float | None) -> dict:
    if target is None:
        return {}
    out: dict = {"mdot_target_g_s": target * 1e3}
    if isinstance(result, two_phase.TwoPhaseResult):
        diameters = {}
        for label, mdot in (
            ("spi", result.mdot_spi_kg_s),
            ("nhne", result.mdot_nhne_kg_s),
            ("hem", result.mdot_hem_kg_s),
        ):
            if mdot is not None and mdot > 0.0:
                diameters[label] = diameter_for_mass_flow(result.d_m, mdot, target) * 1e3
        out["d_for_target_mm"] = diameters
        return out
    mdot = getattr(result, "mdot_kg_s", None)
    if mdot is None:
        mdot = getattr(result, "mdot_real_kg_s", None)
    if mdot:
        out["d_for_target_mm"] = diameter_for_mass_flow(result.d_m, mdot, target) * 1e3
    return out


def main(argv: list[str] | None = None) -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Orifice mass-flow sizing")
    parser.add_argument("config", nargs="?", default=str(here / "configs" / "orifice_cases.yaml"))
    parser.add_argument("--outdir", default=str(here / "results"))
    parser.add_argument("--case", default=None, help="run only this case name")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = load_config(config_path)
    selected = []
    for case in cfg["cases"]:
        if args.case is None or case.get("name") == args.case:
            selected.append(case)
    if not selected:
        known = ", ".join(case.get("name", "?") for case in cfg["cases"])
        raise SystemExit(f"No case named {args.case!r}. Cases in this file: {known}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in selected:
        result, text, data = run_case(case)
        print(text)
        print()
        if not args.no_plot:
            html_path = outdir / f"{case['name']}.html"
            try:
                if plots is None:
                    raise ImportError("plotly")
                plots.write_plot(result, html_path, case["name"])
                print(f"  plot:    {html_path}")
            except ImportError:
                print("  plotly is not installed; skipping the figure")
        results.append(data)

    stem = config_path.stem + (f"_{args.case}" if args.case else "")
    yaml_path = outdir / f"{stem}_results.yaml"
    yaml_path.write_text(yaml.safe_dump({"cases": results}, sort_keys=False), encoding="utf-8")
    print(f"  results: {yaml_path}")
    return {"cases": results}


if __name__ == "__main__":
    main()
