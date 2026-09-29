"""Load the shared baseline (h2_baseline.yaml) and its engine throttle table.

Tools add this folder to sys.path and import the module:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
    import e2_baseline

The module name is unique in the repository, so it cannot clash with a tool's
own modules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT = BASELINE_DIR / "h2_baseline.yaml"
REGENERATE = "cd h2cea && python run_engine_table.py"


def load(path=None) -> dict:
    """The baseline as a dict; the file path is stored under '_path'."""
    path = Path(path) if path else DEFAULT
    bl = yaml.safe_load(path.read_text(encoding="utf-8"))
    bl["_path"] = str(path.resolve())
    return bl


def resolve(ref, relative_to) -> Path:
    """Path of a `baseline:` entry in a config file, relative to that file."""
    p = Path(ref)
    return p if p.is_absolute() else (Path(relative_to).resolve().parent / p).resolve()


def describe(bl: dict) -> dict:
    """Short record of the baseline for results files."""
    m = bl.get("meta", {})
    return {"file": Path(bl["_path"]).name, "revision": m.get("revision"), "date": str(m.get("date"))}


# ----------------------------------------------------------------------------
# Engine throttle table
# ----------------------------------------------------------------------------
def table_inputs(bl: dict) -> dict:
    """The baseline values the engine table depends on."""
    return {"propellants": bl["propellants"], "engine": bl["engine"],
            "tank_pressure_bar": bl["feed"]["tank_pressure_bar"]}


def table_path(bl: dict) -> Path:
    return Path(bl["_path"]).parent / bl["engine"]["table_file"]


def engine_table(bl: dict) -> dict:
    """The throttle table, after checking that it matches the baseline."""
    path = table_path(bl)
    if not path.exists():
        raise SystemExit(f"{path} is missing. Generate it with:  {REGENERATE}")
    table = yaml.safe_load(path.read_text(encoding="utf-8"))
    if table.get("inputs") != table_inputs(bl):
        raise SystemExit(f"{path.name} does not match {Path(bl['_path']).name} "
                         f"(the engine, propellant or tank pressure inputs changed). "
                         f"Regenerate it with:  {REGENERATE}")
    return table


def table_point(table: dict, throttle: float) -> dict:
    """Row of the throttle table, interpolated linearly in throttle."""
    rows = table["rows"]
    thr = [r["throttle"] for r in rows]
    if not thr[0] - 1e-9 <= throttle <= thr[-1] + 1e-9:
        raise ValueError(f"throttle {throttle} is outside the engine table ({thr[0]}-{thr[-1]})")
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {k: float(np.interp(throttle, thr, [r[k] for r in rows])) for k in keys}


# ----------------------------------------------------------------------------
# Filling a tool config from the baseline
# ----------------------------------------------------------------------------
def _get(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fill(cfg: dict, values: dict, label: str = "config") -> list:
    """Set cfg[dotted key] = value for every entry of `values` that the config
    does not set itself. A config value that differs from the baseline is kept
    and reported. Returns the list of override notes (also printed)."""
    notes = []
    for dotted, value in values.items():
        current = _get(cfg, dotted)
        if current is None:
            node = cfg
            *parents, last = dotted.split(".")
            for k in parents:
                node = node.setdefault(k, {})
            node[last] = value
        elif not _same(current, value):
            notes.append(f"NOTE: {label} sets {dotted} = {current}; the baseline has {value}")
    for n in notes:
        print(n)
    return notes


def _same(a, b) -> bool:
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-9, atol=0))
    except (TypeError, ValueError):
        return a == b
