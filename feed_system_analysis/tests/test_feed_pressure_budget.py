"""Checks of the feed pressure budget (pytest), run from feed_system_analysis/."""
import copy
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import feed_pressure_budget as fpb  # noqa: E402

CONFIG = HERE / "configs" / "h2_feed_v12.yaml"


@pytest.fixture(scope="module")
def cfg():
    return fpb.load_config(CONFIG)


@pytest.fixture(scope="module")
def rev_b(cfg):
    """The Rev B inputs: nitrogen of Tank Sizing Rev C, HP side at 130 bar / -40 C."""
    c = copy.deepcopy(cfg)
    c["nitrogen"].update(flow_ox_kg_s=0.116, flow_fuel_kg_s=0.029,
                         storage={"pressure_bar": 130.0, "temperature_C": -40.0})
    return fpb.run(c)


def test_reproduces_rev_b_pressurant_side(rev_b):
    """Pressure Drop Analysis Rev B, table 4.1: 92.5 / 95.7 bar a."""
    assert rev_b["pressurant"]["ox_tank_bar_a"] == pytest.approx(92.5, abs=0.1)
    assert rev_b["pressurant"]["fuel_tank_bar_a"] == pytest.approx(95.7, abs=0.1)
    lines = {k: v["loss_bar"] for k, v in rev_b["nitrogen_lines"].items()}
    assert lines["n2_ox_standpipe"] == pytest.approx(3.1, abs=0.05)
    assert lines["n2_fuel_standpipe"] == pytest.approx(0.8, abs=0.05)
    assert lines["n2_hp"] == pytest.approx(0.8, abs=0.05)


def test_reproduces_rev_b_venturis(rev_b):
    """Rev B 4.2 and 4.3: 85.0 bar a, 84 % at 20 C, 3 bar choking margin, fuel 94.2 bar a."""
    ox = rev_b["oxidizer"]
    assert ox["inlet_bar_a"] == pytest.approx(85.0, abs=0.1)
    assert ox["tank_to_venturi_bar"] == pytest.approx(7.5, abs=0.1)
    assert ox["flow_fraction"][20] == pytest.approx(0.84, abs=0.005)
    assert ox["inlet_required_bar_a"][20] == pytest.approx(98.7, abs=0.15)
    assert ox["choking_margin_bar"] == pytest.approx(3.0, abs=0.1)
    assert rev_b["fuel"]["venturi_inlet_bar_a"] == pytest.approx(94.2, abs=0.1)


def test_required_cv_matches_rev_b(cfg):
    """Rev B 4.5: liquid Cv at the allowances and the regulator at 120 bar a, -40 C."""
    c = copy.deepcopy(cfg)
    c["nitrogen"].update(flow_ox_kg_s=0.116, flow_fuel_kg_s=0.029,
                         storage={"pressure_bar": 130.0, "temperature_C": -40.0})
    cv = fpb.run(c)["required_cv"]
    assert cv["ox_filter"] == pytest.approx(5.6, abs=0.05)
    assert cv["ox_main_valve"] == pytest.approx(3.9, abs=0.05)
    assert cv["ox_throttle_valve"] == pytest.approx(2.8, abs=0.05)
    assert cv["fuel_filter"] == pytest.approx(1.7, abs=0.05)
    assert cv["fuel_main_valve"] == pytest.approx(0.9, abs=0.05)
    assert cv["regulator_at_sizing_limit"] == pytest.approx(0.35, abs=0.01)


def test_more_nitrogen_flow_lowers_the_venturi_inlet(cfg, rev_b):
    assert fpb.run(cfg)["oxidizer"]["inlet_bar_a"] < rev_b["oxidizer"]["inlet_bar_a"]


def test_options_raise_the_venturi_inlet(cfg):
    r = fpb.run(cfg)
    inlets = [r["oxidizer"]["inlet_bar_a"]] + [o["inlet_bar_a"] for o in r["options"]]
    assert inlets == sorted(inlets)
