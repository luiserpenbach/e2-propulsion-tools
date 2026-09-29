"""Regression and verification tests for the regen extension (pytest)."""
import numpy as np
import pytest

from h2cea import config as C
from h2cea.contour import e2_reg1_asbuilt, wall_stations
from h2cea.coolant import Coolant, h_from_Tp
from h2cea.jacket import e2_reg1
from h2cea.operating_line import state
from h2cea.propellants import n2o_card
from h2cea.regen import march, verify
from h2cea import section2d as S2
from h2cea import structure as ST


@pytest.fixture(scope="module")
def setup():
    c = e2_reg1_asbuilt()
    s = state(0.874, np.pi * C.R_T ** 2, eps=C.EPS_E2, ox=n2o_card(C.T_OX_NOM, 70.0))
    h = h_from_Tp("water", 293.15, 30e5)
    return c, s, h


def test_contour_matches_cad(setup):
    c, _, _ = setup
    from h2cea.contour import chamber_volume
    assert abs(c["xt"] * 1e3 - 136.5) < 0.1
    assert abs(chamber_volume(c) * 1e6 - 748) < 3
    assert abs(wall_stations(c)["s"][-1] * 1e3 - 190.2) < 0.5


def test_handbook_verification(setup):
    c, s, h = setup
    v = verify(lambda n: wall_stations(c, n), s["pt"], s["pc"], e2_reg1(), "water", 1.0, 30.0, h)
    assert v["energy_balance_err"] < 0.005          # handbook 10.8: 0.5 %
    assert v["dTwg_max_K_halving"] < 5.0            # handbook 10.8: 5 K


def test_water_100_matches_hand_check(setup):
    c, s, h = setup
    r = march(wall_stations(c), s["pt"], s["pc"], e2_reg1(), "water", 1.0, 30.0, h)
    i = r.at_x(c["xt"])
    assert 205e3 < r["Q"] < 235e3                   # hand check 219 kW
    assert 690 < r["Twg"][i] - 273.15 < 750         # hand check 720 C
    assert 3.5 < r["dp_bar"] < 4.3                  # hand check 3.9 bar


def test_section_agrees_with_1d(setup):
    c, s, h = setup
    r = march(wall_stations(c), s["pt"], s["pc"], e2_reg1(), "water", 1.0, 30.0, h)
    i = r.at_x(c["xt"])
    sec = S2.section_at(r, i, d=0.04e-3)
    sm = S2.summary(sec)
    assert abs(sm["q_mean"] / r["q"][i] - 1) < 0.03
    blk = S2.summary(S2.section_at(r, i, blocked=True, d=0.04e-3))
    assert blk["Ts_max"] > sm["Ts_max"] + 300


def test_n2o_transport_dilute_gas():
    c = Coolant("n2o")
    st = c.state(h_from_Tp("n2o", 300.0, 1e5), 1e5)
    assert abs(st.mu / 14.9e-6 - 1) < 0.05          # literature ~14.9 uPa s at 300 K


def test_hot_wall_bending():
    j = e2_reg1()
    s = ST.pressure_stresses(48.5e-3, j, 30.0)
    assert abs(s["hot_wall_bending"] / 1e6 - 185) < 3
