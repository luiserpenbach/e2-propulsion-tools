"""Checks of the boiling-regime helpers (twophase.py) and of the film boiling march."""
import numpy as np
import pytest

from h2cea import twophase as TP
from h2cea import config as C
from h2cea import n2o
from h2cea.coolant import Coolant
from h2cea.contour import e2_reg1_asbuilt, wall_stations
from h2cea.jacket import e2_reg1
from h2cea.operating_line import state
from h2cea.propellants import n2o_card
from h2cea.regen import march


def test_superheat_limit_between_saturation_and_critical():
    Tc = 309.52
    for Tsat in (250.0, 280.0, 300.0, 309.0):
        Tsl = TP.T_superheat_limit(Tsat, Tc)
        assert Tsat < Tsl <= Tc
    # approaches the critical temperature as saturation approaches it
    assert TP.T_superheat_limit(309.5, Tc) - 309.5 < 0.2


def test_nucleate_limit_is_inverse_of_cooper():
    pr, dT = 0.6, 8.0
    q = TP.q_nb_max(pr, dT)
    assert q / TP.h_cooper(q, pr) == pytest.approx(dT, rel=1e-6)


def test_y_factor_and_vapour_limit():
    assert TP.y_factor(4.0, 1.0) == pytest.approx(1.0)
    assert TP.y_factor(4.0, 0.2) < 1.0
    # Miropolskii at x = 1 is Dittus-Boelter on the vapour with Pr_w^0.8
    cool = Coolant("n2o")
    props = TP.Props(cool)
    sat = props.sat(50e5)
    w = props.at_T(500.0, 50e5)
    G, D = 5000.0, 1e-3
    h = TP.h_miropolskii(G, D, 1.0, sat, w)
    ref = 0.023 * (G * D / sat["mu_v"]) ** 0.8 * w.Pr ** 0.8 * sat["k_v"] / D
    assert h == pytest.approx(ref, rel=1e-9)


def test_dryout_incipience_negative_at_e2_throat():
    """At the E2 full-thrust throat (Bo ~ 7e-3, low surface tension) the
    Kim-Mudawar dryout quality is negative: no wetted annular film."""
    props = TP.Props(Coolant("n2o"))
    sat = props.sat(51.7e5)
    assert TP.x_dryout_incipience(18600.0, 0.93e-3, sat, 22e6, 0.68) < 0


@pytest.mark.parametrize("model", ["miropolskii", "jackson"])
def test_film_march_energy_balance(model):
    contour = e2_reg1_asbuilt()
    wall = wall_stations(contour, 200)
    ops = state(0.874, np.pi * C.R_T ** 2, eps=C.EPS_E2, ox=n2o_card(C.T_OX_NOM, 70.0))
    h_tank = n2o.h_TP(C.T_OX_NOM, C.P_TANK)
    r = march(wall, ops["pt"], ops["pc"], e2_reg1(), "n2o", 0.874, 58.0, h_tank, tp_model=model)
    assert r["energy_balance_err"] < 5e-3
    assert r["Twg_max"] > 1050.0          # film boiling: throat above the IN718 limit
