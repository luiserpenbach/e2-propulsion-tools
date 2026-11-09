"""Tests for the variable (axial-profile) jacket: jacket.Jacket.with_profile and its use in regen.march."""
import numpy as np
import pytest

from h2cea import config as C
from h2cea.contour import e2_reg1_asbuilt, wall_stations
from h2cea.coolant import h_from_Tp
from h2cea.jacket import Jacket, e2_reg1, channels
from h2cea.operating_line import state
from h2cea.propellants import n2o_card
from h2cea.regen import march


@pytest.fixture(scope="module")
def setup():
    c = e2_reg1_asbuilt()
    w = wall_stations(c, 300)
    s = state(0.874, np.pi * C.R_T ** 2, eps=C.EPS_E2, ox=n2o_card(C.T_OX_NOM, 70.0))
    h = h_from_Tp("water", 293.15, 30e5)
    return c, w, s, h


def _run(w, s, h, j, mdot=1.0):
    return march(w, s["pt"], s["pc"], j, "water", mdot, 30.0, h)


def test_constant_profile_reproduces_constant_jacket(setup):
    c, w, s, h = setup
    j0 = e2_reg1()
    x = np.linspace(0, c["x"][-1], 7)
    j1 = Jacket.with_profile(x, h=j0.h, t_rib=j0.t_rib, t_wall=j0.t_wall, n=j0.n,
                             t_closeout=j0.t_closeout, roughness=j0.roughness)
    r0, r1 = _run(w, s, h, j0), _run(w, s, h, j1)
    for k in ("Q", "Twg_max", "dp_bar", "T_out"):
        assert abs(r1[k] - r0[k]) <= 1e-9 * max(1.0, abs(r0[k]))


def test_share_equals_double_flow_trick(setup):
    """Half the channels with share 0.5 at the real flow == all channels at twice the flow
    (the earlier work-around for alternating jackets)."""
    c, w, s, h = setup
    j0 = e2_reg1()
    x = np.linspace(0, c["x"][-1], 5)
    jh = Jacket.with_profile(x, t_rib=j0.t_rib, share=0.5, n=j0.n // 2,
                             h=j0.h, t_wall=j0.t_wall, t_closeout=j0.t_closeout, roughness=j0.roughness)
    ra = _run(w, s, h, jh, mdot=0.5)          # one pass: 25 channels, half the wall
    rb = _run(w, s, h, j0, mdot=1.0)          # 50 channels, full wall, same flow per channel
    assert abs(ra["Twg_max"] - rb["Twg_max"]) < 1e-6
    assert abs(ra["T_out"] - rb["T_out"]) < 1e-6
    assert abs(ra["Q"] - 0.5 * rb["Q"]) < 1e-6 * rb["Q"]
    assert ra["energy_balance_err"] < 0.005


def test_variable_height_raises_throat_mass_flux(setup):
    c, w, s, h = setup
    j0 = e2_reg1()
    xt = c["xt"]
    x = np.array([0.0, xt - 0.02, xt, xt + 0.02, c["x"][-1]])
    hh = np.array([j0.h, j0.h, 0.5e-3, j0.h, j0.h])
    jv = Jacket.with_profile(x, h=hh, t_rib=j0.t_rib, t_wall=j0.t_wall, n=j0.n,
                             t_closeout=j0.t_closeout, roughness=j0.roughness)
    r0, rv = _run(w, s, h, j0), _run(w, s, h, jv)
    it = int(np.argmin(np.abs(rv["x"] - xt)))
    assert rv["G"][it] > 1.4 * r0["G"][it]
    assert rv["Twg"][it] < r0["Twg"][it]           # faster coolant, cooler throat wall
    assert rv["dp_bar"] > r0["dp_bar"]
    assert rv["energy_balance_err"] < 0.005


def test_width_profile_and_errors():
    x = np.array([0.0, 0.1])
    j = Jacket.with_profile(x, w=[1.0e-3, 2.0e-3], h=1e-3, n=20)
    g = channels(np.array([0.01, 0.01]), j, x)
    assert np.allclose(g["w"], [1e-3, 2e-3])
    assert np.allclose(g["t_rib"], g["pitch"] - g["w"])
    with pytest.raises(ValueError):
        Jacket.with_profile(x, w=1e-3, t_rib=1e-3)
    with pytest.raises(ValueError):
        Jacket.with_profile(x, share=1.5)
    with pytest.raises(ValueError):
        channels(np.array([0.01]), j)          # profile jacket needs x
