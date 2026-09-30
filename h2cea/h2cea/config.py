"""Inputs for the H2 lander thrust chamber analyses.

The engine design point, propellant states and feed pressures come from the
shared baseline, baseline/h2_baseline.yaml at the repository root. Change them
there. The E2-REG-1 hardware (contour, jacket) and the regen analysis settings
below are specific to h2cea and are set here.

Sources
-------
Baseline: see the `meta.sources` list in baseline/h2_baseline.yaml
E2-TCA-DOC-001 Rev C and Thrust Chamber E2-REG-1-A (as-built contour table)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baseline"))
import e2_baseline  # noqa: E402

BASELINE = e2_baseline.load()
_E, _P, _F = BASELINE["engine"], BASELINE["propellants"], BASELINE["feed"]

G0 = 9.80665                 # m/s^2, standard gravity
P_AMB = _E["ambient_pressure_bar"]   # bar

# ---- Engine design point (baseline) ---------------------------------------------
F_MAX = _E["rated_thrust_N"]                 # N, sea-level thrust at 100 %
PC_MAX = _E["chamber_pressure_rated_bar"]    # bar, chamber pressure at 100 %
EPS = _E["expansion_ratio"]                  # nozzle area ratio (E2-REG-1 as built is 3.0)
MDOT_FUEL = _E["fuel_flow_kg_s"]             # kg/s, fixed by the fuel cavitating venturi
THROTTLE_FLOOR = _E["throttle_min"]          # fraction of F_MAX

# Efficiencies. The split between combustion and nozzle matters for the throat area
# and for the throttled points, so it is explicit:
ETA_CSTAR = _E["eta_cstar"]                  # c* efficiency (energy release, mixing)
ETA_CF_VAC = _E["eta_cf_vac"]                # applied to the ideal VACUUM thrust coefficient;
                                             # the ambient pressure term -pa*Ae carries no loss

# ---- Propellant states (baseline) -------------------------------------------------
T_OX_NOM = _P["oxidizer"]["temperature_nominal_C"] + 273.15     # K, N2O at start
T_OX_RANGE = (_P["oxidizer"]["temperature_min_C"] + 273.15,
              _P["oxidizer"]["temperature_max_C"] + 273.15)
P_TANK = _F["tank_pressure_bar"]             # bar, regulated tank set pressure
ETHANOL_WT = _P["fuel"]["ethanol_wt_pct"]    # wt% ethanol in the fuel

# ---- Feed pressure model at 100 % (baseline) ----------------------------------------
P_INJ_IN_NOM = _F["ox_injector_inlet_bar"]   # bar, oxidiser injector inlet at nominal flow
DP_JACKET_NOM = tuple(_F["jacket_pressure_drop_bar"])   # bar, jacket pressure drop estimate
P_VENTURI_IN = _F["venturi_inlet_bar"]       # bar, venturi inlet after lines, main valve, throttle
VENTURI_RECOVERY = _F["venturi_pressure_recovery"]   # max outlet/inlet pressure ratio

# ---- Reference contour: E2-REG-1 as built --------------------------------------
# The as-built table lists "D_c 48.5 mm", but only R_c = 48.5 mm closes the listed
# chamber length of 136.5 mm with the listed arcs and 35 deg cone, and gives the
# listed contraction ratio of 12. The table label is therefore taken as a radius.
R_T = 28.14e-3 / 2           # m, throat radius
R_C = 48.5e-3                # m, chamber radius (see note above)
L_CYL = 54.68e-3             # m, cylindrical length
R_CONV1 = 82.45e-3           # m, blend radius cylinder -> cone
THETA_CONV = 35.0            # deg, convergent half angle
R_UP = 1.5 * R_T             # m, throat upstream radius
R_DN = 0.382 * R_T           # m, throat downstream radius
BELL_FRACTION = 0.80         # Rao bell length, fraction of 15 deg cone
THETA_N, THETA_E = 21.0, 14.0  # deg, Rao angles for eps 4, 80 % bell (chart values)

# ---- E2-REG-1 as built: nozzle and cooling jacket (Thrust Chamber E2-REG-1-A) ---
# The as-built nozzle is not the eps-4 Rao bell above: it ends at r = 24.33 mm
# (eps 2.99) after 43.6 mm, with 18.8 deg entry and 8 deg exit angles.
R_E_E2 = 24.33e-3            # m, exit radius as built
EPS_E2 = (R_E_E2 / R_T) ** 2 # 2.99
L_BELL_E2 = 43.6e-3          # m, throat to exit plane
THETA_N_E2, THETA_E_E2 = 18.8, 8.0   # deg
RC_THROAT_BARTZ = None       # m; None = mean of upstream/downstream throat radius (handbook 9.2)

N_CH = 50                    # channels
T_WALL = 0.5e-3              # m, hot wall
T_RIB = 0.6e-3               # m, rib, constant
H_CH = 0.75e-3               # m, channel height, constant
T_CLOSEOUT = 1.0e-3          # m, closeout (TBC against CAD: 1.0 or 1.25 mm)
ROUGHNESS = 10e-6            # m, absolute channel roughness (assumed; take from flow test)
WALL_MATERIAL = "IN718"

# ---- Regen analysis defaults ------------------------------------------------------
N_STATIONS = 600             # 1D march stations along the wall
BARTZ_FACTOR = 1.0           # gas-side multiplier; CEA transport already matches the
                             # handbook 9500 W/m2K at the full-thrust throat
T_WG_LIMIT = 1050.0          # K, IN718 gas-side wall limit (handbook 11.1)
T_WC_LIMIT_N2O = 573.0       # K, coolant-side wall limit with N2O (handbook 10.3)
CHF_RATIO_LIMIT = 0.5        # q / q_CHF (handbook 10.3)
V_MIN_N2O = 5.0              # m/s, minimum coolant velocity with N2O (handbook 10.3)

# N2O-cooled design case (run_regen.py): throttle points as fractions of rated thrust
# (oxidiser flow from the baseline engine table) and the coolant-side model for boiling
# N2O (twophase.MODELS). The jacket heat is returned to the chamber (regen.n2o_design_case).
N2O_THROTTLE = {"100": 1.00, "81": 0.81, "74": 0.74, "50": 0.50}
N2O_TP_MODEL = "regime"

# Water-cooled test configuration (E2-REG-1-A, test plan)
WATER_MDOT = 1.0             # kg/s
WATER_P_IN = 30.0            # bar
WATER_T_IN = 293.15          # K
