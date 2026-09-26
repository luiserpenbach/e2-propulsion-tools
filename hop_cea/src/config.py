"""Baseline inputs for the H2 lander thrust chamber analyses.

Every number here is traceable to a project document. Change them here, not in the
analysis scripts.

Sources
-------
H2-PRP-DOC-001 Rev A (Propulsion System Architecture, 2026-09-24)
H2-PRP-001 (2250 N max thrust at pc 25 bar), H2-PRP-036 (ox flow range)
E2-TCA-DOC-001 Rev C and Thrust Chamber E2-REG-1-A (as-built contour table)
LDR-PRP-TNK-003 Rev B (tank sizing: 100 bar set pressure)
"""
G0 = 9.80665                 # m/s^2, standard gravity
P_AMB = 1.01325              # bar, sea level

# ---- Engine design point (architecture decisions of 2026-09-24) ----------------
F_MAX = 2250.0               # N, sea-level thrust at 100 % (H2-PRP-001)
PC_MAX = 25.0                # bar, chamber pressure at 100 %
EPS = 4.0                    # nozzle area ratio (baseline; E2-REG-1 as built is 3.0)
MDOT_FUEL = 0.20             # kg/s, fixed by the fuel cavitating venturi
THROTTLE_FLOOR = 0.50        # fraction of F_MAX

# Efficiencies. The architecture sets the delivered sea-level Isp to 93 % of ideal.
# The split between combustion and nozzle matters for the throat area and for the
# throttled points, so it is made explicit here:
ETA_CSTAR = 0.955            # c* efficiency (energy release, mixing)
ETA_CF_VAC = 0.977           # applied to the ideal VACUUM thrust coefficient
                             # (divergence, boundary layer, kinetics); the ambient
                             # pressure term -pa*Ae is exact and carries no loss.
# 0.955 x 0.977 on vacuum thrust gives 0.930 on sea-level Isp at the 100 % point.

# ---- Propellant states ----------------------------------------------------------
T_OX_NOM = 293.15            # K, N2O temperature at start (architecture: 5-25 C)
T_OX_RANGE = (278.15, 298.15)
P_TANK = 100.0               # bar, regulated tank set pressure
ETHANOL_WT = 100.0           # wt% ethanol in the fuel (state the grade!)

# ---- Feed pressure model at 100 % (architecture section 7.1) -------------------
P_INJ_IN_NOM = 40.0          # bar, oxidiser injector inlet at nominal flow
DP_JACKET_NOM = (20.0, 30.0) # bar, jacket pressure drop estimate at nominal flow
P_VENTURI_IN = 93.0          # bar, venturi inlet after lines, main valve, throttle
VENTURI_RECOVERY = 0.80      # max outlet/inlet pressure ratio that keeps cavitation

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
