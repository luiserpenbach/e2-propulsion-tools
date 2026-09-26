# H-1B Pressurization & Tank Sizing Analysis

Physics-based sizing and transient simulation for the H-1B propellant tanks and shared nitrogen pressurant system. The analysis sizes capsule tank geometry (ethanol fuel + N2O oxidizer), runs coupled pressurization transients at multiple propellant temperatures, and sizes the pressurant bottle against regulator inlet requirements.

## Overview

The vehicle uses **externally pressurized blowdown** with a single N2 bottle feeding both tanks through regulators held at a common set pressure (default 100 bar). The oxidizer tank is the hard problem: N2O is subcooled at tank pressure but continuously evaporates into the growing ullage as liquid drains, coupling pressurant demand to liquid cooling. The fuel tank is modeled as a single-phase liquid with negligible vapor pressure.

The main script performs four steps:

1. **Tank geometry sizing** - Fixed inner diameter (300 mm), hemispherical domes; cylinder length from liquid load, worst-case (warmest) fill density, and minimum ullage fraction.
2. **Coupled pressurization** - Ethanol and N2O tank models are co-stepped against one shared pressurant bottle (adiabatic blowdown, isenthalpic regulator inlet).
3. **Pressurant bottle sizing** - Bisection on bottle volume using the recorded N2 demand profile; margin applied to the worst temperature case.
4. **Reporting** - Plotly HTML report and structured YAML results.

## Repository layout

`
press_analysis/
|-- flight_config_sizing.py   # Main entry point: sizing, coupling, bottle sizing, reports
|-- ethanol_press.py       # Single-phase liquid tank pressurization model (fuel)
|-- n2o_press.py           # Two-phase N2O tank pressurization model (oxidizer)
|-- mixing_crosscheck.py   # Standalone PR/Dalton ullage mixing cross-check (n2o_press)
|-- configs/
|   -- n2o_press.yaml     # Mission, tank, operating, and model configuration
|-- results/               # Generated outputs (not source)
|   |-- h1b_press_report.html
|   -- h1b_press_results.yaml
-- README.md
`

ethanol_press.py and 
2o_press.py are intentionally independent modules (geometry helpers are duplicated by project convention). Each exposes init_state, step, and 
un_standalone for use outside the main analysis.

## Requirements

- Python 3.10+ (uses | union types in annotations)
- [CoolProp](https://coolprop.org/) - real-fluid thermodynamic properties
- PyYAML
- Plotly - HTML report generation
- NumPy - used by `mixing_crosscheck.py`

Install dependencies:

`ash
pip install CoolProp pyyaml plotly numpy
`

## Quick start

From the press_analysis directory:

`ash
python flight_config_sizing.py configs/n2o_press.yaml --outdir results
`

Arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| config | configs/n2o_press.yaml | Path to YAML configuration |
| --outdir | . | Output directory for report and results YAML |

A full run takes on the order of **2-3 minutes** on a typical laptop (CoolProp property lookups dominate runtime).

### Outputs

- **h1b_press_report.html** - Summary tables plus per-case Plotly charts (temperatures, N2 flow, N2O partial pressures, bottle state).
- **h1b_press_results.yaml** - Tank sizing, propellant loads, per-case pressurant statistics, and bottle sizing (required volume, margin, loaded mass).

Open the HTML report in a browser. The YAML file is suitable for downstream sizing spreadsheets or configuration management.

## Configuration

All mission and model inputs live in a single YAML file. See configs/n2o_press.yaml for the reference H-1B case.

| Section | Purpose |
|---------|---------|
| mission | Burn time, total mass flow, O/F ratio |
| propellants | CoolProp fluid names for oxidizer (N2O) and fuel (ethanol) |
| 	anks | Fixed diameter, ullage rules, wall thermal properties, oxidizer evaporation makeup toggle |
| operating | Regulator set pressure, propellant temperature envelope, ambient temperature |
| pressurant | N2 bottle initial state, regulator minimum inlet pressure, blowdown mode, volume margin |
| model | Time step, fixed-point iterations (N2O), evaporation model, heat-transfer mode |
| cases | Named analysis cases with propellant initial temperature |

Key defaults for the reference configuration:

- 40 s burn at 1.0 kg/s total flow, O/F = 4.0
- 100 bar tank pressure, 300 bar bottle at 20 C
- 120 bar minimum bottle pressure for regulation
- 1.30x margin on required bottle volume
- Cold (5 C) and hot (25 C) propellant cases

### Oxidizer evaporation makeup

When 	anks.oxidizer_evap_makeup is 	rue, the analysis iterates oxidizer load: delivered mass plus an allowance for N2O vapor lost to ullage during the burn (not available to the engine). This increases tank volume and liquid load until the allowance converges.

## Physical models

### Ethanol tank (ethanol_press.py)

Single-phase liquid expulsion at constant tank pressure. Control volumes: ullage gas (real-gas N2), lumped dry wall, isothermal liquid. Ullage energy balance is an open system at constant pressure; natural-convection correlations (or fixed coefficients) couple gas, liquid, and wall nodes.

### N2O tank (
2o_press.py)

Two-phase, two-node model with equilibrium evaporation at the liquid surface:

- Ullage: Dalton mixing of N2 and N2O vapor; P_vap = Psat(T_liq), P_N2 = P_set - P_vap
- Liquid: energy balance with evaporative cooling and ambient/wetted-wall heat transfer
- Per-step fixed-point iteration couples ullage temperature, liquid temperature, and N2 demand

Documented limitations (conservative for sizing): no N2 dissolution (supercharging), instantaneous interface equilibrium, bulk-mixed liquid, N2 transport properties used for ullage convection when N2O properties are unavailable in CoolProp. See **mixing cross-check** below for quantifying Dalton vs mixture-EOS error on ullage density and dissolved N2.

### Shared bottle (flight_config_sizing.py)

Rigid vessel with adiabatic (or isothermal) discharge. Pre-pressurization draw is isothermal; burn-time draw uses d(m u)/dt = -mdot * h_bottle. Bottle volume is sized by bisection so end pressure stays above the regulator minimum inlet pressure for the full demand profile.

## Using the tank models standalone

Both modules can be run independently with constant inlet enthalpy (e.g. for unit checks or single-tank studies):

`python
import ethanol_press as ep

cfg = ep.LiquidTankConfig(...)  # see dataclass fields in ethanol_press.py
result = ep.run_standalone(cfg, t_burn=40.0, dt=0.05)
# result keys: prepress_mass_kg, expulsion_mass_kg, total_mass_kg, timeseries, final_state
`

`python
import n2o_press as np2

cfg = np2.N2OTankConfig(...)
result = np2.run_standalone(cfg, t_burn=40.0, dt=0.05)
# additionally: evaporated_n2o_kg
`

For coupled simulation with a shared bottle and geometry sizing, use flight_config_sizing.py.

## Mixing cross-check (`mixing_crosscheck.py`)

Standalone sanity check for the **Dalton (additive partial-pressure)** ullage rule in `n2o_press.py`. At fixed tank set pressure and liquid temperature, it compares:

- **(a)** Dalton with CoolProp pure-fluid densities (what the transient model uses)
- **(b)** Dalton with Peng–Robinson pure-fluid densities
- **(c)** PR mixture at the real-gas Dalton composition (mixing correction only)
- **(d)** PR full vapor–liquid equilibrium (mixing + dissolved N2 in the liquid)

`k12` for N2–N2O is swept over a plausible range. Console output only (no config file).

From the `press_analysis` directory:

```bash
python mixing_crosscheck.py --p-bar 100 --t-c 0 10 20 30
```

Optional: `--k12 -0.02 0.0 0.03 0.06` to override interaction-parameter sweep.

## Reference results (default config)

With configs/n2o_press.yaml, typical outputs are:

| Item | Value |
|------|-------|
| N2O tank | L_cyl ~ 565 mm, V ~ 54 L, 36 kg loaded (32 kg delivered + ~4 kg evap allowance) |
| Ethanol tank | Spherical (L_cyl = 0), V ~ 14 L, 8 kg fuel |
| Worst-case bottle | Cold case: ~61 L required -> **~79 L** with 1.30x margin (~24 kg N2 at 300 bar) |
| Peak N2 flow | ~158 g/s (cold), ~166 g/s (hot) near end of burn |

Exact numbers depend on config edits and CoolProp version; re-run the analysis after changing inputs.

## Development notes

- **SI units** throughout (pressures in Pa internally; config uses bar where noted).
- **Stateless solvers** - state is passed explicitly to step(); no global simulation state.
- **Caching** - CoolProp calls are LRU-cached on rounded (P, T) grids for performance.
- **Conservative assumptions** - isothermal pre-press charge, equilibrium N2O evaporation, adiabatic bottle blowdown, and configurable margins are intended to bound sizing risk, not predict flight telemetry exactly.

When changing mission parameters, start with configs/n2o_press.yaml, run the main script, and review both the console summary and h1b_press_report.html for temperature and flow transients before adopting new tank or bottle sizes.
