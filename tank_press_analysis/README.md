# Tank sizing and pressurization analysis

Physics-based sizing and transient simulation of the propellant tanks and the shared nitrogen pressurant system. The analysis sizes capsule tank geometry (ethanol fuel + N2O oxidizer), runs coupled pressurization transients at several propellant temperatures, and sizes the pressurant bottle against the regulator inlet requirement.

The tool was written for H-1B and is now also used for the H2 lander (`configs/h2_tank_config_mission1.yaml`).

## Overview

The vehicle uses **externally pressurized blowdown** with a single N2 bottle feeding both tanks through regulators held at a common set pressure (default 100 bar). The oxidizer tank is the hard problem: N2O is subcooled at tank pressure but continuously evaporates into the growing ullage as liquid drains, coupling pressurant demand to liquid cooling. The fuel tank is modeled as a single-phase liquid with negligible vapor pressure.

The main script performs four steps:

1. **Tank geometry sizing** - fixed inner diameter (300 mm), hemispherical domes; cylinder length from liquid load, worst-case (warmest) fill density, and minimum ullage fraction.
2. **Coupled pressurization** - ethanol and N2O tank models are co-stepped against one shared pressurant bottle (adiabatic blowdown, isenthalpic regulator inlet).
3. **Pressurant bottle sizing** - bisection on bottle volume using the recorded N2 demand profile; margin applied to the worst temperature case.
4. **Reporting** - Plotly HTML report and structured YAML results.

## Layout

```
tank_press_analysis/
|-- flight_config_sizing.py   # entry point: sizing, coupling, bottle sizing, reports
|-- ethanol_press.py          # single-phase liquid tank pressurization model (fuel)
|-- n2o_press.py              # two-phase N2O tank pressurization model (oxidizer)
|-- mixing_crosscheck.py      # standalone PR/Dalton ullage mixing cross-check (n2o_press)
|-- configs/
|   |-- h2_tank_config_mission1.yaml   # H2 lander, hover mission (14.5 s, 1.1 kg/s) - default
|   `-- n2o_press.yaml                 # H-1B reference case (40 s, 1.0 kg/s)
|-- results/                  # generated, not in git
`-- README.md
```

`ethanol_press.py` and `n2o_press.py` are intentionally independent modules (geometry helpers are duplicated by project convention). Each exposes `init_state`, `step`, and `run_standalone` for use outside the main analysis.

## Requirements

- Python 3.10+ (uses `|` union types in annotations)
- [CoolProp](https://coolprop.org/) - real-fluid thermodynamic properties
- PyYAML
- Plotly - HTML report generation
- NumPy - used by `mixing_crosscheck.py`

```bash
pip install CoolProp pyyaml plotly numpy
```

## Quick start

From this directory:

```bash
python flight_config_sizing.py                                  # H2 config, writes to results/
python flight_config_sizing.py configs/n2o_press.yaml           # H-1B reference case
python flight_config_sizing.py <config.yaml> --outdir <folder>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `config` | `configs/h2_tank_config_mission1.yaml` | YAML configuration |
| `--outdir` | `results/` next to the script | output folder |

The H2 case takes about 40 s, the 40 s H-1B case 1-2 minutes (CoolProp lookups dominate).

### Outputs

Both files are named after the config file, so runs of different configs do not overwrite each other:

- **`<config>_report.html`** - summary tables plus per-case Plotly charts (temperatures, N2 flow, N2O partial pressures, bottle state).
- **`<config>_results.yaml`** - config name, date and convergence flag, tank sizing, propellant loads, per-case pressurant statistics, and bottle sizing (required volume, margin, loaded mass).

If the outer load / bottle iteration does not converge in 5 passes, the console prints a WARNING and `meta.converged` is `false`. Set `model.ox_load_guess_kg` close to the converged oxidizer load and re-run.

## Configuration

All mission and model inputs for one case live in a single YAML file.

| Section | Purpose |
|---------|---------|
| `mission` | Burn time, total mass flow, O/F ratio |
| `propellants` | CoolProp fluid names for oxidizer (N2O) and fuel (ethanol) |
| `tanks` | Fixed diameter, ullage rules, residuals, wall thermal properties, oxidizer evaporation makeup toggle |
| `operating` | Regulator set pressure, propellant temperature envelope, ambient temperature |
| `pressurant` | N2 bottle initial state, regulator minimum inlet pressure, blowdown mode, volume margin |
| `model` | Time step, warm starts, fixed-point iterations (N2O), evaporation model, heat-transfer mode |
| `cases` | Named analysis cases with propellant initial temperature |

`propellants.*.name`, `tanks.dome` and `tanks.wall.material` are informational and not read by the code. `operating.temp_min_C` / `temp_max_C` must be kept consistent with the `cases` by hand.

### Oxidizer evaporation makeup

When `tanks.oxidizer_evap_makeup` is `true`, the analysis iterates the oxidizer load: delivered mass plus an allowance for N2O vapor lost to ullage during the burn (not available to the engine). This increases tank volume and liquid load until the allowance converges.

## Physical models

### Ethanol tank (`ethanol_press.py`)

Single-phase liquid expulsion at constant tank pressure. Control volumes: ullage gas (real-gas N2), lumped dry wall, isothermal liquid. Ullage energy balance is an open system at constant pressure; natural-convection correlations (or fixed coefficients) couple gas, liquid, and wall nodes.

### N2O tank (`n2o_press.py`)

Two-phase, two-node model with equilibrium evaporation at the liquid surface:

- Ullage: Dalton mixing of N2 and N2O vapor; P_vap = Psat(T_liq), P_N2 = P_set - P_vap
- Liquid: energy balance with evaporative cooling and ambient/wetted-wall heat transfer
- Per-step fixed-point iteration couples ullage temperature, liquid temperature, and N2 demand

Documented limitations (conservative for sizing): no N2 dissolution (supercharging) beyond the fixed allowance, instantaneous interface equilibrium, bulk-mixed liquid, N2 transport properties used for ullage convection. See the mixing cross-check below for the Dalton vs mixture-EOS error on ullage density and dissolved N2.

### Shared bottle (`flight_config_sizing.py`)

Rigid vessel with adiabatic (or isothermal) discharge. Pre-pressurization draw is isothermal; burn-time draw uses d(m u)/dt = -mdot * h_bottle. Bottle volume is sized by bisection so the end pressure stays above the regulator minimum inlet pressure for the full demand profile.

## Known limits

- The tool always sizes **one** 300 mm capsule per propellant. It cannot evaluate a fixed tank volume or several tanks (e.g. the present H2 hardware, 2 x 18 L N2O + 6.9 L ethanol).
- It uses a fixed O/F and total flow, not the engine throttle line or the mission propellant budget. The handover from `mission_analysis/mission_results.yaml` is manual.
- **The reported peak N2 flow can be a numerical start-up transient.** In the first few steps the N2O tank step oscillates (and negative steps are clipped), so the peak at t ≈ 0.1-0.2 s is not physical. Read the steady flow from the report plots; do not size the regulator on the reported peak.
- Tank pressure, regulator minimum inlet and bottle pressure in the H2 config are the H-1B values, not derived from the H2 feed-pressure budget.

## Using the tank models standalone

Both modules can be run independently with constant inlet enthalpy (e.g. for unit checks or single-tank studies):

```python
import ethanol_press as ep

cfg = ep.LiquidTankConfig(...)  # see dataclass fields in ethanol_press.py
result = ep.run_standalone(cfg, t_burn=40.0, dt=0.05)
# result keys: prepress_mass_kg, expulsion_mass_kg, total_mass_kg, timeseries, final_state
```

```python
import n2o_press as np2

cfg = np2.N2OTankConfig(...)
result = np2.run_standalone(cfg, t_burn=40.0, dt=0.05)
# additionally: evaporated_n2o_kg
```

For coupled simulation with a shared bottle and geometry sizing, use `flight_config_sizing.py`.

## Mixing cross-check (`mixing_crosscheck.py`)

Standalone sanity check for the **Dalton (additive partial-pressure)** ullage rule in `n2o_press.py`. At fixed tank set pressure and liquid temperature, it compares:

- **(a)** Dalton with CoolProp pure-fluid densities (what the transient model uses)
- **(b)** Dalton with Peng–Robinson pure-fluid densities
- **(c)** PR mixture at the real-gas Dalton composition (mixing correction only)
- **(d)** PR full vapor–liquid equilibrium (mixing + dissolved N2 in the liquid)

`k12` for N2–N2O is swept over a plausible range. Console output only (no config file). It is not part of the main run.

```bash
python mixing_crosscheck.py --p-bar 100 --t-c 0 10 20 30
```

Optional: `--k12 -0.02 0.0 0.03 0.06` to override the interaction-parameter sweep.

## Reference results

Re-run after changing inputs; exact numbers depend on the CoolProp version.

| Item | H2, `h2_tank_config_mission1.yaml` | H-1B, `n2o_press.yaml` |
|------|------|------|
| N2O tank | L_cyl 125 mm, 23.0 L, 15.7 kg loaded (12.8 delivered + 0.4 residual + 2.5 vapor) | L_cyl 600 mm, 56.6 L, 38.7 kg loaded (32.0 + 0.4 + 6.3) |
| Ethanol tank | sphere, 14.1 L (70 % ullage), 3.3 kg | sphere, 14.1 L, 8.1 kg |
| Bottle, worst case (cold) | 32.0 L required -> 41.6 L with 1.30 margin, 12.6 kg N2 at 300 bar | 65.7 L -> 85.4 L, 25.8 kg N2 |

## Development notes

- **SI units** throughout (pressures in Pa internally; the config uses bar and °C where the key name says so).
- **Stateless solvers** - state is passed explicitly to `step()`; no global simulation state.
- **Caching** - CoolProp calls are LRU-cached on rounded (P, T) grids for performance.
- **Conservative assumptions** - isothermal pre-press charge, equilibrium N2O evaporation, adiabatic bottle blowdown, and configurable margins are intended to bound sizing risk, not predict flight telemetry exactly.
