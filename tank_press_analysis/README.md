# Tank sizing and pressurization analysis

Physics-based sizing and transient simulation of the propellant tanks and the shared nitrogen pressurant system. The analysis either sizes new capsule tanks or takes existing tank hardware (ethanol fuel + N2O oxidizer), runs coupled pressurization transients at several propellant temperatures, and sizes the pressurant bottle against the regulator inlet requirement.

The tool was written for H-1B and is now also used for the H2 lander (`configs/h2_tank_config_mission1.yaml`).

## Overview

The vehicle uses **externally pressurized blowdown** with a single N2 bottle feeding both tanks through regulators held at a common set pressure (default 100 bar). The oxidizer tank is the hard problem: N2O is subcooled at tank pressure but continuously evaporates into the growing ullage as liquid drains, coupling pressurant demand to liquid cooling. The fuel tank is modeled as a single-phase liquid with negligible vapor pressure.

The main script performs four steps:

1. **Tank geometry** - either sized (one capsule per propellant, fixed inner diameter, hemispherical domes; cylinder length from liquid load, worst-case (warmest) fill density, and minimum ullage fraction) or fixed hardware (count, volume and diameter per propellant; see [Fixed tanks](#fixed-tanks)).
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
|   |-- h2_tank_config_mission1.yaml        # H2 lander, hover mission, present hardware - default
|   |-- h2_tank_config_mission1_sized.yaml  # same mission, minimum sized tanks (D 200 mm)
|   `-- n2o_press.yaml                      # H-1B reference case (40 s, 1.0 kg/s)
|-- results/                  # generated; only the H2 mission1 results are in git
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

The H2 case takes 1-2 minutes, the 40 s H-1B case about 2 minutes (CoolProp lookups dominate). The H2 config reads the mission tool's results, so run `mission_analysis/mission_sizing.py` first (see [Mission handover](#mission-handover)).

### Outputs

Both files are named after the config file, so runs of different configs do not overwrite each other:

- **`<config>_report.html`** - summary tables plus per-case Plotly charts (temperatures, N2 flow, N2O partial pressures, bottle state).
- **`<config>_results.yaml`** - config name, baseline, date and convergence flag, tank sizing, propellant loads, per-case pressurant statistics, bottle sizing (required volume, margin, loaded mass), and the `handback` section for the mission tool.

If the outer load / bottle iteration does not converge in `model.outer_iterations` passes (default 10), the console prints a WARNING and `meta.converged` is `false`. Set `model.ox_load_guess_kg` close to the converged oxidizer load, or raise `model.outer_iterations`, and re-run.

## Mission handover

The propellant to deliver can come from the mission tool instead of being typed in:

```yaml
mission:
  from_mission_results: ../../mission_analysis/mission_results.yaml
```

The tool then takes the usable N2O and ethanol of the mission's sizing case (the `handover` section) and delivers them at the mission's mean flow over the equivalent constant-flow duration. `from_mission_results` replaces `burn_time_s`, `mdot_total_kg_s` and `of_ratio`; use one or the other. The H2 config uses the handover; the H-1B config keeps its own values, and any config can do the same for a stand-alone study.

The results file returns a `handback` section: the loaded N2 (bottle charge including `margin_factor`), the N2O vapour make-up, and the tank volumes they were computed for. The mission tool reads it through `pressurization.from_tank_results` and scales both values to its own tank volumes.

Run order, from the repository root:

```bash
cd mission_analysis && python mission_sizing.py            # 1. handover (first time: manual pressurization values)
cd ../tank_press_analysis && python flight_config_sizing.py   # 2. tank sizing from the handover
cd ../mission_analysis && python mission_sizing.py         # 3. mission with the tank handback
```

The liftoff mass is fixed by the liftoff thrust-to-weight, so the pressurant mass changes the ballast but not the propellant: one pass is enough. The mission tool prints a WARNING if the tank results were computed for other propellant loads (for example after a mission input changed); then repeat steps 2 and 3.

## Configuration

All mission and model inputs for one case live in a single YAML file. A config with a
`baseline:` key (the H2 config) takes the tank pressure, propellant temperature window,
cold / hot cases, residuals and fluid names from the shared baseline
(`baseline/h2_baseline.yaml`); a value set in the config overrides the baseline and the
tool prints a NOTE. The H-1B reference config has no `baseline:` key and sets everything
itself.

| Section | Purpose |
|---------|---------|
| `mission` | Burn time, total mass flow, O/F ratio - or `from_mission_results` (see [Mission handover](#mission-handover)) |
| `propellants` | CoolProp fluid names for oxidizer (N2O) and fuel (ethanol) |
| `tanks` | Fixed hardware (`present_hardware`, or `oxidizer` / `fuel` blocks) or the sizing diameter and ullage rule; residuals, wall thermal properties, oxidizer evaporation makeup toggle |
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

## Fixed tanks

Instead of sizing new capsules, the tool can simulate existing tanks. Per propellant:

```yaml
tanks:
  oxidizer: {count: 2, volume_each_L: 18.0, diameter_inner_m: 0.240}
  fuel:     {count: 1, volume_each_L: 6.9,  diameter_inner_m: 0.175}
```

Each tank is a vertical capsule (cylinder with hemispherical domes); the cylinder length follows from the volume and the diameter. Several identical tanks of one propellant are fed in parallel from the same regulator: one tank is simulated with 1/count of the load and flow, and its N2 draw is counted `count` times. A propellant without a block is sized as before, so fixed and sized tanks can be mixed.

With `tanks.present_hardware: true` in a config that names the baseline, both blocks come from the `tanks` section of `baseline/h2_baseline.yaml` (the present H2 hardware). The H2 config does this. Set it to `false` to size new tanks for the same mission.

The pressurant bottles work the same way: `pressurant.bottle_count` and `pressurant.bottle_volume_each_L` (from `tanks.pressurant` of the baseline with `present_hardware: true`) fix the bottle volume. The bottles are treated as one vessel of the total volume. The cases are then simulated with that volume, the required volume is still computed, and the results report the available margin (`bottle.margin_available` = available / required volume), the end pressure, and the N2 mass loaded in the fixed bottles, which is also the handback. The tool warns if the bottles are smaller than required or below `margin_factor`. Without these keys the bottle is sized as before.

The tool stops if the load does not fit the tanks at hot fill, and warns if the hot-fill ullage is below `ullage_min_frac`. The oxidizer vapour make-up is iterated as for sized tanks, so a partly filled tank gets a larger vapour allowance.

## Known limits

- Tanks are capsules with hemispherical domes; other shapes (ellipsoidal domes, spheres of a given size) are approximated by a capsule of the same volume and diameter.
- Parallel tanks are assumed identical and equally loaded.
- The present-tank diameters in the baseline are estimates (capsule with length = 2 x diameter) until they are replaced with the drawings.
- It drains both tanks at constant flow (the mission's mean flow when it uses the handover), not along the mission's throttle profile.
- With sized tanks, the handback is for those tanks and the mission tool scales it to its present tanks by volume (first order). With `present_hardware: true` no scaling is needed.
- **The reported peak N2 flow can be a numerical start-up transient.** In the first few steps the N2O tank step oscillates (and negative steps are clipped), so the peak at t ≈ 0.1-0.2 s is not physical. Read the steady flow from the report plots; do not size the regulator on the reported peak.
- The regulator minimum inlet and bottle pressure in the H2 config are the H-1B values, not derived from the H2 feed-pressure budget. The 100 bar tank pressure comes from the baseline.

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
| N2O tank | present 2 x 12 L (D 209 mm est.), 24.0 L, 15.2 kg loaded (11.6 delivered + 0.8 residual + 2.8 vapor), hot-fill ullage 20 % | sized, L_cyl 600 mm, 56.6 L, 38.7 kg loaded (32.0 + 0.4 + 6.3) |
| Ethanol tank | present 1 x 6 L (D 166 mm est.), 3.6 kg, hot-fill ullage 24 % | sized, sphere, 14.1 L, 8.1 kg |
| Bottle, worst case (cold) | 25.6 L required; present 2 x 15 L = 30.0 L, margin 1.17 (below the 1.30 target), 9.1 kg N2 at 300 bar, 140 bar at the end | 65.7 L -> 85.4 L, 25.8 kg N2 |

The H2 column is for the hover handover of the mission tool (11.6 kg N2O and 3.4 kg ethanol usable, delivered over 16.9 s) in the present hardware (baseline revision C). Sized for the same mission instead (`h2_tank_config_mission1_sized.yaml`, D 200 mm), the tanks would be 21.6 L N2O + 5.1 L ethanol with a 31.9 L bottle (9.6 kg N2) and 2.4 kg vapour.

## Development notes

- **SI units** throughout (pressures in Pa internally; the config uses bar and °C where the key name says so).
- **Stateless solvers** - state is passed explicitly to `step()`; no global simulation state.
- **Caching** - CoolProp calls are LRU-cached on rounded (P, T) grids for performance.
- **Conservative assumptions** - isothermal pre-press charge, equilibrium N2O evaporation, adiabatic bottle blowdown, and configurable margins are intended to bound sizing risk, not predict flight telemetry exactly.
