# LCSC element sizing

`lcsc_sizing.py` sizes and analyses one liquid-centred swirl coaxial element (fuel swirl in the centre, gaseous or two-phase oxidizer through a coaxial annulus). Inputs are a YAML file; outputs are a Markdown report, a JSON file with every number, and a CSV of the operating envelope.

```
pip install CoolProp numpy scipy pyyaml
python lcsc_sizing.py configs/e2_lcsc_p04_6x_design.yaml     # new 6-element design
python lcsc_sizing.py configs/e2_lcsc_p03_3x_as_built.yaml   # analyse existing P03 hardware
python lcsc_sizing.py --check                                # validation cases
```

Results go to `results/` in this folder (or `-o <folder>`), named after the `name` field in the config. `results/` is not in git.

## Optional app

```
pip install streamlit pandas
streamlit run app.py        # run from this folder
```

- **Design**: pick a base config, change the main inputs, size the element and see geometry, operating envelope and warnings. The design-space section sweeps one parameter (spray angle, gas velocity, pressure drops, oxidizer temperature, element or inlet count) and plots geometry and J.
- **Testing**: pick a config with an `as_built` block, enter measured chamber pressure, flows, manifold pressures and temperatures. The app returns the measured swirl and hole flow coefficients, the suggested calibration values and the model prediction at the measured flows. Entries can be appended to `test_log.csv` in this folder.

Edits in the app are not written back to the YAML files; copy the suggested calibration block into the config.

## What it computes


| Step                    | Model                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fuel swirl element      | `bazarov` (default): Abramovich maximum-flow theory with Bazarov's viscous correction (wall friction via equivalent characteristic A_eq, inlet losses). Geometry is iterated until the predicted spray half-angle and flow coefficient match the targets. `nardi`: the vom Schemm thesis procedure used for P03/P04. Either way the final geometry is analysed with the viscous model. |
| Gas annulus             | Exit area from the chosen exit velocity; exit state at chamber pressure with h = h0 - v²/2.                                                                                                                                                                                                                                                                                            |
| Recess                  | Critical recess flow (recess angle = 2 × spray half-angle) or a fixed recess ratio.                                                                                                                                                                                                                                                                                                    |
| Oxidizer metering holes | HEM: isentropic homogeneous equilibrium expansion with real-fluid N₂O properties, including choking. Valid for gas, supercritical, liquid and two-phase inlet. `spi` (incompressible) for comparison.                                                                                                                                                                                  |
| Operating envelope      | For each throttle point: fuel Δp, oxidizer feed pressure, stiffness, film thickness (theory and three correlations), momentum flux ratio J, recess regime, warnings.                                                                                                                                                                                                                   |


## Config essentials

- `operating_point`: chamber pressure, total mass flow and O/F of the whole head; `elements` splits it.
- `baseline` (optional, no config uses it at present): path of the shared H2 baseline, e.g. `../../baseline/h2_baseline.yaml`. Leave `operating_point` out and it comes from the rated row of the engine table; each throttle point's `fraction` then becomes a fraction of rated thrust, with chamber pressure, O/F and total flow from the table (fuel flow fixed, oxidizer throttled, as the flight engine does). Use it to check a design against the current flight engine. Without the key the tool does not read the baseline at all.
- `oxidizer.temperature_K` or `oxidizer.quality` defines the state at the metering holes. Throttle points can override it with `ox_temperature_K` or `ox_quality` (e.g. heat-sink tests without preheating).
- `as_built`: any geometry value in mm replaces the sized value before the analysis. Use it to evaluate real hardware or CAD rounding.
- `calibration.swirl_cd_factor`: measured / predicted swirl flow coefficient. `calibration.spray_half_angle_measured_deg`: measured J = 0 half-angle, used for the recess and the recess regime.

## Calibration workflow

1. Flow-test elements (water for the fuel side, nitrogen for the gas side) or use hot-fire data: measured fuel flow and fuel Δp give the swirl C_D; oxidizer flow and manifold pressure give the hole C_D.
2. Put the ratio measured / predicted into `swirl_cd_factor` and the hole C_D into `ox_holes.discharge_coefficient`.
3. Put the measured spray half-angle into `spray_half_angle_measured_deg`.
4. Re-run the sizing for the next revision.

## Known limits

- The viscous model predicts C_D = 0.073 for the thesis LCSC geometry (water, 20 g/s); the thesis measured about 0.057 on SLA parts. It also predicts about 61° half-angle where 50–55° was measured. Calibrate before trusting absolute values.
- Film thickness is not validated; J is therefore reported as a range over four film models.
- Part-thrust chamber pressure defaults to fraction × nominal unless given per point.
- Without `baseline`, throttle points scale fuel and oxidizer flow together at the nominal O/F. The E2 flight engine holds the fuel flow fixed and throttles only the oxidizer, so part-thrust fuel Δp and stiffness differ from the engine's; add `baseline:` to see the engine's throttle line.
- Background and test data: Notion, E2 Engine › Injector Head › Master Thesis Summary – Coaxial Swirl Injectors (vom Schemm, 2024).

