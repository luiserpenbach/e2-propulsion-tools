# Feed system pressure budget

Pressure budget of the H2 vehicle fluid system at full thrust, for P&ID H2 / CFG-01 V1.2: from the main regulator through the nitrogen lines and standpipes into the propellant tanks, and from the tanks through lines, filters and valves to the cavitating venturis. It answers how much pressure is left at the venturi inlets, what flow they deliver at each N₂O temperature, how much choking margin the oxidizer venturi keeps, which flow coefficients the components need, and what the cheaper hardware changes would gain.

Documentation: Notion, Vehicle Fluid System (H2) / Pressurant and Feed System Pressure Drop Analysis (H2-FLD-ANA-003) and Feed Line Sizing (H2-FLD-ANA-002).

## Run

```bash
pip install CoolProp pyyaml
python feed_pressure_budget.py                                # configs/h2_feed_v12.yaml
python feed_pressure_budget.py configs/<case>.yaml --outdir results
python -m pytest tests                                        # reproduces the Rev B tables
```

The report goes to the console and `results/<config>_results.yaml` (not in git).

## Inputs

`configs/h2_feed_v12.yaml` names the shared baseline, which supplies the rated N₂O and fuel flow (100 % row of `baseline/h2_baseline_engine_table.yaml`), the regulator set pressure (`feed.tank_pressure_bar`) and the ambient pressure. Everything else is in the config:

| Section | Content |
|---------|---------|
| `nitrogen` | Flow to the oxidizer and fuel tanks, regulated state, storage state for the high-pressure line check |
| `regulator` | Drop-out margin, bottle sizing limit (120 bar), expected end-of-burn storage pressure |
| `lines` | Per line: tube OD and wall, length, sum of fitting K, parallel count |
| `n2o_liquid`, `fuel_liquid` | Liquid state for the line losses; N₂O viscosity is an input (CoolProp has none) |
| `allowances_bar` | Regulator droop and component losses until datasheet values exist |
| `venturis` | Throat, Cd; oxidizer pressure recovery limit and the outlet pressure the engine needs |
| `options` | Hardware changes on the oxidizer side, applied cumulatively |

The nitrogen flow is not computed here. It comes from the tank tool (`tank_press_analysis`): the steady end-of-burn flow at the mission's mean flow, scaled to the full-thrust propellant volume flow. Update `nitrogen` when the tank sizing changes.

## Method

- **Lines.** Darcy–Weisbach with the Swamee–Jain friction factor (roughness 1.5 µm, drawn stainless tube), plus K times the dynamic pressure for the fittings. Nitrogen at the regulated state (96 bar, 0 °C), the high-pressure line at the storage state.
- **Components.** Allowances, except the throttle valve and option valves, whose loss follows from the liquid Cv equation.
- **Venturis.** `mdot = Cd A sqrt(2 rho (p_in - p_v))` with the density of the compressed liquid at the inlet and the saturation pressure at the liquid temperature. Choking margin = recovery limit × inlet pressure − outlet pressure needed.
- **Required Cv.** Liquid Cv equation at each allowance; IEC 60534-2-1 for the gas valves and for the wide-open regulator at the sizing limit and at the expected end of burn.

`tests/` checks that the method reproduces every value of the Rev B tables (run 2026-10-21 with the nitrogen flow of Tank Sizing Rev C) to within 0.1 bar.

## Known limits

- Fitting counts and line lengths are typical values until the routing exists in CAD; component losses are allowances. The cold-flow test (HAZOP A-32) replaces both.
- Full thrust only; the 50 % throttle point is not covered yet.
- The interim RSH4 restrictor (PS-OR-12) is not modelled; Rev C scales its Rev B loss with the square of the flow.
- The regulator droop is a fixed allowance, independent of flow and inlet pressure.
