# e2-propulsion-tools

Analysis and sizing tools for the E2 engine and the H2 lander propulsion system (N₂O / ethanol, N₂ pressure-fed): mission and propellant budget, thrust chamber and regenerative cooling, torch igniter, liquid-centred swirl injector, orifices, and tank pressurization. The tools are for preliminary analysis and for quick checks during a test campaign.

Each folder is an independent tool with its own README. Run the commands from that folder.

## Shared baseline

[baseline/](baseline/) holds every value that more than one tool uses: engine design point, propellant states, feed pressures and residuals (`h2_baseline.yaml`), and the engine throttle table computed from it (`h2_baseline_engine_table.yaml`). Change those values there, not in the tool configs. See [baseline/README.md](baseline/README.md).

After changing the engine, propellant or tank pressure inputs, regenerate the table:

```bash
cd h2cea && python run_engine_table.py
```

## Tools

| Folder | What it does | How to run |
|--------|--------------|------------|
| [mission_analysis/](mission_analysis/) | H2 hover and hop missions: throttle profile, propellant budget, mass point, tank check. | `python mission_sizing.py --max-apex --plot` |
| [h2cea/](h2cea/) | Thrust chamber library (SI RocketCEA with real-fluid N₂O card, throttle line with fixed fuel flow, contour, Bartz), the engine table for the baseline, and the E2-REG-1 regenerative cooling analysis. | `python run_engine_table.py`, `python run_regen.py`, `python run_boiling.py`, `python -m pytest tests` |
| [torch_igniter_analysis/](torch_igniter_analysis/) | Torch igniter chamber sizing (CEA) and Bartz wall heat flux / energy budget. | `python igniter_combustion_analysis.py --config configs/igniter_example.yaml`, `python igniter_thermal_analysis.py --config configs/igniter_example.yaml` |
| [swirl_injector_analysis/](swirl_injector_analysis/) | Liquid-centred swirl coaxial element: Bazarov or Nardi swirl sizing, oxidizer annulus, HEM metering holes, throttle envelope. Streamlit app for design sweeps and test-point calibration. | `python lcsc_sizing.py configs/e2_lcsc_p04_6x_design.yaml`, `python lcsc_sizing.py --check`, `streamlit run app.py` |
| [orifice_lib/](orifice_lib/) | Single-orifice mass flow: incompressible liquid, real gas, and flashing flow (SPI / HEM / NHNE). | `python flight_config_sizing.py configs/orifice_cases.yaml` |
| [tank_press_analysis/](tank_press_analysis/) | Capsule tank sizing, coupled N₂O / ethanol pressurization transient, pressurant bottle sizing. | `python flight_config_sizing.py configs/h2_tank_config_mission1.yaml` |

## Inputs and outputs

- **Shared inputs** are in `baseline/`. Tool configs name the baseline with a `baseline:` key and only hold tool-specific values. A config that sets a baseline value itself keeps it, and the tool prints a `NOTE`.
- **Tool inputs** are YAML files in each tool's `configs/` folder (`h2_mission_inputs.yaml` for the mission tool). `h2cea` keeps its hardware (contour, jacket) in `h2cea/h2cea/config.py`.
- **Outputs** go to `<tool>/results/` (mission tool: next to its inputs file) and are not in git. They are named after the config file or the `name` in it, and record the baseline revision. Re-run the tool to regenerate them.
- `h2cea/out/` is an older committed snapshot of the cooling results; re-run before quoting numbers.
- For a design document, archive the results together with the config file, the baseline revision and the git commit.

Using the baseline is opt-in for the tank and injector tools: only configs with a `baseline:` key read it (at present the H2 tank config). All other configs, including the injector P03 / P04 configs and the H-1B tank case, are independent of it. For what-if studies, copy the baseline to a new file instead of editing the shared one; see [baseline/README.md](baseline/README.md).

## Data flow

```
baseline/h2_baseline.yaml ──> h2cea/run_engine_table.py ──> baseline/h2_baseline_engine_table.yaml
      │                                                          │
      ├─> h2cea (config.py)                                      │
      ├─> swirl_injector_analysis (opt-in) <─────────────────────┤
      ├─> tank_press_analysis (opt-in)                           │
      └─> mission_analysis/mission_sizing.py <───────────────────┘
                 │
                 └─> mission_results.yaml ──(handover, manual)──> tank config `mission` block
                                                                        │
                           pressurant mass, vapour make-up (manual) <───┘
                           back into h2_mission_inputs.yaml
```

## Requirements

Python 3.10+.

| Tool | Packages |
|------|----------|
| mission_analysis | `numpy`, `pyyaml`, `matplotlib` (the engine table comes from h2cea) |
| h2cea | `rocketcea`, `CoolProp`, `numpy`, `scipy`, `pyyaml`, `pandas`, `matplotlib`, `pytest` |
| torch igniter | `rocketcea`, `CoolProp`, `numpy`, `scipy`, `pyyaml`, `plotly` (static images also need `kaleido`; HTML is written either way) |
| swirl injector | `CoolProp`, `numpy`, `scipy`, `pyyaml`; the app also needs `streamlit` and `pandas` |
| orifice_lib | `CoolProp`, `numpy`, `scipy`, `pyyaml`, `plotly` |
| tank pressurization | `CoolProp`, `pyyaml`, `plotly`, `numpy` |

```bash
pip install rocketcea CoolProp numpy scipy pyyaml pandas matplotlib plotly pytest
```

`rocketcea` builds from source on Linux and needs a Fortran compiler (`gfortran`).

## Torch igniter

Both scripts read a YAML case (`--config`, default `configs/igniter_example.yaml`); `configs/igniter_thermal_legacy.yaml` keeps the earlier 50 g/s point. They print a report and write HTML to `torch_igniter_analysis/results/`, prefixed with the case `name`.

- `igniter_combustion_analysis.py`: N₂O / ethanol CEA at a fixed throat. Writes `<name>_combustion.html` and `<name>_contour.html`.
- `igniter_thermal_analysis.py`: chamber pressure from the throat and mass flow, then Bartz heat flux, energy budget and heat soak. Writes `<name>_thermal.html`.

Both use RocketCEA's built-in ideal-gas `N2O` card, not the liquid card of `h2cea`. Referenced from IGN-DOC-001.
