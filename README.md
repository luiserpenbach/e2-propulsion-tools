# e2-propulsion-tools

Analysis and sizing tools for the E2 engine and the H2 lander propulsion system (N₂O / ethanol, N₂ pressure-fed): mission and propellant budget, thrust chamber and regenerative cooling, torch igniter, liquid-centred swirl injector, orifices, and tank pressurization. The tools are for preliminary analysis and for quick checks during a test campaign.

Each folder is an independent tool with its own README. Run the commands from that folder.

## Tools

| Folder | What it does | How to run |
|--------|--------------|------------|
| [mission_analysis/](mission_analysis/) | H2 hover and hop missions: throttle profile, propellant budget, mass point, tank check. | `python mission_sizing.py --max-apex --plot` |
| [h2cea/](h2cea/) | Thrust chamber library (SI RocketCEA with real-fluid N₂O card, throttle line with fixed fuel flow, contour, Bartz) and the E2-REG-1 regenerative cooling analysis. | `python run_regen.py`, `python run_boiling.py`, `python -m pytest tests` |
| [torch_igniter_analysis/](torch_igniter_analysis/) | Torch igniter chamber sizing (CEA) and Bartz wall heat flux / energy budget. | `python igniter_combustion_analysis.py --config configs/igniter_example.yaml`, `python igniter_thermal_analysis.py --config configs/igniter_example.yaml` |
| [swirl_injector_analysis/](swirl_injector_analysis/) | Liquid-centred swirl coaxial element: Bazarov or Nardi swirl sizing, oxidizer annulus, HEM metering holes, throttle envelope. Streamlit app for design sweeps and test-point calibration. | `python lcsc_sizing.py configs/e2_lcsc_p04_6x_design.yaml`, `python lcsc_sizing.py --check`, `streamlit run app.py` |
| [orifice_lib/](orifice_lib/) | Single-orifice mass flow: incompressible liquid, real gas, and flashing flow (SPI / HEM / NHNE). | `python flight_config_sizing.py configs/orifice_cases.yaml` |
| [tank_press_analysis/](tank_press_analysis/) | Capsule tank sizing, coupled N₂O / ethanol pressurization transient, pressurant bottle sizing. | `python flight_config_sizing.py configs/h2_tank_config_mission1.yaml` |

## Inputs and outputs

- **Inputs** are YAML files in each tool's `configs/` folder (`h2_mission_inputs.yaml` for the mission tool). `h2cea` keeps its baseline in `h2cea/h2cea/config.py`.
- **Outputs** go to `<tool>/results/` (mission tool: next to its inputs file) and are not in git. Output files are named after the config file or the `name` in it, so different cases do not overwrite each other. Re-run the tool to regenerate them.
- `h2cea/out/` is an older committed snapshot of the cooling results; re-run before quoting numbers.
- For a design document, archive the results together with the config file and the git commit.

### The design point is not shared yet

Each tool still has its own copy of the engine operating point, and the copies differ:

| Where | Total flow at 100 % | O/F at 100 % |
|-------|--------------------|--------------|
| `h2cea/h2cea/config.py` (2250 N, 25 bar, 0.20 kg/s fuel) | 1.074 kg/s | 4.37 |
| `mission_analysis` (`engine_performance.py`, gaseous N₂O card) | 1.056 kg/s | 4.28 |
| `swirl_injector_analysis/configs/*.yaml` | 1.0 kg/s | 4.0 |
| `tank_press_analysis/configs/h2_tank_config_mission1.yaml` | 1.1 kg/s | 4.0 |

When you change the engine, update every tool's config, and check the numbers against `h2cea`, which is the reference engine model.

## Data flow

```
mission_analysis/h2_mission_inputs.yaml
        │
        ├─> engine_performance.py ─> engine_performance.yaml ─┐
        │                                                     v
        └──────────────────────────────────────────> mission_sizing.py ─> mission_results.yaml
                                                                              │ handover (manual)
                                                                              v
                                   tank_press_analysis/configs/*.yaml ─> flight_config_sizing.py
                                                                              │ pressurant mass,
                                                                              v vapour make-up (manual)
                                                               back into h2_mission_inputs.yaml
```

## Requirements

Python 3.10+.

| Tool | Packages |
|------|----------|
| mission_analysis | `rocketcea`, `numpy`, `pyyaml`, `matplotlib` |
| h2cea | `rocketcea`, `CoolProp`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pytest` |
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
