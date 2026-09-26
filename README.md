# e2-propulsion-tools

Analysis and sizing tools for the Hopper E2 / H2 propulsion system: thrust-chamber performance, the torch igniter, liquid-centred swirl injectors, orifices, and propellant-tank pressurization. The tools are for preliminary analysis and for quick checks during a test campaign.

Each folder is independent. Run the commands from that folder.

## Tools

| Path | What it does | How to run |
|------|----------------|------------|
| [hop_cea/](hop_cea/) | SI RocketCEA library for the N₂O / ethanol thrust chamber. Real-fluid reactant enthalpies, throttle line with fixed fuel flow, chamber contour, Bartz heat flux, N₂O coolant properties. | Import `src` (no CLI) |
| [torch_igniter_analysis/](torch_igniter_analysis/) | Torch igniter chamber sizing (CEA) and Bartz wall heat flux / energy budget. | `python igniter_combustion_analysis.py --config configs/igniter_example.yaml` and `python igniter_thermal_analysis.py --config configs/igniter_example.yaml` |
| [swirl_injector_analysis/](swirl_injector_analysis/) | Liquid-centred swirl coaxial element: Bazarov or Nardi swirl sizing, oxidizer annulus, HEM metering holes, throttle envelope. Streamlit app for design sweeps and test-point calibration. | `python lcsc_sizing.py configs/e2_lcsc_p04_6x_design.yaml` or `streamlit run app.py` |
| [orifice_lib/](orifice_lib/) | Single-orifice mass flow: incompressible liquid, real gas, and flashing flow (SPI / HEM / NHNE). | `python flight_config_sizing.py configs/orifice_cases.yaml --outdir results` |
| [tank_press_analysis/](tank_press_analysis/) | H-1B ethanol and N₂O tank sizing, coupled N₂ blowdown, pressurant bottle sizing. | `python flight_config_sizing.py configs/n2o_press.yaml --outdir results` |

`press_analysis/` is the earlier copy of the tank analysis (`H1B_Analysis_Main.py` is only there). `injector_sizing/` holds the same LCSC scripts as `swirl_injector_analysis/`, plus the old injector template.

## Requirements

Python 3.10+ is what the newer modules are written for. The LCSC app also runs on 3.9.

| Tool | Packages |
|------|----------|
| hop_cea, torch igniter | `rocketcea`, `CoolProp`, `numpy`, `scipy`, `pyyaml`. Igniter plots need `plotly`. A static image also needs `kaleido`; HTML is written either way. |
| swirl injector | `CoolProp`, `numpy`, `scipy`, `pyyaml`. The app also needs `streamlit` and `pandas`. |
| orifice_lib | `CoolProp`, `numpy`, `scipy`, `pyyaml`, `plotly` |
| tank pressurization | `CoolProp`, `pyyaml`, `plotly`, `numpy` |

## hop_cea

Library only. Inputs for the H2 lander chamber live in `hop_cea/src/config.py` (thrust, chamber pressure, area ratio, fixed fuel flow, efficiencies, E2-REG-1 contour). Propellant cards in `propellants.py` use the compressed-liquid N₂O enthalpy rather than RocketCEA's ideal-gas N₂O card.

From `hop_cea/`:

```python
from src.cea_si import point
from src.operating_line import state
from src.contour import e2_contour
from src.bartz import bartz_profile

p = point(pc=25.0, mr=4.0, eps=4.0)   # bar, O/F, area ratio
```

`point` returns chamber, throat, and exit state in SI units. `state(mox, At)` is one point on the throttle line with fuel flow fixed. `n2o.py` has saturation properties and a first-cut coolant check.

## Torch igniter

Both scripts have the geometry and the operating point at the top of the file (throat diameter, mass flow, O/F, c* efficiency). They print a report and write HTML next to the script.

- `igniter_combustion_analysis.py` — N₂O / ethanol CEA at a fixed throat. Writes `cea_combustion_analysis.html` and `chamber_contour.html`.
- `igniter_thermal_analysis.py` — chamber pressure from the throat and mass flow, then Bartz heat flux and an energy budget. Writes `torch_thermal.html`.

Referenced from IGN-DOC-001.

## Swirl injector

See [swirl_injector_analysis/readme.md](swirl_injector_analysis/readme.md).

```bash
python lcsc_sizing.py configs/e2_lcsc_p04_6x_design.yaml --out results
python lcsc_sizing.py configs/e2_lcsc_p03_3x_as_built.yaml --out results
python lcsc_sizing.py --check
streamlit run app.py
```

The 6-element file is a new design (Bazarov). The 3-element file analyses the P03 hardware through its `as_built` block. `python lcsc_sizing.py --check` runs the built-in comparisons with the thesis geometry.

## Orifice library

See [orifice_lib/README.md](orifice_lib/README.md). Three cases are in `configs/orifice_cases.yaml`: ethanol liquid, nitrogen pressurant, and a subcooled N₂O injector orifice. `--case <name>` runs one of them.

## Tank pressurization

See [tank_press_analysis/README.md](tank_press_analysis/README.md). A full run is a few minutes; CoolProp dominates. Outputs are `results/h1b_press_report.html` and `results/h1b_press_results.yaml`. `mixing_crosscheck.py` is a standalone check of the ullage mixing model and is not part of that run.
