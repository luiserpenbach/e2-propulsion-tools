# Torch igniter analysis

N₂O / ethanol torch igniter: chamber sizing with CEA, Bartz wall heat flux, energy budget and heat soak. Referenced from IGN-DOC-001.

| File | Role |
|------|------|
| `app.py` | Streamlit app: design exploration and test-point analysis |
| `igniter_combustion_analysis.py` | CEA at a fixed throat, O/F sweep, chamber contour (HTML report) |
| `igniter_thermal_analysis.py` | Bartz heat flux, energy budget, heat soak (HTML report) |
| `igniter_model.py` | Calculations used by the app, built on the two scripts' functions |
| `igniter_config.py` | Loads a case (YAML file or dict) and derives the chamber geometry |
| `configs/*.yaml` | Cases: `igniter_example.yaml`, `igniter_thermal_legacy.yaml` (earlier 50 g/s point) |
| `test_log.csv` | Test points saved from the app (created on first save; commit it) |

## App

```bash
pip install rocketcea CoolProp numpy scipy pyyaml plotly streamlit pandas
streamlit run app.py          # from this folder
```

Streamlit 1.50 or newer.

**Design tab**
- Pick a base config and change the operating point (total flow, nominal and other O/F points, c* efficiency, ambient pressure), the chamber and nozzle (throat and chamber diameter, L* or cylinder length, area ratio), the angles and the wall (initial temperature, thickness). Press **Update**.
- Headline numbers at the nominal O/F with the change against the base config: chamber pressure, flame temperature, ambient Isp, throat heat flux, total wall heat, total length.
- Contour (edited and base), cold-wall heat flux along the chamber for each O/F, a table of all operating points with the time for the throat wall to reach the material limit (stainless, copper, graphite), and an O/F sweep of chamber pressure and temperature.
- Warnings for a long chamber (L_cyl / Dc > 8) and for heat-soak times beyond the validity of the semi-infinite-wall estimate.
- **Design sweep:** vary one parameter (flow, throat or chamber diameter, c* efficiency, O/F, area ratio, L*) and plot chamber pressure, heat flux, wall heat, Isp or length.
- **Download this case as YAML** and save it in `configs/` to keep a design; the scripts and the app read that folder.

**Testing tab**
- Pick the hardware config (throat, propellants, design c* efficiency) and the throat diameter as measured. Tick *Chamber pressure is gauge* if it is.
- Enter steady-state test points in the table: ox and fuel flow in g/s, chamber pressure in bar, thrust in N (optional). Or open *Average a window of a time-series CSV*, pick the time, pressure and flow columns, drag the window over the steady part of the run, and add the mean as a test point.
- For each point the app computes the measured c* (pc·At/ṁ), the CEA ideal c* at the measured O/F and pressure, the measured c* efficiency, the model's chamber pressure at the measured flows with the design efficiency, and, with thrust, Cf and Isp.
- Plots: c* efficiency vs O/F against the design value, c* vs O/F against the CEA and model lines, measured vs model chamber pressure.
- **Save to test_log.csv** keeps the table; **Download results CSV** exports the analysis. Put the measured c* efficiency into `eta_cstar` of the config for the next design iteration.

## Scripts

```bash
python igniter_combustion_analysis.py --config configs/igniter_example.yaml
python igniter_thermal_analysis.py --config configs/igniter_example.yaml
```

Both print a report and write HTML to `results/`, prefixed with the case `name` (static images need `kaleido`).

## Config

Units are in the key names. Set exactly one of `lstar_m` or `lcyl_mm`; the convergent length follows from the cone angle.

## Model limits

- RocketCEA's built-in `N2O` card is ideal gas at 298 K, not the liquid card of `h2cea`.
- Chamber pressure is solved from the throat, the total flow and the c* efficiency; the nozzle thrust coefficient is ideal (Isp = eta_c* × CEA Isp).
- Heat flux is Bartz for a cold wall at the initial wall temperature; the heat soak is a semi-infinite wall with constant flux, valid only while the heat has not reached the outer surface (`t_slab_valid`).
- The test analysis assumes steady state and an absolute (or gauge-corrected) chamber pressure measured in the chamber.
