# h2cea – thrust chamber reference analyses (Handbook H2-PRP-HBK-001 Rev B)

Thermochemistry, first-order sizing and the regenerative cooling analysis of the
E2 chamber. All baseline inputs live in `h2cea/config.py`.

    pip install rocketcea CoolProp numpy scipy matplotlib pandas
    python run_regen.py            # ~2 min, cooling analyses -> out/regen/
    python run_regen.py --viewer   # + 2D sections along the channel and the 3D viewer (~8 min)
    python run_boiling.py          # N2O boiling regime, all coolant-side models -> out/boiling/
    python -m pytest tests         # regression and handbook 10.8 verification checks

Run the commands from this folder (`h2cea/`), so that `import h2cea` finds the package.

The thermochemistry and sizing script of the handbook (`run_all.py`, figures 01–11) and
`notion_figs.py` are not in this repository. The library functions they used are still
here: `cea_si.point()`, `operating_line.size_throat()` / `thrust_to_mox()`, `bartz.py`,
`gasdyn.py`, `contour.py`.

## Library

```python
from h2cea.cea_si import point
from h2cea.operating_line import size_throat, state, thrust_to_mox

p = point(pc=25.0, mr=4.0, eps=4.0)   # bar, O/F, area ratio -> chamber, throat, exit state in SI
mr, At = size_throat()                 # rated point from config.py
s = state(thrust_to_mox(1125.0, At), At)   # 50 % thrust on the fixed-fuel-flow throttle line
```

`propellants.py` builds RocketCEA cards from real-fluid states: the N₂O card uses the
compressed-liquid enthalpy, not RocketCEA's built-in ideal-gas `N2O` card (about 1.5 % in c*).

## run_regen.py (cooling; replaces resa-v3 for E2)
Geometry is the E2-REG-1 as built (`contour.e2_reg1_asbuilt()`, eps 2.99, 43.6 mm bell)
with the jacket in `jacket.e2_reg1()` (50 channels, 0.5 / 0.6 / 0.75 mm, closeout 1.0 mm TBC).
Operating points come from `operating_line.state()` with the as-built throat.

| Output | Content |
|---|---|
| `water_*_profile.csv`, `n2o_*_profile.csv` | every march station: gas side, wall, coolant, regime, CHF ratio |
| `water_sections.csv` | 2D sections at throat and cylinder, normal and blocked |
| `water_flow_sweep.csv`, `water_sensitivities.csv` | flow 0.4–1.6 kg/s; Bartz ±20 %, roughness, wall thickness |
| `structure_pressure_cases.csv` | hot wall, rib and closeout stresses for the pressure cases |
| `results_regen.json` | all key numbers, handbook checks and the verification result |
| `R01`–`R06 *.png` | axial profiles (water, N2O), cross-sections, blocked channel, flow sweep, N2O p–h path |
| `channel_viewer.html` | interactive 3D sector of 15 channels with a movable cut (`--viewer`) |

**The N2O cases in `run_regen.py` use the liquid-only coolant model. For boiling N2O
this is optimistic, not conservative.** With the regime-switching model
(`run_boiling.py`) the gas-side wall at 100 % reaches about 1330 K against the
1050 K limit, where the liquid-only march gives about 990 K. Quote the regime model
for any N2O design statement.

Modules
- `coolant.py` – water (CoolProp) and N2O (CoolProp EOS; transport by CO2 corresponding states)
- `jacket.py` – channel geometry; `materials.py` – IN718 k, E, alpha, yield vs temperature
- `regen.py` – 1D march: Bartz with CEA transport and sigma at the wall temperature,
  wall conductivity at the mean wall temperature, Gnielinski with rib fin efficiency,
  Haaland friction plus acceleration, Hall–Mudawar CHF (x <= 0.05), ONB flag;
  `march_to_outlet_pressure()` finds the jacket inlet pressure for a given injector inlet pressure.
  Both iterations print a `RuntimeWarning` if they stop without converging.
- `twophase.py` – coolant-side two-phase models (`MODELS`), used with `march(..., tp_model=...)`
- `section2d.py` – 2D finite-volume conduction of the channel cross-section, optional blocked channel
- `structure.py` – pressure and thermal stresses; `regen_plots.py`, `viewer.py` – figures and the 3D page

Model limits (state them in any design document that uses the numbers)
- `tp_model="liquid_only"` (the default) gives no boiling credit, but it also ignores
  the loss of heat transfer in film boiling, so it underpredicts the N2O wall
  temperature. Dryout CHF in saturated flow boiling is not evaluated; Hall–Mudawar
  is used only up to x = 0.05, as a fluid-to-fluid estimate for N2O.
- The CEA card for the N2O cases is the tank-state liquid. Jacket heat returned to
  the chamber (about 250 kJ/kg at 100 %) is not added, so Tc and the heat load are
  somewhat low (`n2o_card(extra_h_kJkg=...)` exists for this).
- N2O viscosity and conductivity from CO2 corresponding states: about 10–15 % away from
  the critical point, worse near it.
- Supercritical states use Gnielinski with bulk properties (no heat transfer deterioration model).
- Manifold, volute and inlet-tube losses are not included in the pressure drop.
- Gas-side coefficient: Bartz with CEA frozen transport; it reproduces the handbook
  9500 W/m²K at the full-thrust throat. Calibrate `BARTZ_FACTOR` with hot-fire calorimetry.

For design documents: archive `results_regen.json`, the config values and the git commit.
The committed files in `out/` are a snapshot and can be older than the code; re-run
before quoting numbers.

## run_boiling.py (N2O boiling regime)
Marches the N2O design case once per coolant-side model in `twophase.MODELS`
(liquid-only, nucleate, Jackson-type near-critical, Miropolskii and Groeneveld film
boiling, regime switching) and tabulates the regime indicators along the jacket
(reduced pressure, mass flux, boiling/Weber/confinement numbers, homogeneous Mach
number, limiting liquid superheat, Cooper nucleate limit, Hall–Mudawar CHF,
Kim–Mudawar dryout quality). Output in `out/boiling`: `summary.csv/json`,
`regime_*.csv`, one profile CSV per point and model, figures B01–B03.
`python run_boiling.py regime jackson groeneveld` runs a subset of models; the
liquid-only march is always added because the regime table is built from it.

`regen.march(..., tp_model="regime")` is the recommended N2O design case; the
liquid-only default is kept for water and for comparison.
