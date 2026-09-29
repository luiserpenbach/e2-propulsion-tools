# h2cea – thrust chamber reference analyses (Handbook H2-PRP-HBK-001 Rev B)

Thermochemistry, first-order sizing and the regenerative cooling analysis of the
E2 chamber. All baseline inputs live in `h2cea/config.py`.

    pip install rocketcea CoolProp numpy scipy matplotlib pandas
    python run_all.py          # ~25 s, thermochemistry and sizing -> out/
    python run_regen.py        # ~2 min, cooling analyses -> out/regen/
    python run_regen.py --viewer   # + 2D sections along the channel and the 3D viewer (~8 min)
    python -m pytest tests     # regression and handbook 10.8 verification checks

## run_all.py
01 O/F sweep (liquid vs gaseous N2O card), 02 operating line vs architecture Rev A,
03 chamber species, 04 oxidiser enthalpy sensitivity, 05 area ratio / separation,
06 CEA transport vs textbook approximations, 07 Bartz profiles on E2-REG-1,
08 jacket pressure regime map, 09 Hall–Mudawar CHF estimate, 10 finite-area combustor,
decomposition temperatures, acoustics, 11 jacket outlet states and p–h path.
`notion_figs.py` builds the compact SVGs used in the Notion handbook.

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

Modules
- `coolant.py` – water (CoolProp) and N2O (CoolProp EOS; transport by CO2 corresponding states)
- `jacket.py` – channel geometry; `materials.py` – IN718 k, E, alpha, yield vs temperature
- `regen.py` – 1D march: Bartz with CEA transport and sigma at the wall temperature,
  wall conductivity at the mean wall temperature, Gnielinski with rib fin efficiency,
  Haaland friction plus acceleration, Hall–Mudawar CHF (x <= 0.05), ONB flag;
  `march_to_outlet_pressure()` finds the jacket inlet pressure for a given injector inlet pressure
- `section2d.py` – 2D finite-volume conduction of the channel cross-section, optional blocked channel
- `structure.py` – pressure and thermal stresses; `regen_plots.py`, `viewer.py` – figures and the 3D page

Model limits (state them in any design document that uses the numbers)
- Two-phase N2O uses a liquid-only Gnielinski coefficient (no boiling credit, conservative
  for the wall). Dryout CHF in saturated flow boiling is not evaluated; Hall–Mudawar is
  used only up to x = 0.05, as a fluid-to-fluid estimate for N2O.
- N2O viscosity and conductivity from CO2 corresponding states: about 10–15 % away from
  the critical point, worse near it.
- Supercritical states use Gnielinski with bulk properties (no heat transfer deterioration model).
- Manifold, volute and inlet-tube losses are not included in the pressure drop.
- Gas-side coefficient: Bartz with CEA frozen transport; it reproduces the handbook
  9500 W/m²K at the full-thrust throat. Calibrate `BARTZ_FACTOR` with hot-fire calorimetry.

For design documents: archive `results_regen.json`, the config values and the git commit.

## run_boiling.py (N2O boiling regime)
Marches the N2O design case once per coolant-side model in `twophase.MODELS`
(liquid-only, nucleate, Jackson-type near-critical, Miropolskii and Groeneveld film
boiling, regime switching) and tabulates the regime indicators along the jacket
(reduced pressure, mass flux, boiling/Weber/confinement numbers, homogeneous Mach
number, limiting liquid superheat, Cooper nucleate limit, Hall–Mudawar CHF,
Kim–Mudawar dryout quality). Output in `out/boiling`: `summary.csv/json`,
`regime_*.csv`, one profile CSV per point and model, figures B01–B03.
`python run_boiling.py regime jackson groeneveld` runs a subset of models.

`regen.march(..., tp_model="regime")` is the recommended N2O design case; the
liquid-only default is kept for water and for comparison.
