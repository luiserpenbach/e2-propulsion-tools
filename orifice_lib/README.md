# Orifice library

Mass flow through a single orifice for the three fluids this feed system actually sees: incompressible liquid (ethanol), real gas (nitrogen or helium pressurant), and flashing liquid (nitrous oxide). One YAML file drives all three. The same functions can be imported by another tool.

## Models

**Liquid.** Bernoulli, upstream velocity neglected:

```
mdot = Cd * A * sqrt(2 * rho * dP)
```

Density is the upstream liquid. If the downstream pressure is below the vapor pressure the report says so and points at the two-phase model.

**Gas.** Real-gas isentropic expansion with CoolProp. Choked when the throat velocity equals the local speed of sound; mass flux is then `rho* * c*`. Subsonic, the fluid expands to the downstream pressure. A constant-k closed form, using the real upstream density and the upstream isentropic exponent, is printed next to it. Use the real-gas number when the two differ. If the isentrope hits the saturation curve before a gas-phase sonic point, the gas model stops and says to use the two-phase model.

**Two-phase.** Three numbers:

| Model | Role |
|-------|------|
| SPI | No flashing. Upper mass flow for a liquid. |
| HEM | Isentropic equilibrium flashing. Mass flux `G = rho * sqrt(2*(h0-h))`, choked at the maximum of G. |
| NHNE | Best estimate. Dyer / Solomon blend of SPI and HEM. |

```
kappa = sqrt( (P_up - P_down) / (P_sat - P_down) )
mdot  = (kappa * mdot_SPI + mdot_HEM) / (1 + kappa)
```

`kappa = 1` for saturated liquid. A subcooled liquid weights SPI more heavily. No flashing (downstream pressure at or above the vapor pressure) returns SPI. Orifice L/D is printed as a sanity check; it is not inside the equations. The proportionality constant in Dyer's time-scale ratio is taken as 1, following Solomon (2011) and Waxman (AIAA 2013-3636).

Cd is never predicted. It multiplies the geometric area in every model.

Mass flow is proportional to area, so a target mass flow is turned into a diameter by scaling the solved point. For a flashing orifice the report gives the SPI, NHNE, and HEM diameters. NHNE is the estimate. The HEM diameter is the larger hole if the target is a minimum flow under equilibrium flashing.

## Layout

```
orifice_lib/
|-- flight_config_sizing.py    # entry point
|-- liquid.py
|-- gas.py
|-- two_phase.py
|-- common.py                  # CoolProp state, area, units
|-- plots.py
|-- configs/orifice_cases.yaml
|-- results/                   # generated; not source
`-- README.md
```

## Requirements

- Python 3.10+
- CoolProp
- NumPy, SciPy, PyYAML
- Plotly, for the HTML figures

```bash
pip install CoolProp numpy scipy pyyaml plotly
```

## Quick start

From this directory:

```bash
python flight_config_sizing.py configs/orifice_cases.yaml --outdir results
```

| Argument | Default | |
|----------|---------|---|
| config | `configs/orifice_cases.yaml` | YAML case list |
| `--outdir` | `results/` in this folder | HTML figures and `<config>_results.yaml` (`<config>_<case>_results.yaml` with `--case`) |
| `--case NAME` | all cases | run one case |
| `--no-plot` | plots on | skip the HTML figures |

The two-phase case takes a few seconds. The HEM critical pressure is a property scan.

## Calling it from another script

From the repository root:

```python
from orifice_lib import liquid_orifice, gas_orifice, two_phase_orifice

result = liquid_orifice("Ethanol", T_K=293.15, P_down_Pa=1e5, d_m=1e-3, Cd=0.8, P_up_Pa=100e5)
print(result.mdot_kg_s)
```

Pressures and diameters passed to the functions are SI. The YAML file is in bar, °C, and mm.
