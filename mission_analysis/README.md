# H2 lander mission analysis

Propellant, mass and tank sizing for the two H2 missions (10 s hover, 10 m hop).
Documentation: Notion, 02 Engineering / Systems Engineering / Mission Analysis.

## Files

| File | Role |
|---|---|
| `h2_mission_inputs.yaml` | All inputs: engine, vehicle, tanks, pressurization, allowances, missions |
| `engine_performance.py` | Estimates the engine throttle line with RocketCEA (fixed fuel flow, oxidizer throttled) |
| `engine_performance.yaml` | Generated throttle line: O/F, chamber pressure, flows, Isp per throttle point |
| `mission_sizing.py` | Flies both missions, builds the propellant budget, mass point and tank check |
| `mission_results.yaml` | Generated results; the `handover` section is the input to the tank sizing |
| `plot_profiles_svg.py` | Figure for the Notion page (`mission_profiles.svg`) |

## Data flow

```
h2_mission_inputs.yaml ──> engine_performance.py ──> engine_performance.yaml
          │                                                   │
          └──────────────> mission_sizing.py <────────────────┘
                                   │
                                   └──> mission_results.yaml ──> tank / pressurization sizing
                                                                  (returns pressurant mass and
                                                                   vapour make-up to the inputs)
```

## Run

```bash
pip install numpy pyyaml matplotlib rocketcea
python engine_performance.py                 # only when engine inputs change
python mission_sizing.py --max-apex --sweep --plot
python mission_sizing.py --size-tanks        # first-order tanks for both missions
python mission_sizing.py --set missions.hop.apex_m=12 --set vehicle.dry_mass_kg=130
```
