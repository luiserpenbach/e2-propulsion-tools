# Shared baseline

The single source for every value that more than one tool uses: the engine design point, propellant states, feed pressures and residuals of the H2 lander / E2 flight engine.

| File | What it is |
|------|------------|
| `h2_baseline.yaml` | Hand-edited inputs. Change values here, not in the tool configs. |
| `engine_table.yaml` | Engine throttle table (thrust, O/F, chamber pressure, flows, Isp per throttle point). Generated from `h2_baseline.yaml` by the h2cea engine model. Do not edit by hand. |
| `e2_baseline.py` | Small loader the tools import: reads the baseline, checks the table is current, fills tool configs. |

## Who reads what

| Tool | How | Takes from the baseline |
|------|-----|--------------------------|
| `h2cea` | `h2cea/h2cea/config.py` | engine, propellants, feed pressures |
| `mission_analysis` | `baseline:` in `h2_mission_inputs.yaml` | engine table, throttle limits, residuals, loading temperature |
| `swirl_injector_analysis` | `baseline:` in the config (P04 design) | operating point and throttle points from the engine table |
| `tank_press_analysis` | `baseline:` in the config (H2) | tank pressure, temperature window and cold/hot cases, residuals, fluid names |

A tool config without a `baseline:` key (for example the H-1B tank case or the P03 as-built injector) is used as it is. A config that names the baseline and also sets one of the baseline values keeps its own value, and the tool prints a `NOTE` with both numbers.

## Changing the baseline

1. Edit `h2_baseline.yaml`, raise `meta.revision` and update `meta.date`.
2. If you changed `propellants`, `engine` or `feed.tank_pressure_bar`, regenerate the table:

   ```bash
   cd h2cea && python run_engine_table.py
   ```

   The mission, injector and tank tools stop with a message if the table does not match the baseline.
3. Re-run the tools you rely on and commit the baseline, the table and your results together.

Results files record the baseline file and revision they were computed with (`meta.baseline` / `baseline_info`).

## Not in the baseline yet

- The tank tool's `mission` block (burn time, flow, O/F) is a hand copy of the mission handover.
- Tank tool: pressurant bottle, regulator and ullage rules. Mission tool: vehicle, present tank volumes, pressurization. Injector: oxidizer state at the holes and pressure drops. These are used by one tool each and stay in that tool's config.
- `torch_igniter_analysis` and `orifice_lib` are independent of the flight engine and do not use the baseline.
