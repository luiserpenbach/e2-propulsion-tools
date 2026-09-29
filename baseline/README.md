# Shared baseline

The agreed H2 lander / E2 flight engine numbers that more than one tool uses: engine design point, propellant states, feed pressures and residuals. It is there so that system-level results (mission, tank sizing, chamber) use the same engine. It is **not** required for analysis work: every tool can still be run from its own config, independent of this folder.

| File | What it is |
|------|------------|
| `h2_baseline.yaml` | Hand-edited inputs. |
| `h2_baseline_engine_table.yaml` | Engine throttle table (thrust, O/F, chamber pressure, flows, Isp per throttle point). Generated from `h2_baseline.yaml` by the h2cea engine model. Do not edit by hand. |
| `e2_baseline.py` | Small loader the tools import when a config asks for the baseline. |

## Who uses it

| Tool | Uses the baseline | Independent analysis |
|------|-------------------|----------------------|
| `h2cea` | always, through `h2cea/h2cea/config.py` | run with a private copy (`E2_BASELINE`, below), or call the library functions with your own arguments, e.g. `size_throat(F=..., pc=...)` |
| `mission_analysis` | through `baseline:` in `h2_mission_inputs.yaml` | point `baseline:` at a private copy, or `--set baseline=<file>` |
| `tank_press_analysis` | only configs with a `baseline:` key (the H2 config) | any config without the key, e.g. `n2o_press.yaml` (H-1B) |
| `swirl_injector_analysis` | only configs with a `baseline:` key (none at present) | all configs, including P03 and P04, set their own operating point |
| `torch_igniter_analysis`, `orifice_lib` | never | always |

A config without a `baseline:` key never reads this folder; the tank and injector tools only load `e2_baseline.py` when a config names a baseline.

### Opting in with a tool config

Add `baseline: ../../baseline/h2_baseline.yaml` (path relative to the config) to a tank or injector config and leave out the values it should take from the baseline:

- **Injector:** the operating point comes from the rated row of the engine table, and each throttle point's `fraction` becomes a fraction of rated thrust with chamber pressure, O/F and total flow from the table (fuel flow fixed, oxidizer throttled, as the engine does).
- **Tanks:** tank pressure, temperature window and cold/hot cases, residuals and fluid names.

A value that the config still sets itself wins, and the tool prints a `NOTE` with both numbers.

## Changing the shared baseline

1. Edit `h2_baseline.yaml`, raise `meta.revision` and update `meta.date`.
2. If you changed `propellants`, `engine` or `feed.tank_pressure_bar`, regenerate the table:

   ```bash
   cd h2cea && python run_engine_table.py
   ```

   Tools that use the baseline stop with a message if the table does not match it.
3. Re-run the tools you rely on and commit the baseline, the table and your results together.

Results files record the baseline file and revision they were computed with.

## What-if studies: private baselines

Do not edit the shared file for a trade study. Copy it instead:

```bash
cp baseline/h2_baseline.yaml baseline/study_pc30.yaml     # edit the copy
cd h2cea
E2_BASELINE=../baseline/study_pc30.yaml python run_engine_table.py   # -> baseline/study_pc30_engine_table.yaml
E2_BASELINE=../baseline/study_pc30.yaml python run_regen.py
```

On Windows PowerShell: `$env:E2_BASELINE="..\baseline\study_pc30.yaml"; python run_engine_table.py`.

Every baseline file has its own table (`<name>_engine_table.yaml`), so a study cannot overwrite the shared H2 table. Mission, tank and injector configs use a study baseline through their `baseline:` key (mission tool: `--set baseline=../baseline/study_pc30.yaml`). Commit a study baseline only if it is meant to be shared.

## Not in the baseline

- Values used by one tool only stay in that tool's config: pressurant bottle, regulator and ullage rules (tanks); vehicle, present tank volumes, pressurization (mission); oxidizer state at the holes and pressure drops (injector).
