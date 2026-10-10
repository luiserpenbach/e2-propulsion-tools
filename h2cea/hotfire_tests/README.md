# E2 hot-fire test points

One YAML file per test, written by the E2 Test Explorer (`streamlit run test_explorer.py`
from `h2cea/`). The files hold the inputs only (steady-state means, metadata, notes); the
app recomputes every result with the current model when it loads them, so a model or
baseline change shows up in all tests at once.

Commit the files you want the team to compare against. Edit them by hand if needed;
the keys are listed in `h2cea/hotfire.py` (`INPUT_FIELDS`).
