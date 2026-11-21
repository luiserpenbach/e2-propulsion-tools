"""
E2 Test Explorer: hot-fire test points against the h2cea models.

    python test_explorer.py            (run from h2cea/; opens http://localhost:8600)
    python test_explorer.py --port 8700 --no-browser
    python test_explorer.py --host 0.0.0.0     (reachable from other machines)

Needs only the h2cea requirements (no Streamlit, no plotly). The server is
h2cea/explorer.py, the page h2cea/templates/test_explorer.html, and every number
comes from h2cea/hotfire.py.

Test        type the steady-state means of a test analysed in hopper-test-lab; get
            eta c*, eta Cf, Isp, pc and thrust against the model at the measured flows,
            the nozzle separation check, injector stiffness and CdA, and the water-jacket
            heat with the Bartz factor that matches it. Save writes hotfire_tests/.
Campaign    every saved test recomputed with the current model: table, trend, scatter,
            side-by-side comparison, CSV export.
Model       throttle line of the hardware, and the CEA map export.
"""
from h2cea.explorer import main

if __name__ == "__main__":      # the guard is required: the workers are spawned processes
    main()
