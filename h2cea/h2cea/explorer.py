"""Local server behind the E2 Test Explorer (started by test_explorer.py).

A small HTTP server from the standard library that keeps the h2cea model loaded and
answers JSON requests of one static page (templates/test_explorer.html). All numbers
come from hotfire.py; this module only moves them.

Why it is fast: the model is imported once and stays in memory, a test point is
analysed in about 50 ms and cached, and the slow parts (the water-jacket march with
the Bartz fit, the throttle line, the CEA map) run in worker processes, so the page
shows the results at once and fills those parts in when they arrive.

RocketCEA is Fortran with global state: every CEA call in this process is made under
one lock, and the workers are separate processes with their own copy.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import threading
import time
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import config as C
from . import hotfire as H

PAGE = Path(__file__).resolve().parent / "templates" / "test_explorer.html"

# Where each input is found in a test analysed with the "E2 engine hot fire" template of
# hopper-test-lab (src/lib/template-presets.ts). Shown next to the inputs for the manual
# transfer, and the mapping for pulling an analysed test over its API later.
HTL_SOURCE = {
    "pc_bar": "pc_mean (gauge in the E2 template)",
    "ox_flow_kg_s": "mdot_ox_mean",
    "fuel_flow_kg_s": "mdot_fu_mean",
    "thrust_N": "thrust_mean",
    "duration_s": "steady-state duration",
    "ox_temp_C": "condition N2O temperature",
    "p_ox_inj_bar": "pc_mean + dp_ox_inj_mean",
    "p_fuel_inj_bar": "pc_mean + dp_fu_inj_mean",
}

_LOCK = threading.RLock()          # every CEA / CoolProp call of this process
_CACHE: dict = {}                  # (test key, eta_c, eta_f) -> (R, error)
_JOBS: dict = {}                   # worker results, by (kind, arguments)
_POOL: ProcessPoolExecutor | None = None
_SAVED: list | None = None
_DEMO: list | None = None


# ----------------------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------------------
def clean(o, nd=6):
    """JSON-safe copy: arrays to lists, NaN and inf to null, floats to nd significant digits."""
    if isinstance(o, dict):
        return {str(k): clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v, nd) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        if not math.isfinite(f):
            return None
        return float(f"{f:.{nd}g}")
    if o is None or isinstance(o, str):
        return o
    return str(o)


def thin(a, n=320):
    """At most about n points of an array (first and last kept) for the plots."""
    a = np.asarray(a)
    if len(a) <= n:
        return a
    idx = np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))
    return a[idx]


def contour_json(c):
    return {"x_mm": thin(c["x"] * 1e3), "r_mm": thin(c["r"] * 1e3), "xt_mm": c["xt"] * 1e3, "Rt_mm": c["Rt"] * 1e3}


def test_key(test):
    return json.dumps({k: test.get(k) for k in ("hardware", "pressures_gauge", "inputs")}, sort_keys=True, default=str)


def normalise(test: dict) -> dict:
    """A test record as sent by the page, with every input present and numeric or None."""
    t = H.new_test()
    t.update({k: v for k, v in (test or {}).items() if k != "inputs"})
    for k in H.INPUTS:
        t["inputs"][k] = H._num((test or {}).get("inputs", {}).get(k))
    if t.get("hardware") not in H.hardware_presets():
        t["hardware"] = "E2-REG-1"
    if t.get("outcome") not in H.OUTCOMES:
        t["outcome"] = "pass"
    t["test_id"] = str(t.get("test_id") or "").strip()
    t["date"] = str(t.get("date") or "")[:10]
    t["pressures_gauge"] = bool(t.get("pressures_gauge"))
    return t


# ----------------------------------------------------------------------------------------
# Model calls of this process (fast)
# ----------------------------------------------------------------------------------------
def analyse(test, eta_c, eta_f):
    key = (test_key(test), round(eta_c, 5), round(eta_f, 5))
    with _LOCK:
        if key not in _CACHE:
            if len(_CACHE) > 2000:
                _CACHE.clear()
            try:
                _CACHE[key] = (H.analyse(test, eta_c, eta_f), None)
            except Exception as e:      # shown on the page
                _CACHE[key] = (None, str(e))
        return _CACHE[key]


def saved_tests(reload=False):
    global _SAVED
    with _LOCK:
        if _SAVED is None or reload:
            _SAVED = H.load_tests(H.TEST_DIR)
        return _SAVED


def demo_tests():
    global _DEMO
    with _LOCK:
        if _DEMO is None:
            _DEMO = H.demo_tests()
        return _DEMO


def pool_tests(demo):
    return list(saved_tests()) + (list(demo_tests()) if demo else [])


def next_id(ids):
    nums = [(m.group(1), int(m.group(2)), len(m.group(2))) for i in ids
            if (m := re.match(r"^(.*?)(\d+)$", i)) and not i.startswith("DEMO")]
    if not nums:
        return "E2-HF-001"
    pre, n, w = max(nums, key=lambda x: x[1])
    return f"{pre}{n + 1:0{w}d}"


def reference_for(test, demo, eta_c, eta_f):
    """Mean results of the other tests on the same hardware (drift checks)."""
    rows = []
    for t in pool_tests(demo):
        if t["test_id"] == test["test_id"] or t.get("hardware") != test["hardware"]:
            continue
        R, _ = analyse(t, eta_c, eta_f)
        if R:
            rows.append(R)
    if not rows:
        return None
    keys = ("eta_cstar", "eta_cf_vac", "CdA_ox_mm2", "CdA_fuel_mm2", "isp_meas_s", "pc_dev_pct")
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}


def last_test(test, demo, eta_c, eta_f):
    """The previous test on the same hardware, as a table row."""
    cands = [t for t in pool_tests(demo) if t["test_id"] != test["test_id"] and t.get("hardware") == test["hardware"]
             and str(t.get("date", "")) <= str(test.get("date") or "9999")]
    for t in sorted(cands, key=lambda t: (str(t.get("date")), t["test_id"]), reverse=True):
        R, _ = analyse(t, eta_c, eta_f)
        if R:
            return H.flat(t, R)
    return None


def result_json(R):
    out = {k: v for k, v in R.items() if k not in ("pt", "ox_card", "nozzle")}
    nz = R["nozzle"]
    out["nozzle"] = {k: thin(nz[k]) for k in ("x_mm", "p_wall_bar", "p_sep_bar", "M")}
    out["nozzle"].update({k: nz[k] for k in ("x_sep_mm", "gamma", "x_throat_mm", "x_exit_mm")})
    return out


def api_analyse(body):
    test = normalise(body.get("test"))
    eta_c, eta_f, demo = float(body.get("eta_c", C.ETA_CSTAR)), float(body.get("eta_f", C.ETA_CF_VAC)), bool(body.get("demo"))
    t0 = time.perf_counter()
    R, err = analyse(test, eta_c, eta_f)
    if R is None:
        return {"error": err}
    with _LOCK:
        ref = reference_for(test, demo, eta_c, eta_f)
        last = last_test(test, demo, eta_c, eta_f)
        obs = H.observations(R, None, eta_c, eta_f, reference=ref)
        hw = H.hardware_presets()[test["hardware"]]
        dt_mm = test["inputs"].get("throat_d_mm")
        cont = hw.contour(dt_mm) if dt_mm else None
    out = {"R": result_json(R), "obs": obs, "row": H.flat(test, R), "last": last, "reference": ref,
           "jacket": bool("Q_meas_kW" in R and hw.regen_model),
           "ms": round(1e3 * (time.perf_counter() - t0), 1)}
    if cont is not None:      # a measured throat changes the contour drawn in the nozzle view
        out["contour"] = contour_json(cont)
    return out


def api_tests(q):
    eta_c, eta_f = float(q.get("eta_c", C.ETA_CSTAR)), float(q.get("eta_f", C.ETA_CF_VAC))
    demo = q.get("demo") == "1"
    saved = saved_tests(reload=q.get("reload") == "1")
    saved_ids = {t["test_id"] for t in saved}
    out = []
    for t in pool_tests(demo):
        R, err = analyse(t, eta_c, eta_f)
        hw = H.hardware_presets().get(t.get("hardware"))
        out.append({"test": {k: v for k, v in t.items() if not k.startswith("_")},
                    "saved": t["test_id"] in saved_ids and not t.get("demo"),
                    "row": H.flat(t, R) if R else None, "error": err,
                    "jacket": bool(R and "Q_meas_kW" in R and hw and hw.regen_model)})
    return {"tests": out, "next_id": next_id([t["test_id"] for t in saved])}


def api_meta():
    hws = {}
    with _LOCK:
        for k, hw in H.hardware_presets().items():
            hws[k] = {"key": hw.key, "label": hw.label, "throat_d_mm": hw.throat_d_mm, "eps": hw.eps,
                      "regen_model": hw.regen_model, "contour": contour_json(hw.contour())}
    bl = C.BASELINE["meta"]

    def decimals(fmt):
        m = re.search(r"\.(\d+)f", fmt)
        return int(m.group(1)) if m else 0
    return {
        "fields": [{"key": f.key, "label": f.label, "unit": f.unit, "group": f.group, "default": f.default,
                    "help": f.help, "decimals": decimals(f.fmt), "required": f.key in H.REQUIRED,
                    "htl": HTL_SOURCE.get(f.key, "")} for f in H.INPUT_FIELDS],
        "results": {k: {"label": v[0], "unit": v[1], "decimals": decimals(v[2]), "signed": "+" in v[2]}
                    for k, v in H.RESULT_LABELS.items()},
        "outcomes": H.OUTCOMES, "hardware": hws,
        "eta_c": C.ETA_CSTAR, "eta_f": C.ETA_CF_VAC, "stiffness_min_pct": 100 * H.STIFFNESS_MIN,
        "T_wg_limit_K": C.T_WG_LIMIT, "F_rated_N": C.F_MAX, "pc_rated_bar": C.PC_MAX,
        "fuel_flow_kg_s": C.MDOT_FUEL, "T_ox_nom_C": C.T_OX_NOM - 273.15, "p_amb_bar": C.P_AMB,
        "p_tank_bar": C.P_TANK, "ethanol_wt": C.ETHANOL_WT,
        "baseline": {"revision": bl.get("revision"), "date": str(bl.get("date"))},
        "test_dir": str(H.TEST_DIR),
    }


def api_save(body):
    test = normalise(body.get("test"))
    if not test["test_id"]:
        raise ValueError("A test needs a test ID.")
    if test["test_id"].upper().startswith("DEMO"):
        raise ValueError("Demo points are not saved. Give the test its own ID.")
    with _LOCK:
        existed = any(t["test_id"] == test["test_id"] for t in saved_tests())
        test.pop("demo", None)
        path = H.save_test(test, H.TEST_DIR)
        saved_tests(reload=True)
    return {"file": path.name, "updated": existed}


def api_delete(body):
    with _LOCK:
        H.delete_test(str(body.get("test_id") or ""), H.TEST_DIR)
        saved_tests(reload=True)
    return {"ok": True}


# ----------------------------------------------------------------------------------------
# Worker processes (slow model calls)
# ----------------------------------------------------------------------------------------
def _warm():
    H.hardware_presets()
    return True


def job_jacket(test, eta_c, eta_f, fit, n, reference):
    R = H.analyse(test, eta_c, eta_f)
    W = H.water_jacket(test, R, n_stations=n, fit=fit)
    if W is None:
        return None
    out = {k: v for k, v in W.items() if k not in ("model", "fit")}
    out["Q_ratio"] = R["Q_meas_kW"] / W["Q_model_kW"]
    if fit:
        def profile(r):
            return {"x_mm": r["x"] * 1e3, "Twg_K": r["Twg"], "Twc_K": r["Twc"], "q_MW_m2": r["q"] / 1e6,
                    "T_out_C": r["T_out"] - 273.15, "dp_bar": float(r["dp_bar"])}
        out["model"] = profile(W["model"])
        if W.get("fit") is not None:
            out["fit"] = profile(W["fit"])
        out["obs"] = H.observations(R, W, eta_c, eta_f, reference=reference)
    return clean(out, 5)


def job_line(hw, mf, T_ox, pa, dt_mm, eta_c, eta_f, lo, hi):
    return clean(H.model_line(hw, mf, T_ox, pa, dt_mm, eta_c, eta_f, (lo, hi)), 6)


def job_cea_map(hw_key, T_ox, pa):
    return H.cea_map(H.hardware_presets()[hw_key].eps, T_ox, pa)


def run_job(kind, fn, *args):
    """Run fn(*args) in a worker, once per distinct argument set."""
    key = (kind, json.dumps(args, sort_keys=True, default=str))
    with _LOCK:
        fut = _JOBS.get(key)
        if fut is None:
            if len(_JOBS) > 600:
                for k in [k for k, f in _JOBS.items() if f.done()][:300]:
                    del _JOBS[k]
            fut = _JOBS[key] = _POOL.submit(fn, *args)
    try:
        return fut.result()
    except Exception:
        with _LOCK:
            _JOBS.pop(key, None)
        raise


def api_jacket(body):
    test = normalise(body.get("test"))
    eta_c, eta_f = float(body.get("eta_c", C.ETA_CSTAR)), float(body.get("eta_f", C.ETA_CF_VAC))
    fit = bool(body.get("fit", True))
    ref = reference_for(test, bool(body.get("demo")), eta_c, eta_f) if fit else None
    slim = {k: test[k] for k in ("hardware", "pressures_gauge", "inputs")}
    return {"W": run_job("jacket", job_jacket, slim, eta_c, eta_f, fit, 200 if fit else 120, ref)}


def api_line(body):
    return {"line": run_job("line", job_line, body["hardware"], float(body["fuel_flow_kg_s"]), float(body["T_ox_K"]),
                            float(body.get("pa_bar", C.P_AMB)), H._num(body.get("throat_d_mm")),
                            float(body.get("eta_c", C.ETA_CSTAR)), float(body.get("eta_f", C.ETA_CF_VAC)),
                            float(body.get("lo", 1.6)), float(body.get("hi", 5.2)))}


def api_cea_map(q):
    return run_job("cea", job_cea_map, q.get("hardware", "E2-REG-1"), float(q.get("T_ox_K", C.T_OX_NOM)),
                   float(q.get("pa_bar", C.P_AMB)))


# ----------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------
GET = {"/api/meta": lambda q: api_meta(), "/api/tests": api_tests, "/api/cea_map": api_cea_map}
POST = {"/api/analyse": api_analyse, "/api/jacket": api_jacket, "/api/line": api_line,
        "/api/save": api_save, "/api/delete": api_delete}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _reply(self, fn, arg):
        try:
            self._send(200, json.dumps(clean(fn(arg)), ensure_ascii=False).encode())
        except Exception as e:
            self._send(400, json.dumps({"error": str(e) or type(e).__name__}).encode())

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html")
        if u.path in GET:
            return self._reply(GET[u.path], {k: v[0] for k, v in parse_qs(u.query).items()})
        self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in POST:
            return self._send(404, b'{"error": "not found"}')
        # the page is served from this host; refuse requests that another site makes a browser send
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._send(403, b'{"error": "cross-origin request refused"}')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self._send(400, b'{"error": "invalid JSON"}')
        self._reply(POST[u.path], body)


def start_workers(workers=2):
    global _POOL
    _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    for _ in range(workers):
        _POOL.submit(_warm)


def make_server(host="127.0.0.1", port=8600):
    """The HTTP server on the first free port from `port` (serve_forever not yet called)."""
    last = None
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, p), Handler)
        except OSError as e:
            last = e
    raise last


def main(argv=None):
    ap = argparse.ArgumentParser(description="E2 Test Explorer: hot-fire test points against the h2cea models.")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines on the network")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    start_workers()
    srv = make_server(a.host, a.port)
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{srv.server_address[1]}"
    print(f"E2 Test Explorer on {url}   (Ctrl+C stops it)")
    threading.Thread(target=lambda: [analyse(t, C.ETA_CSTAR, C.ETA_CF_VAC) for t in saved_tests()], daemon=True).start()
    if not a.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        _POOL.shutdown(wait=False, cancel_futures=True)
