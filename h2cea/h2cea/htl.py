"""Pull an analysed test from hopper-test-lab into a test point of hotfire.py.

hopper-test-lab stays generic: it finds the steady state and stores the result formulas
of its template. This module reads a saved test over its API (the one the browser uses)
and maps the results of the template "E2 engine hot fire" onto the inputs of
hotfire.analyse. Nothing is written to hopper-test-lab.

Connection: h2cea/hopper_test_lab.yaml (not committed, see hopper_test_lab.example.yaml)

    url: https://your-hopper-test-lab.example
    password: the shared team password

or the environment variables HTL_URL and HTL_PASSWORD, which win over the file.

Mapping (result key of the template -> input), see to_test():

    pc_mean -> pc_bar                      mdot_ox_mean -> ox_flow_kg_s
    thrust_mean -> thrust_N                mdot_fu_mean -> fuel_flow_kg_s
    pc_mean + dp_ox_inj_mean -> p_ox_inj_bar      (or p_ox_inj_mean)
    pc_mean + dp_fu_inj_mean -> p_fuel_inj_bar    (or p_fu_inj_mean)
    steady_duration_s -> duration_s        condition "N2O temperature" -> ox_temp_C
    derived constant pa -> p_amb_bar; pa > 0 means the pressures are gauge
    derived constant d_throat -> throat_d_mm, if it differs from the hardware nominal

Water calorimetry is not in the template. A test whose template has these formulas
gets it pulled as well: mdot_w_mean [kg/s], T_w_in_mean and T_w_out_mean [°C],
p_w_in_mean and p_w_out_mean [bar].
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from . import hotfire as H

CONFIG = Path(__file__).resolve().parents[1] / "hopper_test_lab.yaml"
RESULT_FOR = {          # input key -> result key of the hopper-test-lab template, taken as is
    "pc_bar": "pc_mean", "ox_flow_kg_s": "mdot_ox_mean", "fuel_flow_kg_s": "mdot_fu_mean",
    "thrust_N": "thrust_mean", "duration_s": "steady_duration_s",
    "water_flow_kg_s": "mdot_w_mean", "water_T_in_C": "T_w_in_mean", "water_T_out_C": "T_w_out_mean",
    "water_p_in_bar": "p_w_in_mean", "water_p_out_bar": "p_w_out_mean",
}
INJECTOR = {"p_ox_inj_bar": ("p_ox_inj_mean", "dp_ox_inj_mean"), "p_fuel_inj_bar": ("p_fu_inj_mean", "dp_fu_inj_mean")}
OX_TEMP_CONDITIONS = ("oxTempC", "tankT_C")
UNIT_SCALE = {"g/s": 1e-3, "kN": 1e3, "mbar": 1e-3, "kPa": 1e-2, "MPa": 10.0, "psi": 0.0689476}


class HtlError(Exception):
    pass


def config():
    """{'url', 'password'} or None if hopper-test-lab is not set up on this machine."""
    cfg = {}
    if CONFIG.exists():
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    url = os.environ.get("HTL_URL") or cfg.get("url")
    password = os.environ.get("HTL_PASSWORD") or cfg.get("password")
    if not url or not password:
        return None
    return {"url": str(url).rstrip("/"), "password": str(password)}


# ----------------------------------------------------------------------------------------
# Mapping (no network)
# ----------------------------------------------------------------------------------------
def derived_constant(text, name):
    """Value of a constant line `name = number [unit]` of the derived channels, or None."""
    for line in (text or "").splitlines():
        m = re.match(rf"\s*{re.escape(name)}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:\[[^\]]*\])?\s*(?:#.*)?$", line)
        if m:
            return float(m.group(1))
    return None


def to_test(bundle: dict) -> dict:
    """Map a hopper-test-lab test (the JSON of GET /api/runs/<id>) to a test point.

    Returns {"test", "pulled": [[input label, source, value]], "missing": [input labels],
    "notes": [text]}. Inputs that the test does not hold stay empty."""
    run = bundle.get("run") or {}
    an = run.get("analysis") or {}
    res = {}
    for r in bundle.get("results") or []:
        v = H._num(r.get("value"))
        if v is not None:
            res[r.get("key")] = v * UNIT_SCALE.get((r.get("unit") or "").strip(), 1.0)
    if an.get("steadyDurationS") is not None:
        res.setdefault("steady_duration_s", H._num(an.get("steadyDurationS")))

    article = str(run.get("article") or "")
    notes = []
    if article.upper().startswith("E2-REG-1"):
        hardware = "E2-REG-1"
    elif "flight" in article.lower():
        hardware = "E2 flight"
    else:
        hardware = "E2-REG-1"
        notes.append(f"Test article '{article or 'not set'}' is not a known hardware; E2-REG-1 assumed.")
    outcome = run.get("outcome") if run.get("outcome") in H.OUTCOMES else "pass"
    text = "; ".join(x for x in (str(run.get("notes") or "").strip(), ", ".join(an.get("observations") or [])) if x)
    test = H.new_test(test_id=str(run.get("id") or ""), date=str(run.get("startedAt") or "")[:10], hardware=hardware,
                      outcome=outcome, notes=text,
                      source=f"hopper-test-lab {run.get('id')}" + (f" ({an['templateName']})" if an.get("templateName") else ""))
    for k in H.INPUTS:                      # only what the test holds, no defaults of the form
        test["inputs"][k] = None
    pulled = []

    def put(key, value, source):
        if value is None:
            return
        test["inputs"][key] = float(f"{value:.6g}")
        pulled.append([H.INPUTS[key].label, source, test["inputs"][key]])

    for key, rk in RESULT_FOR.items():
        put(key, res.get(rk), rk)
    for key, (direct, dp) in INJECTOR.items():
        if direct in res:
            put(key, res[direct], direct)
        elif dp in res and "pc_mean" in res:
            put(key, res["pc_mean"] + res[dp], f"pc_mean + {dp}")
    cond = an.get("conditions") or {}
    for ck in OX_TEMP_CONDITIONS:
        v = H._num(str(cond.get(ck, "")).replace(",", ".") or None)
        if v is not None:
            put("ox_temp_C", v, "condition N2O temperature")
            break
    derived = an.get("derivedText") or ""
    pa = derived_constant(derived, "pa")
    if pa is None:
        notes.append("No constant 'pa' in the derived channels: check whether the pressures are gauge or absolute.")
    elif pa > 0:
        test["pressures_gauge"] = True
        put("p_amb_bar", pa, "derived constant pa (pressures are gauge)")
    else:
        notes.append("pa = 0 in the derived channels: pressures taken as absolute.")
    d = derived_constant(derived, "d_throat")
    if d is not None and abs(d * 1e3 - H.hardware_presets()[hardware].throat_d_mm) > 0.005:
        put("throat_d_mm", d * 1e3, "derived constant d_throat")
    for key, f in H.INPUTS.items():         # what the form would default, where the test is silent
        if test["inputs"][key] is None and f.default is not None and key not in ("fuel_flow_kg_s",):
            test["inputs"][key] = f.default
    missing = [H.INPUTS[k].label for k in H.REQUIRED if test["inputs"][k] is None]
    if run.get("type") not in (None, "engine"):
        notes.append(f"This is a test of type '{run.get('type')}', not an engine hot fire.")
    return {"test": test, "pulled": pulled, "missing": missing, "notes": notes}


# ----------------------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------------------
class Client:
    """Signs in with the shared password and reads campaigns and tests. Read-only."""

    def __init__(self, url, password, timeout=20):
        self.url, self.password, self.timeout = url.rstrip("/"), password, timeout
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self._runs = (0.0, None)

    def _open(self, path, body=None):
        req = urllib.request.Request(self.url + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Accept": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as r:
            return json.loads(r.read() or b"{}")

    def login(self):
        try:
            self._open("/api/login", {"password": self.password, "name": "E2 Test Explorer"})
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise HtlError("Hopper Test Lab refused the password in hopper_test_lab.yaml.") from None
            if e.code == 429:
                raise HtlError("Hopper Test Lab blocked the sign-in after too many attempts; wait ten minutes.") from None
            raise HtlError(f"Hopper Test Lab sign-in failed (HTTP {e.code}).") from None

    def get(self, path):
        try:
            try:
                return self._open(path)
            except urllib.error.HTTPError as e:
                if e.code != 401:
                    raise
                self.login()                 # no session yet, or it has expired
                return self._open(path)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise HtlError("Hopper Test Lab does not have this test.") from None
            raise HtlError(f"Hopper Test Lab answered HTTP {e.code} for {path}.") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise HtlError(f"Hopper Test Lab at {self.url} cannot be reached ({getattr(e, 'reason', e)}).") from None
        except ValueError:
            raise HtlError(f"{self.url} did not answer as Hopper Test Lab; check the url.") from None

    def runs(self, max_age_s=20):
        """Engine tests of all campaigns, newest first."""
        at, cached = self._runs
        if cached is not None and time.time() - at < max_age_s:
            return cached
        out = []
        for c in self.get("/api/campaigns").get("campaigns", []):
            if not c.get("testCount", 1):
                continue
            b = self.get("/api/campaigns/" + urllib.parse.quote(str(c["id"]), safe=""))
            for r in b.get("runs", []):
                if r.get("type") == "engine":
                    out.append({"id": r["id"], "campaign": c.get("name", ""), "date": str(r.get("startedAt") or "")[:10],
                                "article": r.get("article", ""), "outcome": r.get("outcome", ""),
                                "started_at": str(r.get("startedAt") or "")})
        out.sort(key=lambda r: r["started_at"], reverse=True)
        self._runs = (time.time(), out)
        return out

    def test(self, run_id):
        return to_test(self.get("/api/runs/" + urllib.parse.quote(str(run_id), safe="")))
