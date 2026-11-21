"""Checks of the hopper-test-lab pull (h2cea/htl.py): the mapping and the read-only client."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from h2cea import config as C
from h2cea import hotfire as H
from h2cea import htl

# The derived channels of the hopper-test-lab template "E2 engine hot fire" (the lines that matter here)
DERIVED = """pc = {PT-CC-01} [bar]
# E2-REG-1-A test hardware. Pressures are taken as gauge: set pa = 0 for absolute transducers.
pa = 1.01325 [bar]
d_throat = 0.02814 [m]
A_throat = pi * d_throat^2 / 4 [m²]
pc_abs = pc + pa [bar]"""


def bundle(**over):
    res = {"pc_mean": (19.4, "bar"), "mdot_ox_mean": (0.62, "kg/s"), "mdot_fu_mean": (0.2, "kg/s"),
           "thrust_mean": (1650.0, "N"), "dp_ox_inj_mean": (9.1, "bar"), "dp_fu_inj_mean": (6.4, "bar"),
           "eta_cstar_mean": (0.93, ""), "steady_duration_s": (4.2, "s")}
    res.update(over.pop("results", {}))
    run = {"id": "E2-HF-007", "type": "engine", "article": "E2-REG-1-A", "outcome": "anomaly", "notes": "Hard start",
           "startedAt": "2026-11-20T14:03:11.000Z",
           "analysis": {"templateName": "E2 engine hot fire", "derivedText": DERIVED, "observations": ["rough"],
                        "conditions": {"oxTempC": "17,5", "targetPcBar": "25"}, "steadyDurationS": 4.2}}
    run.update(over)
    return {"run": run, "results": [{"key": k, "value": v, "unit": u, "phaseName": "Steady state", "formula": ""}
                                    for k, (v, u) in res.items() if v is not None]}


def test_template_results_map_to_the_inputs():
    out = htl.to_test(bundle())
    t, i = out["test"], out["test"]["inputs"]
    assert (t["test_id"], t["date"], t["hardware"], t["outcome"]) == ("E2-HF-007", "2026-11-20", "E2-REG-1", "anomaly")
    assert t["pressures_gauge"] is True and i["p_amb_bar"] == pytest.approx(1.01325)
    assert (i["pc_bar"], i["ox_flow_kg_s"], i["fuel_flow_kg_s"], i["thrust_N"]) == (19.4, 0.62, 0.2, 1650.0)
    assert i["p_ox_inj_bar"] == pytest.approx(28.5) and i["p_fuel_inj_bar"] == pytest.approx(25.8)
    assert i["ox_temp_C"] == 17.5 and i["duration_s"] == 4.2
    assert i["throat_d_mm"] is None             # the template throat is the hardware nominal
    assert out["sources"]["pc_bar"] == "pc_mean" and out["sources"]["p_ox_inj_bar"] == "pc_mean + dp_ox_inj_mean"
    assert "throat_d_mm" not in out["sources"] and len(out["sources"]) == len(out["pulled"])
    assert i["water_flow_kg_s"] is None and out["missing"] == [] and out["notes"] == []
    assert "Hard start" in t["notes"] and "rough" in t["notes"] and "E2-HF-007" in t["source"]
    R = H.analyse(t)                            # gauge pc plus ambient, as hopper-test-lab's pc_abs
    assert R["pc_abs_bar"] == pytest.approx(19.4 + 1.01325)
    assert R["dp_ox_bar"] == pytest.approx(9.1)


def test_absolute_pressures_other_throat_units_and_water():
    d = DERIVED.replace("pa = 1.01325", "pa = 0").replace("0.02814", "0.02830")
    b = bundle(results={"mdot_ox_mean": (620.0, "g/s"), "dp_ox_inj_mean": (None, ""), "p_ox_inj_mean": (30.0, "bar"),
                        "mdot_w_mean": (1.0, "kg/s"), "T_w_in_mean": (18.0, "°C"), "T_w_out_mean": (49.0, "°C")})
    b["run"]["analysis"]["derivedText"] = d
    out = htl.to_test(b)
    i = out["test"]["inputs"]
    assert out["test"]["pressures_gauge"] is False and i["p_amb_bar"] == C.P_AMB
    assert i["ox_flow_kg_s"] == pytest.approx(0.62) and i["p_ox_inj_bar"] == 30.0
    assert i["throat_d_mm"] == pytest.approx(28.3) and i["water_T_out_C"] == 49.0
    assert any("absolute" in n for n in out["notes"])
    assert "Q_meas_kW" in H.analyse(out["test"])


def test_missing_results_are_named_not_defaulted():
    b = bundle(results={"mdot_fu_mean": (None, ""), "pc_mean": (None, "")}, article="IGN-C02", type="igniter")
    b["run"]["analysis"]["derivedText"] = ""
    out = htl.to_test(b)
    assert out["missing"] == ["Chamber pressure", "Fuel flow"]
    assert out["test"]["inputs"]["fuel_flow_kg_s"] is None and out["test"]["inputs"]["p_ox_inj_bar"] is None
    assert len(out["notes"]) == 3               # unknown article, no pa, not an engine test


class FakeLab(BaseHTTPRequestHandler):
    """The parts of hopper-test-lab the client uses: password login, session cookie, 401 without it."""
    logins = 0

    def log_message(self, *a):
        pass

    def _json(self, code, obj, cookie=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/login" and body.get("password") == "secret":
            FakeLab.logins += 1
            return self._json(200, {"ok": True}, "htl_session=tok; Path=/; HttpOnly; SameSite=lax")
        self._json(401, {"error": "Wrong password"})

    def do_GET(self):
        if "htl_session=tok" not in (self.headers.get("Cookie") or ""):
            return self._json(401, {"error": "Unauthorized"})
        if self.path == "/api/campaigns":
            return self._json(200, {"campaigns": [{"id": "c 1", "name": "E2 hot fire", "testCount": 2}, {"id": "c2", "name": "empty", "testCount": 0}]})
        if self.path == "/api/campaigns/c%201":
            return self._json(200, {"runs": [{"id": "IGN-1", "type": "igniter", "startedAt": "2026-11-21T10:00:00Z"},
                                             {"id": "E2-HF-006", "type": "engine", "startedAt": "2026-11-19T10:00:00Z", "outcome": "pass"},
                                             {"id": "E2-HF-007", "type": "engine", "startedAt": "2026-11-20T10:00:00Z", "outcome": "anomaly"}]})
        if self.path == "/api/runs/E2-HF-007":
            return self._json(200, bundle())
        self._json(404, {"error": "Test not found."})


@pytest.fixture()
def lab(monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    FakeLab.logins = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLab)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_client_signs_in_once_and_reads(lab):
    c = htl.Client(lab, "secret")
    assert [r["id"] for r in c.runs()] == ["E2-HF-007", "E2-HF-006"]        # engine tests only, newest first
    assert c.test("E2-HF-007")["test"]["inputs"]["pc_bar"] == 19.4
    assert FakeLab.logins == 1
    with pytest.raises(htl.HtlError, match="does not have this test"):
        c.test("nope")


def test_client_errors_are_readable(lab):
    with pytest.raises(htl.HtlError, match="refused the password"):
        htl.Client(lab, "wrong").runs()
    with pytest.raises(htl.HtlError, match="cannot be reached"):
        htl.Client("http://127.0.0.1:9", "secret", timeout=2).runs()


def test_config_from_file_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(htl, "CONFIG", tmp_path / "hopper_test_lab.yaml")
    monkeypatch.delenv("HTL_URL", raising=False)
    monkeypatch.delenv("HTL_PASSWORD", raising=False)
    assert htl.config() is None
    htl.CONFIG.write_text("url: https://lab.example/\npassword: pw\n")
    assert htl.config() == {"url": "https://lab.example", "password": "pw"}
    monkeypatch.setenv("HTL_PASSWORD", "from-env")
    assert htl.config()["password"] == "from-env"
