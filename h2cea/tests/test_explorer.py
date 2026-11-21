"""Checks of the E2 Test Explorer server (h2cea/explorer.py): the JSON it sends to the page."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from h2cea import config as C
from h2cea import explorer as X
from h2cea import hotfire as H


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "TEST_DIR", tmp_path)
    monkeypatch.setattr(X, "_SAVED", None)
    return tmp_path


def demo(i=2):
    t = H.demo_tests()[i]
    return {k: v for k, v in t.items() if k != "demo"}


def test_clean_is_strict_json():
    out = X.clean({"a": np.array([1.0, np.inf, np.nan]), "b": np.float64(1.23456789), "c": np.bool_(True), 3: None})
    assert out == {"a": [1.0, None, None], "b": 1.23457, "c": True, "3": None}
    json.dumps(out, allow_nan=False)


def test_analyse_matches_hotfire(store):
    t = demo()
    out = X.clean(X.api_analyse({"test": t}))
    json.dumps(out, allow_nan=False)
    R = H.analyse(X.normalise(t))
    assert out["R"]["eta_cstar"] == pytest.approx(R["eta_cstar"], rel=1e-5)
    assert out["R"]["thrust_model_N"] == pytest.approx(R["thrust_model_N"], rel=1e-5)
    assert out["jacket"] is True and "pt" not in out["R"]
    assert len(out["R"]["nozzle"]["x_mm"]) == len(out["R"]["nozzle"]["p_wall_bar"]) <= 320
    assert [o[0] for o in out["obs"]] == [o[0] for o in H.observations(R)]
    assert {o[2] for o in out["obs"]} <= {"performance", "nozzle", "injector", "jacket"}
    assert out["level"] == X.worst(out["obs"]) and out["level"] in ("crit", "warn", "good")


def test_findings_are_filed_under_the_part_of_the_engine():
    obs = [("warn", "η c* 0.9 vs 0.955 assumed"), ("info", "Jacket heat 130 kW measured"), ("good", "Hot-wall maximum 780 K"),
           ("warn", "Nozzle flow likely separated"), ("good", "Ox injector stiffness 50 %"), ("warn", "Fuel CdA +6 % against the mean")]
    assert [o[2] for o in X.tagged(obs)] == ["performance", "jacket", "jacket", "nozzle", "injector", "injector"]
    assert X.worst(obs) == "warn" and X.worst(obs[1:3]) == "good" and X.worst(obs + [("crit", "x")]) == "crit"


def test_missing_inputs_are_reported_not_raised(store):
    out = X.api_analyse({"test": {"test_id": "X", "inputs": {"pc_bar": "", "ox_flow_kg_s": None}}})
    assert "Chamber pressure" in out["error"]


def test_text_and_comma_inputs_are_rejected_as_missing(store):
    t = X.normalise({"inputs": {"pc_bar": "abc", "ox_flow_kg_s": float("nan")}, "hardware": "nope", "outcome": "?"})
    assert t["inputs"]["pc_bar"] is None and t["inputs"]["ox_flow_kg_s"] is None
    assert t["hardware"] == "E2-REG-1" and t["outcome"] == "pass"


def test_save_list_reference_delete(store):
    a, b = demo(1), demo(3)
    a["test_id"], b["test_id"] = "E2-HF-001", "E2-HF-002"
    assert X.api_save({"test": a}) == {"file": "E2-HF-001.yaml", "updated": False}
    assert X.api_save({"test": a})["updated"] is True
    lst = X.api_tests({})
    assert [x["test"]["test_id"] for x in lst["tests"]] == ["E2-HF-001"] and lst["next_id"] == "E2-HF-002"
    assert lst["tests"][0]["saved"] and lst["tests"][0]["row"]["r_eta_cstar"] > 0.8
    out = X.api_analyse({"test": b})          # the saved test is the previous one and the drift reference
    assert out["last"]["test_id"] == "E2-HF-001" and "eta_cstar" in out["reference"]
    X.api_delete({"test_id": "E2-HF-001"})
    assert X.api_tests({})["tests"] == [] and not list(store.glob("*.yaml"))


def test_demo_points_and_empty_ids_are_not_saved(store):
    with pytest.raises(ValueError):
        X.api_save({"test": demo() | {"test_id": "DEMO-HF-003"}})
    with pytest.raises(ValueError):
        X.api_save({"test": demo() | {"test_id": "  "}})
    assert not list(store.glob("*.yaml"))


def test_jacket_job_and_line_job_are_json(store):
    t = X.normalise(demo())
    W = X.job_jacket({k: t[k] for k in ("hardware", "pressures_gauge", "inputs")}, C.ETA_CSTAR, C.ETA_CF_VAC, False, 60, None)
    json.dumps(W, allow_nan=False)
    assert 0.3 < W["Q_ratio"] < 3 and "model" not in W
    line = X.job_line("E2-REG-1", 0.2, C.T_OX_NOM, C.P_AMB, None, C.ETA_CSTAR, C.ETA_CF_VAC, 2.0, 4.0)
    json.dumps(line, allow_nan=False)
    assert line[0]["OF"] == pytest.approx(2.0) and line[-1]["pc_model"] > line[0]["pc_model"]


def test_http_round_trip(store):
    srv = X.make_server(port=8650)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert b"E2 Test Explorer" in opener.open(base + "/").read()
        meta = json.loads(opener.open(base + "/api/meta").read())
        assert {f["key"] for f in meta["fields"]} == set(H.INPUTS) and "E2-REG-1" in meta["hardware"]

        def post(headers):
            req = urllib.request.Request(base + "/api/analyse", json.dumps({"test": demo()}).encode(), headers)
            return opener.open(req)
        assert json.loads(post({}).read())["R"]["OF"] > 1
        with pytest.raises(urllib.error.HTTPError) as e:      # a page of another site cannot post to the server
            post({"Origin": "http://example.com"})
        assert e.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()
