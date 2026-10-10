"""
E2 Test Explorer (Streamlit): hot-fire test points against the h2cea models.

    pip install streamlit plotly pandas      (plus the h2cea requirements)
    streamlit run test_explorer.py           (run from h2cea/)

Hot fire   drop the DAQ CSV (or type the steady-state means), get eta c*, eta Cf, Isp,
           pc and thrust against the model at the measured flows, the nozzle separation
           check, injector stiffness and CdA, and the water-jacket heat with the Bartz
           factor that matches it. Save the point to hotfire_tests/.
Campaign   every saved test recomputed with the current model: table, trends,
           scatter, side-by-side comparison, CSV export.
Model      throttle line of the hardware, and the CEA map export for hopper-test-lab.

All numbers come from h2cea/hotfire.py, which has no UI code.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from h2cea import config as C
from h2cea import hotfire as H
from h2cea.gasdyn import mach_from_area

st.set_page_config(page_title="E2 Test Explorer", page_icon="🔥", layout="wide")

# ----------------------------------------------------------------------------------------
# Look
# ----------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURFACE = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#151514"
MEAS, MODEL, IDEAL, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"
OTHER = "#5d5c58"
STATUS = {"good": ("#0ca30c", "✓", "OK"), "warn": ("#fab219", "!", "Check"),
          "crit": ("#d03b3b", "✕", "Off"), "info": ("#3987e5", "i", "Note")}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

st.markdown(f"""
<style>
  .block-container {{padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1500px;}}
  [data-testid="stSidebar"] {{border-right: 1px solid rgba(255,255,255,.06);}}
  .e2-hero {{display:flex; align-items:center; gap:16px; margin: 0 0 6px 0;}}
  .e2-hero h1 {{font-size: 1.55rem; font-weight: 750; letter-spacing: .14em; margin: 0; padding: 0;
               background: linear-gradient(90deg, #ffd2a8 0%, #ff7a2f 45%, #f0efec 100%);
               -webkit-background-clip: text; background-clip: text; color: transparent;}}
  .e2-hero .sub {{color:{MUTED}; font-size:.82rem; letter-spacing:.04em;}}
  .chips {{display:flex; flex-wrap:wrap; gap:6px; margin: 2px 0 14px 0;}}
  .chip {{font-size:.72rem; color:{INK2}; border:1px solid rgba(255,255,255,.10); border-radius:999px;
          padding:2px 10px; background:rgba(255,255,255,.02);}}
  .chip b {{color:{INK}; font-weight:600;}}
  .kpi-grid {{display:grid; grid-template-columns: repeat(auto-fill, minmax(178px, 1fr)); gap:10px; margin-bottom: 10px;}}
  .kpi {{background:{SURFACE}; border:1px solid rgba(255,255,255,.07); border-radius:12px;
         padding:11px 14px 10px 16px; position:relative; overflow:hidden;}}
  .kpi::before {{content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--s, #383835);}}
  .kpi .lbl {{font-size:.74rem; letter-spacing:.03em; color:{MUTED}; display:flex;
              justify-content:space-between; gap:6px;}}
  .kpi .tag {{font-size:.64rem; letter-spacing:.04em; color: var(--s); text-transform:none;}}
  .kpi .val {{font-size:1.65rem; font-weight:650; color:{INK}; line-height:1.25; margin-top:2px;}}
  .kpi .val .u {{font-size:.8rem; font-weight:500; color:{INK2}; margin-left:3px;}}
  .kpi .sub {{font-size:.76rem; color:{INK2}; margin-top:1px;}}
  .kpi .last {{font-size:.7rem; color:{MUTED}; margin-top:3px;}}
  .obs {{display:flex; gap:10px; align-items:flex-start; padding:7px 10px; border-radius:9px;
         background:rgba(255,255,255,.025); margin-bottom:5px; font-size:.86rem; color:{INK2};}}
  .obs .ic {{flex:0 0 auto; width:20px; height:20px; border-radius:50%; display:flex; align-items:center;
             justify-content:center; font-size:.72rem; font-weight:800; color:#0d0d0d; background:var(--s);
             margin-top:1px;}}
  .obs .lv {{color: var(--s); font-weight:650; font-size:.7rem; letter-spacing:.06em; text-transform:uppercase;
             margin-right:4px;}}
  .sec {{font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; margin: 14px 0 6px 0;
         border-bottom: 1px solid rgba(255,255,255,.06); padding-bottom:4px;}}
  .demo-banner {{font-size:.8rem; color:#0d0d0d; background:#fab219; border-radius:8px; padding:4px 10px;
                 display:inline-block; font-weight:600; margin-bottom:8px;}}
  div[data-testid="stNumberInput"] label p, div[data-testid="stTextInput"] label p {{font-size:.8rem; color:{INK2};}}
</style>
""", unsafe_allow_html=True)

NOZZLE_SVG = """<svg width="38" height="38" viewBox="0 0 38 38" aria-hidden="true">
<defs><linearGradient id="fl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffe0bf"/>
<stop offset=".45" stop-color="#ff7a2f"/><stop offset="1" stop-color="#ff7a2f" stop-opacity="0"/></linearGradient></defs>
<path d="M11 2h16v7c0 3-4 4-4 7l6 9H9l6-9c0-3-4-4-4-7z" fill="#2a2a28" stroke="#c3c2b7" stroke-width="1.2"/>
<path d="M10 26h18l-3 11h-12z" fill="url(#fl)"/></svg>"""


def theme(fig, h=340, legend=True, hover="closest", subplots=False):
    """Dark chart style. subplots=True leaves room for the legend above the subplot titles."""
    fig.update_layout(
        height=h, margin=dict(l=10, r=10, t=(84 if subplots else 64) if legend else 36, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK2, size=12), hovermode=hover,
        hoverlabel=dict(bgcolor="#222221", bordercolor="#383835", font=dict(color=INK, family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.07 if subplots else 1.01, xanchor="left", x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)") if legend else None, showlegend=legend,
        title=dict(font=dict(size=13, color=INK), x=0.005, xanchor="left", yref="container", y=0.985, yanchor="top"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#383835", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#383835", tickfont=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"scale": 2}})


def fmt(key, val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "–"
    return H.RESULT_LABELS[key][2].format(val)


# ----------------------------------------------------------------------------------------
# Cached model calls
# ----------------------------------------------------------------------------------------
def _key(test):
    return json.dumps({k: test.get(k) for k in ("hardware", "pressures_gauge", "inputs")}, sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=500)
def analyse_cached(test_json: str, eta_c: float, eta_f: float):
    t = H.new_test()
    t.update(json.loads(test_json))
    try:
        return H.analyse(t, eta_c, eta_f), None
    except Exception as e:    # shown in the UI, never crashes the page
        return None, str(e)


def analyse(test, eta_c, eta_f):
    return analyse_cached(_key(test), eta_c, eta_f)


@st.cache_data(show_spinner="Marching the water jacket and fitting the Bartz factor ...", max_entries=200)
def jacket_cached(test_json: str, eta_c: float, eta_f: float, fit: bool, n: int):
    t = H.new_test()
    t.update(json.loads(test_json))
    R, err = analyse_cached(test_json, eta_c, eta_f)
    if R is None:
        return None
    try:
        return H.water_jacket(t, R, n_stations=n, fit=fit)
    except Exception:
        return None


@st.cache_data(show_spinner="Throttle line ...", max_entries=50)
def line_cached(hw, mf, T_ox, pa, dt_mm, eta_c, eta_f, lo, hi):
    return pd.DataFrame(H.model_line(hw, mf, T_ox, pa, dt_mm, eta_c, eta_f, (lo, hi)))


@st.cache_data(show_spinner="Time-resolved c* ...", max_entries=20)
def trace_eta_cached(t, pc, mox, mf, hw, T_ox, pa, dt_mm):
    return pd.DataFrame(H.eta_cstar_trace(t, pc, mox, mf, hw, T_ox, pa, dt_mm))


@st.cache_data(show_spinner="Computing the CEA map (about 10 s) ...")
def cea_map_cached(eps, T_ox, pa):
    return json.dumps(H.cea_map(eps, T_ox, pa), indent=1)


@st.cache_data(show_spinner=False)
def demo_tests_cached():
    return H.demo_tests()


@st.cache_data(show_spinner=False)
def demo_trace_csv():
    return H.demo_trace().to_csv(index=False).encode()


# ----------------------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------------------
SS = st.session_state
META_DEFAULTS = {"m_test_id": "", "m_date": dt.date.today(), "m_hardware": "E2-REG-1", "m_outcome": "pass",
                 "m_gauge": False, "m_notes": "", "m_source": ""}


def next_id(ids):
    nums = [(m.group(1), int(m.group(2)), len(m.group(2))) for i in ids
            if (m := re.match(r"^(.*?)(\d+)$", i)) and not i.startswith("DEMO")]
    if not nums:
        return "E2-HF-001"
    pre, n, w = max(nums, key=lambda x: x[1])
    return f"{pre}{n + 1:0{w}d}"


def saved_tests():
    if "saved" not in SS:
        SS.saved = H.load_tests()
    return SS.saved


def reset_form(keep_hw=True):
    hw = SS.get("m_hardware", "E2-REG-1")
    for k, v in META_DEFAULTS.items():
        SS[k] = v
    if keep_hw:
        SS.m_hardware = hw
    SS.m_test_id = next_id([t["test_id"] for t in saved_tests()])
    for f in H.INPUT_FIELDS:
        SS[f"in_{f.key}"] = f.default
    SS.pop("trace_sig", None)


def load_into_form(test):
    SS.m_test_id = test["test_id"]
    try:
        SS.m_date = dt.date.fromisoformat(str(test.get("date"))[:10])
    except ValueError:
        SS.m_date = dt.date.today()
    SS.m_hardware = test.get("hardware", "E2-REG-1")
    SS.m_outcome = test.get("outcome", "pass")
    SS.m_gauge = bool(test.get("pressures_gauge"))
    SS.m_notes = test.get("notes", "") or ""
    SS.m_source = test.get("source", "") or ""
    for f in H.INPUT_FIELDS:
        v = test["inputs"].get(f.key)
        SS[f"in_{f.key}"] = None if v is None else float(v)


def form_test():
    t = H.new_test(test_id=SS.m_test_id.strip(), date=str(SS.m_date), hardware=SS.m_hardware,
                   outcome=SS.m_outcome, pressures_gauge=SS.m_gauge, notes=SS.m_notes, source=SS.m_source)
    for f in H.INPUT_FIELDS:
        t["inputs"][f.key] = SS.get(f"in_{f.key}")
    return t


if "m_test_id" not in SS:
    reset_form()


def _open_selected():
    sel = SS.get("open_sel")
    pool = {t["test_id"]: t for t in saved_tests() + (demo_tests_cached() if SS.get("demo_on") else [])}
    if sel in pool:
        load_into_form(pool[sel])
        SS.pop("trace_sig", None)
        SS.trace = None


# ----------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------
HW = H.hardware_presets()
with st.sidebar:
    st.markdown(f"<div class='e2-hero'>{NOZZLE_SVG}<div><div style='font-weight:700;letter-spacing:.12em;"
                f"color:{INK}'>E2</div><div class='sub'>Test Explorer</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>Model assumptions</div>", unsafe_allow_html=True)
    eta_c_model = st.number_input("η c* (model)", 0.70, 1.05, float(C.ETA_CSTAR), 0.005, format="%.3f",
                                  help="Baseline value. The model pc and thrust at the measured flows use it.")
    eta_f_model = st.number_input("η Cf,vac (model)", 0.70, 1.05, float(C.ETA_CF_VAC), 0.005, format="%.3f",
                                  help="Applied to the ideal vacuum Cf; the -pa·Ae term carries no loss (as the baseline).")
    st.markdown("<div class='sec'>Tests</div>", unsafe_allow_html=True)
    demo_on = st.toggle("Show demo campaign", key="demo_on",
                        help="Six synthetic test points built from the model, to try the app. Never saved.")
    pool_ids = [t["test_id"] for t in saved_tests()] + ([t["test_id"] for t in demo_tests_cached()] if demo_on else [])
    st.selectbox("Open a test", [""] + pool_ids[::-1], key="open_sel", on_change=_open_selected,
                 format_func=lambda x: x or "—")
    c1, c2 = st.columns(2)
    c1.button("New test", on_click=reset_form, width="stretch")
    if c2.button("Reload", width="stretch", help="Re-read hotfire_tests/ (after a git pull)."):
        SS.pop("saved", None)
        st.rerun()
    st.caption(f"{len(saved_tests())} saved in `h2cea/hotfire_tests/`")
    bl = C.BASELINE["meta"]
    st.markdown("<div class='sec'>Baseline</div>", unsafe_allow_html=True)
    st.caption(f"h2_baseline rev **{bl.get('revision')}** ({bl.get('date')}) · F rated {C.F_MAX:.0f} N · "
               f"pc {C.PC_MAX:g} bar · fuel {C.MDOT_FUEL:g} kg/s · CEA liquid N2O card")

# ----------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------
st.markdown(f"<div class='e2-hero'>{NOZZLE_SVG}<div><h1>E2 TEST EXPLORER</h1>"
            f"<div class='sub'>Hot-fire points against CEA, the h2cea engine model and the E2-REG-1 jacket</div>"
            f"</div></div>", unsafe_allow_html=True)
hw = HW[SS.m_hardware]
st.markdown(
    "<div class='chips'>"
    f"<span class='chip'>Hardware <b>{hw.key}</b></span>"
    f"<span class='chip'>Throat <b>{hw.throat_d_mm:.2f} mm</b> · ε <b>{hw.eps:.2f}</b></span>"
    f"<span class='chip'>Model η c* <b>{eta_c_model:.3f}</b> · η Cf,vac <b>{eta_f_model:.3f}</b></span>"
    f"<span class='chip'>Baseline rev <b>{C.BASELINE['meta'].get('revision')}</b></span>"
    + ("<span class='chip' style='border-color:#fab219;color:#fab219'>Demo campaign on</span>" if demo_on else "")
    + "</div>", unsafe_allow_html=True)

tab_fire, tab_camp, tab_model = st.tabs(["🔥  Hot fire", "📊  Campaign", "🧮  Model"])


# ----------------------------------------------------------------------------------------
# Helpers for the campaign pool
# ----------------------------------------------------------------------------------------
def pool_tests(include_current=None):
    tests = list(saved_tests()) + (list(demo_tests_cached()) if demo_on else [])
    if include_current is not None:
        tests = [t for t in tests if t["test_id"] != include_current["test_id"]] + [dict(include_current, _current=True)]
    return tests


def reference_for(test, hw_key):
    """Mean results of the other saved (or demo) tests on the same hardware."""
    rows = []
    for t in pool_tests():
        if t["test_id"] == test["test_id"] or t.get("hardware") != hw_key:
            continue
        R, _ = analyse(t, eta_c_model, eta_f_model)
        if R:
            rows.append(R)
    if not rows:
        return None, None
    keys = ("eta_cstar", "eta_cf_vac", "CdA_ox_mm2", "CdA_fuel_mm2", "isp_meas_s", "pc_dev_pct")
    ref = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}
    return ref, rows


def last_test(test, hw_key):
    cands = [t for t in pool_tests() if t["test_id"] != test["test_id"] and t.get("hardware") == hw_key
             and str(t.get("date", "")) <= str(test.get("date", "9999"))]
    if not cands:
        return None, None
    t = sorted(cands, key=lambda t: (str(t.get("date")), t["test_id"]))[-1]
    R, _ = analyse(t, eta_c_model, eta_f_model)
    return t, R


# ========================================================================================
# HOT FIRE
# ========================================================================================
UNIT_SCALE = {"g/s": 1e-3, "kg/s": 1.0}


def unit_of(col):
    m = re.search(r"\[([^\]]+)\]|\(([^)]+)\)", str(col))
    return (m.group(1) or m.group(2)).strip() if m else ""


def conversion(key, col):
    """(scale, offset, note) from the unit in the column name."""
    u = unit_of(col).lower()
    if key.endswith("_kg_s") and u in ("g/s", "gps"):
        return 1e-3, 0.0, "g/s → kg/s"
    if key.endswith("_C") and u == "k":
        return 1.0, -273.15, "K → °C"
    if key.endswith("_bar") and u in ("psi", "psia", "psig"):
        return 0.0689476, 0.0, "psi → bar"
    if key.endswith("_bar") and u in ("kpa",):
        return 0.01, 0.0, "kPa → bar"
    if key.endswith("_bar") and u in ("mpa",):
        return 10.0, 0.0, "MPa → bar"
    return 1.0, 0.0, ""


def apply_means(means, t0, t1, src, gauge):
    for k, v in means.items():
        if k in H.INPUTS and v is not None:
            f = H.INPUTS[k]
            SS[f"in_{k}"] = float(round(v, 4 if f.step < 0.01 else 3))
    SS.m_source = f"{src}, {t0:.2f}–{t1:.2f} s"
    SS.m_gauge = gauge


with tab_fire:
    # ---------------------------------------------------------------- DAQ trace
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown("**① DAQ trace** <span style='color:#898781;font-size:.85rem'>— optional: drop the export, "
                    "the steady window is found and averaged for you</span>", unsafe_allow_html=True)
        if h2.button("Load demo trace", width="stretch", help="Synthetic 6 s burn from the model, for trying the app."):
            SS.trace = {"name": "DEMO_trace.csv", "data": demo_trace_csv()}
            SS.pop("trace_sig", None)
        up = st.file_uploader("DAQ CSV", type=["csv", "txt"], label_visibility="collapsed")
        if up is not None:
            data = up.getvalue()
            if SS.get("last_upload") != (up.name, len(data), hash(data)):   # only a new upload replaces the trace
                SS.last_upload = (up.name, len(data), hash(data))
                SS.trace = {"name": up.name, "data": data}
                SS.pop("trace_sig", None)
        tr = SS.get("trace")
        if tr:
            try:
                raw = pd.read_csv(io.BytesIO(tr["data"]), sep=None, engine="python")
            except Exception as e:
                st.error(f"Could not read {tr['name']}: {e}")
                raw = None
            if raw is not None:
                num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
                guess = H.guess_columns(num)
                sig = (tr["name"], len(tr["data"]))
                if SS.get("trace_sig") != sig:          # new file: take the guesses
                    for k in H.CHANNEL_HINTS:
                        SS[f"map_{k}"] = guess.get(k, "—")
                with st.expander("Channel mapping", expanded=not {"time", "pc_bar", "ox_flow_kg_s"} <= set(guess)):
                    keys = list(H.CHANNEL_HINTS)
                    cols = st.columns(4)
                    for i, k in enumerate(keys):
                        lab = "Time [s]" if k == "time" else f"{H.INPUTS[k].label} [{H.INPUTS[k].unit}]"
                        cols[i % 4].selectbox(lab, ["—"] + num, key=f"map_{k}")
                    gauge_guess = any(re.search(r"barg|\(g\)|gauge|psig", str(SS.get(f"map_{k}")), re.I)
                                      for k in H.PRESSURE_KEYS)
                    tr_gauge = st.checkbox("Pressures in the file are gauge", value=gauge_guess, key=f"tg_{sig}")
                mapping = {k: SS[f"map_{k}"] for k in H.CHANNEL_HINTS if SS.get(f"map_{k}") not in (None, "—")}
                if not {"time", "pc_bar"} <= set(mapping):
                    st.warning("Map at least the time and the chamber pressure columns.")
                else:
                    df = raw.copy()
                    notes = []
                    for k, col in mapping.items():
                        if k == "time":
                            continue
                        s, o, n = conversion(k, col)
                        df[col] = df[col] * s + o
                        if n:
                            notes.append(f"{col}: {n}")
                    t = df[mapping["time"]].to_numpy(float)
                    pcs = df[mapping["pc_bar"]].to_numpy(float)
                    win = H.burn_window(t, pcs)
                    if win is None:
                        st.warning("No burn found in the chamber pressure trace.")
                    else:
                        b0, b1, s0, s1 = win
                        tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
                        if SS.get("trace_sig") != sig:
                            SS.win = (round(s0, 3), round(s1, 3))
                        w = st.slider("Steady-state window [s]", tmin, tmax, key="win", step=0.01,
                                      help=f"Auto: burn {b0:.2f}–{b1:.2f} s ({b1 - b0:.2f} s), "
                                           f"start transient and the last 10 % removed.")
                        means = H.window_means(df, mapping, w[0], w[1])
                        means["duration_s"] = round(b1 - b0, 2)
                        if SS.get("trace_sig") != sig:   # new file: fill the form once
                            apply_means(means, w[0], w[1], tr["name"], tr_gauge)
                            SS.trace_sig = sig
                            if not SS.m_test_id or SS.m_test_id in [x["test_id"] for x in saved_tests()]:
                                stem = re.sub(r"(_export)?\.(csv|txt)$", "", tr["name"], flags=re.I)
                                SS.m_test_id = stem if not stem.startswith("DEMO_") else "DEMO-TRACE"
                        rows_k = [k for k in ("pc_bar", "thrust_N", "ox_flow_kg_s") if k in mapping]
                        extra = [("Flows", [k for k in ("ox_flow_kg_s", "fuel_flow_kg_s") if k in mapping]),
                                 ("Injector manifolds", [k for k in ("p_ox_inj_bar", "p_fuel_inj_bar") if k in mapping]),
                                 ("Water temperatures", [k for k in ("water_T_in_C", "water_T_out_C") if k in mapping])]
                        panels = [("Chamber pressure [bar]", ["pc_bar"])]
                        if "thrust_N" in mapping:
                            panels.append(("Thrust [N]", ["thrust_N"]))
                        panels += [(f"{n} [{H.INPUTS[ks[0]].unit}]", ks) for n, ks in extra if ks]
                        colors = [MEAS, MODEL]
                        short = {"ox_flow_kg_s": "ox", "fuel_flow_kg_s": "fuel", "p_ox_inj_bar": "ox",
                                 "p_fuel_inj_bar": "fuel", "water_T_in_C": "in", "water_T_out_C": "out"}

                        def ptitle(name, ks):    # series named in the panel title, in their colour
                            if len(ks) == 1:
                                return f"<b>{name}</b>"
                            return f"<b>{name}</b>   " + "  ".join(
                                f"<span style='color:{colors[j]}'>━ {short.get(k, k)}</span>" for j, k in enumerate(ks))
                        fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                            subplot_titles=[ptitle(*p) for p in panels])
                        step = max(1, len(df) // 4000)
                        for i, (_, ks) in enumerate(panels, start=1):
                            for j, k in enumerate(ks):
                                fig.add_trace(go.Scattergl(
                                    x=df[mapping["time"]].iloc[::step], y=df[mapping[k]].iloc[::step],
                                    name=H.INPUTS[k].label, line=dict(width=1.4, color=colors[j % 2])), row=i, col=1)
                        for i in range(1, len(panels) + 1):
                            fig.add_vrect(x0=w[0], x1=w[1], fillcolor=MEAS, opacity=0.13, line_width=0, row=i, col=1)
                            fig.add_vline(x=b0, line=dict(color=MUTED, dash="dot", width=1), row=i, col=1)
                            fig.add_vline(x=b1, line=dict(color=MUTED, dash="dot", width=1), row=i, col=1)
                        theme(fig, h=70 + 112 * len(panels), legend=False, hover="x unified")
                        fig.update_layout(margin=dict(t=24))
                        fig.update_annotations(font=dict(size=11, color=INK2), x=0, xanchor="left")
                        fig.update_xaxes(title_text="time [s]", row=len(panels), col=1)
                        show(fig)
                        a1, a2, a3 = st.columns([2, 1, 1])
                        a1.caption(f"Window {w[0]:.2f}–{w[1]:.2f} s · {means['_n']} samples · burn {b1 - b0:.2f} s"
                                   + (f" · {', '.join(notes)}" if notes else ""))
                        a2.button("Use window means ↓", type="primary", width="stretch", on_click=apply_means,
                                  args=(means, w[0], w[1], tr["name"], tr_gauge))
                        show_eta = a3.toggle("η c* over time", value=False)
                        if show_eta and {"ox_flow_kg_s", "fuel_flow_kg_s"} <= set(mapping):
                            sel = df[(df[mapping["time"]] >= b0) & (df[mapping["time"]] <= b1)]
                            pa_tr = SS.get("in_p_amb_bar") or C.P_AMB
                            add = pa_tr if tr_gauge else 0.0
                            T_ox = (SS.get("in_ox_temp_C") if SS.get("in_ox_temp_C") is not None else 20.0) + 273.15
                            te = trace_eta_cached(tuple(sel[mapping["time"]]), tuple(sel[mapping["pc_bar"]] + add),
                                                  tuple(sel[mapping["ox_flow_kg_s"]]), tuple(sel[mapping["fuel_flow_kg_s"]]),
                                                  SS.m_hardware, T_ox, pa_tr, SS.get("in_throat_d_mm"))
                            if len(te):
                                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                                    subplot_titles=["η c* (measured c* / CEA c*)", "O/F"])
                                fig.add_trace(go.Scatter(x=te["t"], y=te["eta_cstar"], name="η c*",
                                                         line=dict(color=MEAS, width=1.6)), row=1, col=1)
                                fig.add_hline(y=eta_c_model, line=dict(color=MODEL, dash="dash", width=1.2), row=1, col=1,
                                              annotation_text=f"model {eta_c_model:.3f}", annotation_font_color=INK2)
                                fig.add_trace(go.Scatter(x=te["t"], y=te["OF"], name="O/F",
                                                         line=dict(color=VIOLET, width=1.6)), row=2, col=1)
                                for r in (1, 2):
                                    fig.add_vrect(x0=w[0], x1=w[1], fillcolor=MEAS, opacity=0.10, line_width=0, row=r, col=1)
                                theme(fig, h=360, legend=False, hover="x unified")
                                fig.update_annotations(font=dict(size=11, color=INK2))
                                fig.update_xaxes(title_text="time [s]", row=2, col=1)
                                show(fig)
                                st.caption("Flow-meter lag makes the start and shutdown values meaningless; read the plateau.")
        else:
            st.caption("Long or wide CSV with a time column. Columns are matched by name "
                       "(pc / PT-CC, FM-OX, FM-FU, LC / thrust, PT-OX-INJ, TT-W-IN …); units in brackets "
                       "(g/s, K, psi, kPa) are converted.")

    # ---------------------------------------------------------------- inputs | results
    left, right = st.columns([1, 2.35], gap="large")
    with left:
        st.markdown("**② Test point**")
        a, b = st.columns([1.4, 1])
        a.text_input("Test ID", key="m_test_id")
        b.date_input("Date", key="m_date")
        a, b = st.columns([1.4, 1])
        a.selectbox("Hardware", list(HW), key="m_hardware", format_func=lambda k: HW[k].key,
                    help="\n".join(f"{h.key}: {h.label}" for h in HW.values()))
        b.selectbox("Outcome", list(H.OUTCOMES), key="m_outcome", format_func=H.OUTCOMES.get)

        def num(field_key, col):
            f = H.INPUTS[field_key]
            dec = int(f.fmt.split(".")[1].rstrip("f}")) if "." in f.fmt else 0
            col.number_input(f"{f.label} [{f.unit}]", key=f"in_{field_key}", step=f.step, format=f"%.{dec}f",
                             help=f.help or None, placeholder="—")

        st.markdown("<div class='sec'>Operating point (steady state)</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        for i, k in enumerate(["pc_bar", "thrust_N", "ox_flow_kg_s", "fuel_flow_kg_s", "duration_s", "p_amb_bar"]):
            num(k, (a, b)[i % 2])
        st.toggle("Pressures are gauge (adds ambient)", key="m_gauge")
        with st.expander("Injector, propellant, throat", expanded=any(SS.get(f"in_{k}") is not None
                                                                       for k in ("p_ox_inj_bar", "p_fuel_inj_bar"))):
            a, b = st.columns(2)
            for i, k in enumerate(["p_ox_inj_bar", "p_fuel_inj_bar", "ox_temp_C", "throat_d_mm"]):
                num(k, (a, b)[i % 2])
        with st.expander("Water jacket (calorimetry)", expanded=SS.get("in_water_flow_kg_s") is not None):
            a, b = st.columns(2)
            for i, k in enumerate(["water_flow_kg_s", "water_p_in_bar", "water_T_in_C", "water_p_out_bar", "water_T_out_C"]):
                num(k, (a, b)[i % 2])
        st.text_area("Notes", key="m_notes", height=80, placeholder="What happened, what to change next …")
        if SS.m_source:
            st.caption(f"Source: {SS.m_source}")

    test = form_test()
    R, err = analyse(test, eta_c_model, eta_f_model)

    with left:
        is_demo = test["test_id"].upper().startswith("DEMO")
        c1, c2 = st.columns(2)
        if c1.button("💾 Save test", type="primary", width="stretch", disabled=R is None or is_demo or not test["test_id"],
                     help="Writes hotfire_tests/<test id>.yaml (overwrites a test with the same ID)." +
                          (" Demo points are not saved; give it a real test ID." if is_demo else "")):
            existed = any(t["test_id"] == test["test_id"] for t in saved_tests())
            path = H.save_test(test)
            SS.pop("saved", None)
            st.toast(f"{'Updated' if existed else 'Saved'} {path.name}", icon="✅")
        c2.download_button("⬇ YAML", yaml.safe_dump({k: v for k, v in test.items() if not k.startswith("_")},
                                                      sort_keys=False, allow_unicode=True),
                           file_name=f"{test['test_id'] or 'test'}.yaml", width="stretch")

    with right:
        if R is None:
            st.markdown("**③ Results**")
            st.info(f"Enter chamber pressure, oxidiser and fuel flow (or drop a DAQ trace) to see the results. ({err})")
        else:
            W = None
            if "Q_meas_kW" in R and hw.regen_model:
                W = jacket_cached(_key(test), eta_c_model, eta_f_model, True, 200)
            ref, _ = reference_for(test, test["hardware"])
            lt, LR = last_test(test, test["hardware"])
            obs = H.observations(R, W, eta_c_model, eta_f_model, reference=ref)

            hdr = f"**③ Results — {test['test_id'] or 'unnamed'}**"
            if is_demo:
                hdr += "  <span class='demo-banner'>DEMO DATA</span>"
            st.markdown(hdr, unsafe_allow_html=True)

            def lvl_of(text_start):
                for lv, tx in obs:
                    if tx.startswith(text_start):
                        return lv
                return None

            def tile(label, val, unit, sub, status=None, last=None):
                col, _, tag = STATUS.get(status, ("#383835", "", ""))
                return (f"<div class='kpi' style='--s:{col}'><div class='lbl'><span>{label}</span>"
                        f"<span class='tag'>{tag}</span></div>"
                        f"<div class='val'>{val}<span class='u'>{unit}</span></div><div class='sub'>{sub}</div>"
                        + (f"<div class='last'>{last}</div>" if last else "") + "</div>")

            def vs_last(key, nd=3, pct=False):
                if not LR or key not in LR or key not in R:
                    return None
                d = R[key] - LR[key]
                return (f"vs {lt['test_id']}: {100 * d / LR[key]:+.1f} %" if pct else
                        f"vs {lt['test_id']}: {d:+.{nd}f}")

            tiles = [
                tile("η c*", f"{R['eta_cstar']:.3f}", "",
                     f"model {eta_c_model:.3f} · c* {R['cstar_meas_m_s']:.0f} / {R['cstar_ideal_m_s']:.0f} m/s",
                     lvl_of("η c*"), vs_last("eta_cstar")),
                tile("Chamber pressure", f"{R['pc_abs_bar']:.2f}", "bar",
                     f"model {R['pc_model_bar']:.2f} · {R['pc_dev_pct']:+.1f} %", lvl_of("η c*"), vs_last("pc_abs_bar", 2)),
            ]
            if "thrust_N" in R:
                tiles += [
                    tile("Thrust", f"{R['thrust_N']:.0f}", "N",
                         f"model {R['thrust_model_N']:.0f} · {R['thrust_dev_pct']:+.1f} % · {R['throttle_pct']:.0f} % rated",
                         lvl_of("η Cf"), vs_last("thrust_N", 0)),
                    tile("Isp (sea level)", f"{R['isp_meas_s']:.1f}", "s",
                         f"model {R['isp_model_s']:.1f} · ideal {R['isp_ideal_s']:.1f}", lvl_of("η Cf"),
                         vs_last("isp_meas_s", 1)),
                    tile("η Cf,vac", f"{R['eta_cf_vac']:.3f}", "",
                         f"model {eta_f_model:.3f} · Cf {R['cf_meas']:.3f} / {R['cf_ideal']:.3f}", lvl_of("η Cf"),
                         vs_last("eta_cf_vac")),
                ]
            else:
                tiles.append(tile("Thrust (model)", f"{R['thrust_model_N']:.0f}", "N",
                                  f"{R['throttle_pct']:.0f} % rated · no load cell value", "info"))
            tiles.append(tile("O/F", f"{R['OF']:.2f}", "",
                              f"ṁ {R['mdot_kg_s']:.3f} kg/s · Tc {R['Tc_K']:.0f} K", None, vs_last("OF", 2)))
            sep = R["summerfield_sep"] or R.get("x_sep_mm") is not None
            tiles.append(tile("Nozzle pe / pa", f"{R['pe_over_pa']:.2f}", "",
                              (f"separated at x {R['x_sep_mm']:.0f} mm" if R.get("x_sep_mm") is not None else
                               "separated (Summerfield)" if R["summerfield_sep"] else
                               f"attached · pe {R['pe_bar']:.2f} bar"), "warn" if sep else "good"))
            if "stiff_ox_pct" in R or "stiff_fuel_pct" in R:
                s_ox, s_fu = R.get("stiff_ox_pct"), R.get("stiff_fuel_pct")
                worst = min(x for x in (s_ox, s_fu) if x is not None)
                cda = " · ".join(f"{n} CdA {R[k]:.2f}" for k, n in (("CdA_ox_mm2", "ox"), ("CdA_fuel_mm2", "fuel"))
                                 if k in R)
                tiles.append(tile("Injector stiffness", f"{worst:.0f}", "% pc (min)",
                                  " · ".join(x for x in (f"ox {s_ox:.0f} %" if s_ox is not None else "",
                                                         f"fuel {s_fu:.0f} %" if s_fu is not None else "") if x),
                                  "good" if worst >= 100 * H.STIFFNESS_MIN else "warn",
                                  f"{cda} mm²" + (f" · ox {vs_last('CdA_ox_mm2', pct=True)}" if vs_last("CdA_ox_mm2") else "")))
            if W:
                tiles.append(tile("Jacket heat", f"{R['Q_meas_kW']:.0f}", "kW",
                                  f"model {W['Q_model_kW']:.0f} · Bartz fit ×{W.get('bartz_fit', float('nan')):.2f}",
                                  "info", f"hot wall {W.get('Twg_max_fit_K', float('nan')):.0f} K (limit {C.T_WG_LIMIT:.0f})"))
            elif "Q_meas_kW" in R:
                tiles.append(tile("Jacket heat", f"{R['Q_meas_kW']:.0f}", "kW",
                                  f"{R['q_per_ox_kJ_kg']:.0f} kJ per kg N2O", "info"))
            st.markdown("<div class='kpi-grid'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)

            # observations
            order = {"crit": 0, "warn": 1, "good": 2, "info": 3}
            html = ""
            for lv, tx in sorted(obs, key=lambda o: order[o[0]]):
                col, icon, tag = STATUS[lv]
                html += (f"<div class='obs' style='--s:{col}'><div class='ic'>{icon}</div>"
                         f"<div><span class='lv'>{tag}</span>{tx}</div></div>")
            st.markdown(html, unsafe_allow_html=True)

            # charts
            pv = st.segmented_control("View", ["Engine map", "Nozzle", "Jacket", "Injector", "Summary"],
                                      default="Engine map", key="view", label_visibility="collapsed") or "Engine map"
            pool = [t for t in pool_tests() if t["test_id"] != test["test_id"] and t.get("hardware") == test["hardware"]]
            pool_R = [(t, analyse(t, eta_c_model, eta_f_model)[0]) for t in pool]
            pool_R = [(t, r) for t, r in pool_R if r]

            if pv == "Engine map":
                ycho = st.segmented_control("y", ["Chamber pressure", "Thrust", "c*", "Isp"], default="Chamber pressure",
                                            key="ymap", label_visibility="collapsed") or "Chamber pressure"
                ymap = {"Chamber pressure": ("pc", "pc_abs_bar", "pc [bar]"), "Thrust": ("thrust", "thrust_N", "thrust [N]"),
                        "c*": ("cstar", "cstar_meas_m_s", "c* [m/s]"), "Isp": ("isp", "isp_meas_s", "Isp sea level [s]")}
                lk, rk, ylab = ymap[ycho]
                ofs = [R["OF"]] + [r["OF"] for _, r in pool_R]
                lo, hi = max(1.2, min(ofs) - 0.4), min(7.5, max(ofs) + 0.4)
                ln = line_cached(test["hardware"], float(test["inputs"]["fuel_flow_kg_s"]), R["T_ox_K"], R["pa_bar"],
                                 test["inputs"].get("throat_d_mm"), eta_c_model, eta_f_model, round(lo, 1), round(hi, 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ln["OF"], y=ln[f"{lk}_ideal"], name="CEA ideal",
                                         line=dict(color=IDEAL, width=2, dash="dot"),
                                         hovertemplate="O/F %{x:.2f}<br>ideal %{y:.4g}<extra></extra>"))
                fig.add_trace(go.Scatter(x=ln["OF"], y=ln[f"{lk}_model"], name=f"model (η c* {eta_c_model:.3f})",
                                         line=dict(color=MODEL, width=2),
                                         hovertemplate="O/F %{x:.2f}<br>model %{y:.4g}<extra></extra>"))
                pts = [(t["test_id"], r["OF"], r.get(rk)) for t, r in pool_R if r.get(rk) is not None]
                if pts:
                    fig.add_trace(go.Scatter(x=[p[1] for p in pts], y=[p[2] for p in pts], mode="markers",
                                             name="other tests", text=[p[0] for p in pts],
                                             marker=dict(size=10, color=OTHER, line=dict(color=SURFACE, width=2)),
                                             hovertemplate="%{text}<br>O/F %{x:.2f}<br>%{y:.4g}<extra></extra>"))
                if R.get(rk) is not None:
                    fig.add_trace(go.Scatter(x=[R["OF"]], y=[R[rk]], mode="markers+text", name=test["test_id"] or "this test",
                                             text=[test["test_id"]], textposition="top center",
                                             textfont=dict(color=INK, size=11),
                                             marker=dict(size=15, color=MEAS, line=dict(color=INK, width=2)),
                                             hovertemplate="%{text}<br>O/F %{x:.2f}<br>%{y:.4g}<extra></extra>"))
                theme(fig, h=400)
                fig.update_layout(title=f"{ycho} along the throttle line (fuel {test['inputs']['fuel_flow_kg_s']:.3f} kg/s)",
                                  xaxis_title="O/F", yaxis_title=ylab)
                show(fig)
                st.caption("Lines: the hardware at the measured fuel flow, oxidiser throttled (fixed-fuel-flow line of "
                           "operating_line.py). Other tests at different fuel flows sit off these lines.")

            elif pv == "Nozzle":
                nz = R["nozzle"]
                cont = hw.contour(test["inputs"].get("throat_d_mm"))
                xs, rs, xt, Rt = cont["x"] * 1e3, cont["r"] * 1e3, cont["xt"] * 1e3, cont["Rt"] * 1e3
                g = nz["gamma"]
                xg = np.linspace(xs.min(), xs.max(), 320)
                rg = np.interp(xg, xs, rs)
                Mg = np.array([mach_from_area(max((r / Rt) ** 2, 1 + 1e-9), g, x > xt) for x, r in zip(xg, rg)])
                yy = np.linspace(-rs.max() * 1.05, rs.max() * 1.05, 140)
                Z = np.where(np.abs(yy)[:, None] <= rg[None, :], Mg[None, :], np.nan)
                if nz["x_sep_mm"] is not None:     # separated: show the separated core as dark
                    Z = np.where((xg[None, :] > nz["x_sep_mm"]) & (np.abs(yy)[:, None] > 0.55 * rg[None, :]), np.nan, Z)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.55, 0.45],
                                    subplot_titles=["Mach number (1D, γ matched to the CEA exit pressure)",
                                                    "Wall pressure vs Schmucker separation pressure [bar]"])
                fig.add_trace(go.Heatmap(x=xg, y=yy, z=Z, colorscale=[[0, "#2b1206"], [0.35, "#8a3410"], [0.7, "#ff7a2f"],
                                                                      [1, "#ffe3c4"]],
                                         colorbar=dict(title=dict(text="M", font=dict(color=INK2)), len=0.5, y=0.78,
                                                       tickfont=dict(color=MUTED), thickness=10),
                                         hovertemplate="x %{x:.1f} mm<br>M %{z:.2f}<extra></extra>"), row=1, col=1)
                for sgn in (1, -1):
                    fig.add_trace(go.Scatter(x=xs, y=sgn * rs, mode="lines", line=dict(color=INK2, width=2),
                                             showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=nz["x_mm"], y=nz["p_wall_bar"], name="wall pressure",
                                         line=dict(color=MEAS, width=2)), row=2, col=1)
                psep = np.where(np.isfinite(nz["p_sep_bar"]), nz["p_sep_bar"], np.nan)
                fig.add_trace(go.Scatter(x=nz["x_mm"], y=psep, name="separation (Schmucker)",
                                         line=dict(color="#fab219", width=2, dash="dash")), row=2, col=1)
                fig.add_hline(y=R["pa_bar"], line=dict(color=MUTED, dash="dot", width=1), row=2, col=1,
                              annotation_text=f"pa {R['pa_bar']:.3f}", annotation_font_color=MUTED)
                if nz["x_sep_mm"] is not None:
                    for r in (1, 2):
                        fig.add_vline(x=nz["x_sep_mm"], line=dict(color="#fab219", width=1.5, dash="dash"), row=r, col=1)
                fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1, showgrid=False)
                fig.update_yaxes(range=[0, max(np.nanmax(nz["p_wall_bar"][len(nz["p_wall_bar"]) // 3:]) * 1.6,
                                               R["pa_bar"] * 1.3)], row=2, col=1)
                theme(fig, h=490, subplots=True)
                fig.update_annotations(font=dict(size=11, color=INK2), x=0, xanchor="left")
                fig.update_xaxes(title_text="x from injector face [mm]", row=2, col=1)
                show(fig)
                st.caption(f"pe {R['pe_bar']:.3f} bar at ε {R['eps']:.2f}; Summerfield (pe < 0.4 pa, as the engine table): "
                           f"{'separated' if R['summerfield_sep'] else 'attached'}; Schmucker: "
                           f"{'separated at x ' + format(nz['x_sep_mm'], '.0f') + ' mm' if nz['x_sep_mm'] is not None else 'attached'}.")

            elif pv == "Jacket":
                if not W:
                    st.info("Enter the water flow and the inlet and outlet temperatures (Water jacket) for the "
                            "calorimetric heat and the model comparison. The jacket model is the E2-REG-1 as built.")
                else:
                    m, f_ = W["model"], W.get("fit")
                    xm = m["x"] * 1e3
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                        subplot_titles=["Wall temperature [K]", "Heat flux [MW/m²]"])
                    if f_ is not None:
                        fig.add_trace(go.Scatter(x=xm, y=f_["Twg"], name=f"hot wall, Bartz ×{W['bartz_fit']:.2f} (fit)",
                                                 line=dict(color=MODEL, width=2.2)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=xm, y=f_["Twc"], name="coolant-side wall (fit)",
                                                 line=dict(color=MEAS, width=2)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=xm, y=f_["q"] / 1e6, name="q (fit)", showlegend=False,
                                                 line=dict(color=MODEL, width=2)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=xm, y=m["Twg"], name="hot wall, Bartz ×1.00",
                                             line=dict(color=MODEL, width=1.4, dash="dot")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=xm, y=m["q"] / 1e6, name="q (×1.00)", showlegend=False,
                                             line=dict(color=MODEL, width=1.2, dash="dot")), row=2, col=1)
                    fig.add_hline(y=C.T_WG_LIMIT, line=dict(color="#d03b3b", dash="dash", width=1.2), row=1, col=1,
                                  annotation_text=f"IN718 limit {C.T_WG_LIMIT:.0f} K", annotation_font_color=INK2)
                    fig.add_vline(x=hw.contour()["xt"] * 1e3, line=dict(color=MUTED, dash="dot", width=1))
                    theme(fig, h=480, hover="x unified", subplots=True)
                    fig.update_annotations(font=dict(size=11, color=INK2), x=0, xanchor="left")
                    fig.update_xaxes(title_text="x from injector face [mm] (dotted: throat)", row=2, col=1)
                    show(fig)
                    jt = pd.DataFrame([
                        ["Jacket heat [kW]", R["Q_meas_kW"], W["Q_model_kW"], W.get("Q_fit_kW")],
                        ["Water outlet [°C]", test["inputs"]["water_T_out_C"], W["T_out_model_C"],
                         W["fit"]["T_out"] - 273.15 if W.get("fit") else None],
                        ["Jacket dp [bar]", R.get("dp_jacket_meas_bar"), W["dp_jacket_model_bar"],
                         float(W["fit"]["dp_bar"]) if W.get("fit") else None],
                        ["Hot wall max [K]", None, W["Twg_max_K"], W.get("Twg_max_fit_K")],
                    ], columns=["", "measured", "model ×1.00", f"model ×{W.get('bartz_fit', 1):.2f}"])
                    st.dataframe(jt.round(2), hide_index=True, width="stretch")
                    st.caption("regen.march (Bartz with CEA transport, Gnielinski, 200 stations) at the measured pc, O/F "
                               "and η c*. The fitted factor is the BARTZ_FACTOR in config.py that reproduces the "
                               "calorimetric heat; manifold losses and heat to the injector face are not in the model.")

            elif pv == "Injector":
                if "dp_ox_bar" not in R and "dp_fuel_bar" not in R:
                    st.info("Enter the ox and fuel injector inlet pressures (Injector, propellant, throat) for the "
                            "pressure drop, stiffness and CdA.")
                else:
                    cA, cB = st.columns([1, 1.4])
                    with cA:
                        names, vals = [], []
                        for c, n in (("ox", "Ox"), ("fuel", "Fuel")):
                            if f"stiff_{c}_pct" in R:
                                names.append(n)
                                vals.append(R[f"stiff_{c}_pct"])
                        fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker=dict(
                            color=[STATUS["good" if v >= 100 * H.STIFFNESS_MIN else "warn"][0] for v in vals],
                            cornerradius=4), text=[f"{v:.0f} %" for v in vals], textposition="outside",
                            textfont=dict(color=INK), hovertemplate="%{y}: %{x:.1f} % of pc<extra></extra>"))
                        fig.add_vline(x=100 * H.STIFFNESS_MIN, line=dict(color=MUTED, dash="dash", width=1.2),
                                      annotation_text=f"{100 * H.STIFFNESS_MIN:.0f} %", annotation_font_color=MUTED)
                        theme(fig, h=220, legend=False)
                        fig.update_layout(title="Stiffness dp / pc [%]", xaxis_range=[0, max(vals + [20]) * 1.25],
                                          bargap=0.45)
                        show(fig)
                    with cB:
                        hist = sorted([(str(t.get("date")), t["test_id"], r) for t, r in pool_R] +
                                      [(str(test.get("date")), test["test_id"], R)], key=lambda x: (x[0], x[1]))
                        fig = make_subplots(rows=1, cols=2, subplot_titles=["Ox CdA [mm²]", "Fuel CdA [mm²]"],
                                            horizontal_spacing=0.12)
                        for j, k in enumerate(("CdA_ox_mm2", "CdA_fuel_mm2"), start=1):
                            pts = [(i, r[k]) for _, i, r in hist if k in r]
                            if not pts:
                                continue
                            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                                     line=dict(color=OTHER, width=1.5),
                                                     marker=dict(size=9, color=[MEAS if p[0] == test["test_id"] else OTHER
                                                                                for p in pts],
                                                                 line=dict(color=SURFACE, width=2)),
                                                     hovertemplate="%{x}<br>%{y:.3f} mm²<extra></extra>",
                                                     showlegend=False), row=1, col=j)
                        theme(fig, h=220, legend=False)
                        fig.update_annotations(font=dict(size=11, color=INK2), x=0.0, xanchor="left")
                        fig.update_annotations(selector=dict(text="Fuel CdA [mm²]"), x=0.56)
                        fig.update_xaxes(showticklabels=len(hist) < 9, tickangle=-35, tickfont=dict(size=9))
                        show(fig)
                    st.caption("CdA = ṁ / √(2 ρ dp) with liquid density (N2O below saturation: saturated liquid). A "
                               "CdA shift between tests points at a blocked or eroded element or a sensor problem.")

            else:   # Summary
                lines = [f"{test['test_id']} ({test['date']}, {test['hardware']}) — {H.OUTCOMES[test['outcome']]}",
                         f"pc {R['pc_abs_bar']:.2f} bar abs ({R['pc_dev_pct']:+.1f} % vs model) · O/F {R['OF']:.2f} · "
                         f"ṁ {R['mdot_kg_s']:.3f} kg/s",
                         f"η c* {R['eta_cstar']:.3f} (model {eta_c_model:.3f}) · c* {R['cstar_meas_m_s']:.0f} m/s"]
                if "thrust_N" in R:
                    lines.append(f"F {R['thrust_N']:.0f} N ({R['throttle_pct']:.0f} %, {R['thrust_dev_pct']:+.1f} % vs model) · "
                                 f"Isp {R['isp_meas_s']:.1f} s · η Cf,vac {R['eta_cf_vac']:.3f}")
                if "stiff_ox_pct" in R or "stiff_fuel_pct" in R:
                    lines.append("Injector: " + " · ".join(
                        f"{n} dp {R[f'dp_{c}_bar']:.1f} bar ({R[f'stiff_{c}_pct']:.0f} %), CdA {R.get(f'CdA_{c}_mm2', float('nan')):.2f} mm²"
                        for c, n in (("ox", "ox"), ("fuel", "fuel")) if f"dp_{c}_bar" in R))
                if W:
                    lines.append(f"Jacket: {R['Q_meas_kW']:.0f} kW measured / {W['Q_model_kW']:.0f} kW model, "
                                 f"Bartz fit ×{W.get('bartz_fit', float('nan')):.2f}, hot wall {W.get('Twg_max_fit_K', float('nan')):.0f} K")
                lines += [f"[{lv}] {tx}" for lv, tx in obs if lv in ("warn", "crit")]
                if test.get("notes"):
                    lines.append(f"Notes: {test['notes']}")
                st.caption("Copy for the test channel or the log book (button at the top right of the box).")
                st.code("\n".join(lines), language=None, wrap_lines=True)
                full = pd.DataFrame([(H.RESULT_LABELS[k][0], fmt(k, R.get(k) if k in R else (W or {}).get(k)),
                                      H.RESULT_LABELS[k][1]) for k in H.RESULT_LABELS
                                     if R.get(k) is not None or (W and W.get(k) is not None)],
                                    columns=["result", "value", "unit"])
                st.dataframe(full, hide_index=True, width="stretch", height=min(36 * (len(full) + 1), 560))


# ========================================================================================
# CAMPAIGN
# ========================================================================================
with tab_camp:
    current = form_test()
    Rc, _ = analyse(current, eta_c_model, eta_f_model)
    tests = pool_tests(include_current=current if Rc and current["test_id"] else None)
    rows = []
    for t in tests:
        R_, e_ = analyse(t, eta_c_model, eta_f_model)
        if R_ is None:
            continue
        row = H.flat(t, R_)
        if "Q_meas_kW" in R_ and HW[t["hardware"]].regen_model:
            Wq = jacket_cached(_key(t), eta_c_model, eta_f_model, False, 120)
            if Wq:
                row["r_Q_ratio"] = R_["Q_meas_kW"] / Wq["Q_model_kW"]
        row["current"] = bool(t.get("_current"))
        row["saved"] = any(s["test_id"] == t["test_id"] for s in saved_tests())
        rows.append(row)
    if not rows:
        st.info("No tests yet. Save a test point on the Hot fire tab, or switch on the demo campaign in the sidebar.")
    else:
        df = pd.DataFrame(rows).sort_values(["date", "test_id"]).reset_index(drop=True)
        f1, f2, f3 = st.columns([1.2, 1.2, 2])
        hws = f1.multiselect("Hardware", sorted(df["hardware"].unique()), default=sorted(df["hardware"].unique()))
        outs = f2.multiselect("Outcome", list(H.OUTCOMES.values()), default=[v for v in H.OUTCOMES.values() if v != "Invalid"])
        df = df[df["hardware"].isin(hws) & df["outcome"].isin(outs)].reset_index(drop=True)
        f3.caption(f"{len(df)} tests · all results recomputed with the current model (η c* {eta_c_model:.3f}, "
                   f"η Cf,vac {eta_f_model:.3f}) · ● = the test on the Hot fire tab (unsaved changes included)")

        METRICS = {"η c*": ("r_eta_cstar", eta_c_model), "η Cf,vac": ("r_eta_cf_vac", eta_f_model),
                   "pc vs model [%]": ("r_pc_dev_pct", 0.0), "Thrust vs model [%]": ("r_thrust_dev_pct", 0.0),
                   "Isp [s]": ("r_isp_meas_s", None), "c* [m/s]": ("r_cstar_meas_m_s", None), "O/F": ("r_OF", None),
                   "pc [bar]": ("r_pc_abs_bar", None), "Thrust [N]": ("r_thrust_N", None),
                   "Ox CdA [mm²]": ("r_CdA_ox_mm2", None), "Fuel CdA [mm²]": ("r_CdA_fuel_mm2", None),
                   "Ox stiffness [%]": ("r_stiff_ox_pct", 100 * H.STIFFNESS_MIN),
                   "Fuel stiffness [%]": ("r_stiff_fuel_pct", 100 * H.STIFFNESS_MIN),
                   "Jacket heat [kW]": ("r_Q_meas_kW", None), "Jacket heat meas / model": ("r_Q_ratio", 1.0),
                   "Throttle [%]": ("r_throttle_pct", None)}
        METRICS = {k: v for k, v in METRICS.items() if v[0] in df and df[v[0]].notna().any()}

        show_cols = ["test_id", "date", "hardware", "outcome", "r_throttle_pct", "r_pc_abs_bar", "r_OF",
                     "r_eta_cstar", "r_eta_cf_vac", "r_isp_meas_s", "r_pc_dev_pct", "r_thrust_dev_pct",
                     "r_CdA_ox_mm2", "r_CdA_fuel_mm2", "r_Q_ratio", "notes"]
        show_cols = [c for c in show_cols if c in df]
        disp = df[show_cols].copy()
        disp.insert(0, "●", np.where(df["current"], "●", np.where(df["saved"], "", "demo")))
        cc = {"r_throttle_pct": st.column_config.NumberColumn("thr %", format="%.0f"),
              "r_pc_abs_bar": st.column_config.NumberColumn("pc bar", format="%.2f"),
              "r_OF": st.column_config.NumberColumn("O/F", format="%.2f"),
              "r_eta_cstar": st.column_config.ProgressColumn("η c*", format="%.3f", min_value=0.8, max_value=1.0),
              "r_eta_cf_vac": st.column_config.NumberColumn("η Cf,vac", format="%.3f"),
              "r_isp_meas_s": st.column_config.NumberColumn("Isp s", format="%.1f"),
              "r_pc_dev_pct": st.column_config.NumberColumn("pc Δ%", format="%+.1f"),
              "r_thrust_dev_pct": st.column_config.NumberColumn("F Δ%", format="%+.1f"),
              "r_CdA_ox_mm2": st.column_config.NumberColumn("CdA ox", format="%.2f"),
              "r_CdA_fuel_mm2": st.column_config.NumberColumn("CdA fu", format="%.2f"),
              "r_Q_ratio": st.column_config.NumberColumn("Q meas/mod", format="%.2f"),
              "test_id": st.column_config.TextColumn("test", width="medium"),
              "notes": st.column_config.TextColumn("notes", width="large")}
        sel = st.dataframe(disp, hide_index=True, width="stretch", column_config=cc, on_select="rerun",
                           selection_mode="multi-row", key="camp_table",
                           height=min(38 * (len(disp) + 1) + 4, 420))
        picked = [df.loc[i, "test_id"] for i in sel.selection.rows] if sel and sel.selection else []
        st.caption("Select rows to highlight them in the plots and compare them side by side.")

        def color_for(tid, cur):
            if tid in picked:
                return VIOLET
            return MEAS if cur else OTHER

        g1, g2 = st.columns(2)
        with g1:
            mname = st.selectbox("Trend", list(METRICS), index=0)
            mk, mref = METRICS[mname]
            d = df[df[mk].notna()]
            fig = go.Figure()
            if len(d) >= 3:
                mu, sd = d[mk].mean(), d[mk].std()
                fig.add_hrect(y0=mu - 2 * sd, y1=mu + 2 * sd, fillcolor=MEAS, opacity=0.07, line_width=0)
                fig.add_hline(y=mu, line=dict(color=MEAS, width=1, dash="dot"),
                              annotation_text=f"mean {mu:.3g} ± 2σ", annotation_position="bottom left",
                              annotation_font_color=MUTED)
            if mref is not None:
                fig.add_hline(y=mref, line=dict(color=MODEL, width=1.4, dash="dash"),
                              annotation_text=f"model {mref:.3g}", annotation_position="bottom right",
                              annotation_font_color=INK2)
            fig.add_trace(go.Scatter(x=d["test_id"], y=d[mk], mode="lines+markers", line=dict(color="#45443f", width=1.4),
                                     marker=dict(size=[14 if c else 10 for c in d["current"]],
                                                 color=[color_for(i, c) for i, c in zip(d["test_id"], d["current"])],
                                                 line=dict(color=SURFACE, width=2)),
                                     customdata=np.stack([d["date"].astype(str), d["r_OF"]], axis=1),
                                     hovertemplate="%{x}<br>%{customdata[0]} · O/F %{customdata[1]:.2f}<br>"
                                                   f"{mname} " + "%{y:.4g}<extra></extra>", showlegend=False))
            theme(fig, h=360, legend=False)
            fig.update_layout(title=f"{mname} by test", yaxis_title=mname)
            fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
            show(fig)
        with g2:
            a, b = st.columns(2)
            xn = a.selectbox("x", list(METRICS), index=list(METRICS).index("O/F") if "O/F" in METRICS else 0)
            yn = b.selectbox("y", list(METRICS), index=0)
            xk, yk = METRICS[xn][0], METRICS[yn][0]
            d = df[df[xk].notna() & df[yk].notna()]
            fig = go.Figure()
            if METRICS[yn][1] is not None:
                fig.add_hline(y=METRICS[yn][1], line=dict(color=MODEL, width=1.4, dash="dash"),
                              annotation_text="model", annotation_font_color=INK2)
            fig.add_trace(go.Scatter(x=d[xk], y=d[yk], mode="markers+text", text=d["test_id"],
                                     textposition="top center", textfont=dict(size=10, color=MUTED), cliponaxis=False,
                                     marker=dict(size=[15 if c else 11 for c in d["current"]],
                                                 color=[color_for(i, c) for i, c in zip(d["test_id"], d["current"])],
                                                 line=dict(color=SURFACE, width=2)),
                                     hovertemplate="%{text}<br>" + f"{xn} " + "%{x:.4g}<br>" + f"{yn} " + "%{y:.4g}<extra></extra>",
                                     showlegend=False))
            theme(fig, h=360, legend=False)
            fig.update_layout(title=f"{yn} vs {xn}", xaxis_title=xn, yaxis_title=yn)
            show(fig)

        if len(picked) >= 2:
            st.markdown("<div class='sec'>Side by side</div>", unsafe_allow_html=True)
            sub = df.set_index("test_id").loc[picked]
            keys = [k for k in ["r_throttle_pct", "r_pc_abs_bar", "r_OF", "r_mdot_kg_s", "r_eta_cstar", "r_eta_cf_vac",
                                "r_isp_meas_s", "r_thrust_N", "r_pc_dev_pct", "r_thrust_dev_pct", "r_Tc_K",
                                "r_pe_over_pa", "r_stiff_ox_pct", "r_stiff_fuel_pct", "r_CdA_ox_mm2",
                                "r_CdA_fuel_mm2", "r_Q_meas_kW", "r_Q_ratio"] if k in sub]
            comp = pd.DataFrame({tid: [sub.loc[tid, k] for k in keys] for tid in picked},
                                index=[(H.RESULT_LABELS.get(k[2:], ("Jacket heat meas / model", "-"))[0] + " ["
                                        + H.RESULT_LABELS.get(k[2:], ("", "-"))[1] + "]") for k in keys])
            base = picked[0]
            for tid in picked[1:]:
                comp[f"Δ {tid}"] = comp[tid] - comp[base]
            st.dataframe(comp.astype(float).round(4), width="stretch")
            st.caption(f"Δ columns: against {base} (the first selected row).")

        st.download_button("⬇ Campaign CSV (inputs and results)", df.drop(columns=["current", "saved"]).to_csv(index=False),
                           file_name="e2_hotfire_campaign.csv")


# ========================================================================================
# MODEL
# ========================================================================================
with tab_model:
    c1, c2, c3, c4 = st.columns(4)
    m_hw = c1.selectbox("Hardware", list(HW), index=list(HW).index(SS.m_hardware), key="mod_hw")
    m_mf = c2.number_input("Fuel flow [kg/s]", 0.05, 0.5, float(C.MDOT_FUEL), 0.005, format="%.3f", key="mod_mf")
    m_T = c3.number_input("N2O temperature [°C]", -20.0, 35.0, float(C.T_OX_NOM - 273.15), 0.5, key="mod_T")
    m_pa = c4.number_input("Ambient pressure [bar]", 0.0, 1.2, float(C.P_AMB), 0.01, format="%.3f", key="mod_pa")
    ln = line_cached(m_hw, m_mf, m_T + 273.15, m_pa, None, eta_c_model, eta_f_model, 1.6, 5.4)
    ln["throttle_pct"] = 100 * ln["thrust_model"] / C.F_MAX
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ln["ox_flow_kg_s"], y=ln["thrust_ideal"], name="CEA ideal",
                                 line=dict(color=IDEAL, dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=ln["ox_flow_kg_s"], y=ln["thrust_model"], name="model", line=dict(color=MODEL, width=2)))
        for f in (0.5, 1.0):
            fig.add_hline(y=f * C.F_MAX, line=dict(color=MUTED, dash="dot", width=1),
                          annotation_text=f"{100 * f:.0f} % rated", annotation_font_color=MUTED)
        theme(fig, h=340, hover="x unified")
        fig.update_layout(title="Thrust vs oxidiser flow", xaxis_title="ox flow [kg/s]", yaxis_title="thrust [N]")
        show(fig)
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ln["ox_flow_kg_s"], y=ln["pc_ideal"], name="CEA ideal",
                                 line=dict(color=IDEAL, dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=ln["ox_flow_kg_s"], y=ln["pc_model"], name="model", line=dict(color=MODEL, width=2)))
        theme(fig, h=340, hover="x unified")
        fig.update_layout(title="Chamber pressure vs oxidiser flow", xaxis_title="ox flow [kg/s]", yaxis_title="pc [bar]")
        show(fig)
    tbl = ln[["throttle_pct", "ox_flow_kg_s", "OF", "pc_model", "thrust_model", "isp_model", "cstar_model", "Tc_model"]]
    tbl.columns = ["throttle %", "ox kg/s", "O/F", "pc bar", "F N", "Isp s", "c* m/s (delivered)", "Tc K"]
    with st.expander("Throttle line table"):
        st.dataframe(tbl.round(3), hide_index=True, width="stretch")
    st.markdown("<div class='sec'>Export for hopper-test-lab</div>", unsafe_allow_html=True)
    st.caption("hopper-test-lab runs in the browser (TypeScript) and cannot call CEA. The CEA map is the ideal "
               "c*, Tc, Cf_vac, Isp_vac and exit pressure on a pc × O/F grid for this hardware's area ratio; "
               "bilinear interpolation reproduces the h2cea values, so a derived channel there gives the same η c*.")
    if st.button("Build CEA map JSON"):
        SS.cea_json = cea_map_cached(HW[m_hw].eps, m_T + 273.15, m_pa)
        SS.cea_name = f"e2_cea_map_{m_hw.replace(' ', '_')}_eps{HW[m_hw].eps:.2f}.json"
    if SS.get("cea_json"):
        st.download_button(f"⬇ {SS.cea_name}", SS.cea_json, file_name=SS.cea_name, mime="application/json")
    with st.expander("Model notes"):
        st.markdown(f"""
- **CEA**: shifting equilibrium, real-fluid liquid N2O card at the N2O temperature and the supply pressure
  ({C.P_TANK:g} bar; the feed throttles isenthalpically), ethanol {C.ETHANOL_WT:g} wt%. Same as the engine table.
- **η c\\*** = pc·At/ṁ / c\\*_CEA(pc, O/F). **η Cf,vac** = (F + pa·Ae)/(pc·At) / Cf_vac,CEA, the split used
  by the baseline (η c\\* {C.ETA_CSTAR}, η Cf,vac {C.ETA_CF_VAC}). A measured throat diameter changes At and ε.
- **Model at the measured flows**: pc from ṁ·η c\\*·c\\*(pc)/At (fixed point), thrust from η Cf,vac.
- **Nozzle**: wall pressure from the 1D area ratio with a constant γ that matches the CEA exit pressure;
  Schmucker separation criterion; Summerfield pe < 0.4 pa as in run_engine_table.py.
- **Jacket**: regen.march for the E2-REG-1 as built (water, liquid-only), Bartz factor fitted to the calorimetric heat.
- **Injector CdA** with liquid density; for N2O below its saturation pressure the saturated liquid density
  (compare between tests, not with a single-phase design value).
""")
