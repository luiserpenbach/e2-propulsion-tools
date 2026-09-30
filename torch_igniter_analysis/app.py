"""
Torch igniter explorer (Streamlit)

    pip install streamlit pandas plotly
    streamlit run app.py          (run from this folder)

Design tab: start from a config in configs/, change operating point and geometry,
            see chamber pressure, temperature, heat flux and heat soak, sweep one
            parameter, compare with the base config, download the case as YAML.
Test tab:   enter measured flows and chamber pressure (or average a window of a
            time-series CSV), compare measured c* efficiency and chamber pressure
            with the model, keep a test log (test_log.csv in this folder).

The numbers come from igniter_model.py, which uses the same functions as
igniter_combustion_analysis.py and igniter_thermal_analysis.py.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

import igniter_model as M

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
TEST_LOG = HERE / "test_log.csv"

st.set_page_config(page_title="Torch igniter", layout="wide")


def load_raw(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Running CEA and the heat-flux model ...")
def analyse_cached(raw_yaml: str, sweep: bool = True) -> dict:
    return M.analyse(yaml.safe_load(raw_yaml), sweep=sweep)


def run(raw: dict, sweep: bool = True) -> dict:
    return analyse_cached(yaml.safe_dump(raw, sort_keys=True), sweep)


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float, np.floating)) else str(x)


configs = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
if not configs:
    st.error(f"No configs in {CONFIG_DIR}")
    st.stop()

st.title("Torch igniter")
tab_design, tab_test = st.tabs(["Design", "Testing"])

# ============================== DESIGN ======================================
with tab_design:
    base_name = st.selectbox("Base config", configs, key="design_base",
                             help="Starting values. Edits are not written to the file; use the download button.")
    base = load_raw(base_name)

    with st.form("design_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Operating point**")
            mdot = st.number_input("Total mass flow [g/s]", 1.0, 1000.0, 1e3 * float(base["mdot_kg_s"]), step=1.0)
            of_nom = st.number_input("Nominal O/F", 0.5, 10.0, float(base["of_nominal"]), step=0.1)
            of_pts = st.text_input("Other O/F points (comma separated)",
                                   ", ".join(f"{x:g}" for x in base["of_points"]))
            eta = st.number_input("c* efficiency", 0.3, 1.0, float(base["eta_cstar"]), step=0.01)
            pa = st.number_input("Ambient pressure [bar]", 0.0, 2.0, float(base["pa_bar"]))
        with c2:
            st.markdown("**Chamber and nozzle**")
            dt_mm = st.number_input("Throat diameter [mm]", 0.5, 50.0, float(base["dt_mm"]), step=0.1)
            dc_mm = st.number_input("Chamber diameter [mm]", 1.0, 200.0, float(base["dc_mm"]), step=0.5)
            by_lstar = st.radio("Chamber length from", ["L*", "cylinder length"], horizontal=True,
                                index=0 if base.get("lstar_m") is not None else 1)
            if by_lstar == "L*":
                lstar = st.number_input("L* [m]", 0.05, 5.0, float(base.get("lstar_m") or 1.0), step=0.05)
                lcyl = None
            else:
                lcyl = st.number_input("Cylinder length [mm]", 1.0, 500.0, float(base.get("lcyl_mm") or 30.0))
                lstar = None
            eps = st.number_input("Nozzle area ratio", 1.0, 10.0, float(base["eps"]), step=0.1)
        with c3:
            st.markdown("**Angles and wall**")
            th_c = st.number_input("Convergent half-angle [deg]", 10.0, 80.0, float(base["theta_conv_deg"]))
            th_d = st.number_input("Divergent half-angle [deg]", 5.0, 30.0, float(base["theta_div_deg"]))
            rc_ratio = st.number_input("Throat curvature radius / Dt", 0.2, 3.0,
                                       float(base.get("rc_curv_over_dt", 1.0)))
            t0 = st.number_input("Initial wall temperature [K]", 200.0, 800.0, float(base.get("wall_t0_K", 293.0)))
            wall = st.number_input("Wall thickness [mm]", 0.5, 20.0, float(base.get("wall_thickness_mm", 2.0)))
        st.form_submit_button("Update", type="primary")

    try:
        of_list = [float(x) for x in of_pts.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        st.error("O/F points must be numbers separated by commas.")
        st.stop()

    raw = {k: v for k, v in base.items() if k not in ("lstar_m", "lcyl_mm")}
    raw.update(name=f"{base.get('name', Path(base_name).stem)}_edit", mdot_kg_s=mdot / 1e3,
               of_nominal=of_nom, of_points=of_list, eta_cstar=eta, pa_bar=pa, dt_mm=dt_mm, dc_mm=dc_mm,
               eps=eps, theta_conv_deg=th_c, theta_div_deg=th_d, rc_curv_over_dt=rc_ratio,
               wall_t0_K=t0, wall_thickness_mm=wall)
    raw.update({"lstar_m": lstar} if lstar is not None else {"lcyl_mm": lcyl})

    try:
        res = run(raw)
        res_base = run(base, sweep=False)
    except Exception as e:                      # invalid geometry, CEA failure
        st.error(f"Analysis failed: {e}")
        st.stop()

    nom, nom_b = M.nominal_point(res), M.nominal_point(res_base)
    geo, geo_b = M.geometry_summary(res["case"]), M.geometry_summary(res_base["case"])

    st.subheader(f"Nominal point, O/F {of_nom:g}")
    m = st.columns(6)
    for col, (label, key, src, nd) in zip(m, [
            ("Chamber pressure [bar]", "pc_bar", "p", 2), ("Tc [K]", "Tc_K", "p", 0),
            ("Isp amb [s]", "isp_amb_s", "p", 1), ("Throat q [MW/m²]", "q_throat_MW_m2", "p", 2),
            ("Wall heat [kW]", "P_wall_kW", "p", 2), ("Total length [mm]", "L_total_mm", "g", 1)]):
        v, vb = (nom[key], nom_b[key]) if src == "p" else (geo[key], geo_b[key])
        d = round(v - vb, nd)
        col.metric(label, fmt(v, nd), None if d == 0 else f"{d:+.{nd}f} vs base")

    warn = []
    if geo["L_cyl_over_Dc"] > 8:
        warn.append(f"Long chamber: L_cyl / Dc = {geo['L_cyl_over_Dc']:.1f}.")
    for name, t_lim in ((k.split("_")[2], v) for k, v in nom.items() if k.startswith("t_limit_")):
        if t_lim > res["t_slab_valid_s"][name]:
            warn.append(f"{name}: time to limit {t_lim:.2f} s is beyond the semi-infinite slab validity "
                        f"({res['t_slab_valid_s'][name]:.3f} s for this wall); the estimate is optimistic.")
    for w in warn:
        st.warning(w)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        c = res["contour"]
        fig.add_trace(go.Scatter(x=c["x_mm"], y=c["r_mm"], name="wall", line=dict(color="#555")))
        fig.add_trace(go.Scatter(x=c["x_mm"], y=-c["r_mm"], showlegend=False, line=dict(color="#555")))
        cb = res_base["contour"]
        fig.add_trace(go.Scatter(x=cb["x_mm"], y=cb["r_mm"], name="base", line=dict(color="#aaa", dash="dot")))
        fig.update_layout(title="Contour", xaxis_title="x from injector [mm]", yaxis_title="r [mm]",
                          yaxis_scaleanchor="x", height=330, margin=dict(t=40, b=30))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        for of, pr in res["profiles"].items():
            fig.add_trace(go.Scatter(x=pr["x_mm"], y=pr["q_MW_m2"], name=f"O/F {of:g}"))
        fig.update_layout(title="Wall heat flux (cold wall, Bartz)", xaxis_title="x [mm]",
                          yaxis_title="q [MW/m²]", height=330, margin=dict(t=40, b=30))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Operating points**")
    cols = ["O/F", "pc_bar", "Tc_K", "cstar_del_m_s", "isp_amb_s", "mdot_ox_g_s", "mdot_fuel_g_s",
            "q_throat_MW_m2", "P_wall_kW", "wall_loss_pct"] + [k for k in nom if k.startswith("t_limit_")]
    st.dataframe(pd.DataFrame(res["points"])[cols].round(3), hide_index=True, width="stretch")
    st.caption("t_limit: time for the inner wall at the throat to reach the material limit "
               "(semi-infinite slab, constant cold-wall heat flux). "
               + ", ".join(f"{n} {v['T_limit']:.0f} K" for n, v in res["materials"].items()))

    with st.expander("Geometry"):
        st.dataframe(pd.DataFrame([{"quantity": k, "edited": v, "base": geo_b[k]} for k, v in geo.items()]).round(3),
                     hide_index=True)

    sw = pd.DataFrame(res["sweep"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sw["O/F"], y=sw["pc_bar"], name="pc [bar]"))
    fig.add_trace(go.Scatter(x=sw["O/F"], y=sw["Tc_K"], name="Tc [K]", yaxis="y2"))
    fig.update_layout(title="O/F sweep at this flow and throat", xaxis_title="O/F", yaxis_title="pc [bar]",
                      yaxis2=dict(title="Tc [K]", overlaying="y", side="right"), height=330,
                      margin=dict(t=40, b=30))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Design sweep")
    params = {"Total mass flow [g/s]": ("mdot_kg_s", 1e-3), "Throat diameter [mm]": ("dt_mm", 1.0),
              "Chamber diameter [mm]": ("dc_mm", 1.0), "c* efficiency": ("eta_cstar", 1.0),
              "Nominal O/F": ("of_nominal", 1.0), "Nozzle area ratio": ("eps", 1.0)}
    if lstar is not None:
        params["L* [m]"] = ("lstar_m", 1.0)
    s1, s2, s3, s4 = st.columns([2, 1, 1, 1])
    p_name = s1.selectbox("Parameter", list(params))
    key, scale = params[p_name]
    cur = raw[key] / scale
    lo = s2.number_input("From", value=round(0.7 * cur, 4), key="sw_lo")
    hi = s3.number_input("To", value=round(1.3 * cur, 4), key="sw_hi")
    n = s4.number_input("Points", 3, 25, 9)
    if st.button("Run sweep"):
        rows = []
        for v in np.linspace(lo, hi, int(n)):
            r = dict(raw, **{key: float(v) * scale})
            if key == "of_nominal":
                r["of_points"] = []
            try:
                out = run(r, sweep=False)
                p, g = M.nominal_point(out), M.geometry_summary(out["case"])
                rows.append({p_name: v, "pc_bar": p["pc_bar"], "Tc_K": p["Tc_K"],
                             "q_throat_MW_m2": p["q_throat_MW_m2"], "P_wall_kW": p["P_wall_kW"],
                             "isp_amb_s": p["isp_amb_s"], "L_total_mm": g["L_total_mm"]})
            except Exception as e:
                st.warning(f"{p_name} = {v:g}: {e}")
        if rows:
            df = pd.DataFrame(rows)
            ycols = st.multiselect("Show", [c for c in df.columns if c != p_name],
                                   default=["pc_bar", "q_throat_MW_m2"])
            fig = go.Figure([go.Scatter(x=df[p_name], y=df[c], name=c, mode="lines+markers") for c in ycols])
            fig.update_layout(xaxis_title=p_name, height=330, margin=dict(t=20, b=30))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df.round(3), hide_index=True)

    st.download_button("Download this case as YAML", yaml.safe_dump(raw, sort_keys=False),
                       file_name=f"{raw['name']}.yaml", mime="text/yaml",
                       help="Save it in configs/ to keep it; the scripts and this app read that folder.")

# ============================== TESTING =====================================
with tab_test:
    t1, t2, t3 = st.columns(3)
    hw_name = t1.selectbox("Hardware config", configs, key="test_cfg",
                           help="Throat, propellants and the design c* efficiency to compare with.")
    hw = load_raw(hw_name)
    dt_meas = t2.number_input("Throat diameter as measured [mm]", 0.5, 50.0, float(hw["dt_mm"]), step=0.01)
    gauge = t3.checkbox("Chamber pressure is gauge", value=False,
                        help=f"Adds the config's ambient pressure ({hw['pa_bar']} bar).")

    if "tests" not in st.session_state:
        if TEST_LOG.exists():
            st.session_state.tests = pd.read_csv(TEST_LOG)
        else:
            st.session_state.tests = pd.DataFrame(columns=M.TEST_COLUMNS)

    with st.expander("Average a window of a time-series CSV"):
        up = st.file_uploader("CSV with a time column, chamber pressure and flows", type=["csv", "txt"])
        if up is not None:
            ts = pd.read_csv(up)
            num = [c for c in ts.columns if pd.api.types.is_numeric_dtype(ts[c])]
            if len(num) < 3:
                st.error("Need at least three numeric columns.")
            else:
                a, b, c_, d = st.columns(4)
                tcol = a.selectbox("Time", num)
                pcol = b.selectbox("Chamber pressure [bar]", num, index=min(1, len(num) - 1))
                ocol = c_.selectbox("Ox flow [g/s]", num, index=min(2, len(num) - 1))
                fcol = d.selectbox("Fuel flow [g/s]", num, index=min(3, len(num) - 1))
                tmin, tmax = float(ts[tcol].min()), float(ts[tcol].max())
                win = st.slider("Steady-state window", tmin, tmax, (tmin + 0.4 * (tmax - tmin), tmin + 0.8 * (tmax - tmin)))
                sel = ts[(ts[tcol] >= win[0]) & (ts[tcol] <= win[1])]
                fig = go.Figure([go.Scatter(x=ts[tcol], y=ts[col], name=col) for col in (pcol, ocol, fcol)])
                fig.add_vrect(x0=win[0], x1=win[1], fillcolor="#4a9eff", opacity=0.15, line_width=0)
                fig.update_layout(height=280, margin=dict(t=10, b=30), xaxis_title=tcol)
                st.plotly_chart(fig, width="stretch")
                mean = sel[[pcol, ocol, fcol]].mean()
                st.write(f"Window mean ({len(sel)} samples): pc {mean[pcol]:.3f} bar, "
                         f"ox {mean[ocol]:.3f} g/s, fuel {mean[fcol]:.3f} g/s")
                tid = st.text_input("Test ID for this point", Path(up.name).stem)
                if st.button("Add window mean as test point"):
                    row = {"test_id": tid, "date": str(dt.date.today()), "mdot_ox_g_s": mean[ocol],
                           "mdot_fuel_g_s": mean[fcol], "pc_bar": mean[pcol], "thrust_N": None,
                           "ignition": "", "notes": f"{up.name}, {win[0]:g}-{win[1]:g}"}
                    st.session_state.tests = pd.concat([st.session_state.tests, pd.DataFrame([row])],
                                                       ignore_index=True)
                    st.rerun()

    st.markdown("**Test points** (steady state; add rows at the bottom of the table)")
    edited = st.data_editor(
        st.session_state.tests, num_rows="dynamic", width="stretch", key="test_editor",
        column_config={"mdot_ox_g_s": st.column_config.NumberColumn("ox flow [g/s]"),
                       "mdot_fuel_g_s": st.column_config.NumberColumn("fuel flow [g/s]"),
                       "pc_bar": st.column_config.NumberColumn("pc [bar]"),
                       "thrust_N": st.column_config.NumberColumn("thrust [N] (optional)")})

    b1, b2 = st.columns(2)
    if b1.button("Save to test_log.csv"):
        edited.to_csv(TEST_LOG, index=False)
        st.session_state.tests = edited
        st.success(f"Saved {len(edited)} rows to {TEST_LOG.name}")

    rows = edited.to_dict("records")
    try:
        res_t = M.analyse_test_points(hw, rows, dt_mm=dt_meas, pc_gauge=gauge)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        res_t = []
    if not res_t:
        st.info("Enter at least one test point with ox flow, fuel flow and chamber pressure.")
    else:
        df = pd.DataFrame(res_t)
        st.dataframe(df.round(3), hide_index=True, width="stretch")
        b2.download_button("Download results CSV", df.to_csv(index=False), file_name="igniter_test_results.csv")

        eta_d = float(hw["eta_cstar"])
        line = pd.DataFrame(M.of_line(hw, 1.0, max(0.5, df["O/F"].min() - 0.5), df["O/F"].max() + 0.5))
        l, r = st.columns(2)
        with l:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["O/F"], y=df["eta_cstar_meas"], mode="markers+text", text=df["test_id"],
                                     textposition="top center", name="measured"))
            fig.add_hline(y=eta_d, line_dash="dash", annotation_text=f"design {eta_d:g}")
            fig.update_layout(title="c* efficiency", xaxis_title="O/F", yaxis_title="eta c*", height=350,
                              margin=dict(t=40, b=30))
            st.plotly_chart(fig, width="stretch")
        with r:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["O/F"], y=df["cstar_meas_m_s"], mode="markers", name="measured c*"))
            fig.add_trace(go.Scatter(x=line["O/F"], y=line["cstar_ideal_m_s"], name="CEA ideal"))
            fig.add_trace(go.Scatter(x=line["O/F"], y=eta_d * line["cstar_ideal_m_s"], name=f"model, eta {eta_d:g}",
                                     line=dict(dash="dash")))
            fig.update_layout(title="c* vs O/F", xaxis_title="O/F", yaxis_title="c* [m/s]", height=350,
                              margin=dict(t=40, b=30))
            st.plotly_chart(fig, width="stretch")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["pc_model_bar"], y=df["pc_abs_bar"], mode="markers+text", text=df["test_id"],
                                 textposition="top center", name="tests"))
        lim = [0, 1.1 * max(df["pc_model_bar"].max(), df["pc_abs_bar"].max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dot", color="#888"), name="1:1"))
        fig.update_layout(title="Chamber pressure: measured vs model at the measured flows",
                          xaxis_title="model pc [bar]", yaxis_title="measured pc [bar]", height=350,
                          margin=dict(t=40, b=30))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Mean measured c* efficiency {df['eta_cstar_meas'].mean():.3f} "
                   f"(design {eta_d:g}). Put the measured value into eta_cstar of the config for the next design "
                   f"iteration. Chamber pressure must be absolute or marked as gauge above.")
