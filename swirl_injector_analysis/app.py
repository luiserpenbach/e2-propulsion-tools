"""
Optional Streamlit front end for lcsc_sizing.py

    pip install streamlit pandas
    streamlit run app.py          (run from this folder)

Design tab:  size an element from a base config and sweep one parameter across the design space.
Testing tab: pick an as-built config, enter measured values, compare with the model and keep a
             simple test log (test_log.csv next to this file).
"""

from __future__ import annotations

import datetime as dt
import glob
import math
import os

import pandas as pd
import streamlit as st

import lcsc_sizing as L

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "test_log.csv")
BAR, MM = L.BAR, L.MM

st.set_page_config(page_title="LCSC sizing", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1300px}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
@st.cache_data
def load_configs() -> dict:
    cfgs = {}
    paths = glob.glob(os.path.join(HERE, "*.yaml")) + glob.glob(os.path.join(HERE, "configs", "*.yaml"))
    for p in sorted(paths):
        try:
            cfgs[os.path.basename(p)] = L.load_config(p)
        except Exception as e:  # show broken files instead of failing
            cfgs[os.path.basename(p)] = {"_error": str(e)}
    return cfgs


def geometry_table(g: dict, g2: dict | None = None, labels=("Value", None)) -> pd.DataFrame:
    rows = []
    for k, lab, unit in L.GEOM_LABELS:
        row = {"Feature": lab, labels[0]: L.fmt_geom(g, k)}
        if g2 is not None:
            row[labels[1]] = L.fmt_geom(g2, k)
        row["Unit"] = unit
        rows.append(row)
    return pd.DataFrame(rows)


def envelope_table(points: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "Thrust frac.": p["fraction"], "pc [bar]": p["pc_bar"],
        "Fuel Δp [bar]": p["fuel_dp_bar"], "Fuel stiffness": p["fuel_stiffness"],
        "Ox feed [bar]": p["ox_feed_bar"], "Ox stiffness": p["ox_stiffness"],
        "Ox choked": "yes" if p["ox_holes_choked"] else "no",
        "v_ox [m/s]": p["ox_exit_velocity_m_s"], "J (S-L)": p["J_SL"],
        "J min": p["J_min"], "J max": p["J_max"], "Recess": p["recess_regime"],
    } for p in points]).round(3)


@st.cache_data(show_spinner="Sizing ...")
def size_and_analyse(cfg: dict):
    g, info = L.size_element(cfg)
    g_r = L.round_geometry(g, cfg)
    pts = [L.analyse_point(cfg, g_r, p) for p in cfg["throttle_points"]]
    warn = L.check_manufacturing(g_r, cfg) + [f"[{p['fraction']:.2f}] {w}" for p in pts for w in p["warnings"]]
    return g, g_r, info, pts, warn


@st.cache_data(show_spinner="Sweeping design space ...")
def sweep(cfg: dict, path: tuple, values: tuple) -> pd.DataFrame:
    rows = []
    for v in values:
        upd = {path[0]: v} if len(path) == 1 else {path[0]: {path[1]: v}}
        c = L.deep_merge(cfg, upd)
        try:
            g, _ = L.size_element(c)
            g = L.round_geometry(g, c)
            p = L.analyse_point(c, g, {"fraction": 1.0})
            rows.append({"value": v, "d_out [mm]": g["d_out"] / MM, "d_inlet [mm]": g["d_inlet"] / MM,
                         "d_swirl_chamber [mm]": g["d_swirl_chamber"] / MM, "d_gas_out [mm]": g["d_gas_out"] / MM,
                         "recess [mm]": g["recess"] / MM, "d_ox_hole [mm]": g["d_ox_hole"] / MM,
                         "J (S-L)": p["J_SL"], "J min": p["J_min"], "J max": p["J_max"],
                         "spray half-angle pred [deg]": p["spray_half_angle_pred_deg"]})
        except Exception as e:
            rows.append({"value": v, "error": str(e)})
    return pd.DataFrame(rows).set_index("value")


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------
st.title("LCSC element sizing")
cfgs = load_configs()
good = {k: v for k, v in cfgs.items() if "_error" not in v}
if not good:
    st.error("No valid YAML config found next to app.py.")
    st.stop()

tab_design, tab_test = st.tabs(["Design", "Testing"])

# ============================== DESIGN ======================================
with tab_design:
    base_name = st.selectbox("Base config", list(good), key="design_base",
                             help="Starting values; edits below are not written back to the file.")
    base = good[base_name]

    with st.form("design_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Operating point (head)**")
            n_el = st.number_input("Elements", 1, 50, int(base["elements"]))
            pc = st.number_input("Chamber pressure [bar]", 1.0, 200.0, float(base["operating_point"]["chamber_pressure_bar"]))
            mdot = st.number_input("Total mass flow [kg/s]", 0.01, 50.0, float(base["operating_point"]["mdot_total_kg_s"]))
            of = st.number_input("O/F", 0.5, 20.0, float(base["operating_point"]["of_ratio"]))
        with c2:
            st.markdown("**Fuel**")
            T_f = st.number_input("Fuel temperature [K]", 200.0, 500.0, float(base["fuel"]["temperature_K"]))
            dp_f = st.number_input("Fuel Δp [bar]", 0.5, 100.0, float(base["fuel"]["pressure_drop_bar"]))
            alpha = st.number_input("Spray half-angle [deg]", 30.0, 70.0, float(base["swirl"]["spray_half_angle_deg"]))
            n_in = st.number_input("Tangential inlets", 1, 12, int(base["swirl"]["n_inlets"]))
            method = st.selectbox("Swirl method", ["bazarov", "nardi"],
                                  index=0 if base["swirl"]["method"] == "bazarov" else 1)
        with c3:
            st.markdown("**Oxidizer**")
            ox_mode = st.radio("Hole inlet state", ["Temperature", "Saturated (quality)"], horizontal=True,
                               index=0 if base["oxidizer"]["temperature_K"] is not None else 1)
            if ox_mode == "Temperature":
                T_o = st.number_input("Ox temperature [K]", 180.0, 900.0, float(base["oxidizer"]["temperature_K"] or 500))
                q_o = None
            else:
                q_o = st.number_input("Ox quality [-]", 0.0, 1.0, float(base["oxidizer"]["quality"] or 0.0))
                T_o = None
            dp_o = st.number_input("Ox Δp [bar]", 0.5, 100.0, float(base["oxidizer"]["pressure_drop_bar"]))
            v_ox = st.number_input("Gas exit velocity [m/s]", 10.0, 400.0, float(base["gas_annulus"]["exit_velocity_m_s"]))
        with c4:
            st.markdown("**Holes and calibration**")
            n_h = st.number_input("Ox metering holes", 1, 24, int(base["ox_holes"]["n_holes"]))
            cd_h = st.number_input("Ox hole C_D", 0.3, 1.0, float(base["ox_holes"]["discharge_coefficient"]))
            t_post = st.number_input("Post thickness [mm]", 0.1, 3.0, float(base["gas_annulus"]["post_thickness_mm"]))
            cdf = st.number_input("Swirl C_D factor", 0.3, 1.5, float(base["calibration"]["swirl_cd_factor"]))
        st.form_submit_button("Size element", type="primary")

    cfg = L.deep_merge(base, {
        "elements": int(n_el), "as_built": {},
        "operating_point": {"chamber_pressure_bar": pc, "mdot_total_kg_s": mdot, "of_ratio": of},
        "fuel": {"temperature_K": T_f, "pressure_drop_bar": dp_f},
        "oxidizer": {"temperature_K": T_o, "quality": q_o, "pressure_drop_bar": dp_o},
        "swirl": {"method": method, "spray_half_angle_deg": alpha, "n_inlets": int(n_in)},
        "gas_annulus": {"exit_velocity_m_s": v_ox, "post_thickness_mm": t_post},
        "ox_holes": {"n_holes": int(n_h), "discharge_coefficient": cd_h},
        "calibration": {"swirl_cd_factor": cdf},
    })
    # throttle points defined by absolute chamber pressure in the base file are rescaled to the new pc
    pc0 = base["operating_point"]["chamber_pressure_bar"]
    cfg["throttle_points"] = [
        {**p, "chamber_pressure_bar": p.get("chamber_pressure_bar", pc0 * p.get("fraction", 1.0)) * pc / pc0}
        for p in base["throttle_points"] if "ox_quality" not in p and "ox_temperature_K" not in p
    ] or [{"fraction": 1.0}]

    try:
        g, g_r, info, pts, warn = size_and_analyse(cfg)
    except Exception as e:
        st.error(f"Sizing failed: {e}")
        st.stop()

    nom = pts[0]
    m = st.columns(6)
    m[0].metric("Fuel outlet", f"{g_r['d_out'] / MM:.2f} mm")
    m[1].metric("Fuel inlets", f"{g_r['n_inlets']} × {g_r['d_inlet'] / MM:.2f} mm")
    m[2].metric("Gas outlet", f"{g_r['d_gas_out'] / MM:.2f} mm")
    m[3].metric("Ox holes", f"{g_r['n_ox_holes']} × {g_r['d_ox_hole'] / MM:.2f} mm")
    m[4].metric("J nominal (S-L)", f"{nom['J_SL']:.2f}")
    m[5].metric("Swirl C_D", f"{info['swirl_at_sizing']['mu']:.4f}")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Geometry**")
        st.dataframe(geometry_table(g, g_r, ("Sized", "Rounded")), hide_index=True)
    with right:
        st.markdown("**Operating envelope**")
        st.dataframe(envelope_table(pts), hide_index=True)
        st.markdown("**Warnings**")
        for w in warn or ["none"]:
            st.caption(f"• {w}")

    st.divider()
    st.subheader("Design space")
    params = {
        "Spray half-angle [deg]": (("swirl", "spray_half_angle_deg"), 45.0, 65.0),
        "Gas exit velocity [m/s]": (("gas_annulus", "exit_velocity_m_s"), 60.0, 200.0),
        "Fuel Δp [bar]": (("fuel", "pressure_drop_bar"), 5.0, 25.0),
        "Ox Δp [bar]": (("oxidizer", "pressure_drop_bar"), 5.0, 25.0),
        "Ox temperature [K]": (("oxidizer", "temperature_K"), 300.0, 600.0),
        "Elements": (("elements",), 3, 12),
        "Tangential inlets": (("swirl", "n_inlets"), 2, 6),
    }
    s1, s2, s3, s4 = st.columns([2, 1, 1, 1])
    pname = s1.selectbox("Parameter", list(params))
    path, lo, hi = params[pname]
    is_int = isinstance(lo, int)
    lo_v = s2.number_input("From", value=lo, key="sw_lo")
    hi_v = s3.number_input("To", value=hi, key="sw_hi")
    n_v = s4.number_input("Steps", 2, 15, 6 if not is_int else int(hi - lo + 1), key="sw_n")
    if path == ("oxidizer", "temperature_K") and cfg["oxidizer"]["temperature_K"] is None:
        st.info("The oxidizer is defined by quality; switch to temperature to sweep it.")
    elif st.button("Run sweep"):
        vals = sorted({int(round(lo_v + i * (hi_v - lo_v) / (n_v - 1))) for i in range(int(n_v))}) if is_int \
            else [lo_v + i * (hi_v - lo_v) / (n_v - 1) for i in range(int(n_v))]
        df = sweep(cfg, path, tuple(vals))
        df.index.name = pname
        if "error" in df.columns:
            for v, e in df["error"].dropna().items():
                st.warning(f"{pname} = {v}: {e}")
            df = df.drop(columns="error").dropna()
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Geometry [mm]**")
            st.line_chart(df[["d_out [mm]", "d_inlet [mm]", "d_gas_out [mm]", "d_ox_hole [mm]", "recess [mm]"]])
        with g2:
            st.markdown("**Momentum flux ratio at nominal thrust**")
            st.line_chart(df[["J (S-L)", "J min", "J max"]])
        st.dataframe(df.round(3))

# ============================== TESTING =====================================
with tab_test:
    built = {k: v for k, v in good.items() if v.get("as_built")}
    if not built:
        st.info("No config with an as_built block found. Add one (see configs/e2_lcsc_p03_3x_as_built.yaml).")
        st.stop()

    name = st.selectbox("As-built configuration", list(built), key="test_cfg")
    cfg_t = built[name]
    g_t, _ = L.size_element(cfg_t)
    g_t = L.apply_as_built(L.round_geometry(g_t, cfg_t), cfg_t)
    n_el_t = cfg_t["elements"]

    with st.expander("Geometry and model settings", expanded=False):
        st.dataframe(geometry_table(g_t), hide_index=True)
        st.caption(f"Elements: {n_el_t} · swirl C_D factor {cfg_t['calibration']['swirl_cd_factor']} · "
                   f"ox hole C_D {cfg_t['ox_holes']['discharge_coefficient']} · hole model {cfg_t['ox_holes']['model']}")

    # These two switch the defaults of the inputs below, so they sit outside the form
    # (widgets inside a form do not rerun the page until the form is submitted).
    m1, m2 = st.columns(2)
    per_el = m1.checkbox("Flows are per element", value=False,
                         help=f"Unchecked: enter totals for the head, divided by {n_el_t} elements.")
    ox_mode_t = m2.radio("Ox state at holes", ["Temperature", "Saturated (quality)"], horizontal=True)

    with st.form("test_form"):
        a, b, c = st.columns(3)
        with a:
            test_id = st.text_input("Test ID", placeholder="e.g. C03-R2")
            date = st.date_input("Date", dt.date.today())
            pc_m = st.number_input("Chamber pressure [bar]", 0.5, 200.0, float(cfg_t["operating_point"]["chamber_pressure_bar"]))
        with b:
            st.markdown("**Fuel**")
            mf_m = st.number_input("Fuel mass flow [g/s]", 0.1, 20000.0,
                                   1000 * L.per_element_flows(cfg_t)[1] * (1 if per_el else n_el_t))
            pf_m = st.number_input("Fuel manifold pressure [bar]", 0.5, 300.0,
                                   float(cfg_t["operating_point"]["chamber_pressure_bar"] + cfg_t["fuel"]["pressure_drop_bar"]))
            Tf_m = st.number_input("Fuel temperature [K]", 200.0, 500.0, float(cfg_t["fuel"]["temperature_K"]))
        with c:
            st.markdown("**Oxidizer**")
            mo_m = st.number_input("Ox mass flow [g/s]", 0.1, 50000.0,
                                   1000 * L.per_element_flows(cfg_t)[0] * (1 if per_el else n_el_t))
            po_m = st.number_input("Ox manifold pressure [bar]", 0.5, 300.0,
                                   float(cfg_t["operating_point"]["chamber_pressure_bar"] + cfg_t["oxidizer"]["pressure_drop_bar"]))
            ox_val = st.number_input("Ox temperature [K] or quality [-]", 0.0, 900.0,
                                     float(cfg_t["oxidizer"]["temperature_K"] or 0.0) if ox_mode_t == "Temperature" else 0.0)
        notes = st.text_input("Notes")
        submitted = st.form_submit_button("Evaluate", type="primary")

    if submitted:
        k = 1 if per_el else n_el_t
        T_ox = ox_val if ox_mode_t == "Temperature" else None
        x_ox = ox_val if ox_mode_t != "Temperature" else None
        if x_ox is not None and not 0 <= x_ox <= 1:
            st.error("Quality must be between 0 and 1.")
            st.stop()
        try:
            r = L.evaluate_test_point(cfg_t, g_t, pc_m * BAR, mf_m / 1e3 / k, pf_m * BAR, Tf_m,
                                      mo_m / 1e3 / k, po_m * BAR, T_ox=T_ox, x_ox=x_ox)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()
        st.session_state["last_eval"] = {"r": r, "row": {
            "logged": dt.datetime.now().isoformat(timespec="seconds"), "config": name, "test_id": test_id,
            "date": str(date), "notes": notes, "pc_bar": pc_m, "mdot_fuel_g_s_head": mf_m * n_el_t / k,
            "p_fuel_bar": pf_m, "T_fuel_K": Tf_m, "mdot_ox_g_s_head": mo_m * n_el_t / k, "p_ox_bar": po_m,
            "ox_T_K": T_ox, "ox_quality": x_ox}}

    if "last_eval" in st.session_state and st.session_state["last_eval"]["row"]["config"] == name:
        r, row = st.session_state["last_eval"]["r"], st.session_state["last_eval"]["row"]
        m = st.columns(5)
        m[0].metric("Fuel C_D measured", f"{r['fuel_cd_meas']:.4f}", f"model {r['fuel_cd_model']:.4f}", delta_color="off")
        m[1].metric("Suggested swirl C_D factor", f"{r['swirl_cd_factor_suggested']:.2f}")
        m[2].metric("Ox hole C_D measured", f"{r['ox_hole_cd_meas']:.3f}",
                    f"config {cfg_t['ox_holes']['discharge_coefficient']}", delta_color="off")
        m[3].metric("O/F", f"{r['of_ratio']:.2f}")
        m[4].metric("J (S-L)", f"{r['J_SL']:.2f}")

        cmp = pd.DataFrame([
            {"Quantity": "Fuel Δp [bar]", "Measured": r["fuel_dp_meas_bar"], "Model at measured flow": r["fuel_dp_pred_bar"]},
            {"Quantity": "Ox manifold pressure [bar]", "Measured": row["p_ox_bar"], "Model at measured flow": r["ox_feed_pred_bar"]},
            {"Quantity": "Fuel stiffness [-]", "Measured": r["fuel_stiffness_meas"], "Model at measured flow": r["fuel_dp_pred_bar"] / row["pc_bar"]},
            {"Quantity": "Ox stiffness [-]", "Measured": r["ox_stiffness_meas"], "Model at measured flow": (r["ox_feed_pred_bar"] - row["pc_bar"]) / row["pc_bar"]},
        ]).round(3)
        st.dataframe(cmp, hide_index=True)
        if r["ox_hole_cd_meas"] > 1.0:
            st.warning("Oxidizer hole C_D above 1: the measured flow cannot pass the holes in the stated state. "
                       "The oxidizer is probably colder, denser or two-phase at the holes, or the pressure sensor "
                       "is not directly upstream of them.")
        if not 0.3 < r["swirl_cd_factor_suggested"] < 1.5:
            st.warning("Suggested swirl C_D factor far from 1: check the fuel pressure location, the flow reading "
                       "and the as-built fuel outlet and inlet diameters.")
        st.caption(f"Oxidizer at holes: {r['ox_inlet_phase']}, {r['ox_inlet_T_K']:.0f} K · holes "
                   f"{'choked' if r['ox_holes_choked'] else 'not choked'} · annulus exit {r['ox_exit_velocity_m_s']:.0f} m/s, "
                   f"{r['ox_exit_density']:.1f} kg/m³ · predicted spray half-angle {r['spray_half_angle_pred_deg']:.1f}° (J = 0)")
        st.code(f"calibration:\n  swirl_cd_factor: {r['swirl_cd_factor_suggested']:.3f}\n"
                f"ox_holes:\n  discharge_coefficient: {r['ox_hole_cd_meas']:.3f}", language="yaml")

        if st.button("Add to test log"):
            log_row = {**row, **{k2: r[k2] for k2 in ("fuel_cd_meas", "fuel_cd_model", "swirl_cd_factor_suggested",
                                                      "ox_hole_cd_meas", "fuel_dp_pred_bar", "ox_feed_pred_bar",
                                                      "ox_inlet_phase", "J_SL", "of_ratio")}}
            df_new = pd.DataFrame([log_row])
            df_new.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
            st.success(f"Logged to {os.path.basename(LOG_FILE)}")

    st.divider()
    st.subheader("Test log")
    if os.path.exists(LOG_FILE):
        log = pd.read_csv(LOG_FILE)
        show_all = st.checkbox("Show all configs", value=False)
        view = log if show_all else log[log["config"] == name]
        st.dataframe(view.round(4), hide_index=True)
        if len(view) and view["swirl_cd_factor_suggested"].notna().any():
            st.caption(f"Mean suggested swirl C_D factor: {view['swirl_cd_factor_suggested'].mean():.3f} · "
                       f"mean ox hole C_D: {view['ox_hole_cd_meas'].mean():.3f} (n = {len(view)})")
        st.download_button("Download log (CSV)", log.to_csv(index=False), "test_log.csv", "text/csv")
    else:
        st.caption("No entries yet.")