"""Run every thrust chamber reference analysis for the H2 lander.

    python run_all.py            -> out/*.csv, out/*.png, out/results.json

Each analysis is a function; they share the SI CEA front end (h2cea.cea_si) and
the baseline inputs (h2cea.config). Numbers quoted in the Thrust Chamber Design
Handbook (H2-PRP-HBK-001 Rev B) come from out/results.json.
"""
import json
import os

import numpy as np
import pandas as pd

from h2cea import config as C
from h2cea.cea_si import point, species, pinj_over_pcomb, full_output, cea_obj
from h2cea.propellants import n2o_card, ethanol_card, n2o_enthalpy_below_ideal_gas
from h2cea.operating_line import size_throat, state, thrust_to_mox
from h2cea.contour import e2_contour, L_star
from h2cea.bartz import bartz_profile
from h2cea import gasdyn as gd
from h2cea import n2o
from h2cea.plotstyle import plt, SERIES, BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED, GRID, \
    label_end, note

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
R = {}                       # key results for the handbook
LIQ = n2o_card(C.T_OX_NOM, 70.0)
GAS = "N2O"                  # RocketCEA built-in: ideal gas at 298.15 K
MRS = np.round(np.arange(1.5, 7.01, 0.1), 2)
FR = np.array(sorted(set(np.round(np.arange(0.40, 1.061, 0.02), 3)) | {0.5, 0.74, 0.81, 1.0}))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name.replace(".png", ".svg")), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 01
def a01_mixture_ratio_sweep():
    rows = []
    for card, lab in ((LIQ, "liquid 293 K"), (GAS, "gas 298 K")):
        for mr in MRS:
            p = point(C.PC_MAX, float(mr), C.EPS, ox=card)
            pf_isp_amb = (p.isp_vac_frozen * C.G0 / p.cstar - C.P_AMB / p.pc * p.eps) \
                * p.cstar / C.G0
            rows.append({"card": lab, "MR": mr, "Tc_K": p.Tc, "Tt_K": p.Tt,
                         "cstar_ms": p.cstar, "isp_sl_s": p.isp_amb,
                         "isp_sl_frozen_s": pf_isp_amb, "isp_vac_s": p.isp_vac,
                         "cf_sl": p.cf_amb, "cf_vac": p.cf_vac, "pe_bar": p.pe,
                         "Me": p.Me, "mw": p.mw_c, "gam_frozen": p.gam_fr_c,
                         "gam_eq": p.gam_eq_c, "cp_fr": p.cp_fr_c, "mu_Pas": p.mu_c,
                         "k_WmK": p.k_fr_c, "Pr": p.pr_fr_c})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "01_mr_sweep_25bar_eps4.csv"), index=False)
    liq = df[df.card == "liquid 293 K"].set_index("MR")
    gas = df[df.card == "gas 298 K"].set_index("MR")
    R["mr_opt_cstar"] = float(liq.cstar_ms.idxmax())
    R["mr_opt_isp_sl"] = float(liq.isp_sl_s.idxmax())
    R["mr_peak_Tc"] = float(liq.Tc_K.idxmax())
    R["ref_table_liquid"] = liq.loc[[3.0, 3.5, 4.0, 4.5, 5.0, 5.7],
                                    ["Tc_K", "Tt_K", "mw", "gam_frozen", "cstar_ms",
                                     "pe_bar", "isp_sl_s", "cf_sl", "isp_sl_frozen_s"]
                                    ].round(4).reset_index().to_dict("records")
    R["gas_vs_liquid_at_MR4"] = {"dTc": float(gas.Tc_K[4.0] - liq.Tc_K[4.0]),
                                 "dcstar_pct": float(100 * (gas.cstar_ms[4.0] / liq.cstar_ms[4.0] - 1)),
                                 "disp_pct": float(100 * (gas.isp_sl_s[4.0] / liq.isp_sl_s[4.0] - 1))}
    R["frozen_vs_eq_isp_sl_pct_MR4"] = float(100 * (1 - liq.isp_sl_frozen_s[4.0] / liq.isp_sl_s[4.0]))
    R["frozen_vs_eq_isp_sl_pct_MR2.4"] = float(100 * (1 - liq.isp_sl_frozen_s[2.4] / liq.isp_sl_s[2.4]))

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title, unit in ((axs[0], "Tc_K", "Chamber temperature", "K"),
                                 (axs[1], "cstar_ms", "Characteristic velocity c*", "m/s"),
                                 (axs[2], "isp_sl_s", "Sea-level Isp, ideal", "s")):
        ax.axvspan(2.35, 4.37, color=GRID, alpha=0.6, lw=0)
        ax.plot(liq.index, liq[col], color=BLUE)
        ax.plot(gas.index, gas[col], color=ORANGE, lw=1.5, ls="--")
        ax.axvline(5.73, color=MUTED, lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Mixture ratio O/F")
        ax.set_ylabel(unit)
    y = liq["isp_sl_s"]
    label_end(axs[2], 7.0, y[7.0] - 2.5, "liquid N₂O\n(tank state)", BLUE)
    label_end(axs[2], 7.0, gas["isp_sl_s"][7.0] + 3.5, "gaseous N₂O\n(built-in card)", ORANGE)
    axs[0].text(3.36, axs[0].get_ylim()[0] + 40, "throttle range\n50–100 %", ha="center",
                fontsize=8, color=INK2)
    axs[0].text(5.8, axs[0].get_ylim()[0] + 40, "stoich.", fontsize=8, color=MUTED)
    for ax in axs:
        ax.set_xlim(1.5, 8.6)
    note(fig, "RocketCEA, shifting equilibrium, pc 25 bar, ε 4, pa 1.013 bar; ethanol 100 % liquid 298 K.")
    fig.tight_layout()
    save(fig, "01_mixture_ratio_sweep.png")


# --------------------------------------------------------------------------- 02
def a02_operating_line():
    mr_d, At = size_throat(ox=LIQ)
    Dt = 2 * np.sqrt(At / np.pi)
    mr_a, At_a = size_throat(ox=GAS, flat_eta_isp=0.93)
    fr = FR
    rows = []
    for f in fr:
        m = thrust_to_mox(f * C.F_MAX, At, ox=LIQ)
        s = state(m, At, ox=LIQ)
        ma = thrust_to_mox(f * C.F_MAX, At_a, ox=GAS, flat_eta_isp=0.93)
        sa = state(ma, At_a, ox=GAS, flat_eta_isp=0.93)
        rows.append({"thrust_frac": f, "F_N": s["F"], "mox": s["mox"], "mdot": s["mdot"],
                     "MR": s["MR"], "pc_bar": s["pc"], "isp_del_s": s["isp"],
                     "isp_ideal_s": s["isp_ideal"], "eta_isp": s["eta_isp"],
                     "Tc_K": s["Tc"], "pe_bar": s["pe"], "Me": s["Me"],
                     "arch_mox": sa["mox"], "arch_MR": sa["MR"], "arch_pc": sa["pc"],
                     "arch_isp": sa["isp"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "02_operating_line.csv"), index=False)
    key = {}
    for f in (1.0, 0.81, 0.74, 0.50):
        m = thrust_to_mox(f * C.F_MAX, At, ox=LIQ)
        s = state(m, At, ox=LIQ)
        key[f"{int(f*100)}"] = {k: round(float(s[k]), 4) for k in
                                ("F", "mox", "mdot", "MR", "pc", "isp", "isp_ideal", "eta_isp", "Tc", "pe")}
        ma = thrust_to_mox(f * C.F_MAX, At_a, ox=GAS, flat_eta_isp=0.93)
        sa = state(ma, At_a, ox=GAS, flat_eta_isp=0.93)
        key[f"{int(f*100)}"]["arch"] = {k: round(float(sa[k]), 4) for k in ("mox", "MR", "pc", "isp")}
    R["operating_line"] = key
    R["throat_liquid_split"] = {"MR": mr_d, "Dt_mm": Dt * 1e3, "At_m2": At}
    R["throat_arch_method"] = {"MR": mr_a, "Dt_mm": 2e3 * np.sqrt(At_a / np.pi)}
    # E2 as-built throat
    At_e2 = np.pi * C.R_T ** 2
    s25 = None
    m = thrust_to_mox(C.F_MAX, At_e2, ox=LIQ)
    s = state(m, At_e2, ox=LIQ)
    mrs = np.linspace(3.5, 5.0, 61)
    F25 = []
    for mr in mrs:
        from scipy.optimize import brentq
        mm = brentq(lambda x: state(x, At_e2, ox=LIQ)["pc"] - C.PC_MAX, 0.4, 1.3)
        s25 = state(mm, At_e2, ox=LIQ)
        break
    R["e2_asbuilt_throat"] = {"Dt_mm": 28.14, "F_at_25bar": s25["F"], "mox_at_25bar": s25["mox"],
                              "pc_for_2250N": s["pc"], "mox_for_2250N": s["mox"], "MR_for_2250N": s["MR"]}

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    x = df.F_N / 1e3
    pairs = ((axs[0, 0], "pc_bar", "arch_pc", "Chamber pressure", "bar"),
             (axs[0, 1], "MR", "arch_MR", "Mixture ratio O/F", "–"),
             (axs[1, 0], "mox", "arch_mox", "Oxidiser flow", "kg/s"),
             (axs[1, 1], "isp_del_s", "arch_isp", "Delivered sea-level Isp", "s"))
    for ax, c1, c2, t, u in pairs:
        for fx, lab in ((0.5, "50 %"), (0.74, ""), (0.81, "hover"), (1.0, "100 %")):
            ax.axvline(fx * C.F_MAX / 1e3, color=GRID, lw=1.2, zorder=0)
        ax.plot(x, df[c1], color=BLUE)
        ax.plot(x, df[c2], color=ORANGE, lw=1.5, ls="--")
        ax.set_title(t)
        ax.set_ylabel(u)
    axs[1, 1].plot(x, df.isp_ideal_s, color=AQUA, lw=1.5)
    label_end(axs[1, 1], x.iloc[-1], df.isp_ideal_s.iloc[-1], "ideal (CEA)", AQUA)
    label_end(axs[1, 1], x.iloc[-1], df.isp_del_s.iloc[-1] - 3, "delivered", BLUE)
    label_end(axs[1, 0], x.iloc[-1], df.mox.iloc[-1], "this handbook", BLUE, dy=7)
    label_end(axs[1, 0], x.iloc[-1], df.arch_mox.iloc[-1], "architecture Rev A", ORANGE, dy=-9)
    for ax in axs[1]:
        ax.set_xlabel("Sea-level thrust, kN")
        ax.set_xlim(0.85, 2.75)
    for f, lab in ((0.5, "50 %"), (0.775, "hover"), (1.0, "100 %")):
        axs[0, 0].text(f * C.F_MAX / 1e3, 26.3, lab, ha="center", fontsize=8, color=INK2)
    axs[0, 0].set_ylim(9, 27.5)
    note(fig, f"Fuel fixed at {C.MDOT_FUEL} kg/s, throat fixed by 2250 N at 25 bar. Blue: liquid-N₂O card, "
              f"η_c* {C.ETA_CSTAR}, η_CF,vac {C.ETA_CF_VAC}. Dashed: architecture Rev A method "
              f"(gaseous card, flat 93 % of Isp).")
    fig.tight_layout()
    save(fig, "02_operating_line.png")
    return At


# --------------------------------------------------------------------------- 03
def a03_species():
    sp_rows = []
    for mr in np.round(np.arange(1.5, 7.01, 0.25), 2):
        d = species(C.PC_MAX, float(mr), C.EPS, ox=LIQ, min_fraction=1e-5)
        d["MR"] = mr
        sp_rows.append(d)
    df = pd.DataFrame(sp_rows).fillna(0.0).set_index("MR")
    df.to_csv(os.path.join(OUT, "03_species_chamber.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    major = ["N2", "H2O", "CO", "CO2", "H2"]
    minor = ["OH", "H", "NO", "O2", "O"]
    for ax, names, title in ((axs[0], major, "Major species"), (axs[1], minor, "Dissociation products")):
        ax.axvspan(2.35, 4.37, color=GRID, alpha=0.6, lw=0)
        for i, n in enumerate(names):
            if n in df:
                ax.plot(df.index, df[n], color=SERIES[i], lw=1.8)
                label_end(ax, df.index[-1], df[n].iloc[-1], n, SERIES[i])
        ax.set_title(title)
        ax.set_xlabel("Mixture ratio O/F")
        ax.set_ylabel("Mole fraction, chamber")
        ax.set_xlim(1.5, 7.6)
    note(fig, "Chamber equilibrium at 25 bar, liquid N₂O / ethanol. Shaded: mixture ratios crossed between 50 % and 100 % thrust.")
    fig.tight_layout()
    save(fig, "03_species.png")
    R["species_MR4.37"] = {k: round(float(v), 4) for k, v in species(C.PC_MAX, 4.37, C.EPS, ox=LIQ).items()}


# --------------------------------------------------------------------------- 04
def a04_oxidiser_enthalpy():
    rows = []
    for T in (273.15, 278.15, 288.15, 293.15, 298.15, 303.15):
        card = n2o_card(T, 100.0)
        p = point(C.PC_MAX, 4.37, C.EPS, ox=card)
        rows.append({"case": "liquid at tank T, 100 bar", "T_K": T, "added_kJkg": 0.0,
                     "h_below_ig_kJkg": n2o_enthalpy_below_ideal_gas(T, 100.0) / 1e3,
                     "Tc": p.Tc, "cstar": p.cstar})
    for dh in (0, 25, 50, 100, 150, 200, 260):
        card = n2o_card(C.T_OX_NOM, 70.0, float(dh))
        p = point(C.PC_MAX, 4.37, C.EPS, ox=card)
        rows.append({"case": "liquid 293 K + returned heat", "T_K": C.T_OX_NOM, "added_kJkg": dh,
                     "h_below_ig_kJkg": n2o_enthalpy_below_ideal_gas(C.T_OX_NOM, 70) / 1e3 - dh,
                     "Tc": p.Tc, "cstar": p.cstar})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "04_oxidiser_enthalpy.csv"), index=False)
    b = df[df.case == "liquid 293 K + returned heat"].set_index("added_kJkg")
    R["dTc_per_50kJkg_MR4.37"] = float(b.Tc[50] - b.Tc[0])
    R["dcstar_pct_per_50kJkg"] = float(100 * (b.cstar[50] / b.cstar[0] - 1))
    t = df[df.case == "liquid at tank T, 100 bar"].set_index("T_K")
    R["tank_T_effect_278_298"] = {"dTc": float(t.Tc[298.15] - t.Tc[278.15]),
                                  "dcstar_pct": float(100 * (t.cstar[298.15] / t.cstar[278.15] - 1))}
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(b.index, 100 * (b.cstar / b.cstar[0] - 1), color=BLUE, marker="o", ms=4)
    pg = point(C.PC_MAX, 4.37, C.EPS, ox=GAS)
    dg = 100 * (pg.cstar / b.cstar[0] - 1)
    ax.axhline(dg, color=ORANGE, lw=1.5, ls="--")
    ax.text(5, dg - 0.1, "built-in gaseous card (298 K)", color=INK2, fontsize=8.5)
    ax.axvline(55, color=MUTED, lw=0.9, ls=":")
    ax.text(58, 0.05, "≈ jacket heat picked up\ndownstream of the throat", color=INK2, fontsize=8)
    ax.set_title("c* gained per kJ/kg of oxidiser enthalpy")
    ax.set_xlabel("Enthalpy added to liquid N₂O at 293 K, kJ/kg")
    ax.set_ylabel("Change in ideal c*, %")
    note(fig, "O/F 4.37, pc 25 bar. 260 kJ/kg takes liquid at 293 K / 70 bar to the ideal-gas reference state.")
    fig.tight_layout()
    save(fig, "04_oxidiser_enthalpy.png")


# --------------------------------------------------------------------------- 05
def a05_separation(At):
    fr = FR
    rows = []
    for eps in (3.0, 3.5, 4.0, 4.5, 5.0):
        for f in fr:
            m = thrust_to_mox(f * C.F_MAX, At, ox=LIQ, eps=eps)
            s = state(m, At, ox=LIQ, eps=eps)
            p = s["pt"]
            pe = p.pe * s["pc"] / p.pc
            lim = gd.schmucker_psep_over_pa(p.Me)
            rows.append({"eps": eps, "thrust_frac": f, "pc": s["pc"], "MR": s["MR"],
                         "pe_bar": pe, "pe_over_pa": pe / C.P_AMB, "Me": p.Me,
                         "schmucker": lim, "isp": s["isp"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "05_separation.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    sel = [(3.0, AQUA), (4.0, BLUE), (5.0, ORANGE)]
    for eps, col in sel:
        d = df[df.eps == eps]
        ax.plot(d.thrust_frac * 100, d.pe_over_pa, color=col)
        label_end(ax, 106, d.pe_over_pa.iloc[-1], f"ε = {eps:g}", col)
    d4 = df[df.eps == 4.0]
    ax.plot(d4.thrust_frac * 100, d4.schmucker, color=MUTED, lw=1.2, ls="--")
    ax.axhline(0.45, color=MUTED, lw=0.9, ls=":")
    ax.text(88, 0.465, "design rule 0.45", fontsize=8, color=INK2)
    ax.text(88, d4.schmucker.iloc[0] - 0.045, "Schmucker onset (ε 4)", fontsize=8, color=INK2)
    ax.axvspan(74, 81, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(77.5, 1.7, "hover", ha="center", fontsize=8, color=INK2)
    ax.axvline(50, color=GRID, lw=1.2, zorder=0)
    ax.set_xlim(40, 112)
    ax.set_title("Exit pressure against ambient along the throttle line")
    ax.set_xlabel("Thrust, % of 2250 N")
    ax.set_ylabel("p_e / p_a (1-D, CEA)")
    note(fig, "Fuel fixed at 0.20 kg/s, same throat for every ε; 1-D exit pressure (a bell wall is lower). Sea level.")
    fig.tight_layout()
    save(fig, "05_separation.png")
    res = {}
    for eps in (3.0, 3.5, 4.0, 4.5, 5.0):
        d = df[df.eps == eps].set_index("thrust_frac")
        ok = d[d.pe_over_pa >= 0.45]
        res[str(eps)] = {"pe_pa_at_50": float(d.pe_over_pa[0.5]), "isp_100": float(d.isp[1.0]),
                         "isp_50": float(d.isp[0.5]), "min_frac_rule": float(ok.index.min()) if len(ok) else None,
                         "schmucker_at_50": float(d.schmucker[0.5])}
    R["separation"] = res


# --------------------------------------------------------------------------- 06
def a06_transport_vs_approx():
    rows = []
    for mr in (2.35, 3.0, 4.0, 4.37):
        p = point(C.PC_MAX, mr, C.EPS, ox=LIQ)
        g = p.gam_fr_c
        pr_hh = 4 * g / (9 * g - 5)
        mu_hh = 1.184e-7 * p.mw_c ** 0.5 * p.Tc ** 0.6
        rows.append({"MR": mr, "Tc": p.Tc, "mu_cea": p.mu_c, "mu_HH": mu_hh,
                     "Pr_cea_frozen": p.pr_fr_c, "Pr_approx": pr_hh, "cp_frozen": p.cp_fr_c,
                     "cp_eq": p.cp_eq_c, "k_frozen": p.k_fr_c,
                     "bartz_group_ratio": (p.mu_c ** 0.2 / p.pr_fr_c ** 0.6) / (mu_hh ** 0.2 / pr_hh ** 0.6)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "06_transport_vs_approx.csv"), index=False)
    R["transport"] = df.round(6).to_dict("records")


# --------------------------------------------------------------------------- 07
def a07_bartz(At):
    c = e2_contour()
    R["e2_contour"] = {"L_chamber_mm": c["L_chamber"] * 1e3, "L_nozzle_mm": c["L_nozzle"] * 1e3,
                       "L_star_m": L_star(c), "CR": (C.R_C / C.R_T) ** 2, "Dc_mm": 2e3 * C.R_C}
    pd.DataFrame({"x_mm": c["x"] * 1e3, "r_mm": c["r"] * 1e3}).to_csv(
        os.path.join(OUT, "07_e2_reference_contour.csv"), index=False)
    # Use the E2 throat for the operating points of this contour
    At_e2 = np.pi * C.R_T ** 2
    res, prof = {}, {}
    for f, lab in ((1.0, "100 %"), (0.81, "81 %"), (0.74, "74 %"), (0.5, "50 %")):
        m = thrust_to_mox(f * C.F_MAX, At_e2, ox=LIQ)
        s = state(m, At_e2, ox=LIQ)
        b = bartz_profile(c, s["pt"], s["pc"], Twg=1000.0)
        b7 = bartz_profile(c, s["pt"], s["pc"], Twg=700.0)
        prof[lab] = b
        res[lab] = {"pc": s["pc"], "MR": s["MR"], "mox": s["mox"], "Tc": s["Tc"],
                    "Q_kW": b["Q"] / 1e3, "Q_up_frac": b["Q_upstream"] / b["Q"],
                    "q_throat_MWm2": b["q_max"] / 1e6, "x_qmax_mm": b["x_qmax"] * 1e3,
                    "dh_kJkg": b["Q"] / s["mox"] / 1e3, "dh_kJkg_0.7": 0.7 * b["Q"] / s["mox"] / 1e3,
                    "Q_kW_Twg700": b7["Q"] / 1e3,
                    "q_cyl_MWm2": float(np.interp(0.03, c["x"], b["q"])) / 1e6,
                    "hg_throat": float(b["hg"][np.argmax(b["q"])])}
    R["bartz"] = res
    rows = []
    for lab, b in prof.items():
        for xi, ri, qi, hi, Mi in zip(b["x"][::10], b["r"][::10], b["q"][::10], b["hg"][::10], b["M"][::10]):
            rows.append({"case": lab, "x_mm": xi * 1e3, "r_mm": ri * 1e3, "q_MWm2": qi / 1e6,
                         "hg": hi, "M": Mi})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "07_bartz_profiles.csv"), index=False)
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 2.4]})
    k = 8   # decimate for a light SVG
    ax0.fill_between(c["x"][::k] * 1e3, c["r"][::k] * 1e3, 0, color="#dcdbd6", lw=0)
    ax0.plot(c["x"][::k] * 1e3, c["r"][::k] * 1e3, color=INK2, lw=1.2)
    ax0.set_ylim(0, 55)
    ax0.set_ylabel("r, mm")
    ax0.set_title("E2-REG-1 reference contour (ε 4) and Bartz heat flux")
    ax0.set_aspect("auto")
    for (lab, b), col in zip(prof.items(), (BLUE, ORANGE, AQUA, YELLOW)):
        if lab == "74 %":
            continue
        ax.plot(b["x"][::k] * 1e3, b["q"][::k] / 1e6, color=col, lw=1.8)
        i = np.argmax(b["q"])
        label_end(ax, b["x"][i] * 1e3 + 6, b["q"][i] / 1e6, f"{lab}: {b['q_max']/1e6:.1f} MW/m²", col)
    ax.set_xlabel("Axial position from injector face, mm")
    ax.set_ylabel("Heat flux, MW/m²")
    note(fig, "Bartz, frozen CEA transport at chamber, gas-side wall 1000 K, no film cooling, factor 1.0. "
              "Operating points on the fixed-fuel line with the as-built 28.14 mm throat.")
    fig.tight_layout()
    save(fig, "07_bartz_heat_flux.png")


# --------------------------------------------------------------------------- 08
def a08_jacket_regime():
    At_e2 = np.pi * C.R_T ** 2
    m_nom = thrust_to_mox(C.F_MAX, At_e2, ox=LIQ)
    fr = FR
    rows = []
    for f in fr:
        m = thrust_to_mox(f * C.F_MAX, At_e2, ox=LIQ)
        s = state(m, At_e2, ox=LIQ)
        for dpj in C.DP_JACKET_NOM:
            pin, pout = n2o.jacket_pressures(m, s["pc"], m_nom, dp_jacket_nom=dpj)
            rows.append({"thrust_frac": f, "mox": m, "pc": s["pc"], "dp_jacket_nom": dpj,
                         "p_jacket_in": pin, "p_jacket_out": pout})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "08_jacket_pressure.csv"), index=False)
    ps = {T: n2o.psat(T) for T in (278.15, 293.15, 298.15)}
    R["psat_bar"] = {f"{T-273.15:.0f}C": v for T, v in ps.items()}
    res = {}
    for f in (1.0, 0.81, 0.74, 0.5):
        d = df[np.isclose(df.thrust_frac, f)]
        res[str(f)] = {f"pin_dp{int(r.dp_jacket_nom)}": float(r.p_jacket_in) for r in d.itertuples()}
        res[str(f)]["pout"] = float(d.p_jacket_out.iloc[0])
    R["jacket_pressure"] = res
    R["venturi_max_outlet"] = C.VENTURI_RECOVERY * C.P_VENTURI_IN
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    lo = df[df.dp_jacket_nom == C.DP_JACKET_NOM[0]]
    hi = df[df.dp_jacket_nom == C.DP_JACKET_NOM[1]]
    x = lo.thrust_frac * 100
    ax.fill_between(x, lo.p_jacket_in, hi.p_jacket_in, color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, 0.5 * (lo.p_jacket_in.values + hi.p_jacket_in.values), color=BLUE)
    ax.plot(x, lo.p_jacket_out, color=AQUA)
    ax.plot(x, lo.pc, color=MUTED, lw=1.3)
    label_end(ax, 106, 0.5 * (lo.p_jacket_in.iloc[-1] + hi.p_jacket_in.iloc[-1]),
              "jacket inlet\n(Δp 20–30 bar at 100 %)", BLUE)
    label_end(ax, 106, lo.p_jacket_out.iloc[-1], "jacket outlet =\ninjector inlet", AQUA)
    label_end(ax, 106, lo.pc.iloc[-1], "chamber", MUTED)
    for T, v in ps.items():
        ax.axhline(v, color=ORANGE, lw=0.9, ls="--")
        ax.text(41, v + 0.8, f"N₂O vapour pressure at {T-273.15:.0f} °C", fontsize=7.8, color=INK2)
    ax.axhline(n2o.P_CRIT, color=INK2, lw=0.9, ls=":")
    ax.text(41, n2o.P_CRIT - 3.2, "critical pressure 72.4 bar", fontsize=7.8, color=INK2)
    ax.axvspan(74, 81, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(77.5, 2, "hover", ha="center", fontsize=8, color=INK2)
    ax.axhline(C.VENTURI_RECOVERY * C.P_VENTURI_IN, color=MUTED, lw=0.9, ls="-.")
    ax.text(41, C.VENTURI_RECOVERY * C.P_VENTURI_IN + 0.8, "venturi recovery limit ≈ 74 bar",
            fontsize=7.8, color=INK2)
    ax.set_xlim(40, 128)
    ax.set_ylim(0, 82)
    ax.set_title("Where the jacket sits: throttle upstream of the jacket")
    ax.set_xlabel("Thrust, % of 2250 N")
    ax.set_ylabel("Pressure, bar")
    note(fig, "First-order scaling: injector p_in² − p_c² ∝ ṁ², jacket Δp ∝ ṁ², anchored at 40 bar injector inlet "
              "at 100 %. Below a vapour-pressure line the N₂O leaving the venturi at that temperature is two-phase.")
    fig.tight_layout()
    save(fig, "08_jacket_regime_map.png")


# --------------------------------------------------------------------------- 09
def a09_chf_estimate():
    """Order-of-magnitude CHF at the throat channel against the Bartz throat flux."""
    rows = []
    D = 1.0e-3
    cases = {"100 %: 65 bar, liquid 293 K": (65.0, n2o.h_TP(293.15, 65.0)),
             "hover 74 %: 45 bar, saturated liquid": (45.0, n2o.sat_props(45.0)["h_l"]),
             "supercritical reference: 80 bar, 293 K": None}
    for G in np.linspace(5000, 40000, 36):
        for lab, v in cases.items():
            if v is None:
                continue
            p, h = v
            q, x = n2o.chf_hall_mudawar(G, D, p, h)
            rows.append({"case": lab, "G": G, "q_chf_MWm2": q / 1e6, "x_local": x})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "09_chf_estimate.csv"), index=False)
    R["chf"] = {lab: {"at_G20000": float(df[(df.case == lab) & np.isclose(df.G, 20000)].q_chf_MWm2.iloc[0]),
                      "at_G30000": float(df[(df.case == lab) & np.isclose(df.G, 30000)].q_chf_MWm2.iloc[0])}
                for lab in df.case.unique()}
    fig, ax = plt.subplots(figsize=(7, 4))
    for (lab, d), col, dy in zip(df.groupby("case", sort=False), (BLUE, ORANGE), (-6, 6)):
        ax.plot(d.G / 1e3, d.q_chf_MWm2, color=col)
        label_end(ax, 40.5, d.q_chf_MWm2.iloc[-1], lab.split(":")[0], col, dy=dy)
    qb = R["bartz"]
    ax.axhspan(qb["74 %"]["q_throat_MWm2"] * 0.7, qb["100 %"]["q_throat_MWm2"], color=YELLOW, alpha=0.18, lw=0)
    ax.text(6, qb["100 %"]["q_throat_MWm2"] - 1.2, "throat heat flux, Bartz ×0.7 to ×1.0,\nhover to full thrust",
            fontsize=8, color=INK2)
    ax.set_xlim(5, 48)
    ax.set_ylim(0, max(qb["100 %"]["q_throat_MWm2"] + 2, 10))
    ax.set_title("Critical heat flux estimate at the throat channel")
    ax.set_xlabel("Coolant mass flux G, 10³ kg/(m² s)")
    ax.set_ylabel("Heat flux, MW/m²")
    note(fig, "Hall–Mudawar (2000) water correlation in dimensionless form, applied to N₂O (extrapolated, p/p_crit 0.6–0.9), "
              "D_h 1 mm. Order of magnitude only.")
    fig.tight_layout()
    save(fig, "09_chf_estimate.png")


# --------------------------------------------------------------------------- 11
def a11_coolant_path():
    """Jacket process on the N2O p-h diagram and the outlet state (handbook 10.2)."""
    At_e2 = np.pi * C.R_T ** 2
    m_nom = thrust_to_mox(C.F_MAX, At_e2, ox=LIQ)
    h_tank = n2o.h_TP(C.T_OX_NOM, C.P_TANK)      # venturi and valves are isenthalpic
    rows, paths = [], {}
    for f, lab in ((1.0, "100 %"), (0.81, "81 %"), (0.74, "74 %"), (0.5, "50 %")):
        m = thrust_to_mox(f * C.F_MAX, At_e2, ox=LIQ)
        s = state(m, At_e2, ox=LIQ)
        pin, pout = n2o.jacket_pressures(m, s["pc"], m_nom, dp_jacket_nom=25.0)
        Q = R["bartz"][lab]["Q_kW"] * 1e3
        st_in = n2o.state_hp(h_tank, pin)
        for fac in (1.0, 0.7):
            h_out = h_tank + fac * Q / m
            st = n2o.state_hp(h_out, pout)
            Tsat_out = n2o.tsat(pout)
            rows.append({"case": lab, "bartz_factor": fac, "mox": m, "p_in": pin, "p_out": pout,
                         "x_in": st_in["x"], "T_in": st_in["T"], "dh_kJkg": fac * Q / m / 1e3,
                         "T_out": st["T"], "rho_out": st["rho"], "x_out": st["x"],
                         "superheat_out_K": st["T"] - Tsat_out if (st["x"] or 0) >= 1 else 0.0})
        paths[lab] = (h_tank, pin, h_tank + Q / m, pout)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "11_jacket_outlet_state.csv"), index=False)
    R["jacket_states"] = df.round(4).to_dict("records")
    R["h_tank_kJkg"] = h_tank / 1e3

    # p-h diagram
    from CoolProp.CoolProp import PropsSI
    ps = np.linspace(10e5, n2o.P_CRIT * 1e5 - 2e3, 200)
    hl = [PropsSI("H", "P", p, "Q", 0, "NitrousOxide") / 1e3 for p in ps]
    hv = [PropsSI("H", "P", p, "Q", 1, "NitrousOxide") / 1e3 for p in ps]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(hl, ps / 1e5, color=INK2, lw=1.2)
    ax.plot(hv, ps / 1e5, color=INK2, lw=1.2)
    ax.plot([hl[-1]], [n2o.P_CRIT], "o", color=INK2, ms=4)
    ax.text(hl[-1] + 6, n2o.P_CRIT + 1.5, "critical point", fontsize=8, color=INK2)
    for T in (293.15, 320.0, 360.0, 400.0):
        pp = np.linspace(10e5, 90e5, 120)
        hh = [PropsSI("H", "T", T, "P", p, "NitrousOxide") / 1e3 for p in pp]
        ax.plot(hh, pp / 1e5, color=GRID, lw=1.0, zorder=0)
        ax.text(hh[5] + 3, 12, f"{T:.0f} K", fontsize=7.5, color=MUTED, rotation=90)
    for (lab, (h0, p0, h1, p1)), col in zip(paths.items(), (BLUE, ORANGE, AQUA, YELLOW)):
        if lab == "81 %":
            continue
        ax.annotate("", xy=(h1 / 1e3, p1), xytext=(h0 / 1e3, p0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        ax.plot([h0 / 1e3], [p0], "o", color=col, ms=5)
        ax.text(h1 / 1e3 + 4, p1, lab, fontsize=8.5, color=INK2, va="center")
    ax.plot([h_tank / 1e3], [C.P_TANK], "s", color=INK2, ms=5)
    ax.text(h_tank / 1e3 + 4, C.P_TANK, "tank, 293 K / 100 bar", fontsize=8, color=INK2, va="center")
    ax.plot([h_tank / 1e3] * 2, [C.P_TANK, 25], color=MUTED, lw=0.8, ls=":")
    ax.text(h_tank / 1e3 - 4, 88, "valve + venturi:\nisenthalpic drop", fontsize=7.8, color=INK2, ha="right")
    ax.text(0.5 * (hl[40] + hv[40]), ps[40] / 1e5 - 3, "two-phase dome", fontsize=8, color=MUTED, ha="center")
    ax.set_xlim(180, 560)
    ax.set_ylim(10, 105)
    ax.set_title("Oxidiser path through the cooling jacket (N₂O p–h diagram)")
    ax.set_xlabel("Specific enthalpy, kJ/kg (CoolProp reference)")
    ax.set_ylabel("Pressure, bar")
    note(fig, "Arrows: jacket inlet (after the venturi) to jacket outlet at 100 %, 74 % and 50 % thrust; "
              "heat from Bartz ×1.0 on the E2 contour; jacket Δp 25 bar at 100 %, scaled with ṁ².")
    fig.tight_layout()
    save(fig, "11_jacket_path_ph.png")


# --------------------------------------------------------------------------- 10
def a10_misc():
    # Decomposition temperature of N2O as a monopropellant
    from rocketcea.cea_obj import add_new_propellant
    from rocketcea.cea_obj_w_units import CEA_Obj
    out = {}
    for lab, T, p, extra in (("gas 298 K", 298.15, 1.0, 0.0), ("liquid 293 K", 293.15, 70.0, 0.0),
                             ("heated 400 K, 40 bar", 400.0, 40.0, 0.0)):
        dh = n2o_enthalpy_below_ideal_gas(T, p)
        hf = 19467.0 * 4.184 - dh * 44.0128e-3
        name = f"N2Omono{int(T)}{int(p)}"
        add_new_propellant(name, f"name N2O N 2 O 1 wt%=100.0\nh,cal={hf/4.184:.1f} t(k)={T:.2f}\n")
        c = CEA_Obj(propName=name, pressure_units="bar", temperature_units="K", cstar_units="m/s")
        out[lab] = {"T_ad_K_at_40bar": float(c.get_Tcomb(Pc=40.0, MR=1.0)),
                    "T_ad_K_at_1bar": float(c.get_Tcomb(Pc=1.0, MR=1.0))}
    R["n2o_decomposition"] = out
    # Finite-area combustor loss vs contraction ratio
    fac = []
    for cr in (3, 4, 6, 8, 12):
        r = pinj_over_pcomb(C.PC_MAX, 4.37, float(cr), ox=LIQ)
        g = point(C.PC_MAX, 4.37, C.EPS, ox=LIQ).gam_fr_c
        Mc, loss = gd.rayleigh_p0_loss(cr, g)
        fac.append({"CR": cr, "pinj_over_pcomb_cea": r, "loss_cea_pct": 100 * (1 - 1 / r),
                    "Mc": Mc, "loss_rayleigh_pct": 100 * loss, "loss_gM2_2_pct": 100 * g * Mc ** 2 / 2})
    pd.DataFrame(fac).to_csv(os.path.join(OUT, "10_finite_area_combustor.csv"), index=False)
    R["finite_area"] = fac
    # Chamber acoustic modes over the throttle range (E2 chamber)
    At_e2 = np.pi * C.R_T ** 2
    ac = []
    for f in (1.0, 0.74, 0.5):
        m = thrust_to_mox(f * C.F_MAX, At_e2, ox=LIQ)
        s = state(m, At_e2, ox=LIQ)
        a = s["pt"].a_c
        L_eff = C.L_CYL + 0.5 * (e2_contour()["xt"] - C.L_CYL)
        ac.append({"thrust_frac": f, "a_c": a, "f_1T": 1.841 * a / (2 * np.pi * C.R_C),
                   "f_2T": 3.054 * a / (2 * np.pi * C.R_C), "f_1R": 3.832 * a / (2 * np.pi * C.R_C),
                   "f_1L": a / (2 * L_eff), "L_eff_mm": L_eff * 1e3})
    R["acoustics"] = ac
    # Constant-gamma error check (handbook 4.4)
    p = point(C.PC_MAX, 4.37, C.EPS, ox=LIQ)
    R_gas = 8314.462618 / p.mw_c
    cstar_fr = np.sqrt(R_gas * p.Tc) / gd.big_gamma(p.gam_fr_c)
    cstar_eq = np.sqrt(R_gas * p.Tc) / gd.big_gamma(p.gam_eq_c)
    cf_fr = gd.cf_ideal(p.gam_fr_c, C.EPS, C.PC_MAX, C.P_AMB)
    R["constant_gamma_check"] = {"cstar_cea": p.cstar, "cstar_gamma_frozen": cstar_fr,
                                 "cstar_gamma_eq": cstar_eq, "cf_cea": p.cf_amb, "cf_gamma_frozen": cf_fr,
                                 "gam_fr": p.gam_fr_c, "gam_eq": p.gam_eq_c}
    # Throat conditions
    R["throat_ratios"] = {"Tt_over_Tc": p.Tt / p.Tc}
    with open(os.path.join(OUT, "design_point_full_cea.txt"), "w") as fh:
        fh.write(full_output(C.PC_MAX, 4.37, C.EPS, ox=LIQ))
    with open(os.path.join(OUT, "floor_point_full_cea.txt"), "w") as fh:
        fh.write(full_output(13.9, 2.35, C.EPS, ox=LIQ))


if __name__ == "__main__":
    a01_mixture_ratio_sweep()
    At = a02_operating_line()
    a03_species()
    a04_oxidiser_enthalpy()
    a05_separation(At)
    a06_transport_vs_approx()
    a07_bartz(At)
    a08_jacket_regime()
    a09_chf_estimate()
    a11_coolant_path()
    a10_misc()
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))
