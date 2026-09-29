"""Figures for the regenerative cooling analyses (run_regen.py).

Same visual language as the rest of h2cea (plotstyle): light surface, thin marks,
fixed hue per quantity. Temperature fields use one warm sequential ramp.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from . import config as C
from .plotstyle import plt, BLUE, ORANGE, AQUA, RED, INK, INK2, MUTED, GRID

HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#fdf3e7", "#f9d7b0", "#f4b077", "#ec8545", "#d95f27", "#b3431b", "#842f14", "#55190b"])
WALL, WATERFILL, BLOCKFILL = "#9b958e", "#d6e6f8", "#e9ebee"
TLIM_C = C.T_WG_LIMIT - 273.15


def _c(T):
    return np.asarray(T) - 273.15


def axial_profiles(results, titles, fname, sections=None, suptitle=None):
    """results: list of RegenResult (one column each).
    sections: optional {case: {"x": [...], "nom": [...], "blk": [...]}} gas-side peaks (K)."""
    two_phase = any(r["two_phase_any"] for r in results)
    rows = 6 if two_phase else 5
    hr = [0.8, 1, 1.5, 0.9, 0.8] + ([0.8] if two_phase else [])
    fig, axs = plt.subplots(rows, len(results), figsize=(6.2 * len(results), 2.2 * sum(hr)),
                            sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=hr, hspace=0.3, wspace=0.12))
    for c, (res, title) in enumerate(zip(results, titles)):
        a = res.arrays
        x = a["x"] * 1e3
        r = a["r"] * 1e3
        j = res.jacket
        E = 3.0
        ax = axs[0, c]
        ax.fill_between(x, 0, r, color="#f3f4f6", lw=0)
        ax.fill_between(x, r, r + j.t_wall * 1e3 * E, color=WALL, lw=0, label="hot wall")
        ax.fill_between(x, r + j.t_wall * 1e3 * E, r + (j.t_wall + j.h) * 1e3 * E, color=BLUE, lw=0, label="channel")
        ax.fill_between(x, r + (j.t_wall + j.h) * 1e3 * E, r + j.depth * 1e3 * E, color="#c9c5bf", lw=0,
                        label="closeout (jacket drawn 3× thick)")
        ax.set_ylim(0, r.max() + j.depth * 1e3 * E + 6)
        ax.set_title(title)
        if c == 0:
            ax.set_ylabel("radius [mm]")
            ax.legend(loc="lower left", ncol=3, fontsize=7.5)
        ax = axs[1, c]
        ax.plot(x, a["q"] / 1e6, color=INK)
        i = int(np.nanargmax(a["q"]))
        ax.annotate(f"{a['q'][i]/1e6:.1f} MW/m²", (x[i], a["q"][i] / 1e6), xytext=(-60, -4),
                    textcoords="offset points", color=INK2, fontsize=8)
        if c == 0:
            ax.set_ylabel("heat flux [MW/m²]")
        ax = axs[2, c]
        if sections and res.case in sections:
            sc = sections[res.case]
            if sc.get("blk") is not None:
                ax.plot(np.array(sc["x"]) * 1e3, _c(sc["blk"]), color=RED, lw=1.2, ls=(0, (1, 2)),
                        label="gas wall over a blocked channel (2D)")
            ax.plot(np.array(sc["x"]) * 1e3, _c(sc["nom"]), color=ORANGE, lw=2, label="gas wall, peak over rib (2D)")
        ax.plot(x, _c(a["Twg"]), color=ORANGE, lw=1.2, ls="--", label="gas wall, 1D")
        ax.plot(x, _c(a["Twc"]), color=AQUA, label="coolant wall, 1D")
        if np.any(np.isfinite(a["Tsat"])):
            ax.plot(x, _c(a["Tsat"]), color=MUTED, lw=1.1, ls=(0, (4, 3)), label="saturation (local p)")
        ax.plot(x, _c(a["Tb"]), color=BLUE, label="coolant bulk")
        ax.axhline(TLIM_C, color=RED, lw=0.9, ls="--")
        ax.text(x[0] + 2, TLIM_C + 15, "IN718 limit 1050 K", color=RED, fontsize=8)
        top = max(np.nanmax(_c(a["Twg"])), TLIM_C)
        if sections and res.case in sections and sections[res.case].get("blk") is not None:
            top = max(top, np.nanmax(_c(sections[res.case]["blk"])))
        ax.set_ylim(min(0, np.nanmin(_c(a["Tb"])) - 10), top * 1.1)
        if c == 0:
            ax.set_ylabel("temperature [°C]")
            ax.legend(loc="upper right", fontsize=7.5)
        ax = axs[3, c]
        Rt = a["Rg"] + a["Rw"] + a["Rc"]
        g1 = a["Rg"] / Rt * 100
        g2 = g1 + a["Rw"] / Rt * 100
        ax.fill_between(x, 0, g1, color=ORANGE, alpha=0.85, lw=0, label="gas boundary layer")
        ax.fill_between(x, g1, g2, color=WALL, lw=0, label="wall")
        ax.fill_between(x, g2, 100, color=BLUE, lw=0, label="coolant film")
        ax.set_ylim(0, 100)
        ax.grid(False)
        if c == 0:
            ax.set_ylabel("share of ΔT [%]")
            ax.legend(loc="lower left", ncol=3, fontsize=7.5, labelcolor="white")
        ax = axs[4, c]
        ax.plot(x, a["p"] / 1e5, color=BLUE)
        if c == 0:
            ax.set_ylabel("coolant p [bar]")
        if two_phase:
            ax = axs[5, c]
            xq = np.clip(a["xq"], -0.2, 1.2)
            ax.plot(x, xq, color=INK)
            ax.axhspan(0, 1, color=GRID, alpha=0.6, lw=0)
            ax.set_ylim(-0.25, 1.25)
            if c == 0:
                ax.set_ylabel("quality x")
            ax.text(x[0] + 2, 0.5, "two-phase", color=INK2, fontsize=8, va="center")
        axs[-1, c].set_xlabel("x from injector face [mm]")
        axs[-1, c].set_xlim(x[0], x[-1])
    fig.subplots_adjust(top=1 - 0.5 / fig.get_figheight())
    if suptitle:
        fig.suptitle(suptitle, x=0.01, y=1 - 0.12 / fig.get_figheight(), ha="left", fontsize=12, fontweight="bold")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _draw_section(ax, sec, npitch, norm, title, Tb_label=True):
    G = sec["geom"]
    T = sec["T"] - 273.15
    yc = G["yc"]
    Tf = np.concatenate([T[::-1], T], axis=0)
    yf = np.concatenate([-yc[::-1], yc])
    m = np.abs(yf) <= npitch * G["p"] / 2 + 1e-12
    y, z, Tm = yf[m] * 1e3, G["zc"] * 1e3, Tf[m]
    H = G["H"] * 1e3
    dy = y[1] - y[0]
    im = ax.imshow(np.ma.masked_invalid(Tm.T), origin="lower", cmap=HEAT, norm=norm,
                   extent=[y[0] - dy / 2, y[-1] + dy / 2, 0, H], interpolation="bilinear", aspect="equal")
    ax.contour(y, z, np.ma.masked_invalid(Tm.T), levels=np.arange(0, 1500, 50), colors=INK,
               linewidths=0.45, alpha=0.28)
    p, w = G["p"] * 1e3, G["w"] * 1e3
    tw = (sec.get("t_wall") or 0.5e-3) * 1e3
    hch = (sec.get("h_ch") or 0.75e-3) * 1e3
    for k in range(-3, 4):
        a_, b_ = max(k * p - w / 2, y[0]), min(k * p + w / 2, y[-1])
        if b_ <= a_:
            continue
        blocked = sec["blocked"] and k == 0
        ax.add_patch(Rectangle((a_, tw), b_ - a_, hch, facecolor=BLOCKFILL if blocked else WATERFILL, edgecolor="none"))
        if Tb_label:
            ax.text((a_ + b_) / 2, tw + hch / 2, "blocked" if blocked else f"{sec['Tb']-273.15:.0f} °C",
                    ha="center", va="center", fontsize=7.5, color=MUTED if blocked else BLUE)
    Ts = sec["Ts"] - 273.15
    Tsf = np.concatenate([Ts[::-1], Ts])
    ys = yf * 1e3
    mm = (ys >= y[0]) & (ys <= y[-1])
    jj = int(np.argmax(np.where(mm, Tsf, -1e9)))
    ax.plot([ys[jj]], [0], marker="^", color=INK, ms=6, clip_on=False, zorder=5)
    xl = ys[jj] if abs(ys[jj] - y[0]) > 1e-9 else y[0] + 0.04 * (y[-1] - y[0])
    ax.text(xl, 0.12 * H / 2.25, f"peak {Tsf[jj]:.0f} °C", ha="center" if xl == ys[jj] else "left",
            va="center", fontsize=8, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(y[0], y[-1])
    ax.set_ylim(-0.02, H + 0.02)
    ax.grid(False)
    ax.set_title(title, fontsize=9.5)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    return im


def cross_sections(secs, titles, fname, npitch=1, vmax=None, suptitle=None, true_scale=True):
    """secs: list of rows, each a list of section dicts (e.g. [[thr50, cyl50], [thr100, cyl100]]).
    true_scale=False gives every panel the same width (stretched horizontally), which suits
    wide multi-pitch windows."""
    vmax = vmax or max(np.nanmax(s["Ts"]) for row in secs for s in row) - 273.15
    vmax = float(np.ceil(vmax / 100) * 100)
    norm = Normalize(0, vmax)
    H = secs[0][0]["geom"]["H"]
    hin = 1.8 if true_scale else 1.25
    ncol = max(len(row) for row in secs)
    if true_scale:
        wins = [hin * npitch * s["geom"]["p"] / H for s in secs[0]]
    else:
        wins = [10.0 / ncol] * ncol
    FW = sum(wins) + 0.45 * len(wins) + 2.0
    FH = (hin + 0.9) * len(secs) + 0.8
    fig = plt.figure(figsize=(FW, FH))
    im = None
    for r_, row in enumerate(secs):
        x0 = 0.7
        for c_, sec in enumerate(row):
            ax = fig.add_axes([x0 / FW, (FH - 0.7 - (r_ + 1) * (hin + 0.9) + 0.45) / FH, wins[c_] / FW, hin / FH])
            im = _draw_section(ax, sec, npitch, norm, titles[r_][c_])
            if not true_scale:
                ax.set_aspect("auto")
            if c_ == 0:
                ax.set_ylabel("z [mm]")
            x0 += wins[c_] + 0.45
    cax = fig.add_axes([(FW - 1.1) / FW, 0.45 / FH, 0.12 / FW, (FH - 1.2) / FH])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.outline.set_visible(False)
    cb.set_label("temperature [°C]", color=INK2)
    if vmax > TLIM_C:
        cb.ax.axhline(TLIM_C, color=RED, lw=2)
    if suptitle:
        fig.text(0.01, 1 - 0.15 / FH, suptitle, ha="left", va="top", fontsize=11, fontweight="bold")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def flow_sweep(rows, fname, title="Water flow"):
    """rows: list of dicts {case, mdot, Twg_max, margin_sat_min, dp_bar}."""
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for (case, d), col in zip(df.groupby("case", sort=False), (BLUE, ORANGE, AQUA)):
        axs[0].plot(d.mdot, _c(d.Twg_max), color=col, marker="o", ms=3, label=case)
        axs[1].plot(d.mdot, d.margin_sat_min, color=col, marker="o", ms=3, label=case)
        axs[2].plot(d.mdot, d.dp_bar, color=col, marker="o", ms=3, label=case)
    axs[0].axhline(TLIM_C, color=RED, lw=0.9, ls="--")
    axs[1].axhline(0, color=RED, lw=0.9, ls="--")
    axs[0].set_title("peak gas-side wall (1D)")
    axs[0].set_ylabel("°C")
    axs[1].set_title("smallest boiling margin")
    axs[1].set_ylabel("T_sat − T_wc [K]")
    axs[2].set_title("channel pressure drop")
    axs[2].set_ylabel("bar")
    for ax in axs:
        ax.set_xlabel(f"{title} [kg/s]")
        ax.axvline(C.WATER_MDOT, color=INK, lw=0.8, alpha=0.4)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def n2o_ph_paths(results, labels, fname):
    """Coolant path through the jacket on the N2O p-h diagram."""
    from CoolProp.CoolProp import PropsSI
    F = "NitrousOxide"
    pcrit = PropsSI("pcrit", F)
    ps = np.linspace(10e5, pcrit - 2e3, 200)
    hl = [PropsSI("H", "P", p, "Q", 0, F) / 1e3 for p in ps]
    hv = [PropsSI("H", "P", p, "Q", 1, F) / 1e3 for p in ps]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(hl, ps / 1e5, color=MUTED, lw=1)
    ax.plot(hv, ps / 1e5, color=MUTED, lw=1)
    ax.fill_betweenx(ps / 1e5, hl, hv, color=GRID, alpha=0.5, lw=0)
    ax.text(np.mean(hl + hv), 25, "two-phase", color=INK2, ha="center", fontsize=8.5)
    cols = (BLUE, ORANGE, AQUA, RED)
    for res, lab, col in zip(results, labels, cols):
        a = res.arrays
        ax.plot(a["h"] / 1e3, a["p"] / 1e5, color=col, label=lab)
        i_in = len(a["h"]) - 1 if res["flow"] == "counter" else 0
        ax.plot([a["h"][i_in] / 1e3], [a["p"][i_in] / 1e5], "o", color=col, ms=4)
    ax.axhline(pcrit / 1e5, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("specific enthalpy [kJ/kg]")
    ax.set_ylabel("pressure [bar]")
    ax.set_title("Oxidiser path through the jacket (march result; dot = jacket inlet)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
