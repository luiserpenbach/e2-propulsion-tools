"""Figures for the boiling-regime analysis (run_boiling.py)."""
import numpy as np

from .config import T_WG_LIMIT
from .plotstyle import plt, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, INK, INK2, MUTED, note

MODEL_STYLE = {
    "liquid_only": ("Liquid-only Gnielinski (h2cea baseline)", MUTED, "-"),
    "nucleate": ("Nucleate boiling, Cooper (heat flux above CHF)", GREEN, ":"),
    "jackson": ("Near-critical homogeneous, Jackson-type", BLUE, "-"),
    "miropolskii": ("Film boiling, Miropolskii 1963", ORANGE, "-"),
    "groeneveld": ("Film boiling, Groeneveld 1973", RED, "-"),
    "jackson+sub": ("Jackson-type, also in subcooled flow", BLUE, "--"),
    "miropolskii+sub": ("Miropolskii, also in subcooled flow", ORANGE, "--"),
    "regime": ("Regime switching (recommended N2O case)", INK, "-"),
}


def _c(T):
    return np.asarray(T) - 273.15


def wall_by_model(profiles, point, fname, xt_mm, limit_K=T_WG_LIMIT):
    """Gas-side and coolant-side wall temperature and coolant-side coefficient
    along the jacket for every model at one throttle point."""
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for mdl, (lab, col, ls) in MODEL_STYLE.items():
        r = profiles.get((point, mdl))
        if r is None:
            continue
        a = r.arrays if hasattr(r, "arrays") else r
        x = np.asarray(a["x"]) * 1e3
        axs[0].plot(x, _c(a["Twg"]), color=col, ls=ls, lw=1.8, label=lab)
        axs[1].plot(x, _c(a["Twc"]), color=col, ls=ls, lw=1.8)
        axs[2].plot(x, a["hc"] / 1e3, color=col, ls=ls, lw=1.8)
    r0 = profiles[(point, "liquid_only")]
    r0 = {k: np.asarray(v) for k, v in (r0.arrays if hasattr(r0, "arrays") else r0).items()}
    tp = r0["regime"] == "two-phase"
    for ax in axs:
        ax.axvline(xt_mm, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("x from injector face (mm)")
    axs[0].axhline(limit_K - 273.15, color=INK, lw=0.9)
    axs[0].annotate(f"IN718 limit {limit_K:.0f} K", (2, limit_K - 273.15), xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color=INK2)
    axs[1].plot(r0["x"] * 1e3, _c(r0["T_sl"]), color=VIOLET, lw=1.2, ls="--")
    axs[1].annotate("limiting liquid superheat", (r0["x"][tp][0] * 1e3 if tp.any() else 0, _c(r0["T_sl"][tp][0]) if tp.any() else 0),
                    xytext=(0, 6), textcoords="offset points", fontsize=8, color=VIOLET)
    axs[0].set_title("Gas-side wall temperature (°C)")
    axs[1].set_title("Coolant-side wall temperature (°C)")
    axs[2].set_title("Coolant-side coefficient (kW/m²K)")
    axs[2].set_yscale("log")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"N2O cooling at {point} %: wall temperature by coolant-side model (coolant enters at 180 mm)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def regime_map(tables, fname, xt_mm):
    """Coolant-side heat flux against what nucleate boiling can carry (superheat
    limit) and the Hall-Mudawar CHF, plus quality and homogeneous Mach number."""
    fig, axs = plt.subplots(1, len(tables), figsize=(4.6 * len(tables), 4.0), sharey=True)
    for ax, (lab, t) in zip(np.atleast_1d(axs), tables.items()):
        ax.plot(t.x_mm, t.q_c_MW, color=INK, lw=2.0, label="coolant-side heat flux")
        ax.plot(t.x_mm, t.q_nb_max_MW, color=GREEN, lw=1.6, label="nucleate boiling limit (Cooper to T_sl)")
        ax.plot(t.x_mm, t.chf_hm_MW, color=RED, lw=2.2, label="subcooled CHF, Hall–Mudawar (x ≤ 0.05)")
        xdi = t.x_di.dropna()
        if len(xdi):
            ax.text(0.03, 0.97, f"Kim–Mudawar dryout quality: {xdi.min():.1f} to {xdi.max():.2f}\n(negative: no wetted film)",
                    transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 60)
        ax.axvline(xt_mm, color=MUTED, lw=0.8, ls="--")
        tp = t.regime == "two-phase"
        if tp.any():
            ax.axvspan(t.x_mm[tp].min(), t.x_mm[tp].max(), color="#f1f0ec", zorder=0)
        ax.set_title(f"{lab}")
        ax.set_xlabel("x from injector face (mm)")
        ax2 = ax.twinx()
        ax2.plot(t.x_mm, t.x_eq, color=AQUA, lw=1.2, ls="--")
        ax2.set_ylim(-0.3, 1.4)
        ax2.grid(False)
        ax2.tick_params(colors=AQUA)
        ax2.spines["right"].set_visible(True)
        ax2.set_ylabel("equilibrium quality", color=AQUA)
    np.atleast_1d(axs)[0].set_ylabel("heat flux (MW/m²)")
    np.atleast_1d(axs)[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Heat flux against the nucleate boiling limits (shaded: boiling region)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
