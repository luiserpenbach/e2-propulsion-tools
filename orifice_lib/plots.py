"""Plotly figures for one orifice case. Physics stays in the model modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gas
import liquid
import two_phase
from common import BAR


def write_plot(result, path: Path, title: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if isinstance(result, liquid.LiquidResult):
        fig = _liquid(result, title)
    elif isinstance(result, gas.GasResult):
        fig = _gas(result, title)
    elif isinstance(result, two_phase.TwoPhaseResult):
        fig = _two_phase(result, title)
    else:
        raise TypeError(f"no plot for {type(result).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def _liquid(result: liquid.LiquidResult, title: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dP = result.P_up_Pa - result.P_down_Pa
    dP_axis = np.linspace(0.02 * dP, dP, 200)
    m_dp = result.Cd * result.area_m2 * np.sqrt(2.0 * result.rho * dP_axis)

    d_axis = np.linspace(0.4 * result.d_m, 2.5 * result.d_m, 200)
    m_d = result.mdot_kg_s * (d_axis / result.d_m) ** 2

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mass flow vs pressure drop", "Mass flow vs diameter"),
    )
    fig.add_trace(
        go.Scatter(x=dP_axis / BAR, y=m_dp * 1e3, name="mass flow", line=dict(color="#1f4e79", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[dP / BAR], y=[result.mdot_kg_s * 1e3], mode="markers", name="operating point",
            marker=dict(size=11, symbol="x", color="#c0392b", line=dict(width=2)),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=d_axis * 1e3, y=m_d * 1e3, name="at this ΔP", line=dict(color="#1f4e79", width=2), showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[result.d_m * 1e3], y=[result.mdot_kg_s * 1e3], mode="markers", showlegend=False,
            marker=dict(size=11, symbol="x", color="#c0392b", line=dict(width=2)),
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="ΔP [bar]", row=1, col=1)
    fig.update_yaxes(title_text="mass flow [g/s]", row=1, col=1)
    fig.update_xaxes(title_text="diameter [mm]", row=1, col=2)
    fig.update_yaxes(title_text="mass flow [g/s]", row=1, col=2)
    fig.update_layout(
        title=f"{title}  —  liquid, {result.fluid}, Cd = {result.Cd:.2f}",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def _gas(result: gas.GasResult, title: str):
    import plotly.graph_objects as go

    P_hi = 0.98 * result.P_up_Pa
    P_lo = max(0.02 * result.P_up_Pa, 5.0e4)
    P_axis = np.linspace(P_lo, P_hi, 80)
    m_real, m_k = result.curve(P_axis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P_axis / BAR, y=m_real * 1e3, name="real gas", line=dict(color="#1f4e79", width=2.5)))
    fig.add_trace(go.Scatter(x=P_axis / BAR, y=m_k * 1e3, name="constant k", line=dict(color="#c0392b", width=2, dash="dash")))
    if result.mdot_real_kg_s is not None:
        fig.add_trace(
            go.Scatter(
                x=[result.P_down_Pa / BAR], y=[result.mdot_real_kg_s * 1e3], mode="markers",
                name="operating point", marker=dict(size=11, symbol="x", color="#1f4e79"),
            )
        )
    if result.P_star_Pa is not None:
        fig.add_vline(x=result.P_star_Pa / BAR, line_dash="dot", line_color="#7f8c8d",
                      annotation_text=f"P* = {result.P_star_Pa / BAR:.1f} bar")
    fig.update_layout(
        title=f"{title}  —  gas, {result.fluid}, d = {result.d_m * 1e3:.2f} mm, Cd = {result.Cd:.2f}",
        xaxis_title="downstream pressure [bar]",
        yaxis_title="mass flow [g/s]",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def _two_phase(result: two_phase.TwoPhaseResult, title: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    valid = result.scan_G > 0.0
    P_g = result.scan_P[valid] / BAR
    G = result.scan_G[valid]

    P_lo = max(1.0e5, 0.02 * result.P_up_Pa)
    P_axis = np.linspace(P_lo, 0.98 * result.P_up_Pa, 70)
    spi, hem, nhne = [], [], []
    for P in P_axis:
        a, b, c = result.fluxes(float(P))
        spi.append(a * 1e3)
        hem.append(b * 1e3)
        nhne.append(c * 1e3)
    P_bar = P_axis / BAR

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.14,
        subplot_titles=("HEM mass flux along the isentrope", "Mass flow vs downstream pressure"),
    )
    fig.add_trace(go.Scatter(x=P_g, y=G, name="G", line=dict(color="#1f4e79", width=2)), row=1, col=1)
    if result.P_star_Pa is not None:
        fig.add_vline(x=result.P_star_Pa / BAR, line_dash="dot", line_color="#c0392b", row=1, col=1,
                      annotation_text=f"P* = {result.P_star_Pa / BAR:.1f} bar")
    fig.add_trace(go.Scatter(x=P_bar, y=spi, name="SPI", line=dict(color="#1f4e79", dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=P_bar, y=hem, name="HEM", line=dict(color="#c0392b", dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=P_bar, y=nhne, name="NHNE", line=dict(color="#1e7f4f", width=2.5)), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=[result.P_down_Pa / BAR],
            y=[result.mdot_nhne_kg_s * 1e3],
            mode="markers", name="operating point",
            marker=dict(size=11, symbol="x", color="#1e7f4f"),
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="throat pressure [bar]", row=1, col=1)
    fig.update_yaxes(title_text="G [kg/(m²·s)]", row=1, col=1)
    fig.update_xaxes(title_text="downstream pressure [bar]", row=2, col=1)
    fig.update_yaxes(title_text="mass flow [g/s]", row=2, col=1)
    fig.update_layout(
        title=f"{title}  —  two-phase, {result.fluid}, d = {result.d_m * 1e3:.2f} mm, Cd = {result.Cd:.2f}",
        template="plotly_white",
        height=780,
        legend=dict(orientation="h", y=1.08),
    )
    return fig
