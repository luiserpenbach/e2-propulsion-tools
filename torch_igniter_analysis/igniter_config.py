"""Load a torch-igniter YAML case and the chamber geometry both scripts share."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "igniter_example.yaml"
RESULTS = HERE / "results"


@dataclass
class IgniterCase:
    name: str
    ox: str
    fuel: str
    of_nominal: float
    of_points: list
    mdot_kg_s: float
    eta_cstar: float
    pa_bar: float
    pc_guess_bar: float
    eps: float
    dt_mm: float
    dc_mm: float
    theta_conv_deg: float
    theta_div_deg: float
    lstar_m: float
    lcyl_mm: float
    lconv_mm: float
    ldiv_mm: float
    de_mm: float
    at_mm2: float
    ac_mm2: float
    at_m2: float
    eps_c: float
    vc_cm3: float
    vconv_cm3: float
    wall_t0_k: float
    wall_thickness_m: float
    rc_curv_m: float
    length_from_lstar: bool


def _req(raw: dict, key: str):
    if key not in raw:
        raise KeyError(f"igniter config missing '{key}'")
    return raw[key]


def load(path: str | Path = DEFAULT_CONFIG) -> IgniterCase:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    dt = float(_req(raw, "dt_mm"))
    dc = float(_req(raw, "dc_mm"))
    if dt <= 0 or dc <= dt:
        raise ValueError("dc_mm must be larger than dt_mm")

    theta_c = float(_req(raw, "theta_conv_deg"))
    theta_d = float(_req(raw, "theta_div_deg"))
    eps = float(_req(raw, "eps"))
    rt = dt / 2.0
    rc = dc / 2.0
    at_mm2 = math.pi / 4.0 * dt ** 2
    ac_mm2 = math.pi / 4.0 * dc ** 2

    lconv_mm = (rc - rt) / math.tan(math.radians(theta_c))
    # Frustum volume. Lengths in cm so the result is cm³.
    vconv = (math.pi / 3.0) * (lconv_mm / 10.0) * (
        (rc / 10.0) ** 2 + (rc / 10.0) * (rt / 10.0) + (rt / 10.0) ** 2
    )

    has_lstar = raw.get("lstar_m") is not None
    has_lcyl = raw.get("lcyl_mm") is not None
    if has_lstar == has_lcyl:
        raise ValueError("set exactly one of lstar_m or lcyl_mm")

    if has_lstar:
        lstar = float(raw["lstar_m"])
        # L*[m] * At[mm²] = Vc[cm³], because 1 m · 1 mm² = 1 cm³.
        vc = lstar * at_mm2
        lcyl_mm = (vc - vconv) / (ac_mm2 * 1e-2) * 10.0
        from_lstar = True
    else:
        lcyl_mm = float(raw["lcyl_mm"])
        vcyl = (ac_mm2 * 1e-2) * (lcyl_mm / 10.0)
        vc = vcyl + vconv
        lstar = vc / at_mm2
        from_lstar = False

    if lcyl_mm <= 0:
        raise ValueError("cylindrical length is not positive; lower L* or raise Dc")

    re = rt * math.sqrt(eps)
    ldiv_mm = (re - rt) / math.tan(math.radians(theta_d))
    rc_ratio = float(raw.get("rc_curv_over_dt", 1.0))

    return IgniterCase(
        name=str(raw.get("name", path.stem)),
        ox=str(_req(raw, "ox")),
        fuel=str(_req(raw, "fuel")),
        of_nominal=float(_req(raw, "of_nominal")),
        of_points=[float(x) for x in _req(raw, "of_points")],
        mdot_kg_s=float(_req(raw, "mdot_kg_s")),
        eta_cstar=float(_req(raw, "eta_cstar")),
        pa_bar=float(_req(raw, "pa_bar")),
        pc_guess_bar=float(raw.get("pc_guess_bar", 30.0)),
        eps=eps,
        dt_mm=dt,
        dc_mm=dc,
        theta_conv_deg=theta_c,
        theta_div_deg=theta_d,
        lstar_m=lstar,
        lcyl_mm=lcyl_mm,
        lconv_mm=lconv_mm,
        ldiv_mm=ldiv_mm,
        de_mm=2.0 * re,
        at_mm2=at_mm2,
        ac_mm2=ac_mm2,
        at_m2=at_mm2 * 1e-6,
        eps_c=(dc / dt) ** 2,
        vc_cm3=vc,
        vconv_cm3=vconv,
        wall_t0_k=float(raw.get("wall_t0_K", 293.0)),
        wall_thickness_m=float(raw.get("wall_thickness_mm", 2.0)) * 1e-3,
        rc_curv_m=rc_ratio * dt * 1e-3,
        length_from_lstar=from_lstar,
    )


def write_figure(fig, html_name: str, image_name: str | None = None):
    """Write HTML always. Write a static image when kaleido is installed."""
    RESULTS.mkdir(exist_ok=True)
    html_path = RESULTS / html_name
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    image_path = None
    if image_name:
        try:
            fig.write_image(str(RESULTS / image_name), scale=2)
            image_path = RESULTS / image_name
        except Exception as exc:
            print(f"Static image skipped ({image_name}): {exc}")
    return html_path, image_path
