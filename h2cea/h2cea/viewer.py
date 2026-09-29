"""Interactive 3D channel viewer (self-contained HTML, three.js from a CDN).

    write_viewer(path, cases, sections_along(...), meta)

The page shows a sector of 15 channels from the injector face to a movable cut,
every wall coloured by temperature from the 2D sections, with a linked
cross-section map and axial profiles. Temperatures are quantised to 8 bit over
0..tmax_scale (5 K steps at 1300 °C), which is plenty for a picture.
"""
import base64
import json
import os

import numpy as np

from . import section2d as s2d

TEMPLATE = os.path.join(os.path.dirname(__file__), "templates", "channel_viewer.html")


def sections_along(res, n=121, blocked=(False, True), d=0.04e-3, progress=None):
    """2D sections at n stations equally spaced along the wall.

    Returns {"idx": [...], "nom": {"field": (n,66,27) degC, "Tsmax": [...]}, "blk": {...}}.
    """
    a = res.arrays
    s = a["s"]
    idx = [int(np.argmin(np.abs(s - v))) for v in np.linspace(s[0], s[-1], n)]
    it = int(np.argmin(np.abs(a["x"] - res.gas.get("xt", np.nan)))) if "xt" in res.gas else None
    if it is not None:
        idx[int(np.argmin(np.abs(np.array(idx) - it)))] = it
    out = {"idx": idx}
    for bl in blocked:
        fields, peaks, summ = [], [], []
        for k, i in enumerate(idx):
            sec = s2d.section_at(res, i, blocked=bl, d=d)
            fields.append(s2d.sample(sec, a["r"][i], res.jacket) - 273.15)
            sm = s2d.summary(sec)
            peaks.append(sm["Ts_max"])
            summ.append(sm)
            if progress:
                progress(res.case, bl, k, len(idx))
        out["blk" if bl else "nom"] = {"field": np.array(fields), "Tsmax": np.array(peaks), "summary": summ}
    return out


def write_viewer(path, results, sections, meta, tmax_scale=1300.0):
    """results: list of RegenResult; sections: list of sections_along() outputs (same order);
    meta: dict(title, subtitle, titleblock=[(label, value)], labels=[...], subs=[...])."""
    r0 = results[0]
    a0 = r0.arrays
    idx = sections[0]["idx"]
    j = r0.jacket
    yv, zv = s2d.yz_vertices(a0["r"][0], j)
    assert len(yv) == 66 and len(zv) == 27, "viewer template expects the default segment counts"
    st = {"x": [round(a0["x"][i] * 1e3, 3) for i in idx], "r": [round(a0["r"][i] * 1e3, 3) for i in idx],
          "s": [round(a0["s"][i] * 1e3, 3) for i in idx],
          "nx": [round(a0["nx"][i], 5) for i in idx] if "nx" in a0 else None,
          "nr": [round(a0["nr"][i], 5) for i in idx] if "nr" in a0 else None,
          "p": [round(a0["pitch"][i] * 1e3, 4) for i in idx], "w": [round(a0["w"][i] * 1e3, 4) for i in idx]}
    cases, clist = {}, []
    for k, (res, sec) in enumerate(zip(results, sections)):
        a = res.arrays
        key = f"c{k}"
        c = {f: [round(float(a[f][i]) - 273.15, 1) for i in idx] for f in ("Tb", "Tsat", "Twg", "Twc")}
        c.update(q=[round(float(a["q"][i]) / 1e6, 3) for i in idx], v=[round(float(a["v"][i]), 2) for i in idx],
                 pw=[round(float(a["p"][i]) / 1e5, 2) for i in idx],
                 xq=[round(float(a["xq"][i]), 3) if np.isfinite(a["xq"][i]) else None for i in idx],
                 regime=[str(a["regime"][i]) for i in idx],
                 chf=[round(float(a["chf_ratio"][i]), 3) if np.isfinite(a["chf_ratio"][i]) else None for i in idx])
        for tag in ("nom", "blk"):
            if tag in sec:
                F = np.clip(sec[tag]["field"], 0, tmax_scale) / tmax_scale * 255
                c[tag + "_field"] = base64.b64encode(np.round(F).astype(np.uint8).tobytes()).decode()
                c[tag + "_Tsmax"] = [round(float(v) - 273.15, 1) for v in sec[tag]["Tsmax"]]
        cases[key] = c
        clist.append({"key": key, "label": meta["labels"][k], "sub": meta.get("subs", [""] * len(results))[k]})
    data = {"z": [round(v * 1e3, 4) for v in zv], "n": len(idx), "tmaxScale": tmax_scale, "st": st, "cases": cases,
            "meta": {"title": meta["title"], "subtitle": meta["subtitle"], "titleblock": meta["titleblock"],
                     "cases": clist, "t_wall": j.t_wall * 1e3, "t_rib": j.t_rib * 1e3, "h_ch": j.h * 1e3}}
    html = open(TEMPLATE, encoding="utf-8").read()
    html = html.replace("__TITLE__", meta.get("page_title", meta["title"]))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("NaN", "null"))
    head = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(head + html + "\n</html>\n")
    return path
