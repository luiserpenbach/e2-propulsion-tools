"""Cooling-jacket geometry (handbook sections 10.4 and 10.10).

Constant jacket (default, E2-REG-1): straight axial channels of constant height and
constant rib thickness, so the channel width follows the local circumference. The
width is taken at the channel floor (back of the hot wall).

Variable jacket (`Jacket.with_profile`): any of channel height h, rib thickness t_rib
or channel width w, hot-wall thickness t_wall, channel count n and wall share can vary
along the axis. Values are given at stations x (m, from the injector face, the same
coordinate as the contour) and interpolated linearly; outside the table the end
values hold.

    n      channels that carry THIS march's flow at x (mass flux G = mdot / (n w h))
    share  fraction of the circumference whose heat goes into this flow (0 < share <= 1)

The geometric pitch follows from both: there are n / share channels around the
circumference at x, so pitch = 2 pi (r + t_wall) share / n. A single-pass jacket has
share = 1. A jacket with alternating down/up channels is two marches with
n = N/2, share = 0.5 where both passes run side by side, and share = 1 where only one
pass is left (e.g. the up channels widening over the upper cylinder). Each march then
gets the right mass flux, the right rib between neighbouring channels and the right
heated wall per kg of its coolant. Conduction between neighbouring channels of the two
passes is not modelled.

If the profile gives w, the rib is pitch - w; if it gives t_rib, the width is
pitch - t_rib (as for the constant jacket). Giving both is an error.
"""
from dataclasses import dataclass, asdict, field

import numpy as np

from . import config as C

_PROFILE_KEYS = ("h", "t_rib", "w", "t_wall", "n", "share")


@dataclass(frozen=True)
class Jacket:
    n: int = C.N_CH
    t_wall: float = C.T_WALL          # m
    t_rib: float = C.T_RIB            # m
    h: float = C.H_CH                 # m
    t_closeout: float = C.T_CLOSEOUT  # m
    roughness: float = C.ROUGHNESS    # m
    material: str = C.WALL_MATERIAL
    blocked: tuple = ()               # channel indices without flow (for bookkeeping)
    # axial profile: (("x", (...)), ("h", (...)), ...) in m / counts / fractions; () = constant
    profile: tuple = field(default=())

    def as_dict(self):
        d = asdict(self)
        d["profile"] = {k: list(v) for k, v in self.profile}
        return d

    @property
    def variable(self):
        return bool(self.profile)

    @property
    def depth(self):
        """Total jacket thickness from the gas side to the closeout outer surface
        (constant jacket; for a profile the maximum along the axis)."""
        if not self.profile:
            return self.t_wall + self.h + self.t_closeout
        p = dict(self.profile)
        tw = np.max(p.get("t_wall", (self.t_wall,)))
        hh = np.max(p.get("h", (self.h,)))
        return float(tw + hh + self.t_closeout)

    @classmethod
    def with_profile(cls, x, h=None, t_rib=None, w=None, t_wall=None, n=None, share=None, **const):
        """Jacket with axial profiles. x in m from the injector face (strictly increasing).
        h, t_rib, w, t_wall, share: arrays the length of x, scalars, or None (constant
        field / default). n: scalar (constant channel count) or array (profile).
        Other keyword arguments set the constant fields (t_closeout, roughness, ...)."""
        x = np.asarray(x, float)
        if np.any(np.diff(x) <= 0):
            raise ValueError("profile x must be strictly increasing")
        if t_rib is not None and w is not None:
            raise ValueError("give either t_rib or w in the profile, not both")
        items = [("x", tuple(x.tolist()))]
        given = dict(h=h, t_rib=t_rib, w=w, t_wall=t_wall, share=share)
        if n is not None and np.ndim(n) > 0:
            given["n"] = n
            const["n"] = int(round(float(np.max(n))))
        elif n is not None:
            const["n"] = int(n)
        for k, v in given.items():
            if v is None:
                continue
            v = np.broadcast_to(np.asarray(v, float), x.shape)
            if k == "share" and (np.any(v <= 0) or np.any(v > 1)):
                raise ValueError("share must be in (0, 1]")
            items.append((k, tuple(v.tolist())))
        # constant fields used where a profile key is absent and by the constant-only tools
        for k in ("h", "t_wall", "t_rib"):
            if given.get(k) is not None:
                const.setdefault(k, float(np.max(given[k])))
        return cls(profile=tuple(items), **const)

    def at(self, x):
        """Local geometry parameters at axial position(s) x (m from the injector face)."""
        x = np.atleast_1d(np.asarray(x, float))
        out = {"t_wall": np.full(x.shape, self.t_wall), "t_rib": np.full(x.shape, self.t_rib),
               "h": np.full(x.shape, self.h), "n": np.full(x.shape, float(self.n)),
               "share": np.ones(x.shape), "w": None}
        if self.profile:
            p = dict(self.profile)
            xp = np.asarray(p["x"])
            for k in _PROFILE_KEYS:
                if k in p:
                    out[k] = np.interp(x, xp, np.asarray(p[k]))
        return out


def channels(r, j: Jacket, x=None):
    """Channel geometry at gas-side radius r (m, scalar or array).

    x (m from the injector face, same shape as r) is needed only for a jacket with an
    axial profile. Returns pitch, w, t_rib, h, t_wall, n, share, A_flow (all channels of
    this flow), Dh and A_one (one channel)."""
    r = np.asarray(r, float)
    if j.profile:
        if x is None:
            raise ValueError("this jacket has an axial profile: pass x to channels()")
        g = j.at(np.broadcast_to(np.asarray(x, float), r.shape).ravel())
        g = {k: (v.reshape(r.shape) if v is not None else None) for k, v in g.items()}
    else:
        g = j.at(np.zeros(r.size))
        g = {k: (v.reshape(r.shape) if v is not None else None) for k, v in g.items()}
    tw, n, share = g["t_wall"], g["n"], g["share"]
    pitch = 2 * np.pi * (r + tw) * share / n
    if g["w"] is not None:
        w = g["w"]
        rib = pitch - w
        if np.any(rib <= 0):
            raise ValueError("channel width larger than the pitch somewhere in the profile")
    else:
        rib = g["t_rib"]
        w = pitch - rib
        if np.any(w <= 0):
            raise ValueError("ribs do not fit the circumference: n*(t_rib) > 2 pi (r + t_wall)")
    h = g["h"]
    area = n * w * h
    Dh = 2 * w * h / (w + h)
    return {"pitch": pitch, "w": w, "t_rib": rib, "h": h, "t_wall": tw, "n": n,
            "share": share, "A_flow": area, "Dh": Dh, "A_one": w * h}


def e2_reg1():
    """E2-REG-1-A jacket as designed (closeout thickness TBC)."""
    return Jacket()


def from_station_table(path, x_of_z, which="down", n_total=None, **kw):
    """Jacket profile from a channel-generator station table (regj_generator_v2 stations.csv).

    path     CSV with columns z_mm, h_mm, w_down_mm, w_up_mm, rib_mm (z from the nozzle exit)
    x_of_z   function mapping generator z (m) to h2cea x (m from the injector face),
             e.g. lambda z: L - z
    which    'down' or 'up'
    n_total  total channel count of the generator (down + up); defaults to kw['n'] * 2
    The down pass exists where w_down is finite (share 0.5). The up pass has share 0.5
    where down channels exist and 1.0 above them.
    """
    import csv
    rows = list(csv.DictReader(open(path)))
    z = np.array([float(r_["z_mm"]) for r_ in rows]) * 1e-3
    h = np.array([float(r_["h_mm"]) for r_ in rows]) * 1e-3
    wd = np.array([float(r_["w_down_mm"]) if r_["w_down_mm"] not in ("", "nan") else np.nan
                   for r_ in rows]) * 1e-3
    wu = np.array([float(r_["w_up_mm"]) for r_ in rows]) * 1e-3
    x = np.array([x_of_z(zz) for zz in z])
    o = np.argsort(x)
    x, h, wd, wu = x[o], h[o], wd[o], wu[o]
    n_half = (n_total // 2) if n_total else int(kw.pop("n"))
    has_down = np.isfinite(wd)
    if which == "down":
        m = has_down
        return Jacket.with_profile(x[m], h=h[m], w=wd[m], share=0.5, n=n_half, **kw)
    share = np.where(has_down, 0.5, 1.0)
    return Jacket.with_profile(x, h=h, w=wu, share=share, n=n_half, **kw)
