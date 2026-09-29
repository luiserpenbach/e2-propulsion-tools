"""Cooling-jacket geometry (handbook sections 10.4 and 10.10).

Straight axial channels of constant height and constant rib thickness, so the
channel width follows the local circumference. The width is taken at the channel
floor (back of the hot wall).
"""
from dataclasses import dataclass, asdict

import numpy as np

from . import config as C


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

    def as_dict(self):
        return asdict(self)

    @property
    def depth(self):
        """Total jacket thickness from the gas side to the closeout outer surface."""
        return self.t_wall + self.h + self.t_closeout


def channels(r, j: Jacket):
    """Channel geometry at gas-side radius r (m, scalar or array)."""
    r = np.asarray(r, float)
    pitch = 2 * np.pi * (r + j.t_wall) / j.n
    w = pitch - j.t_rib
    if np.any(w <= 0):
        raise ValueError("ribs do not fit the circumference: n*(t_rib) > 2 pi (r + t_wall)")
    area = j.n * w * j.h
    Dh = 2 * w * j.h / (w + j.h)
    return {"pitch": pitch, "w": w, "A_flow": area, "Dh": Dh, "A_one": w * j.h}


def e2_reg1():
    """E2-REG-1-A jacket as designed (closeout thickness TBC)."""
    return Jacket()
