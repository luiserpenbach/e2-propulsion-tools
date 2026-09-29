"""Two-dimensional conduction in the channel cross-section (calculation note 7.1, 11.2).

The 1D march lumps the channel floor and the ribs into one effective coefficient.
This module solves the steady conduction problem in the cross-section at one
station, with the boundary conditions taken from the march at that station:

    gas side        h_g (sigma updated with the local surface temperature), T_aw
    wetted walls    h_c, T_b on floor, rib flanks and closeout side of every
                    flowing channel; adiabatic walls for a blocked channel
    closeout outer  adiabatic
    side planes     symmetry (adiabatic)

Domain: 2.5 channel pitches from the middle of channel 0 to the middle of the rib
between channels 2 and 3; mirrored this is a five-channel window whose centre
channel can be blocked. Geometry unrolled flat (error of order t/r). Finite
volumes on a square grid, conductivity per cell from the local temperature,
Picard iteration with under-relaxation.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from . import materials as mat
from .jacket import Jacket
from .regen import bartz_sigma

N_PITCH = 2.5
# feature-aligned sample grid for export (vertex counts per segment)
Y_SEGS = (10, 6, 20, 6, 20, 3)
Z_SEGS = (8, 10, 8)


def _geom(r, j: Jacket, d):
    p = 2 * np.pi * (r + j.t_wall) / j.n
    w = p - j.t_rib
    Y, H = N_PITCH * p, j.depth
    ny, nz = int(round(Y / d)), int(round(H / d))
    dy, dz = Y / ny, H / nz
    yc, zc = (np.arange(ny) + 0.5) * dy, (np.arange(nz) + 0.5) * dz
    Yc, Zc = np.meshgrid(yc, zc, indexing="ij")
    chid = -np.ones_like(Yc, int)
    for k in range(3):
        m = (np.abs(Yc - k * p) < w / 2) & (Zc > j.t_wall) & (Zc < j.t_wall + j.h)
        chid[m] = k
    return dict(p=p, w=w, Y=Y, H=H, ny=ny, nz=nz, dy=dy, dz=dz, yc=yc, zc=zc,
                fluid=chid >= 0, chid=chid)


def solve_section(r, jacket: Jacket, hg0, Tc, X, Taw, Tb, hc, blocked=False, d=0.04e-3,
                  tol=0.05, max_iter=80):
    """Solve one cross-section. Returns dict with T (ny x nz, nan in channels),
    surface temperature Ts along y, gas-side flux, grid and geometry."""
    m = mat.get(jacket.material)
    G = _geom(r, jacket, d)
    ny, nz, dy, dz = G["ny"], G["nz"], G["dy"], G["dz"]
    fl, chid = G["fluid"], G["chid"]
    sol = ~fl
    idx = -np.ones((ny, nz), int)
    idx[sol] = np.arange(sol.sum())
    Nn = int(sol.sum())
    T = np.full((ny, nz), 0.5 * (Taw + Tb))
    pairs = []
    for di, dj, A_, dd in ((1, 0, dz, dy), (0, 1, dy, dz)):
        i1, j1 = np.where(np.ones((ny - di, nz - dj), bool))
        pairs.append((i1, j1, i1 + di, j1 + dj, A_, dd))
    for it in range(max_iter):
        k = mat.k(m, T)
        rows, cols, vals = [], [], []
        diag, b = np.zeros(Nn), np.zeros(Nn)
        for i1, j1, i2, j2, A_, dd in pairs:
            s1, s2 = sol[i1, j1], sol[i2, j2]
            mm = s1 & s2
            k1, k2 = k[i1[mm], j1[mm]], k[i2[mm], j2[mm]]
            gc = 2 * k1 * k2 / (k1 + k2) * A_ / dd
            a1, a2 = idx[i1[mm], j1[mm]], idx[i2[mm], j2[mm]]
            rows += [a1, a2, a1, a2]
            cols += [a2, a1, a1, a2]
            vals += [-gc, -gc, gc, gc]
            for sa, ia, ja, ib, jb in ((s1 & ~s2, i1, j1, i2, j2), (~s1 & s2, i2, j2, i1, j1)):
                ci = chid[ib[sa], jb[sa]]
                keep = ~(blocked & (ci == 0))
                ii, jj = ia[sa][keep], ja[sa][keep]
                if len(ii):
                    gcv = 1 / (1 / (hc * A_) + (dd / 2) / (k[ii, jj] * A_))
                    np.add.at(diag, idx[ii, jj], gcv)
                    np.add.at(b, idx[ii, jj], gcv * Tb)
        k0 = k[:, 0]
        Ts = (T[:, 0])
        hg = hg0 * bartz_sigma(Ts, Tc, X)
        gg = 1 / (1 / (hg * dy) + (dz / 2) / (k0 * dy))
        a1 = idx[:, 0]
        np.add.at(diag, a1, gg)
        np.add.at(b, a1, gg * Taw)
        Amat = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(Nn, Nn)) + sp.diags(diag)
        Tn = spla.spsolve(Amat.tocsc(), b)
        Tnew = np.full((ny, nz), np.nan)
        Tnew[sol] = Tn
        dT = float(np.nanmax(np.abs(Tnew - T)[sol]))
        T = np.where(sol, Tnew if it == 0 else 0.6 * Tnew + 0.4 * T, T)
        if it > 0 and dT < tol:
            break
    T = np.where(sol, T, np.nan)
    k0 = mat.k(m, T[:, 0])
    hg = hg0 * bartz_sigma(T[:, 0], Tc, X)
    Ts = (hg * Taw + 2 * k0 / dz * T[:, 0]) / (hg + 2 * k0 / dz)
    return {"T": T, "Ts": Ts, "q_gas": hg * (Taw - Ts), "geom": G, "iterations": it + 1,
            "blocked": blocked, "Tb": Tb, "Taw": Taw, "t_wall": jacket.t_wall, "h_ch": jacket.h}


def section_at(res, i, blocked=False, d=0.04e-3):
    """Solve the cross-section at march station i of a RegenResult."""
    a = res.arrays
    return solve_section(a["r"][i], res.jacket, a["hg0"][i], res.gas["Tc"], a["X"][i],
                         a["Taw"][i], a["Tb"][i], a["hc"][i], blocked=blocked, d=d)


def summary(sec):
    G = sec["geom"]
    p, yc = G["p"], G["yc"]
    Ts = sec["Ts"]
    return {"Ts_max": float(np.nanmax(Ts)), "y_Ts_max": float(yc[int(np.nanargmax(Ts))]),
            "Ts_channel1": float(np.interp(p, yc, Ts)), "Ts_rib": float(np.interp(1.5 * p, yc, Ts)),
            "Ts_centre": float(Ts[0]), "q_mean": float(np.mean(sec["q_gas"])),
            "T_closeout_max": float(np.nanmax(sec["T"][:, -1])), "T_max": float(np.nanmax(sec["T"]))}


def yz_vertices(r, j: Jacket):
    """Feature-aligned sample points: segment boundaries fall on channel and rib edges."""
    p = 2 * np.pi * (r + j.t_wall) / j.n
    w = p - j.t_rib
    edges = [(0, w / 2), (w / 2, w / 2 + j.t_rib), (p - w / 2, p + w / 2),
             (p + w / 2, p + w / 2 + j.t_rib), (2 * p - w / 2, 2 * p + w / 2), (2 * p + w / 2, 2.5 * p)]
    yv = [0.0]
    for (a, b_), nn in zip(edges, Y_SEGS):
        yv += list(np.linspace(a, b_, nn + 1)[1:])
    zb = [0.0, j.t_wall, j.t_wall + j.h, j.depth]
    zv = [0.0]
    for (a, b_), nn in zip(zip(zb[:-1], zb[1:]), Z_SEGS):
        zv += list(np.linspace(a, b_, nn + 1)[1:])
    return np.array(yv), np.array(zv)


def sample(sec, r, j: Jacket):
    """Temperatures at the feature-aligned vertices (channels filled with the
    nearest solid value so interpolation does not bleed water into the wall)."""
    T, G = sec["T"], sec["geom"]
    mask = np.isnan(T)
    ind = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    Tf = T[tuple(ind)]
    Tf[:, 0] = sec["Ts"]
    zc = G["zc"].copy()
    zc[0] = 0.0
    itp = RegularGridInterpolator((G["yc"], zc), Tf, bounds_error=False, fill_value=None)
    yv, zv = yz_vertices(r, j)
    Yv, Zv = np.meshgrid(yv, zv, indexing="ij")
    return itp(np.c_[Yv.ravel(), Zv.ravel()]).reshape(Yv.shape)
