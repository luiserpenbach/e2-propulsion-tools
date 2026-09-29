"""Wall material data (handbook section 11.1).

Wrought, aged Inconel 718. The printed and heat-treated material is assumed to
behave the same until witness-coupon data exist (E2-TCA-DOC-001 REG1-05).
"""
import numpy as np

_T = np.array([293.0, 473.0, 673.0, 873.0, 1073.0, 1273.0])      # K

IN718 = {
    "name": "IN718",
    "k": np.array([11.0, 14.0, 17.0, 20.0, 24.0, 26.0]),          # W/(m K)
    "E": np.array([200e9, 190e9, 180e9, 170e9, 157e9, 140e9]),    # Pa
    "alpha": np.array([13.0e-6, 13.4e-6, 13.9e-6, 14.4e-6, 15.4e-6, 16.2e-6]),  # 1/K, mean from 20 C
    "sy": np.array([1100e6, 1050e6, 1000e6, 980e6, 700e6, 300e6]),  # Pa, 0.2 % yield, aged
    "nu": 0.29,
    "rho": 8190.0,                                                # kg/m3
    "cp": 435.0,                                                  # J/(kg K) near 300 K
    "T_limit": 1050.0,                                            # K, handbook 11.1
}

MATERIALS = {"IN718": IN718}


def get(name):
    return MATERIALS[name]


def k(mat, T):
    return np.interp(T, _T, mat["k"])


def E(mat, T):
    return np.interp(T, _T, mat["E"])


def alpha(mat, T):
    return np.interp(T, _T, mat["alpha"])


def yield_strength(mat, T):
    return np.interp(T, _T, mat["sy"])


def diffusivity(mat, T):
    return k(mat, T) / (mat["rho"] * mat["cp"])
