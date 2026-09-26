"""Orifice mass-flow models for liquid, gas, and flashing flow.

Run ``flight_config_sizing.py`` for a YAML case list. Import the functions
below when another tool needs a single calculation. See README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gas import gas_orifice
from liquid import liquid_orifice
from two_phase import two_phase_orifice

__all__ = ["liquid_orifice", "gas_orifice", "two_phase_orifice"]
