"""Minimal stub for predict_lightcurve_from_fit.

The thesis fitter (fit_combined.py) imports this module as `uvot_util`.
In the full project this module provides UVOT response folding and utilities.

For the generalized *local* MCMC script (and for Ronchini Rc-only usage), the
only required dependency is the keV<->Hz conversion constant.

If you run inside the full project environment, your real module will shadow
this stub.
"""

from __future__ import annotations

# 1 eV corresponds to nu = E/h = 2.417989242e14 Hz.
_HZ_PER_eV = 2.417989242e14

# keV -> Hz conversion factor.
KEV_TO_HZ = 1.0e3 * _HZ_PER_eV
