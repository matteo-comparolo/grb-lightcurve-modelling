#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_combined.py — Joint fitter for structured-jet HLE + forward shock models,
with XRT and optional optical datasets (UVOT or Ronchini Rc).

Command-line flexibility:
  - Robust data readers (CSV/TSV/QDP) for XRT with named columns + set selection (WT/PC/ALL)
  - tmin/tmax windows, arbitrary time masks to exclude ranges
  - Extra fractional error per dataset; optional log-space fit for XRT
  - Robust loss (soft_l1), f_scale, max_nfev
  - “Fast” pass resolution during the fit (lower grids); dense recompute for final plots
  - Dataset weights for residual block composition
  - Break targets (soft constraints) for HLE observed breaks (E_b1, E_b2)

Physics/architecture:
  - Single jet structure in YAML under `shared:` (used by BOTH HLE and FS).
  - HLE spectrum is a 2SBPL.
  - t0_obs is FITTED and applied as t' = t_data − t0_obs (flags can disable per component).
  - HLE→FS consistency link (R shell → FS onset) applied via YAML `consistency:` without altering FS flux code.
  - YAML with nested blocks: shared / hle / fs (+ consistency).
  - Non-numeric fixed YAML fields supported (e.g., strings like medium="ism").

Outputs:
  outdir/
    best_params.yaml
    fit_summary.json
    combined_xrt.png
    combined_uvot_white.png
    xrt_fit_at_data.csv
    uvot_fit_at_data.csv
    gamma_model.csv
    breaks_observed.png, breaks_comoving.png (if available from HLE grid)
"""

from __future__ import annotations
import argparse, os, re, json, inspect, warnings
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from scipy.interpolate import PchipInterpolator

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import yaml

# Local modules (must be importable)
import hle
import fs as fs
import predict_lightcurve_from_fit as uvot_util

# ===========
# Constants
# ===========
C_CGS = 2.99792458e10   # [cm/s]
C_A_HZ = 2.99792458e18  # [A·Hz]   (nu[Hz] = C_A_HZ / lambda[Å])
E_HZ_PER_eV = 2.418e14  # [Hz/eV]
H_ERG_S = 6.62607015e-27  # Planck's constant [erg s]
# ====================
# Plot styling (global)
# ====================
COL_TOT  = 'tab:blue'
COL_HLE  = 'tab:orange'
COL_FS   = 'tab:green'
COL_DATA = 'tab:red'
LS_TOT = '-'
LS_HLE = '--'
LS_FS  = '-.'

# ====================
# YAML load / dump
# ====================
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

# ==========================
# Dust / extinction helpers
# ==========================
def _conf_get_numeric(conf: dict, block: str, key: str, default=None):
    """Extract numeric from YAML either as scalar or {value: ...} record."""
    try:
        v = conf.get(block, {}).get(key, None)
        if isinstance(v, dict) and ("value" in v):
            return float(v["value"])
        if v is not None:
            return float(v)
    except Exception:
        pass
    return default

def _get_ebv_from_conf(conf: dict) -> float:
    """Resolve E(B-V) from YAML: dust.ebv_mode in {'map','value','off'}.
    If 'map', try dustmaps.sfd with optional dust.cache_dir and shared.{ra_deg,dec_deg}.
    Fallbacks: dust.ebv_value (if provided), else 0.0."""
    dust = conf.get("dust", {}) or {}
    mode = str(dust.get("ebv_mode", "off")).lower()
    if mode not in {"map", "value"}:
        return 0.0
    if mode == "value":
        try:
            v = dust.get("ebv_value", None)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0
    # mode == "map"
    ra = _conf_get_numeric(conf, "shared", "ra_deg", None)
    dec = _conf_get_numeric(conf, "shared", "dec_deg", None)
    if (ra is None) or (dec is None):
        return 0.0
    cache_dir = dust.get("cache_dir", None)
    # 1) Try user's predict_* utility (if available)
    for attr in ("get_sfd_ebv", "get_ebv", "ebv_sfd"):
        if hasattr(uvot_util, attr):
            try:
                return float(getattr(uvot_util, attr)(ra=float(ra), dec=float(dec), cache_dir=cache_dir))
            except Exception:
                break
    # 2) Try dustmaps.sfd directly (optional dependency)
    try:
        if cache_dir:
            import os as _os
            _os.environ.setdefault("DUSTMAPS_PATH", str(cache_dir))
        from dustmaps.sfd import SFDQuery
        import astropy.coordinates as asc
        import astropy.units as u
        sfd = SFDQuery()
        coo = asc.SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame="icrs")
        return float(sfd(coo))
    except Exception:
        return 0.0

# ============================================
# GRB coordinates and trigger time utilities
# ============================================
def load_grb_coords_and_triggers(coords_file: str) -> dict:
    """
    Parse a whitespace-separated text file with columns:
      NAME  RA_deg  DEC_deg  TRIGGER_MET_s
    Lines starting with '#' are comments and ignored.

    Returns:
      coords[name] = {"ra": float, "dec": float, "trigger_met": float or None}

    Note: We DO NOT use csv.reader here, because the file is whitespace-separated.
    """
    out = {}
    with open(coords_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()  # split on any whitespace
            if len(parts) < 3:
                continue
            name = parts[0]
            try:
                ra = float(parts[1])
                dec = float(parts[2])
            except ValueError:
                # Skip malformed numeric fields
                continue
            trig = None
            if len(parts) >= 4:
                try:
                    trig = float(parts[3])
                except ValueError:
                    trig = None
            out[name] = {"ra": ra, "dec": dec, "trigger_met": trig}
    return out

def _k_lambda_CCM89_O94(wave_A: np.ndarray, Rv: float = 3.1) -> np.ndarray:
    """Return k(λ) = A_lambda / E(B-V) using CCM89 + O'Donnell (1994) in the optical.
    Input wavelength in Angstrom. Valid ~0.125–3.5 μm."""
    lam_um = np.asarray(wave_A, float) * 1e-4
    x = 1.0 / np.maximum(lam_um, 1e-9)  # [um^-1]
    a = np.zeros_like(x); b = np.zeros_like(x)
    # IR: x < 1.1
    m = (x < 1.1)
    if np.any(m):
        a[m] = 0.574 * x[m]**1.61
        b[m] = -0.527 * x[m]**1.61
    # Optical/NIR: 1.1 <= x < 3.3  (O'Donnell 94)
    m = (x >= 1.1) & (x < 3.3)
    if np.any(m):
        y = x[m] - 1.82
        a[m] = (1. + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
                  + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
        b[m] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3
                  - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    # UV: 3.3 <= x <= 8.0  (CCM89)
    m = (x >= 3.3) & (x <= 8.0)
    if np.any(m):
        xx = x[m]
        Fa = np.zeros_like(xx); Fb = np.zeros_like(xx)
        m2 = (xx > 5.9)
        if np.any(m2):
            y2 = xx[m2] - 5.9
            Fa[m2] = -0.04473*y2**2 - 0.009779*y2**3
            Fb[m2] =  0.2130*y2**2 + 0.1207*y2**3
        a[m] = 1.752 - 0.316*xx - 0.104/((xx-4.67)**2 + 0.341) + Fa
        b[m] = -3.090 + 1.825*xx + 1.206/((xx-4.62)**2 + 0.263) + Fb
    # k_lambda = a*Rv + b
    return a*Rv + b

def _host_transmission(wave_A_obs: np.ndarray, z: float, EBV_host: float, Rv_host: float = 3.1,
                       law: str = "mw") -> np.ndarray:
    """
    Return transmission T_host(λ_obs) for host-galaxy extinction applied in the *rest frame*.
    We currently support 'mw' (CCM89 + O'Donnell 94). Others can be added later.
    """
    EBV = float(max(0.0, EBV_host))
    if EBV <= 0.0:
        return np.ones_like(wave_A_obs, dtype=float)
    z = float(max(0.0, z))
    lam_rest_A = np.asarray(wave_A_obs, float) / (1.0 + z + 1e-12)
    if law.lower() == "mw":
        k = _k_lambda_CCM89_O94(lam_rest_A, Rv=float(Rv_host))
    else:
        # Fallback to MW if an unknown law is requested
        k = _k_lambda_CCM89_O94(lam_rest_A, Rv=float(Rv_host))
    A_lambda = EBV * k
    return 10.0**(-0.4 * A_lambda)

# =================================================
# ParamVector (flatten nested YAML blocks safely)
# =================================================
class PSpec:
    def __init__(self, block: str, name: str, transform: str, fixed: bool, bounds: Tuple[float, float]):
        self.block = block
        self.name = name
        self.transform = transform  # "lin" | "log"
        self.fixed = fixed
        self.bounds = bounds        # in transformed space

class ParamVector:
    """
    Builds a flattened numeric vector for least_squares while:
      - preserving non-numeric fixed fields (kept in self.static)
      - handling lin/log transforms and bounds
    YAML schema: conf["shared"], conf["hle"], conf["fs"] each a dict of scalars OR dicts with keys:
       value, fixed, bounds [lo,hi], transform ("lin"|"log")
    """
    def __init__(self, conf: dict):
        self.conf = conf
        self.specs: List[PSpec] = []
        self.u0: List[float] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.static = {"shared": {}, "hle": {}, "fs": {}}
        self._build()

    @staticmethod
    def _is_number(x) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    def _build(self) -> None:
        for block in ("shared", "hle", "fs"):
            blk = self.conf.get(block, {}) or {}
            for name, entry in blk.items():
                if isinstance(entry, dict) and "value" in entry:
                    vraw = entry["value"]
                    if not self._is_number(vraw):
                        self.static[block][name] = vraw
                        continue
                    val = float(vraw)
                    fixed = bool(entry.get("fixed", False))
                    tr = str(entry.get("transform", "lin"))
                    bnds = entry.get("bounds", None)
                    if bnds is None:
                        lo, hi = -np.inf, np.inf
                    else:
                        lo, hi = float(bnds[0]), float(bnds[1])
                    if tr == "log":
                        if min(val, lo, hi) <= 0:
                            raise ValueError(f"[{block}.{name}] log-transform requires positive values and bounds.")
                        uval, ulb, uub = np.log(val), np.log(lo), np.log(hi)
                    else:
                        uval, ulb, uub = val, lo, hi
                    self.specs.append(PSpec(block, name, tr, fixed, (ulb, uub)))
                    self.u0.append(uval); self.lb.append(ulb); self.ub.append(uub)
                else:
                    # direct scalar (could be non-numeric)
                    if self._is_number(entry):
                        val = float(entry)
                        self.specs.append(PSpec(block, name, "lin", True, (-np.inf, np.inf)))
                        self.u0.append(val); self.lb.append(-np.inf); self.ub.append(+np.inf)
                    else:
                        self.static[block][name] = entry
        self.u0 = np.asarray(self.u0, float)
        self.lb = np.asarray(self.lb, float)
        self.ub = np.asarray(self.ub, float)

    def active_mask(self) -> np.ndarray:
        return np.array([not ps.fixed for ps in self.specs], dtype=bool)

    def names(self) -> List[str]:
        return [f"{ps.block}.{ps.name}" for ps in self.specs]

    def u_to_blocks(self, u: np.ndarray) -> Dict[str, Dict[str, Any]]:
        out = {"shared": dict(self.static["shared"]),
               "hle":    dict(self.static["hle"]),
               "fs":     dict(self.static["fs"])}
        for i, ps in enumerate(self.specs):
            ui = self.u0[i] if ps.fixed else u[i]
            vi = float(np.exp(ui)) if (ps.transform == "log") else float(ui)
            out[ps.block][ps.name] = vi
        return out

# ==========================
# Generic helper utilities
# ==========================
def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _log_interp(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """
    Monotone log–log interpolation using PCHIP.

    Plotting note (important):
      - We *do not* extrapolate to the left (earlier than the model grid) because the HLE/FS
        models are not defined there in this fitter's convention; values are set to 0.
      - We *do* allow a controlled right-side extrapolation (later than the model grid)
        using a log–log linear continuation (i.e. a power-law), to avoid artificial
        vertical drop-offs at the plot boundary when xlim extends slightly beyond the
        model evaluation grid.
    """
    xs = np.asarray(x_src, float)
    ys = np.asarray(y_src, float)
    xt = np.asarray(x_tgt, float)

    m = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if np.count_nonzero(m) < 3:
        return np.zeros_like(xt, float)

    lx = np.log(xs[m])
    ly = np.log(ys[m])

    # enforce strictly increasing x
    order = np.argsort(lx)
    lx = lx[order]
    ly = ly[order]

    lxt = np.log(np.maximum(xt, 1e-12))

    # PCHIP inside the domain (no extrapolation)
    interp = PchipInterpolator(lx, ly, extrapolate=False)
    y = interp(lxt)

    # Left of domain -> 0
    left = lxt < lx[0]
    if np.any(left):
        y[left] = -np.inf

    # Right of domain -> controlled log–log linear extrapolation
    right = lxt > lx[-1]
    if np.any(right):
        # Estimate terminal slope from the last few points in log–log space
        k = int(min(5, lx.size))
        if k >= 2:
            dx = lx[-1] - lx[-k]
            if np.isfinite(dx) and dx != 0.0:
                slope = (ly[-1] - ly[-k]) / dx
            else:
                slope = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
        else:
            slope = 0.0

        y[right] = ly[-1] + slope * (lxt[right] - lx[-1])

    # Any remaining NaNs/Infs -> 0
    y[~np.isfinite(y)] = -np.inf
    return np.exp(y)



def _monochromatic_Fnu_from_components(
    t_eval_hle: np.ndarray,
    t_eval_fs: np.ndarray,
    nu_hz: float,
    shared: dict,
    hle_pars: dict,
    fs_pars: dict,
    return_components: bool = False,
):
    """
    Return monochromatic flux density F_nu(t, nu_hz) in cgs [erg s^-1 cm^-2 Hz^-1]
    from (HLE + FS), evaluated at a single observed frequency.

    Implementation note:
      - We call the underlying model functions on a narrow frequency grid around nu_hz
        and then log–log interpolate in frequency to obtain F_nu at nu_hz.
      - This keeps the change local to this file and does not require changes in hle/fs.
    """
    nu_hz = float(nu_hz)
    if not np.isfinite(nu_hz) or (nu_hz <= 0.0):
        raise ValueError(f"Invalid nu_hz for monochromatic optical model: {nu_hz}")

    # Build a small log-spaced frequency grid around nu_hz (±1 dex).
    # This is only used internally to enable robust log–log interpolation.
    nu_min_hz = nu_hz / 10.0
    nu_max_hz = nu_hz * 10.0
    n_nu = int(max(2, min(256, int(shared.get("n_nu", 64) if isinstance(shared, dict) else 64))))

    def _compute_Fnu_at_nu(comp: str, t_eval: np.ndarray) -> np.ndarray:
        if t_eval is None or (np.asarray(t_eval).size == 0):
            return np.zeros(0, float)

        t_eval = np.asarray(t_eval, float)

        # Convert Hz -> keV using the same constant used elsewhere in this file
        try:
            keV_min = nu_min_hz / float(uvot_util.KEV_TO_HZ)
            keV_max = nu_max_hz / float(uvot_util.KEV_TO_HZ)
        except Exception:
            # Fallback if import path changes
            E_HZ_PER_eV = 2.418e14
            keV_min = nu_min_hz / (1e3 * E_HZ_PER_eV)
            keV_max = nu_max_hz / (1e3 * E_HZ_PER_eV)

        if comp == "hle":
            pars = dict(shared); pars.update(hle_pars)
            pars.update({"nu_min_keV": keV_min, "nu_max_keV": keV_max, "n_nu": n_nu})
            out = hle.compute_hle_lightcurve(**_filter_kwargs(hle.compute_hle_lightcurve, pars))
        elif comp == "fs":
            pars = dict(shared); pars.update(fs_pars)
            pars.update({"nu_min_keV": keV_min, "nu_max_keV": keV_max, "n_nu": n_nu})
            out = fs.compute_fs_lightcurve(**_filter_kwargs(fs.compute_fs_lightcurve, pars))
        else:
            raise ValueError(f"Unknown component for monochromatic model: {comp}")

        # Unpack robustly (same conventions used in UVOTWhite._component_Fnu_on_white)
        if isinstance(out, (list, tuple)) and len(out) >= 6:
            t_grid = np.asarray(out[0], float)
            F_tnu  = np.asarray(out[4], float)
            nu_grid = np.asarray(out[5], float)
        elif isinstance(out, dict) and all(k in out for k in ("t", "F_tnu", "nu_grid")):
            t_grid = np.asarray(out["t"], float)
            F_tnu  = np.asarray(out["F_tnu"], float)
            nu_grid = np.asarray(out["nu_grid"], float)
        else:
            raise RuntimeError(f"{comp}: model output lacks F_tnu/nu_grid for monochromatic evaluation.")

        nu_grid = np.asarray(nu_grid, float)
        if not np.isfinite(nu_grid).any() or (nu_grid.size < 2):
            raise RuntimeError(f"{comp}: invalid frequency grid from model.")

        # If it looks like keV, convert to Hz
        if np.nanmax(nu_grid) < 1e9:
            try:
                nu_grid = nu_grid * float(uvot_util.KEV_TO_HZ)
            except Exception:
                E_HZ_PER_eV = 2.418e14
                nu_grid = nu_grid * (1e3 * E_HZ_PER_eV)

        # Ensure ascending order for interpolation
        s = np.argsort(nu_grid)
        nu_grid = nu_grid[s]
        F_tnu = F_tnu[..., s]

        # Interpolate in frequency (log–log) to nu_hz for each time bin
        # Expect F_tnu shape (Nt, Nnu); handle alternative shapes defensively.
        if F_tnu.ndim != 2:
            F_tnu = np.atleast_2d(F_tnu)

        Nt = F_tnu.shape[0]
        F_at_nu = np.zeros(Nt, float)
        ln_tgt = np.log(nu_hz)

        ln_src = np.log(np.maximum(nu_grid, 1e-300))
        for i in range(Nt):
            yi = np.asarray(F_tnu[i], float)
            yi = np.maximum(yi, 1e-300)
            F_at_nu[i] = np.exp(np.interp(ln_tgt, ln_src, np.log(yi)))

        # Interpolate in time to requested t_eval (log–log, monotone)
        return _log_interp(np.asarray(t_grid, float), np.asarray(F_at_nu, float), t_eval)

    F_hle = _compute_Fnu_at_nu("hle", t_eval_hle)
    F_fs  = _compute_Fnu_at_nu("fs",  t_eval_fs)
    # Combine robustly: allow one component to be empty (used for FS-only / HLE-only shortcuts).
    if F_hle.size == 0:
        F_tot = np.array(F_fs, float, copy=False)
    elif F_fs.size == 0:
        F_tot = np.array(F_hle, float, copy=False)
    else:
        # In the standard use-case, both are evaluated on the same t_eval grid
        # and should have identical shapes.
        F_tot = F_hle + F_fs
        if F_hle.shape != F_fs.shape:
            raise RuntimeError(f"Monochromatic: HLE/FS shapes differ: {F_hle.shape} vs {F_fs.shape}")

    if return_components:
        return F_hle, F_fs, F_tot
    return F_tot

def _filter_kwargs(func, kwargs: dict) -> dict:
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def _shift_times(t: np.ndarray, t0: float, apply: bool) -> np.ndarray:
    t = np.asarray(t, float)
    return np.maximum(t - (t0 if apply else 0.0), 1e-6)

def _keV_bounds_for_nu(nu_hz: np.ndarray) -> Tuple[float, float]:
    nu = np.asarray(nu_hz, float)
    keV = nu / (1e3 * E_HZ_PER_eV)
    return float(np.nanmin(keV)), float(np.nanmax(keV))

# ==============================
# Energy / normalization helpers
# ==============================
def _luminosity_distance_flatlcdm(z: float, H0_kms_Mpc: float = 67.7, Omega_m: float = 0.31) -> float:
    """Flat ΛCDM luminosity distance [cm]."""
    if z <= 0:
        return 0.0
    H0_s = H0_kms_Mpc / 3.085677581e19  # s^-1
    zs = np.linspace(0.0, z, 4000)
    Ez = np.sqrt(Omega_m * (1.0 + zs)**3 + (1.0 - Omega_m))
    Dc = (C_CGS / H0_s) * np.trapz(1.0 / Ez, zs)
    return (1.0 + z) * Dc

def _integrated_prompt_energy_iso(shared: dict, hpars: dict, band_keV: Tuple[float, float]) -> float:
    """Run HLE once and return E_iso in the given OBSERVED band [keV]."""
    t, F, _ = hle_band_flux_curve(_shared_to_hle_kwargs(shared), hpars, band_keV)  # F: [erg s^-1 cm^-2]
    if t.size == 0:
        return 0.0
    E_obs = np.trapz(F, t)  # [erg cm^-2]
    if not np.isfinite(E_obs) or E_obs <= 0.0:
        return 0.0
    z = float(shared.get("z", 0.0))
    DL = _luminosity_distance_flatlcdm(z)
    return 4.0 * np.pi * DL * DL * E_obs / max(1.0 + z, 1.0)  # [erg]

def _calibrate_ip_from_Egamma(shared: dict, hpars: dict, band_keV: Tuple[float, float]) -> float:
    """Return i_p_prime that makes HLE match Egamma_iso_core in 'band_keV'."""
    Eg = shared.get("Egamma_iso_core", None)
    if Eg is None:
        return float(hpars.get("i_p_prime", 1.0))
    try:
        Eg = float(Eg)
    except Exception:
        return float(hpars.get("i_p_prime", 1.0))
    tmp = dict(hpars); tmp["i_p_prime"] = 1.0
    E_per_unit = _integrated_prompt_energy_iso(shared, tmp, band_keV)
    if E_per_unit <= 0.0 or not np.isfinite(E_per_unit):
        return float(hpars.get("i_p_prime", 1.0))
    return Eg / E_per_unit

def _derive_fs_Eiso_from_prompt(Eg: float, eta: float) -> float:
    """
    Deterministically compute the FS isotropic-equivalent kinetic energy from prompt Egamma and efficiency:
        E_k,iso = Egamma * (1 - eta) / eta
    """
    Eg = float(Eg); et = float(eta)
    if not (np.isfinite(Eg) and Eg > 0.0 and 0.0 < et < 1.0):
        raise RuntimeError("Egamma_iso_core and eta_gamma must be finite, with 0<eta_gamma<1.")
    return Eg * (1.0 - et) / et


# =======================================
# HLE→FS onset coupling (consistency)
# =======================================
def _pick_gamma_for_link(shared: dict, name: str) -> float:
    return float(shared.get(name, shared.get("Gamma_c", 300.0)))

def _t_on_from_R(R_shell_cm: float, z: float, Gamma_link: float, eta: float = 1.0) -> float:
    R = max(float(R_shell_cm), 0.0)
    G = max(float(Gamma_link), 1.0)
    return float(eta) * (1.0 + float(z)) * R / (2.0 * C_CGS * G * G)

def _apply_fs_onset_coupling(shared: dict, hpars: dict, fpars_in: dict, cfg: dict) -> dict:
    """Return a COPY of FS params with fs_on_* overridden by YAML 'consistency' rule."""
    fpars = dict(fpars_in)
    cons  = cfg.get("consistency", {}) or {}
    if not cons.get("link_hle_R_to_fs", False):
        return fpars
    mode = str(cons.get("mode", "tobs_floor")).lower()
    z = float(shared.get("z", 0.0))
    R_hle = float(hpars.get("R", 0.0))
    if R_hle <= 0.0:
        return fpars
    if mode == "radius_floor":
        xi = float(cons.get("xi_radius", 1.0))
        fpars["fs_on_mode"] = "radius"
        fpars["fs_on_radius"] = xi * R_hle
        return fpars
    gamma_name = str(cons.get("gamma_for_link", "Gamma_c"))
    Gamma_link = _pick_gamma_for_link(shared, gamma_name)
    eta = float(cons.get("eta_time", 1.0))
    t_min = _t_on_from_R(R_hle, z, Gamma_link, eta)

    # IMPORTANT: convert the floor to the SAME time coordinate used by the FS model.
    # The fitter evaluates models on t' = t_obs - t0_component when apply_t0_to_component=True.
    # Therefore the HLE-based floor (defined relative to the HLE reference) must be shifted by (t0_hle - t0_fs)
    # when we work in FS-relative time. If apply_t0_to_hle/fs are disabled, the corresponding reference is 0.
    apply_t0_hle = bool(cfg.get("fit", {}).get("apply_t0_to_hle", True))
    apply_t0_fs  = bool(cfg.get("fit", {}).get("apply_t0_to_fs",  True))
    t0_hle = float(shared.get("t0_obs", 0.0))
    t0_fs  = float(shared.get("t0_fs_obs", t0_hle))
    t_ref_hle = t0_hle if apply_t0_hle else 0.0
    t_ref_fs  = t0_fs  if apply_t0_fs  else 0.0
    t_min_fs_frame = t_min + (t_ref_hle - t_ref_fs)

    fpars["fs_on_mode"] = "tobs"
    # IMPORTANT: fs_afterglow expects 'fs_on_tobs'. YAML may provide null/strings: handle robustly.
    def _float_or(x, default=0.0):
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default
    t_user = _float_or(fpars.get("fs_on_tobs", None), 0.0)
    fpars["fs_on_tobs"] = max(t_user, t_min_fs_frame)
    return fpars

# ==================
# Data I/O (XRT)
# ==================
def read_table_auto(path: str,
                    col_t: str, col_y: str, col_err: Optional[str] = None,
                    col_ypos: str = None, col_yneg: str = None,
                    col_tpos: str = None, col_tneg: str = None,
                    err_sym: str = "max",
                    qdp_set: str = "ALL"):
    """
    Robust CSV/TSV/QDP reader (named columns; asymmetric errors; QDP set selection).
    """
    import pandas as pd
    df = None
    p = str(path).lower()

    try:
        if p.endswith(".csv"):
            df = pd.read_csv(path, comment="#")
        elif p.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", comment="#")
    except Exception:
        df = None

    # Basic QDP parsing (WT/PC sets)
    if df is None and p.endswith(".qdp"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        header_idx, cols = None, None
        wanted_sets = [
            {"Time","Flux","Fluxpos","Fluxneg"},
            {"Time","Rate","Ratepos","Rateneg"},
            {"Time","T+ve","T-ve","Flux","Fluxpos","Fluxneg"},
            {"Time","T+ve","T-ve","Rate","Ratepos","Rateneg"},
        ]
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith(("!","#")): continue
            toks = [t for t in re.split(r"\s+", s) if t]
            if any(ws.issubset(set(toks)) for ws in wanted_sets):
                header_idx, cols = i, toks; break
        num_re = re.compile(r'^[\+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][\+\-]?\d+)?$')
        rows, set_ids, set_id = [], [], 0
        def is_sep(s: str) -> bool:
            tok = s.upper().split()
            return (len(tok)==2 and tok[0]=="NO" and tok[1]=="NO")
        def is_directive(s: str) -> bool:
            u = s.upper()
            return u.startswith(("READ","LABEL","ERROR","SERR","TERR","MARK","NEW"))
        it = enumerate(lines[header_idx+1:] if header_idx is not None else lines)
        for _, line in it:
            s = line.strip()
            if not s: continue
            if is_sep(s): set_id += 1; continue
            if s.startswith(("!","#")) or is_directive(s): continue
            toks = [t for t in re.split(r"\s+", s) if t]
            nums = [float(t) for t in toks if num_re.match(t)]
            if nums: rows.append(nums); set_ids.append(set_id)
        if not rows: raise ValueError(f"No numeric data rows in {path}")
        import pandas as pd
        lengths = pd.Series([len(r) for r in rows]); L = int(lengths.mode().iloc[0])
        rows = [r for r in rows if len(r)==L]; set_ids = [sid for (sid,r) in zip(set_ids, rows) if len(r)==L]
        if header_idx is not None and cols is not None:
            if len(cols)>L: cols=cols[:L]
            if L==4 and ("Time" in cols): cols=["Time","Y","Ypos","Yneg"]
        else:
            cols=(["Time","T+ve","T-ve","Flux","Fluxpos","Fluxneg"] if L==6 else ["Time","Y","Ypos","Yneg"])
        df = pd.DataFrame(rows, columns=cols); df["set_id"]=np.array(set_ids,int)
        if qdp_set.upper()=="WT": df = df[df["set_id"]==0]
        elif qdp_set.upper()=="PC": df = df[df["set_id"]==1]
        if df.empty: raise ValueError(f"{path}: selection {qdp_set} produced an empty table.")

    if df is None:
        # Space-separated fallback
        import pandas as pd
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="!", on_bad_lines="skip")

    df.columns = [c.strip() for c in df.columns]
    def asf(name): 
        if name not in df.columns: raise ValueError(f"Column '{name}' not found in {path}. Have: {list(df.columns)}")
        return pd.to_numeric(df[name], errors="coerce").to_numpy()

    t = asf(col_t); y = asf(col_y)
    # y-errors: symmetric or from y+ / y-
    if col_ypos and col_yneg and (col_ypos in df.columns) and (col_yneg in df.columns):
        yp = np.abs(asf(col_ypos)); yn = np.abs(asf(col_yneg))
        if   err_sym=="max":  yerr = np.maximum(yp, yn)
        elif err_sym=="mean": yerr = 0.5*(yp+yn)
        elif err_sym=="pos":  yerr = yp
        elif err_sym=="neg":  yerr = yn
        else: raise ValueError(f"Unknown err_sym={err_sym}")
    else:
        if col_err is None or (col_err not in df.columns): raise ValueError("No symmetric error column and no ypos/yneg provided.")
        yerr = np.abs(asf(col_err))

    tpos = asf(col_tpos) if (col_tpos and col_tpos in df.columns) else None
    tneg = asf(col_tneg) if (col_tneg and col_tneg in df.columns) else None

    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if tpos is not None and tneg is not None:
        m &= np.isfinite(tpos) & np.isfinite(tneg)
        return t[m], y[m], yerr[m], tpos[m], tneg[m]
    else:
        return t[m], y[m], yerr[m], None, None

def load_spectral_evolution_txt(path: str) -> dict:
    """
    Load Burst Analyser 'XRT spectral evolution data' (text table).

    Expected (typical) format is the one you pasted:
      time  time_err+ time_err-  ...  Gamma  Gamma_err+ Gamma_err-  ...

    We ignore comment/header lines starting with '!' or '#'
    and skip lines containing 'NO'.

    Returns a dict with arrays:
      t, t_err_plus, t_err_minus, gamma, gamma_err_plus, gamma_err_minus
    """
    import numpy as np

    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!"):
                continue
            if "NO" in s:
                continue
            parts = s.split()
            # Must be numeric row
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(vals)

    if len(rows) == 0:
        raise ValueError(f"No numeric data rows found in {path}")

    arr = np.array(rows, dtype=float)

    # Fallback for Burst Analyser format you pasted:
    # 0: time
    # 1: time_err_plus
    # 2: time_err_minus
    # 12: Gamma
    # 13: Gamma_err_plus
    # 14: Gamma_err_minus
    if arr.shape[1] < 15:
        raise ValueError(
            f"Unexpected number of columns ({arr.shape[1]}) in {path}. "
            "I expected the Burst Analyser table with >= 15 columns."
        )

    out = {
        "t": arr[:, 0],
        "t_err_plus": np.abs(arr[:, 1]),
        "t_err_minus": np.abs(arr[:, 2]),
        "gamma": arr[:, 12],
        "gamma_err_plus": np.abs(arr[:, 13]),
        "gamma_err_minus": np.abs(arr[:, 14]),
    }

    # Keep only finite points
    m = np.isfinite(out["t"]) & np.isfinite(out["gamma"])
    for k in out:
        out[k] = out[k][m]

    return out

# ==================
# UVOT White CSV
# ==================
def load_uvot_white_csv(path: str, col_t="t", col_y="Y", col_err="Yerr"):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    t = np.asarray(arr[col_t], float)
    y = np.asarray(arr[col_y], float)
    yerr = np.asarray(arr[col_err], float)
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    return t[m], y[m], yerr[m]


def load_ronchini_rc_txt(path: str,
                         col_t: str = "ts",
                         col_y_uJy: str = "fkmicJy",
                         col_err_uJy: str = "dfkmicJy",
                         to_cgs: bool = True):
    """Load Ronchini et al. optical Rc light curve from a CSV-like text file.

    Expected header columns (default): ts,fkmicJy,dfkmicJy
      - ts          : time since trigger [s]
      - fkmicJy     : flux density [microJy]
      - dfkmicJy    : 1-sigma uncertainty [microJy]

    Returns (t, y, yerr) where y is Fnu in cgs if to_cgs=True, else in microJy.

    Notes
    -----
    Ronchini optical light curves are typically already corrected (Galactic extinction,
    host/SN subtraction, and shifted to a reference band). Therefore, this loader does
    not apply any calibration steps; it only parses and (optionally) converts units.
    """
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    t = np.asarray(arr[col_t], float)
    y = np.asarray(arr[col_y_uJy], float)
    yerr = np.asarray(arr[col_err_uJy], float)

    # identify upper limits: dfkmicJy <= 0
    is_ul = ~(np.isfinite(yerr) & (yerr > 0))
    
    # keep all finite points
    m = np.isfinite(t) & np.isfinite(y) & (t > 0)
    t, y, yerr, is_ul = t[m], y[m], yerr[m], is_ul[m]

    # Ensure time ordering (keep duplicates; we will decide later whether to merge them)
    if t.size:
        o = np.argsort(t)
        t, y, yerr = t[o], y[o], yerr[o]

    if to_cgs:
        # 1 microJy = 1e-29 erg s^-1 cm^-2 Hz^-1
        conv = 1e-29
        y *= conv
        yerr *= conv

    return t, y, yerr, is_ul

# ===========================
# Model adapters (HLE / FS)
# ===========================
def _shared_to_hle_kwargs(shared: dict) -> dict:
    """Map shared keys to hle.py expected names (k vs k_struct, etc.)."""
    out = dict(shared)
    if "k" not in out and "k_struct" in out:
        out["k"] = out["k_struct"]
    return out

def hle_band_flux_curve(shared: dict, hle_pars: dict,
                        band_keV: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, dict]:
    pars = _shared_to_hle_kwargs(shared)
    pars.update(hle_pars)
    pars["spectrum"] = 2  # enforce 2SBPL
    pars["nu_min_keV"] = float(band_keV[0])
    pars["nu_max_keV"] = float(band_keV[1])
    out = hle.compute_hle_lightcurve(**_filter_kwargs(hle.compute_hle_lightcurve, pars))
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        t_grid = np.asarray(out[0], float); F_band = np.asarray(out[1], float)
        meta = {}
        if len(out) >= 5: meta["F_tnu"] = np.asarray(out[4], float)
        if len(out) >= 6: meta["nu_grid"] = np.asarray(out[5], float)
        if len(out) >= 7: meta["Eb1_obs"] = np.asarray(out[6], float)
        if len(out) >= 8: meta["Eb2_obs"] = np.asarray(out[7], float)
        return t_grid, F_band, meta
    if isinstance(out, dict) and "t" in out and "Fband" in out:
        meta = {}
        for k in ("F_tnu","nu_grid","Eb1_obs","Eb2_obs"):
            if k in out: meta[k] = np.asarray(out[k], float)
        return np.asarray(out["t"], float), np.asarray(out["Fband"], float), meta
    raise RuntimeError("hle.compute_hle_lightcurve returned an unexpected structure.")

def fs_band_flux_curve(shared: dict, fs_pars: dict,
                       band_keV: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, dict]:
    pars = dict(shared); pars.update(fs_pars)
    pars["nu_min_keV"] = float(band_keV[0])
    pars["nu_max_keV"] = float(band_keV[1])
    out = fs.compute_fs_lightcurve(**_filter_kwargs(fs.compute_fs_lightcurve, pars))
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        t_grid = np.asarray(out[0], float); F_band = np.asarray(out[1], float)
        meta = {}
        if len(out) >= 5: meta["F_tnu"] = np.asarray(out[4], float)
        if len(out) >= 6: meta["nu_grid"] = np.asarray(out[5], float)
        # Optional FS break energies (observer frame) [keV]
        if len(out) >= 7: meta["Em_keV"] = np.asarray(out[6], float)
        if len(out) >= 8: meta["Ec_keV"] = np.asarray(out[7], float)
        return t_grid, F_band, meta
    if isinstance(out, dict) and "t" in out and "Fband" in out:
        meta = {}
        for k in ("F_tnu","nu_grid","Em_keV","Ec_keV"):
            if k in out: meta[k] = np.asarray(out[k], float)
        return np.asarray(out["t"], float), np.asarray(out["Fband"], float), meta
    raise RuntimeError("fs.compute_fs_lightcurve returned an unexpected structure.")

# =================================
# UVOT White folding (HLE + FS)
# =================================
class UVOTWhite:
    """Fold (HLE+FS) spectra through WHITE effective area to counts/s (with optional Galactic extinction)."""
    def __init__(self, effarea_dir: str, dust_conf: dict | None = None, shared_conf: dict | None = None):
        wave_A, EA = uvot_util.load_white_effarea(effarea_dir)
        self.wave_A = np.asarray(wave_A, float)  # [Å]
        self.EA = np.asarray(EA, float)          # [cm^2]
        self.nu_white = C_A_HZ / self.wave_A     # [Hz]
        # Precompute extinction transmission if requested
        dust = dust_conf or {}
        use_ext = bool(dust.get("use_galactic_extinction", False))
        if use_ext:
            # Read R_V from YAML (default Milky Way value 3.1)
            Rv = float(dust.get("rv", 3.1))
            # Build a minimal config for EBV look-up that includes BOTH 'dust' and 'shared' 
            conf_for_ebv = {
                "dust": (dust_conf or {}),
                "shared": (shared_conf or {}),
            }
            # If the user forces EBV numerically, honor it; else query SFD via _get_ebv_from_conf(...).
            if str(dust.get("ebv_mode", "off")).lower() == "value" and (dust.get("ebv_value") is not None):
                EBV = float(dust["ebv_value"])
            else:
                EBV = _get_ebv_from_conf(conf_for_ebv)
            # Read R_V from YAML (default: 3.1 for MW) before using it
            Rv = float(dust.get("rv", 3.1))
            k_lambda = _k_lambda_CCM89_O94(self.wave_A, Rv=Rv)          # A_lambda / E(B-V)
            A_lambda = EBV * k_lambda                                   # [mag]
            self.ext_trans = 10.0**(-0.4 * A_lambda)                     # dimensionless
        else:
            self.ext_trans = None

    def _component_Fnu_on_white(self, t_eval: np.ndarray,
                                comp: str,
                                shared: dict,
                                hle_pars: dict,
                                fs_pars: dict) -> Tuple[np.ndarray, np.ndarray]:
        keV_min, keV_max = _keV_bounds_for_nu(self.nu_white)
        n_nu_desired = int(max(16, min(512, self.nu_white.size)))
        if comp == "HLE":
            pars = _shared_to_hle_kwargs(shared); pars.update(hle_pars)
            pars.update({"spectrum":2, "nu_min_keV":keV_min, "nu_max_keV":keV_max, "n_nu":n_nu_desired})
            out = hle.compute_hle_lightcurve(**_filter_kwargs(hle.compute_hle_lightcurve, pars))
        else:
            pars = dict(shared); pars.update(fs_pars)
            pars.update({"nu_min_keV":keV_min, "nu_max_keV":keV_max, "n_nu":n_nu_desired})
            out = fs.compute_fs_lightcurve(**_filter_kwargs(fs.compute_fs_lightcurve, pars))

        # unpack robustly
        if isinstance(out, (list, tuple)) and len(out) >= 6:
            t_grid = np.asarray(out[0], float)
            F_tnu  = np.asarray(out[4], float)
            nu     = np.asarray(out[5], float)
        elif isinstance(out, dict) and all(k in out for k in ("t","F_tnu","nu_grid")):
            t_grid = np.asarray(out["t"], float)
            F_tnu  = np.asarray(out["F_tnu"], float)
            nu     = np.asarray(out["nu_grid"], float)
        else:
            raise RuntimeError(f"{comp}: model output lacks F_tnu/nu_grid.")

        # --- Ensure frequency is in Hz and strictly ascending for log–log interpolation
        nu = np.asarray(nu, dtype=float)
        if not np.isfinite(nu).any() or nu.size < 2:
            raise RuntimeError(f"{comp}: invalid frequency grid from model.")
        # If it looks like keV, convert to Hz using predict_* constants
        if np.nanmax(nu) < 1e9:  # keV -> Hz
            try:
                nu = nu * uvot_util.KEV_TO_HZ
            except Exception:
                # Fallback: define locally if import path changes
                E_HZ_PER_eV = 2.418e14
                nu = nu * (1e3 * E_HZ_PER_eV)

        # Sort ν ascending and reorder spectral axis accordingly
        idx_nu = np.argsort(nu)
        nu = nu[idx_nu]
        F_tnu = np.asarray(F_tnu, dtype=float)
        if F_tnu.shape[1] != idx_nu.size:
            raise RuntimeError(f"{comp}: F_tnu shape mismatch vs nu_grid.")
        F_tnu = F_tnu[:, idx_nu]

        # time interpolation (log–log per frequency)
        t_eval = np.asarray(t_eval, float)
        F_on_t = np.zeros((t_eval.size, F_tnu.shape[1]), float)
        ltg = np.log(np.maximum(t_grid, 1e-12))
        for j in range(F_tnu.shape[1]):
            fj = np.asarray(F_tnu[:, j], float)
            m = (t_grid > 0) & (fj > 0)
            if np.count_nonzero(m) < 2: 
                continue
            F_on_t[:, j] = np.exp(np.interp(np.log(np.maximum(t_eval, 1e-12)), ltg[m], np.log(fj[m])))

        # frequency interpolation to WHITE grid (log–log per time)
        ln_src = np.log(np.maximum(nu, 1e-300))
        ln_tgt = np.log(np.maximum(self.nu_white, 1e-300))
        F_on_white = np.zeros((t_eval.size, self.nu_white.size), float)
        for i in range(t_eval.size):
            yi = np.maximum(F_on_t[i], 1e-300)
            F_on_white[i] = np.exp(np.interp(ln_tgt, ln_src, np.log(yi)))
        return self.nu_white, F_on_white

    def counts_from_components(self, t_eval_hle: np.ndarray,
                               shared: dict, hle_pars: dict, fs_pars: dict,
                               t_eval_fs: Optional[np.ndarray] = None,
                               host_ext: Optional[dict] = None) -> np.ndarray:
        """
        Return UVOT WHITE count rate (counts/s) from (HLE + FS).
        Galactic extinction (if configured) is applied here; optional host extinction
        (rest-frame) is also supported when host_ext is provided.
        """
        if t_eval_fs is None:
            t_eval_fs = t_eval_hle

        # Spectra on WHITE grid
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_h = ex.submit(self._component_Fnu_on_white, np.asarray(t_eval_hle, float), "HLE", shared, hle_pars, fs_pars)
            fut_f = ex.submit(self._component_Fnu_on_white, np.asarray(t_eval_fs,  float), "FS",  shared, hle_pars, fs_pars)
            nu_h, F_h = fut_h.result()
            nu_f, F_f = fut_f.result()

        F_tot = F_h + F_f  # [erg s^-1 cm^-2 Hz^-1]
        lam = self.wave_A[None, :]                             # [Å]
        EA  = self.EA[None, :]                                 # [cm^2]

        # Convert to F_lambda
        F_lambda = F_tot * (C_A_HZ / (lam**2))
        
        # Galactic extinction (data are NOT corrected in 'counts' mode)
        if self.ext_trans is not None:
            F_lambda = F_lambda * (self.ext_trans[None, :])

        # Optional host extinction (rest-frame)
        if host_ext and bool(host_ext.get("use_host_extinction", False)):
            z = float(shared.get("z", 0.0))
            Th = _host_transmission(self.wave_A, z,
                                    float(host_ext.get("ebv_host", 0.0)),
                                    float(host_ext.get("rv_host", 3.1)),
                                    str(host_ext.get("law", "mw")))
            F_lambda = F_lambda * (Th[None, :])

        # Photon conversion
        HC_ERG_A = H_ERG_S * C_A_HZ
        photon_kernel = (lam / HC_ERG_A)
        counts = np.trapz(F_lambda * EA * photon_kernel, self.wave_A, axis=1)
        return counts


    def band_flux_from_components(self,
                                  t_eval_hle: np.ndarray,
                                  shared: dict,
                                  hle_pars: dict,
                                  fs_pars: dict,
                                  t_eval_fs: Optional[np.ndarray] = None,
                                  apply_extinction: bool = False,
                                  host_ext: Optional[dict] = None) -> np.ndarray:
        """
        Band-averaged Fν (erg s^-1 cm^-2 Hz^-1) using WHITE effective area weighting.
        By default (apply_extinction=False) we return the *intrinsic* model, which must be
        compared to dereddened data (predict_* pipeline). If host_ext is provided and
        use_host_extinction=True, we apply the host attenuation here (rest-frame).
        """
        if t_eval_fs is None:
            t_eval_fs = t_eval_hle

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_h = ex.submit(self._component_Fnu_on_white, np.asarray(t_eval_hle, float), "HLE", shared, hle_pars, fs_pars)
            fut_f = ex.submit(self._component_Fnu_on_white, np.asarray(t_eval_fs,  float), "FS",  shared, hle_pars, fs_pars)
            nu_h, F_h = fut_h.result()
            nu_f, F_f = fut_f.result()

        if not np.allclose(nu_h, nu_f):
            F_tot = F_h + np.interp(self.nu_white, nu_f, F_f, left=0.0, right=0.0)
        else:
            F_tot = F_h + F_f

        # Optional Galactic extinction in *flux* mode is normally OFF (data are dereddened).
        if apply_extinction and (self.ext_trans is not None):
            F_tot = F_tot * (self.ext_trans[None, :])

        # Optional host extinction (rest-frame) – applied to the *model* in flux mode
        if host_ext and bool(host_ext.get("use_host_extinction", False)):
            z = float(shared.get("z", 0.0))
            Th = _host_transmission(self.wave_A, z,
                                    float(host_ext.get("ebv_host", 0.0)),
                                    float(host_ext.get("rv_host", 3.1)),
                                    str(host_ext.get("law", "mw")))
            # Convert to F_lambda, attenuate, then average back to Fν through the EA kernel
            lam = self.wave_A[None, :]
            F_lambda = F_tot * (C_A_HZ / (lam**2))
            F_lambda = F_lambda * (Th[None, :])
            # Back to effective Fν averaging through the same helper
            F_tot = uvot_util.bandpass_average_model(F_lambda * (lam**2 / C_A_HZ), self.nu_white, self.wave_A, self.EA)
            return F_tot

        return uvot_util.bandpass_average_model(F_tot, self.nu_white, self.wave_A, self.EA)


# =========================
# Break targets helper (HLE)
# =========================
def parse_targets_list(raw_list: List[str]) -> List[Tuple[float,float,float]]:
    out = []
    for s in (raw_list or []):
        try:
            parts = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x!=""]
            if   len(parts) == 2: t, E = parts; frac = 0.2
            elif len(parts) >= 3: t, E, frac = parts[:3]
            else: continue
            if min(t,E,frac) <= 0: continue
            out.append((t,E,frac))
        except Exception:
            warnings.warn(f"Could not parse target '{s}'. Expected 't,E[,frac]'. Skipped.")
    return out

# ======================
# Residuals assembly
# ======================
def make_residual_fn(conf: dict,
                     xrt: dict, uvw_block: dict,
                     uvw: UVOTWhite,
                     pvec: ParamVector):

    band = tuple(map(float, conf.get("fit", {}).get("xrt_band_keV", [0.3, 10.0])))
    W = conf.get("fit", {}).get("weight_blocks", {"xrt":1.0, "uvot_white":1.0, "targets":1.0})
    apply_t0_hle = bool(conf.get("fit", {}).get("apply_t0_to_hle", True))
    apply_t0_fs  = bool(conf.get("fit", {}).get("apply_t0_to_fs",  True))
    active = pvec.active_mask()

    # Prepare target constraints for HLE observed breaks (if any)
    targets = conf.get("targets", []) or []
    # Allow CLI override via xrt["targets_b1/b2"] already parsed
    targets_b1 = uvw_block.get("targets_b1", [])
    targets_b2 = uvw_block.get("targets_b2", [])

    def _hle_breaks_fast(shared: dict, hpars: dict) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute fast HLE break arrays (observed) on the model's own time grid."""
        pars = _shared_to_hle_kwargs(shared)
        pars.update(hpars); pars["spectrum"] = 2
        pars["nu_min_keV"], pars["nu_max_keV"] = band
        pars["n_mc"] = 0
        out = hle.compute_hle_lightcurve(**_filter_kwargs(hle.compute_hle_lightcurve, pars))
        t_grid = np.asarray(out[0], float)
        Eb1 = out[6] if len(out) >= 7 else None
        Eb2 = out[7] if len(out) >= 8 else None
        Eb1 = None if Eb1 is None else np.asarray(Eb1, float)
        Eb2 = None if Eb2 is None else np.asarray(Eb2, float)
        return t_grid, Eb1, Eb2

    def residuals(u_free: np.ndarray) -> np.ndarray:
        # Recompose blocks from flattened vector
        u_full = pvec.u0.copy(); u_full[active] = u_free
        blocks = pvec.u_to_blocks(u_full)
        shared = blocks["shared"]; hpars = blocks["hle"]; fpars = blocks["fs"]

        # Apply HLE→FS onset coupling (copy)
        fpars_eff = _apply_fs_onset_coupling(shared, hpars, fpars, conf)
        # Fast grid ovveride (fit pass only)
        use_fast = bool(conf.get("fit", {}).get("fast", False))
        if use_fast:
            hpars = dict(hpars)          # copy to avoid mutating the original dicts
            fpars_eff = dict(fpars_eff)
            nt  = int(conf["fit"].get("fast_n_theta", hpars.get("n_theta", 384)))
            np_ = int(conf["fit"].get("fast_n_phi",   hpars.get("n_phi",   384)))
            nb  = int(conf["fit"].get("fast_n_bins",  hpars.get("n_bins",  150)))
            nn  = int(conf["fit"].get("fast_n_nu",    hpars.get("n_nu",      12)))
            nR_fs = int(conf["fit"].get("fast_n_R_fs", fpars_eff.get("n_R", 240)))
            min_grid = int(conf["fit"].get("fast_min_grid", hpars.get("min_grid_counts", 15)))
            # Extra downscale for --ultra-fast
            if bool(conf["fit"].get("ultra_fast", False)):
                def _shrink(x, f=0.4):
                    return max(4, int(x * f))
                nt, np_, nb, nn = _shrink(nt), _shrink(np_), _shrink(nb), _shrink(nn)
                nR_fs = max(8, _shrink(nR_fs))

            # Apply to both components where relevant
            hpars["n_theta"] = nt;     hpars["n_phi"] = np_;     hpars["n_bins"] = nb
            fpars_eff["n_theta"] = nt; fpars_eff["n_phi"] = np_; fpars_eff["n_bins"] = nb
            # Reduce frequency sampling too
            hpars["n_nu"] = nn;        fpars_eff["n_nu"] = nn

            # FS-only: override radial resolution (n_R is only used by the FS model) 
            if "n_R" in fpars_eff:                                                      
                fpars_eff["n_R"] = int(max(8, nR_fs))
            # Lower the MC fallback threshold for HLE (FS does not use it)
            if "min_grid_counts" in hpars:
                hpars["min_grid_counts"] = min_grid

        # (1) FS kinetic energy is *forced* from prompt Egamma & eta_gamma
        fpars_eff["Eiso_core"] = float(_derive_fs_Eiso_from_prompt(
            float(shared["Egamma_iso_core"]), float(shared["eta_gamma"])
        ))

        # Fit-mode switches
        only_hle = bool(conf["fit"].get("only_hle", False))
        only_fs  = bool(conf["fit"].get("only_fs",  False))

        # (2) Calibrate HLE amplitude i_p_prime from Egamma in the XRT band
        # IMPORTANT: skip if we are doing FS-only, otherwise we waste a full HLE evaluation
        # (and it is unnecessary for the likelihood in FS-only mode).
        if not only_fs:
            ip = _calibrate_ip_from_Egamma(shared, hpars, band)
            hpars["i_p_prime"] = float(ip)

        # Time shifts
        # Use separate reference times for HLE and FS (both defined in OBSERVER frame).
        # The model internal times are computed as t' = max(t_obs - t0_component, 1e-6) when apply_t0_to_component=True.
        t0_hle = float(shared.get("t0_obs", 0.0))
        t0_fs  = float(shared.get("t0_fs_obs", t0_hle))
        t_x_hle = _shift_times(xrt["t"], t0_hle, apply_t0_hle)
        t_x_fs  = _shift_times(xrt["t"], t0_fs,  apply_t0_fs)
        if uvw_block["t"].size:
            t_w_hle = _shift_times(uvw_block["t"], t0_hle, apply_t0_hle)
            t_w_fs  = _shift_times(uvw_block["t"], t0_fs,  apply_t0_fs)

        tx_h = Fx_h = tx_f = Fx_f = None

        # XRT model components
        if only_fs:
            # FS only
            tx_f, Fx_f, _ = fs_band_flux_curve(shared, fpars_eff, band)
        elif only_hle:
            # HLE only
            tx_h, Fx_h, _ = hle_band_flux_curve(_shared_to_hle_kwargs(shared), hpars, band)
        else:
            # Both in parallel
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_h = ex.submit(hle_band_flux_curve, _shared_to_hle_kwargs(shared), hpars, band)
                fut_f = ex.submit(fs_band_flux_curve,  shared,                    fpars_eff, band)
                tx_h, Fx_h, _ = fut_h.result()
                tx_f, Fx_f, _ = fut_f.result()

        # XRT model computation
        Fx = 0.0
        if (tx_h is not None) and (Fx_h is not None):
            Fx = Fx + _log_interp(tx_h, Fx_h, t_x_hle)
        if (tx_f is not None) and (Fx_f is not None):
            Fx = Fx + _log_interp(tx_f, Fx_f, t_x_fs)

        # XRT residuals (opt: logspace)
        yx, yxerr, logspace, extra_frac = xrt["y"], xrt["yerr"], xrt["logspace"], xrt["extra_frac"]
        sigx = np.sqrt(yxerr**2 + (extra_frac * yx)**2)
        if logspace:
            m = (yx > 0) & (Fx > 0) & (sigx > 0)
            r_x = np.zeros_like(yx)
            r_x[m] = (np.log10(Fx[m]) - np.log10(yx[m])) / (sigx[m] / (yx[m] * np.log(10.0)))
        else:
            m = np.isfinite(Fx) & np.isfinite(yx) & (sigx > 0)
            r_x = np.zeros_like(yx); r_x[m] = (Fx[m] - yx[m]) / sigx[m]
        r_x *= float(W.get("xrt", 1.0))

        

        # Optical residuals (UVOT WHITE or Ronchini Rc)
        if (uvw_block["t"].size) and (float(W.get("uvot_white", 1.0)) > 0.0):
            kind = str(uvw_block.get("kind", "uvot")).lower()
        
            # observation times for the two components (t0 already applied above)
            y_w, y_werr = uvw_block["y"], uvw_block["yerr"]
            extra_w = float(uvw_block.get("extra_frac", 0.0))
            sigw = np.sqrt(y_werr**2 + (extra_w * y_w)**2)
        
            if kind == "uvot":
                mode = str(conf.get("uvot", {}).get("mode", "flux")).lower()
        
                if mode == "flux":
                    # 'flux' mode: compare dereddened data vs intrinsic flux model (convolved with UVOT response)
                    model_uvot = uvw.band_flux_from_components(
                        t_eval_hle=t_w_hle,
                        shared=shared, hle_pars=hpars, fs_pars=fpars_eff,
                        t_eval_fs=t_w_fs,
                        apply_extinction=False,                           # data are dereddened for Galactic
                        host_ext=conf.get("dust", {}).get("host", None)   # optional host attenuation on model
                    )
                else:
                    # 'counts' mode: compare observed count rate vs folded (extinction-included) model
                    model_uvot = uvw.counts_from_components(
                        t_eval_hle=t_w_hle,
                        shared=shared, hle_pars=hpars, fs_pars=fpars_eff,
                        t_eval_fs=t_w_fs,
                        host_ext=conf.get("dust", {}).get("host", None)
                    )
        
            elif kind == "ronchini_rc":
                # Rc dataset: compare directly to intrinsic monochromatic F_nu(t, nu_Rc)
                nu_hz = float(uvw_block.get("nu_hz", np.nan))

                if only_fs and not only_hle:
                    # FS-only shortcut: do NOT compute HLE at all
                    # (monochromatic helper would otherwise compute both HLE+FS).
                    _, Fw_fs, _ = _monochromatic_Fnu_from_components(
                        t_eval_hle=np.zeros(0, float),   # no HLE eval
                        t_eval_fs=t_w_fs,
                        nu_hz=nu_hz,
                        shared=shared,
                        hle_pars=hpars,
                        fs_pars=fpars_eff,
                        return_components=True
                    )
                    model_uvot = Fw_fs

                elif only_hle and not only_fs:
                    # HLE-only
                    Fw_hle, _, _ = _monochromatic_Fnu_from_components(
                        t_eval_hle=t_w_hle,
                        t_eval_fs=np.zeros(0, float),   # no FS eval
                        nu_hz=nu_hz,
                        shared=shared,
                        hle_pars=hpars,
                        fs_pars=fpars_eff,
                        return_components=True
                    )
                    model_uvot = Fw_hle

                else:
                    # Both components
                    Fw_hle, Fw_fs, Fw_tot = _monochromatic_Fnu_from_components(
                        t_eval_hle=t_w_hle,
                        t_eval_fs=t_w_fs,
                        nu_hz=nu_hz,
                        shared=shared,
                        hle_pars=hpars,
                        fs_pars=fpars_eff,
                        return_components=True
                    )
                    model_uvot = Fw_tot

        
            else:
                raise RuntimeError(f"[OPTICAL] Unknown optical dataset kind: {kind}")
        
            if bool(uvw_block.get("logspace", False)):
                # Log-space residuals with proper weighting
                m = (y_w > 0) & (model_uvot > 0) & (sigw > 0)
                r_w = np.zeros_like(y_w, dtype=float)
                r_w[m] = (np.log10(model_uvot[m]) - np.log10(y_w[m])) / (sigw[m] / (y_w[m] * np.log(10.0)))
            else:
                # Linear-space residuals
                r_w = (model_uvot - y_w) / sigw
        
            # Apply block weight in both branches
            r_w *= float(W.get("uvot_white", 1.0))
        
        else:
            r_w = np.zeros(0, float)



        # Soft targets on observed breaks (HLE only)
        r_t = []
        if (float(W.get("targets", 1.0)) > 0.0) and (targets_b1 or targets_b2):
            t_model, Eb1, Eb2 = _hle_breaks_fast(shared, hpars)
            def _interp_E(tt, arr):
                if arr is None: return np.nan
                val = _log_interp(np.asarray(t_model), np.asarray(arr), np.array([max(tt,1e-12)]))[0]
                return val if np.isfinite(val) and (val > 0) else np.nan
            for (tt, EE, frac) in targets_b1:
                Em = _interp_E(tt, Eb1); 
                if np.isfinite(Em): r_t.append( (np.log10(Em) - np.log10(EE)) / max(np.log10(1.0+frac), 1e-6) )
            for (tt, EE, frac) in targets_b2:
                Em = _interp_E(tt, Eb2);
                if np.isfinite(Em): r_t.append( (np.log10(Em) - np.log10(EE)) / max(np.log10(1.0+frac), 1e-6) )
        r_t = np.asarray(r_t, float) * float(W.get("targets", 1.0))

        return np.concatenate([r_x, r_w, r_t])

    return residuals

# =================
# Plotting outputs
# =================
def make_figures(outdir: str, conf: dict,
                 xrt: dict, uvw_block: dict,
                 u_best: np.ndarray, pvec: ParamVector, uvw: Optional[UVOTWhite],
                 gamma_data_path: str | None = None) -> None:

    blocks = pvec.u_to_blocks(u_best)
    shared = blocks["shared"]; hpars = blocks["hle"]; fpars = blocks["fs"]
    fpars_eff = _apply_fs_onset_coupling(shared, hpars, fpars, conf)
    # Deterministic normalization coupling (same as in residuals)
    Eiso_user = fpars_eff.get("Eiso_core", None)
    fpars_eff["Eiso_core"] = float(_derive_fs_Eiso_from_prompt(
        float(shared["Egamma_iso_core"]), float(shared["eta_gamma"])
    ))

    ip = _calibrate_ip_from_Egamma(shared, hpars, tuple(conf.get("fit", {}).get("xrt_band_keV", [0.3, 10.0])))
    hpars = dict(hpars); hpars["i_p_prime"] = float(ip)
    band = tuple(map(float, conf.get("fit", {}).get("xrt_band_keV", [0.3, 10.0])))
    apply_t0_hle = bool(conf.get("fit", {}).get("apply_t0_to_hle", True))
    apply_t0_fs  = bool(conf.get("fit", {}).get("apply_t0_to_fs",  True))
    t0_hle = float(shared.get("t0_obs", 0.0))
    t0_fs  = float(shared.get("t0_fs_obs", t0_hle))

    # Which emission component(s) are effectively used in the fit?
    # (These flags also control what we plot for clarity.)
    only_hle = bool(conf.get("fit", {}).get("only_hle", False))
    only_fs  = bool(conf.get("fit", {}).get("only_fs",  False))

    # Dataset inclusion flags:
    #  - weight==0 => dataset excluded from fit, BUT we still plot all its data points in red
    #    and we must NOT show any 'excluded from fit' grey points in that case.
    wb_plot = dict(conf.get('fit', {}).get('weight_blocks', {}))
    include_xrt  = (float(wb_plot.get('xrt', 1.0)) != 0.0)
    include_uvot = (float(wb_plot.get('uvot_white', 1.0)) != 0.0)

    def _lbl_fit(base: str, is_fit: bool) -> str:
        return f"{base} (fit)" if is_fit else base
    if only_hle and only_fs:
        # This should not happen; prefer plotting the total if it does.
        print("[WARN] Both only_hle and only_fs are True; plotting will show the total model.")

    # Plot reference: by default we show t_rel = t_obs - t0_hle
    t0_plot_ref = t0_hle

    # Plotting mode: absolute vs relative time
    plot_rel = bool(conf.get("fit", {}).get("plot_time_relative", False))

    def _maybe_rel(t):
        """Return t_rel = t - t0 if plot_rel=True, else t."""
        tt = np.asarray(t, float)
        return (tt - t0_plot_ref) if plot_rel else tt

    # Build plotting time grids (XRT and optical independently).
    # IMPORTANT:
    #   - x-limits must be data-driven (including points excluded from the fit);
    #   - the model must be *evaluated* on the same time range shown in the figure,
    #     otherwise log–log interpolation will return zeros/NaNs outside the model grid
    #     and the curve will look artificially truncated at early/late times.
    pad_dex = float(conf.get("fit", {}).get("plot_pad_dex", 0.25))  # symmetric padding in log10(t)
    rel_floor = float(conf.get("fit", {}).get("plot_rel_floor_s", 1e-3))  # prevents log-axis issues near 0
    # When plotting absolute time (t_obs since trigger), start the axis close to t_trigger=0
    # instead of snapping to the first available data point.
    abs_floor = float(conf.get("fit", {}).get("plot_abs_floor_s", 1e-1))

    def _bounds_from_times(t_obs: np.ndarray) -> Tuple[float, float, float, float]:
        """Return (xmin_plot, xmax_plot, xmin_obs, xmax_obs).

        - xmin_plot/xmax_plot are in the same convention used on the x-axis (relative if plot_rel=True).
        - xmin_obs/xmax_obs are observer times used to build the model evaluation grid.
        """
        tt = np.asarray(t_obs, float)
        tt = tt[np.isfinite(tt) & (tt > 0.0)]
        if tt.size == 0:
            # Fallback: a reasonable default range
            xmin_plot, xmax_plot = (1.0, 3.0e5) if plot_rel else (max(1.0, t0_plot_ref), max(3.0e5, 10.0*t0_plot_ref))
            xmin_obs = xmin_plot + (t0_plot_ref if plot_rel else 0.0)
            xmax_obs = xmax_plot + (t0_plot_ref if plot_rel else 0.0)
            return float(xmin_plot), float(xmax_plot), float(xmin_obs), float(xmax_obs)

        if plot_rel:
            tr = tt - float(t0_plot_ref)
            tr = tr[np.isfinite(tr) & (tr > 0.0)]
            if tr.size == 0:
                # all points are <= t0 (should not happen in normal usage)
                tr = np.array([rel_floor, 10.0*rel_floor], float)

            lg_min = float(np.log10(np.min(tr)))
            lg_max = float(np.log10(np.max(tr)))
            xmin_plot = 10.0 ** (lg_min - pad_dex)
            xmax_plot = 10.0 ** (lg_max + pad_dex)

            # keep away from 0 on log-x
            xmin_plot = max(xmin_plot, rel_floor)
            xmax_plot = max(xmax_plot, 10.0*xmin_plot)

            xmin_obs = float(t0_plot_ref) + xmin_plot
            xmax_obs = float(t0_plot_ref) + xmax_plot
        else:
            lg_min = float(np.log10(np.min(tt)))
            lg_max = float(np.log10(np.max(tt)))
            xmin_obs = 10.0 ** (lg_min - pad_dex)
            xmax_obs = 10.0 ** (lg_max + pad_dex)
            # Force a near-zero left boundary in absolute-time plots (log-x safe).
            if np.isfinite(abs_floor) and (abs_floor > 0.0):
                xmin_obs = min(xmin_obs, abs_floor)
            xmin_obs = max(xmin_obs, 1e-6)
            xmax_obs = max(xmax_obs, 10.0*xmin_obs)
            xmin_plot, xmax_plot = xmin_obs, xmax_obs

        return float(xmin_plot), float(xmax_plot), float(xmin_obs), float(xmax_obs)

    # XRT plot range (include excluded points when available)
    t_x_src = xrt.get("t_raw", xrt.get("t", np.zeros(0)))
    xrt_xmin_plot, xrt_xmax_plot, xrt_xmin_obs, xrt_xmax_obs = _bounds_from_times(np.asarray(t_x_src, float))
    t_plot_xrt = np.geomspace(max(xrt_xmin_obs, 1e-6), xrt_xmax_obs, 420)

    # Optical plot range (if present; used later for Ronchini/UVOT figures)
    t_w_src = uvw_block.get("t_raw", uvw_block.get("t", np.zeros(0)))
    if np.asarray(t_w_src).size:
        opt_xmin_plot, opt_xmax_plot, opt_xmin_obs, opt_xmax_obs = _bounds_from_times(np.asarray(t_w_src, float))
        t_plot_opt = np.geomspace(max(opt_xmin_obs, 1e-6), opt_xmax_obs, 420)
    else:
        opt_xmin_plot, opt_xmax_plot, opt_xmin_obs, opt_xmax_obs = xrt_xmin_plot, xrt_xmax_plot, xrt_xmin_obs, xrt_xmax_obs
        t_plot_opt = t_plot_xrt

    # For downstream code that expects a variable called t_plot, keep it as the XRT grid by default.
    t_plot = t_plot_xrt
    # XRT components & totals
    # Ensure the underlying model grids cover the plotting range (if supported by the backend).
    # NOTE: hle/fs models work in t_rel (after optional t0 subtraction), so we pass shifted bounds.
    t_plot_h = _shift_times(t_plot, t0_hle, apply_t0_hle)
    t_plot_f = _shift_times(t_plot, t0_fs,  apply_t0_fs)
    hpars_plot = dict(hpars)
    fpars_plot = dict(fpars_eff)
    for k in ("tmin","tmax","t_min","t_max","tmin_obs","tmax_obs","t_min_obs","t_max_obs"):
        hpars_plot[k] = float(t_plot_h.min()) if ("min" in k) else float(t_plot_h.max())
    for k in ("tmin","tmax","t_min","t_max","tmin_obs","tmax_obs","t_min_obs","t_max_obs"):
        fpars_plot[k] = float(t_plot_f.min()) if ("min" in k) else float(t_plot_f.max())
    
    tx_h, Fx_h, meta_h = hle_band_flux_curve(_shared_to_hle_kwargs(shared), hpars_plot, band)
    tx_f, Fx_f, meta_f = fs_band_flux_curve(shared, fpars_plot, band)

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # Photon index Γ(t) from the MODEL spectrum near 1 keV
    # (works in all modes: HLE-only, FS-only, or HLE+FS)
    # ---------------------------------------------------------
    t_gamma_plot = None
    Gamma_plot   = None

    have_h = (not only_fs) and ("F_tnu" in meta_h) and ("nu_grid" in meta_h)
    have_f = (not only_hle) and ("F_tnu" in meta_f) and ("nu_grid" in meta_f)

    if have_h or have_f:

        # Component spectral grids
        if have_h:
            nu_h = np.asarray(meta_h["nu_grid"], float)
            Fh   = np.asarray(meta_h["F_tnu"], float)   # shape (Th, Nh)
            th_obs = np.asarray(tx_h, float) + (t0_hle if apply_t0_hle else 0.0)

        if have_f:
            nu_f = np.asarray(meta_f["nu_grid"], float)
            Ff   = np.asarray(meta_f["F_tnu"], float)   # shape (Tf, Nf)
            tf_obs = np.asarray(tx_f, float) + (t0_fs  if apply_t0_fs  else 0.0)

        # Choose plotting times for Γ(t): use the same x-grid used in the LC plot (observer times here)
        t_eval_obs = np.asarray(t_plot, float)
        t_eval_obs = t_eval_obs[np.isfinite(t_eval_obs) & (t_eval_obs > 0.0)]

        # Pivot at 1 keV (Hz)
        nu_piv = (1.0 * 1e3) * getattr(hle, "E_HZ_PER_eV", E_HZ_PER_eV)

        # Use a reference ν-grid (prefer HLE when available, else FS)
        nu_ref = nu_h if have_h else nu_f

        j = int(np.argmin(np.abs(nu_ref - nu_piv)))
        halfwin = 2
        idx_ref = np.arange(max(0, j - halfwin), min(nu_ref.size, j + halfwin + 1))
        nu_loc  = nu_ref[idx_ref]

        # Precompute component local-window spectra on nu_loc
        if have_h:
            if np.allclose(nu_h, nu_ref):
                Fh_loc_full = Fh[:, idx_ref]  # (Th, M)
            else:
                # interpolate HLE spectra to nu_loc for each time step
                Fh_loc_full = np.zeros((Fh.shape[0], nu_loc.size), float)
                for it in range(Fh.shape[0]):
                    Fh_loc_full[it, :] = np.interp(nu_loc, nu_h, Fh[it, :], left=0.0, right=0.0)

        if have_f:
            if np.allclose(nu_f, nu_ref):
                Ff_loc_full = Ff[:, idx_ref]  # (Tf, M)
            else:
                # interpolate FS spectra to nu_loc for each time step
                Ff_loc_full = np.zeros((Ff.shape[0], nu_loc.size), float)
                for it in range(Ff.shape[0]):
                    Ff_loc_full[it, :] = np.interp(nu_loc, nu_f, Ff[it, :], left=0.0, right=0.0)

        # Helper: log-time interpolation for matrix Y(t,nu) -> Y(t_eval,nu)
        def _logtime_interp_matrix(tg, Y, te):
            '''Log-time interpolation of spectra Y(t,nu) onto te.

            We interpolate in (log t, log Y) with PCHIP inside the domain.

            Key implementation detail (to avoid artificial steps in Γ(t)):
              - Some model grids may contain exact zeros / underflows at late times for some nu-bins.
                If we drop those points, each nu-bin can end up with a different effective time-domain,
                which in turn produces discontinuities when we reconstruct the local spectral slope.
              - Therefore, we keep a common time-domain (finite tg) and apply a tiny positive
                floor to Y before taking logs, per nu-bin, based on that bin's smallest positive value.

            Boundary behavior:
              - For te earlier than the first model time: return 0 (no left extrapolation).
              - For te later than the last model time: controlled log–log linear extrapolation in time
                (power-law continuation), analogous to _log_interp() used for light curves.
            '''
            tg = np.asarray(tg, float)
            te = np.asarray(te, float)
            Y  = np.asarray(Y,  float)

            out = np.zeros((te.size, Y.shape[1]), float)

            # Keep only finite, positive times (time grid is common to all nu-bins)
            mt = np.isfinite(tg) & (tg > 0.0)
            tg = tg[mt]
            Y  = Y[mt, :]

            if tg.size < 3:
                return out

            ltg = np.log(tg)
            lte = np.log(np.maximum(te, 1e-12))

            # Identify in-domain / left / right regions
            in_dom  = (te >= tg[0]) & (te <= tg[-1])
            left_of = (te <  tg[0])
            right_of= (te >  tg[-1])

            # In-domain: PCHIP in log-space
            if np.any(in_dom):
                for jnu in range(Y.shape[1]):
                    y = Y[:, jnu]
                    # floor based on smallest positive value in this nu-bin
                    pos = y[(y > 0) & np.isfinite(y)]
                    if pos.size == 0:
                        continue
                    y_floor = max(np.min(pos) * 1e-12, 1e-300)
                    ly = np.log(np.maximum(y, y_floor))
                    interp = PchipInterpolator(ltg, ly, extrapolate=False)
                    out[in_dom, jnu] = np.exp(interp(lte[in_dom]))

            # Left of domain: keep at 0 (no extrapolation)
            # Right of domain: log–log linear extrapolation using last two points in time
            if np.any(right_of):
                if tg.size >= 2:
                    t1, t2 = tg[-2], tg[-1]
                    lt1, lt2 = np.log(t1), np.log(t2)
                    dt = (lt2 - lt1) if (lt2 != lt1) else 1.0
                    for jnu in range(Y.shape[1]):
                        y = Y[:, jnu]
                        pos = y[(y > 0) & np.isfinite(y)]
                        if pos.size == 0:
                            continue
                        y_floor = max(np.min(pos) * 1e-12, 1e-300)
                        y1 = max(y[-2], y_floor)
                        y2 = max(y[-1], y_floor)
                        ly1, ly2 = np.log(y1), np.log(y2)
                        slope = (ly2 - ly1) / dt
                        out[right_of, jnu] = np.exp(ly2 + slope * (lte[right_of] - lt2))

            return out

        # Interpolate component spectra onto the evaluation times and sum
        Ftot_loc = np.zeros((t_eval_obs.size, nu_loc.size), float)
        if have_h:
            Ftot_loc += _logtime_interp_matrix(th_obs, Fh_loc_full, t_eval_obs)
        if have_f:
            Ftot_loc += _logtime_interp_matrix(tf_obs, Ff_loc_full, t_eval_obs)

        # Compute Γ(t) from ln N_E = ln Fν - ln ν + const
        Gamma = np.full(t_eval_obs.size, np.nan)
        for i in range(t_eval_obs.size):
            y = Ftot_loc[i, :]
            good = np.isfinite(y) & (y > 0) & np.isfinite(nu_loc) & (nu_loc > 0)
            if np.count_nonzero(good) < 2:
                continue
            x = np.log(nu_loc[good])
            z = np.log(y[good]) - np.log(nu_loc[good])
            sN = np.polyfit(x, z, 1)[0] if x.size >= 3 else (z[-1] - z[0]) / (x[-1] - x[0])
            Gamma[i] = -sN

        # Convert to plot x-axis convention (relative vs absolute)
        t_gamma_plot = _maybe_rel(t_eval_obs)
        Gamma_plot   = Gamma

        # Suppress the model Γ(t) before the relevant emission start time to avoid misleading pre-emission values.
        # Rule:
        #   - If HLE is being computed (all modes except --only-fs), start Γ(t) at the HLE onset (t0_hle).
        #   - If we are in --only-fs mode, start Γ(t) at the FS onset (t0_fs).
        # We apply this masking in observer time (t_eval_obs), regardless of the x-axis convention.
        t0_gamma_start = None
        if have_h and apply_t0_hle:
            t0_gamma_start = float(t0_hle)
        elif only_fs and have_f and apply_t0_fs:
            t0_gamma_start = float(t0_fs)

        if t0_gamma_start is not None:
            Gamma_plot = np.where(t_eval_obs < t0_gamma_start, np.nan, Gamma_plot)
    Fx_h_plot = _log_interp(tx_h, Fx_h, _shift_times(t_plot, t0_hle, apply_t0_hle))
    Fx_f_plot = _log_interp(tx_f, Fx_f, _shift_times(t_plot, t0_fs, apply_t0_fs))

    # In absolute-time plots, times earlier than the fitted t0_* correspond to pre-emission.
    # The internal shift-time helper clips to a positive floor to keep log grids stable; here
    # we explicitly zero the flux before t0 to avoid unphysical pre-trigger emission in plots.
    if not plot_rel:
        if apply_t0_hle:
            Fx_h_plot = np.where(np.asarray(t_plot, float) < float(t0_hle), 0.0, Fx_h_plot)
        if apply_t0_fs:
            Fx_f_plot = np.where(np.asarray(t_plot, float) < float(t0_fs), 0.0, Fx_f_plot)

    Fx_tot    = Fx_h_plot + Fx_f_plot

    _mkdir(outdir)

    # ---------------------------------------------------------
    # Diagnostic break-frequency plots
    #   - FS synchrotron breaks (nu_m, nu_c) vs time + Ronchini effective ν
    #     Produced only when optical dataset participates in the fit (w_uvot != 0).
    #   - HLE observed breaks (E_b1, E_b2) vs time + XRT band edges
    #     Produced only when XRT participates in the fit (w_xrt != 0).
    # ---------------------------------------------------------
    try:
        # ---- FS (synchrotron) breaks: plot in frequency [Hz] ----
        if include_uvot and (not only_hle):
            t_fs_obs = np.asarray(tx_f, float) + (t0_fs if apply_t0_fs else 0.0)
            Em_keV = np.asarray(meta_f.get("Em_keV", np.full_like(t_fs_obs, np.nan)), float)
            Ec_keV = np.asarray(meta_f.get("Ec_keV", np.full_like(t_fs_obs, np.nan)), float)
            # Convert keV -> Hz
            nu_m = Em_keV * 1e3 * E_HZ_PER_eV
            nu_c = Ec_keV * 1e3 * E_HZ_PER_eV

            # Ronchini effective frequency [Hz]
            nu_rc = 4.550e14

            plt.figure(figsize=(7.2, 4.0))
            m_m = np.isfinite(t_fs_obs) & np.isfinite(nu_m) & (t_fs_obs > 0.0) & (nu_m > 0.0)
            m_c = np.isfinite(t_fs_obs) & np.isfinite(nu_c) & (t_fs_obs > 0.0) & (nu_c > 0.0)
            if np.any(m_m):
                plt.loglog(_maybe_rel(t_fs_obs[m_m]), nu_m[m_m], lw=1.8, color="tab:blue", label=r"$\nu_m$ (FS)")
            if np.any(m_c):
                plt.loglog(_maybe_rel(t_fs_obs[m_c]), nu_c[m_c], lw=1.8, color="tab:orange", label=r"$\nu_c$ (FS)")
            plt.axhline(nu_rc, lw=1.2, ls=":", color="tab:red", alpha=0.9, label=r"Ronchini $\nu_{\rm eff}$")
            plt.xlabel(r"$t_{\rm rel}$ [s]" if plot_rel else r"$t_{\rm obs}$ [s]")
            plt.ylabel(r"Frequency [Hz]")
            plt.grid(True, which="both", ls=":", alpha=0.35)
            plt.legend(loc="best")
            plt.xlim(opt_xmin_plot, opt_xmax_plot)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "fs_breaks.png"), dpi=170)
            plt.close()

        # ---- HLE breaks: plot in energy [keV] ----
        if include_xrt and (not only_fs):
            t_h_obs = np.asarray(tx_h, float) + (t0_hle if apply_t0_hle else 0.0)
            Eb1 = np.asarray(meta_h.get("Eb1_obs", np.full_like(t_h_obs, np.nan)), float)
            Eb2 = np.asarray(meta_h.get("Eb2_obs", np.full_like(t_h_obs, np.nan)), float)

            plt.figure(figsize=(7.2, 4.0))
            m1 = np.isfinite(t_h_obs) & np.isfinite(Eb1) & (t_h_obs > 0.0) & (Eb1 > 0.0)
            m2 = np.isfinite(t_h_obs) & np.isfinite(Eb2) & (t_h_obs > 0.0) & (Eb2 > 0.0)
            if np.any(m1):
                plt.loglog(_maybe_rel(t_h_obs[m1]), Eb1[m1], lw=1.8, color="tab:blue", label=r"$E_{b,1}$ (HLE)")
            if np.any(m2):
                plt.loglog(_maybe_rel(t_h_obs[m2]), Eb2[m2], lw=1.8, color="tab:orange", label=r"$E_{b,2}$ (HLE)")

            # XRT band edges [keV]
            xrt_lo, xrt_hi = 0.3, 10.0
            plt.axhline(xrt_lo, lw=1.2, ls=":", color="0.25", alpha=0.9, label="XRT 0.3 keV")
            plt.axhline(xrt_hi, lw=1.2, ls="--", color="0.25", alpha=0.9, label="XRT 10 keV")

            plt.xlabel(r"$t_{\rm rel}$ [s]" if plot_rel else r"$t_{\rm obs}$ [s]")
            plt.ylabel(r"Energy [keV]")
            plt.grid(True, which="both", ls=":", alpha=0.35)
            plt.legend(loc="best")
            plt.xlim(x_xmin_plot, x_xmax_plot)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "breaks_observed.png"), dpi=170)
            plt.close()
    except Exception as e:
        print(f"[WARN] Could not create break-frequency diagnostic plots: {e}")

    # Photon index data
    gamma_data = None
    if gamma_data_path:
        try:
            gamma_data = load_spectral_evolution_txt(gamma_data_path)
            print(f"[INFO] Loaded gamma-data points: {len(gamma_data['t'])} from {gamma_data_path}")
        except Exception as e:
            print(f"[WARN] Could not load gamma-data file '{gamma_data_path}': {e}")
            gamma_data = None

    # XRT figure (LC + photon index)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08}
    )

    # --- Top: light curve ---
    if only_fs and (not only_hle):
        ax1.loglog(_maybe_rel(t_plot), Fx_f_plot,
                   color=COL_FS, ls=LS_FS, lw=1.8,
                   label=_lbl_fit("FS", include_xrt))
    elif only_hle and (not only_fs):
        ax1.loglog(_maybe_rel(t_plot), Fx_h_plot,
                   color=COL_HLE, ls=LS_HLE, lw=1.8,
                   label=_lbl_fit("HLE", include_xrt))
    else:
        ax1.loglog(_maybe_rel(t_plot), Fx_tot,
                   color=COL_TOT, ls=LS_TOT, lw=1.8,
                   label="HLE+FS")
        ax1.loglog(_maybe_rel(t_plot), Fx_h_plot,
                   color=COL_HLE, ls=LS_HLE, lw=1.6, alpha=0.85,
                   label="HLE")
        ax1.loglog(_maybe_rel(t_plot), Fx_f_plot,
                   color=COL_FS, ls=LS_FS, lw=1.6, alpha=0.85,
                   label="FS")

    # Data points:
    #   - if include_xrt=True, we plot only the points used in the fit in red and any
    #     masked/excluded points in grey (labelled "Excluded from fit");
    #   - if include_xrt=False (weight==0), we plot *all* points in red and do NOT show
    #     any grey points/labels.
    if (not include_xrt) and all(k in xrt for k in ("t_raw", "y_raw", "yerr_raw")):
        ax1.errorbar(_maybe_rel(xrt["t_raw"]), xrt["y_raw"], xrt["yerr_raw"],
                     fmt="o", ms=3, alpha=0.85,
                     mfc=COL_DATA, mec=COL_DATA, ecolor=COL_DATA, color=COL_DATA,
                     label="XRT data")
    else:
        ax1.errorbar(_maybe_rel(xrt["t"]), xrt["y"], xrt["yerr"],
                     fmt="o", ms=3, alpha=0.85,
                     mfc=COL_DATA, mec=COL_DATA, ecolor=COL_DATA, color=COL_DATA,
                     label="XRT data")

    # Y-limits: include ALL plotted XRT points (used + excluded), regardless of fit participation.
    # IMPORTANT: do not force-include points at t_rel<=0 when plot_rel=True (log-x axis).
    if all(k in xrt for k in ("t_raw", "y_raw")):
        t_all = np.asarray(xrt["t_raw"], float)
        y_all = np.asarray(xrt["y_raw"], float)
        t_plot_all = _maybe_rel(t_all)
        mdat = np.isfinite(t_plot_all) & np.isfinite(y_all) & (y_all > 0.0)
        if plot_rel:
            mdat &= (t_plot_all > 0.0)
        if np.any(mdat):
            y_min = float(np.nanmin(y_all[mdat]))
            y_max = float(np.nanmax(y_all[mdat]))
            if np.isfinite(y_min) and np.isfinite(y_max) and (y_min > 0.0) and (y_max > 0.0) and (y_max > y_min):
                log_min = np.floor(np.log10(y_min)) - 0.5
                log_max = np.ceil(np.log10(y_max)) + 0.5
                ax1.set_ylim(10.0**log_min, 10.0**log_max)
    # Also plot excluded points (ONLY if XRT participates in the fit)
    if include_xrt and all(k in xrt for k in ("t_raw","y_raw","yerr_raw","m_use_raw")):
            t_all  = np.asarray(xrt["t_raw"], float)
            y_all  = np.asarray(xrt["y_raw"], float)
            e_all  = np.asarray(xrt["yerr_raw"], float)
            m_use  = np.asarray(xrt["m_use_raw"], bool)
            m_excl = (~m_use) & np.isfinite(t_all) & np.isfinite(y_all) & np.isfinite(e_all) & (y_all > 0)
            if np.any(m_excl):
                ax1.errorbar(
                    _maybe_rel(t_all[m_excl]),
                    y_all[m_excl],
                    e_all[m_excl],
                    fmt="o", ms=3, mfc="none",
                    mec="0.5", ecolor="0.8", color="0.7",
                    alpha=0.65, label="Excluded from fit"
                )

    # Mark CLI fit window (absolute-time version only)
    if not plot_rel:
        if xrt.get("tmin_cli") is not None:
            ax1.axvline(float(xrt["tmin_cli"]), ls="--", lw=1.0, color="0.6")
            ax2.axvline(float(xrt["tmin_cli"]), ls="--", lw=1.0, color="0.6")
        if xrt.get("tmax_cli") is not None:
            ax1.axvline(float(xrt["tmax_cli"]), ls="--", lw=1.0, color="0.6")
            ax2.axvline(float(xrt["tmax_cli"]), ls="--", lw=1.0, color="0.6")

    ax1.set_ylabel("Flux [erg cm$^{-2}$ s$^{-1}$]  (0.3–10 keV)")
    ax1.grid(True, which="both", ls=":", alpha=0.35)
    ax1.legend(loc="best")

    # --- Bottom: photon index ---
    if (t_gamma_plot is not None) and (Gamma_plot is not None):
        tg = np.asarray(t_gamma_plot, float)
        gg = np.asarray(Gamma_plot, float)
        mgg = np.isfinite(tg) & np.isfinite(gg) & (tg > 0)
        if np.any(mgg):
            gamma_label = r"$\Gamma(t)$ at $1\,\mathrm{keV}$"
            if only_hle and (not only_fs):
                gamma_label += " (HLE)"
            else:
                gamma_label += " (HLE+FS)"
            ax2.semilogx(tg[mgg], gg[mgg], lw=2.0,
                         color=(COL_HLE if (only_hle and (not only_fs)) else (COL_FS if (only_fs and (not only_hle)) else COL_TOT)),
                         ls=(LS_HLE if (only_hle and (not only_fs)) else (LS_FS if (only_fs and (not only_hle)) else LS_TOT)),
                         label=gamma_label)

    if gamma_data is not None and len(gamma_data["t"]) > 0:

        # Determine which Γ(t) data points fall inside the XRT fitting window/masks.
        # This is used ONLY for plotting coherence (gamma points are not fitted).
        t_g_obs = np.asarray(gamma_data["t"], dtype=float)
        m_g_valid = np.isfinite(t_g_obs) & np.isfinite(gamma_data["gamma"])

        m_g_fit = np.ones_like(t_g_obs, dtype=bool)
        if include_xrt:
            if xrt.get("tmin_cli") is not None:
                m_g_fit &= (t_g_obs >= float(xrt["tmin_cli"]))
            if xrt.get("tmax_cli") is not None:
                m_g_fit &= (t_g_obs <= float(xrt["tmax_cli"]))
            for (a, b) in (xrt.get("mask_ranges", []) or []):
                m_g_fit &= ~((t_g_obs >= float(a)) & (t_g_obs <= float(b)))

        # Plot times in the same convention used in the LC panel.
        t_g = _maybe_rel(t_g_obs)
        if plot_rel:
            t_g = np.maximum(t_g, 1e-6)  # log-x axis safety

        # time errors (optional). A constant time-shift does not change the error bars.
        xerr = None
        if "t_err_plus" in gamma_data and "t_err_minus" in gamma_data:
            t_err_minus = np.asarray(gamma_data["t_err_minus"], dtype=float)
            t_err_plus  = np.asarray(gamma_data["t_err_plus"], dtype=float)
            if plot_rel:
                # avoid crossing t<=0 on log axis
                t_err_minus = np.minimum(t_err_minus, np.maximum(t_g - 1e-6, 0.0))
            xerr = np.vstack([t_err_minus, t_err_plus])

        yerr = np.vstack([gamma_data["gamma_err_minus"], gamma_data["gamma_err_plus"]])

        if include_xrt:
            m_in  = m_g_valid & m_g_fit
            m_out = m_g_valid & (~m_g_fit)
        else:
            m_in  = m_g_valid
            m_out = np.zeros_like(m_g_valid, dtype=bool)

        # Points used by the fit window (red)
        if np.any(m_in):
            ax2.errorbar(
                t_g[m_in],
                np.asarray(gamma_data["gamma"], float)[m_in],
                xerr=(xerr[:, m_in] if (xerr is not None) else None),
                yerr=yerr[:, m_in],
                fmt="o",
                ms=3.5,
                mfc="none",
                mec=COL_DATA,
                color=COL_DATA,
                ecolor=COL_DATA,
                lw=1.0,
                alpha=0.85,
                label=r"XRT $\Gamma$ (data)",
                zorder=5,
            )

        # Points excluded by the fit window/masks (grey) — only when XRT participates in the fit
        if np.any(m_out):
            ax2.errorbar(
                t_g[m_out],
                np.asarray(gamma_data["gamma"], float)[m_out],
                xerr=(xerr[:, m_out] if (xerr is not None) else None),
                yerr=yerr[:, m_out],
                fmt="o",
                ms=3.5,
                mfc="none",
                mec="0.5",
                color="0.7",
                ecolor="0.8",
                lw=1.0,
                alpha=0.65,
                label="Excluded from fit",
                zorder=4,
            )
    # ---------------------------------------------------------------------
    # IMPORTANT: the shared time axis must be driven ONLY by the flux light
    # curve (XRT LC). The spectral-evolution (Gamma) points can extend to
    # earlier times and would otherwise expand the x-range via autoscaling.
    #
    # We also enforce that the model is evaluated exactly on this same range
    # via t_plot_xrt above, so the HLE curve is not artificially truncated.
    # ---------------------------------------------------------------------
    ax1.set_xlim(xrt_xmin_plot, xrt_xmax_plot)
    ax2.set_xlim(xrt_xmin_plot, xrt_xmax_plot)


    ax2.set_xlabel(r"$t_{\rm rel}=t_{\rm obs}-t_0\ \mathrm{[s]}$" if plot_rel else r"$t_{\rm obs}\ \mathrm{[s]}$")
    ax2.set_ylabel(r"Photon index $\Gamma$")
    ax2.grid(True, which="both", ls=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "combined_xrt.png"), dpi=170)
    plt.close(fig)
    

    # Optical figure (UVOT/WHITE or Ronchini Rc)
    if uvw_block["t"].size:
        kind = str(uvw_block.get("kind", "uvot")).lower()
        t_plot_h = _shift_times(t_plot_opt, t0_hle, apply_t0_hle)
        t_plot_f = _shift_times(t_plot_opt, t0_fs, apply_t0_fs)

        if kind == "ronchini_rc":
            # Ronchini Rc dataset: compare directly to monochromatic intrinsic F_nu(t, nu_Rc)
            nu_hz = float(uvw_block.get("nu_hz", np.nan))

            F_hle, F_fs, F_tot = _monochromatic_Fnu_from_components(
                t_eval_hle=np.asarray(t_plot_h, float),
                t_eval_fs=np.asarray(t_plot_f, float),
                nu_hz=nu_hz,
                shared=shared,
                hle_pars=hpars,
                fs_pars=fpars_eff,
                return_components=True
            )

            plt.figure(figsize=(7.2,4.0))

            F_hle_uJy = np.asarray(F_hle, float) * 1e29
            F_fs_uJy  = np.asarray(F_fs,  float) * 1e29
            F_tot_uJy = np.asarray(F_tot, float) * 1e29

            if only_fs and (not only_hle):
                plt.loglog(_maybe_rel(t_plot_opt), F_fs_uJy,
                           color=COL_FS, ls=LS_FS, lw=1.8,
                           label=_lbl_fit("FS", include_uvot))
            elif only_hle and (not only_fs):
                plt.loglog(_maybe_rel(t_plot_opt), F_hle_uJy,
                           color=COL_HLE, ls=LS_HLE, lw=1.8,
                           label=_lbl_fit("HLE", include_uvot))
            else:
                plt.loglog(_maybe_rel(t_plot_opt), F_tot_uJy,
                           color=COL_TOT, ls=LS_TOT, lw=1.8,
                           label="HLE+FS")
                plt.loglog(_maybe_rel(t_plot_opt), F_hle_uJy,
                           color=COL_HLE, ls=LS_HLE, lw=1.6, alpha=0.85,
                           label="HLE")
                plt.loglog(_maybe_rel(t_plot_opt), F_fs_uJy,
                           color=COL_FS, ls=LS_FS, lw=1.6, alpha=0.85,
                           label="FS")

            ul = np.asarray(uvw_block["upper_limits"], dtype=bool)

            # If the optical dataset participates in the fit (weight>0), we show excluded points in grey.
            # If weight==0, we plot all data points in red (no grey/excluded labels).
            if include_uvot:
                m_use = np.asarray(uvw_block.get("m_use_raw", np.ones_like(ul, dtype=bool)), dtype=bool)
            else:
                m_use = np.ones_like(ul, dtype=bool)

            det = ~ul
            det_used  = det & m_use
            ul_used   = ul  & m_use
            det_excl  = det & (~m_use)
            ul_excl   = ul  & (~m_use)

            # detections
            plt.errorbar(
                _maybe_rel(uvw_block["t_raw"][det_used]),
                (uvw_block["y_raw"][det_used] * 1e29),
                (uvw_block["yerr_raw"][det_used] * 1e29),
                fmt="o", ms=3, alpha=0.85,
                mfc=COL_DATA, mec=COL_DATA, ecolor=COL_DATA, color=COL_DATA,
                label="Rc detections"
            )

            # detections EXCLUDED from fit (grey) — only when include_uvot=True
            if include_uvot and np.any(det_excl):
                plt.errorbar(
                    _maybe_rel(uvw_block["t_raw"][det_excl]),
                    (uvw_block["y_raw"][det_excl] * 1e29),
                    (uvw_block["yerr_raw"][det_excl] * 1e29),
                    fmt="o", ms=3, mfc="none",
                    mec="0.5", ecolor="0.8", color="0.7",
                    alpha=0.65, label="Rc excluded from fit"
                )

            # upper limits
            plt.errorbar(
                _maybe_rel(uvw_block["t_raw"][ul_used]),
                (uvw_block["y_raw"][ul_used] * 1e29),
                yerr=0.3 * (uvw_block["y_raw"][ul_used] * 1e29),
                fmt="v",
                uplims=True,
                alpha=0.85,
                mfc=COL_DATA, mec=COL_DATA, ecolor=COL_DATA, color=COL_DATA,
                label="Rc upper limits"
            )

            # upper limits EXCLUDED from fit (grey) — only when include_uvot=True
            if include_uvot and np.any(ul_excl):
                plt.errorbar(
                    _maybe_rel(uvw_block["t_raw"][ul_excl]),
                    (uvw_block["y_raw"][ul_excl] * 1e29),
                    yerr=0.3 * (uvw_block["y_raw"][ul_excl] * 1e29),
                    fmt="v",
                    uplims=True,
                    mfc="none",
                    mec="0.5", ecolor="0.8", color="0.7",
                    alpha=0.65,
                    label="_nolegend_"
                )

            plt.xlabel(r"$t_{\rm rel}=t_{\rm obs}-t_0\ \mathrm{[s]}$" if plot_rel else r"$t_{\rm obs}\ \mathrm{[s]}$")
            plt.ylabel(r"$F_\nu\ [\mu{\rm Jy}]$  (Rc, Ronchini)")
            plt.grid(True, which="both", ls=":", alpha=0.35)
            plt.legend(loc="best")
            plt.xlim(opt_xmin_plot, opt_xmax_plot)
            plt.tight_layout()
            # Y-limits: include ALL plotted Rc points (detections + upper limits), regardless of fit participation.
            # IMPORTANT: do not force-include points at t_rel<=0 when plot_rel=True (log-x axis).
            if all(k in uvw_block for k in ("t_raw", "y_raw")):
                t_all = np.asarray(uvw_block["t_raw"], float)
                y_all = np.asarray(uvw_block["y_raw"], float) * 1e29  # -> microJy
                t_plot_all = _maybe_rel(t_all)
                mdat = np.isfinite(t_plot_all) & np.isfinite(y_all) & (y_all > 0.0)
                if plot_rel:
                    mdat &= (t_plot_all > 0.0)
                if np.any(mdat):
                    ymin = float(np.nanpercentile(y_all[mdat], 1))
                    ymax = float(np.nanpercentile(y_all[mdat], 99))
                    if np.isfinite(ymin) and np.isfinite(ymax) and (ymin > 0.0) and (ymax > 0.0) and (ymax > ymin):
                        plt.ylim(0.5*ymin, 2.0*ymax)

            plt.savefig(os.path.join(outdir, "combined_rc.png"), dpi=170)
            plt.close()

        else:
            if uvw is None:
                print("[PLOT] UVOT figure requested but UVOTWhite is not available; skipping UVOT plot.")
            else:
                mode = str(conf.get("uvot", {}).get("mode", "flux")).lower()
                t_plot_h = _shift_times(t_plot_opt, t0_hle, apply_t0_hle)
                t_plot_f = _shift_times(t_plot_opt, t0_fs,  apply_t0_fs)

                if mode == "flux":
                    # --- Band-averaged Fnu for each component, then sum ---
                    # Evaluate component spectra on WHITE grid (Fnu on λ-grid mapped to nu_white)
                    nu_h, F_h = uvw._component_Fnu_on_white(np.asarray(t_plot_h, float), "HLE", shared, hpars, fpars_eff)
                    nu_f, F_f = uvw._component_Fnu_on_white(np.asarray(t_plot_f, float),  "FS",  shared, hpars, fpars_eff)
                    # Band-average each component separately using WHITE EA
                    F_h_ba = uvot_util.bandpass_average_model(F_h, uvw.nu_white, uvw.wave_A, uvw.EA)
                    F_f_ba = uvot_util.bandpass_average_model(F_f, uvw.nu_white, uvw.wave_A, uvw.EA)
                    F_tot  = F_h_ba + F_f_ba

                    plt.figure(figsize=(7.2,4.0))
                    if only_fs and (not only_hle):
                        plt.loglog(_maybe_rel(t_plot_opt), F_f_ba, color=COL_FS, ls=LS_FS, lw=1.8, label=_lbl_fit("FS", include_uvot))
                    elif only_hle and (not only_fs):
                        plt.loglog(_maybe_rel(t_plot_opt), F_h_ba, color=COL_HLE, ls=LS_HLE, lw=1.8, label=_lbl_fit("HLE", include_uvot))
                    else:
                        plt.loglog(_maybe_rel(t_plot_opt), F_tot, color=COL_TOT, ls=LS_TOT, lw=1.8, label="HLE+FS")
                        plt.loglog(_maybe_rel(t_plot_opt), F_h_ba, color=COL_HLE, ls=LS_HLE, lw=1.6, alpha=0.85, label="HLE")
                        plt.loglog(_maybe_rel(t_plot_opt), F_f_ba, color=COL_FS, ls=LS_FS, lw=1.6, alpha=0.85, label="FS")
                    plt.errorbar(_maybe_rel(uvw_block["t_raw"] if ((not include_uvot) and ("t_raw" in uvw_block)) else uvw_block["t"]),
                                 (uvw_block["y_raw"] if ((not include_uvot) and ("y_raw" in uvw_block)) else uvw_block["y"]),
                                 (uvw_block["yerr_raw"] if ((not include_uvot) and ("yerr_raw" in uvw_block)) else uvw_block["yerr"]),
                                 fmt="o", ms=3, alpha=0.85, mfc=COL_DATA, mec=COL_DATA, ecolor=COL_DATA, color=COL_DATA,
                                 label="UVOT White")

                    # Y-limits: include ALL plotted UVOT points (used + excluded), regardless of fit participation.
                    # IMPORTANT: do not force-include points at t_rel<=0 when plot_rel=True (log-x axis).
                    ax = plt.gca()
                    if all(k in uvw_block for k in ("t_raw", "y_raw")):
                        t_all = np.asarray(uvw_block["t_raw"], float)
                        y_all = np.asarray(uvw_block["y_raw"], float)
                        t_plot_all = _maybe_rel(t_all)
                        mdat = np.isfinite(t_plot_all) & np.isfinite(y_all) & (y_all > 0.0)
                        if plot_rel:
                            mdat &= (t_plot_all > 0.0)
                        if np.any(mdat):
                            y_min = float(np.nanmin(y_all[mdat]))
                            y_max = float(np.nanmax(y_all[mdat]))
                            if np.isfinite(y_min) and np.isfinite(y_max) and (y_min > 0.0) and (y_max > 0.0) and (y_max > y_min):
                                log_min = np.floor(np.log10(y_min)) - 0.5
                                log_max = np.ceil(np.log10(y_max)) + 0.5
                                ax.set_ylim(10.0**log_min, 10.0**log_max)

                    # Also plot excluded UVOT points (RAW selection not used in the fit)
                    # NOTE: only when include_uvot=True (weight>0). When weight==0 we plot all points in red and do not show grey points.
                    if include_uvot and all(k in uvw_block for k in ("t_raw", "y_raw", "yerr_raw", "m_use_raw")):
                        t_all  = np.asarray(uvw_block["t_raw"],   float)
                        y_all  = np.asarray(uvw_block["y_raw"],   float)
                        e_all  = np.asarray(uvw_block["yerr_raw"],float)
                        m_use  = np.asarray(uvw_block["m_use_raw"], bool)

                        m_excl = (~m_use) & np.isfinite(t_all) & np.isfinite(y_all) & np.isfinite(e_all) & (y_all > 0)
                        if np.any(m_excl):
                            plt.errorbar(
                                _maybe_rel(t_all[m_excl]),
                                y_all[m_excl],
                                e_all[m_excl],
                                fmt="o", ms=3, mfc="none",
                                mec="0.5", ecolor="0.8", color="0.7",
                                alpha=0.65, label="Excluded (UVOT)"
                            )

                        # Mark UVOT CLI window
                        if uvw_block.get("tmin_cli") is not None:
                            ax.axvline(float(uvw_block["tmin_cli"]), ls="--", lw=1.0, color="0.6")
                        if uvw_block.get("tmax_cli") is not None:
                            ax.axvline(float(uvw_block["tmax_cli"]), ls="--", lw=1.0, color="0.6")
                    
                    plt.xlabel(r"$t_{\rm rel}=t_{\rm obs}-t_0\ \mathrm{[s]}$" if plot_rel else r"$t_{\rm obs}\ \mathrm{[s]}$")
                    plt.ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]  (UVOT White)")
                    plt.grid(True, which="both", ls=":", alpha=0.35); plt.legend()
                    plt.xlim(opt_xmin_plot, opt_xmax_plot)
                    plt.tight_layout(); plt.savefig(os.path.join(outdir, "combined_uvot_white.png"), dpi=170); plt.close()

                else:
                    # --- Counts/s for each component, then sum (mirror of counts_from_components) ---
                    nu_h, F_h = uvw._component_Fnu_on_white(np.asarray(t_plot_h, float), "HLE", shared, hpars, fpars_eff)
                    nu_f, F_f = uvw._component_Fnu_on_white(np.asarray(t_plot_f, float),  "FS",  shared, hpars, fpars_eff)

                    lam = uvw.wave_A[None, :]   # [Å]
                    EA  = uvw.EA[None, :]       # [cm^2]
                    # Convert Fnu -> Flambda
                    Fh_lambda = F_h * (C_A_HZ / (lam**2))
                    Ff_lambda = F_f * (C_A_HZ / (lam**2))
                    # Apply Galactic extinction if present (counts-mode compares to *observed* counts)
                    if uvw.ext_trans is not None:
                        Fh_lambda = Fh_lambda * (uvw.ext_trans[None, :])
                        Ff_lambda = Ff_lambda * (uvw.ext_trans[None, :])
                    # Photon conversion kernel and integration
                    HC_ERG_A = H_ERG_S * C_A_HZ
                    photon_kernel = (lam / HC_ERG_A)
                    C_h = np.trapz(Fh_lambda * EA * photon_kernel, uvw.wave_A, axis=1)
                    C_f = np.trapz(Ff_lambda * EA * photon_kernel, uvw.wave_A, axis=1)
                    C_tot = C_h + C_f

                    plt.figure(figsize=(7.2,4.0))
                    if only_fs and (not only_hle):
                        plt.loglog(_maybe_rel(t_plot_opt), C_f, color=COL_FS, ls=LS_FS, lw=1.8, label=_lbl_fit("FS", include_uvot) + " folded")
                    elif only_hle and (not only_fs):
                        plt.loglog(_maybe_rel(t_plot_opt), C_h, color=COL_HLE, ls=LS_HLE, lw=1.8, label=_lbl_fit("HLE", include_uvot) + " folded")
                    else:
                        plt.loglog(_maybe_rel(t_plot_opt), C_tot, color=COL_TOT, ls=LS_TOT, lw=1.8, label="Model (HLE+FS) folded")
                        plt.loglog(_maybe_rel(t_plot_opt), C_h, color=COL_HLE, ls=LS_HLE, lw=1.6, alpha=0.85, label="HLE")
                        plt.loglog(_maybe_rel(t_plot_opt), C_f, color=COL_FS, ls=LS_FS, lw=1.6, alpha=0.85, label="FS")
                    plt.errorbar(_maybe_rel(uvw_block["t_raw"] if ((not include_uvot) and ("t_raw" in uvw_block)) else uvw_block["t"]),
                                 (uvw_block["y_raw"] if ((not include_uvot) and ("y_raw" in uvw_block)) else uvw_block["y"]),
                                 (uvw_block["yerr_raw"] if ((not include_uvot) and ("yerr_raw" in uvw_block)) else uvw_block["yerr"]),
                                 fmt="o", ms=3, alpha=0.85, mfc=COL_DATA, mec=COL_DATA, ecolor=COL_DATA, color=COL_DATA,
                                 label="UVOT White")

                    plt.xlabel(r"$t_{\rm rel}=t_{\rm obs}-t_0\ \mathrm{[s]}$" if plot_rel else r"$t_{\rm obs}\ \mathrm{[s]}$")
                    plt.ylabel("Counts s$^{-1}$  (UVOT White)")
                    plt.grid(True, which="both", ls=":", alpha=0.35); plt.legend()
                    plt.xlim(opt_xmin_plot, opt_xmax_plot)
                    plt.tight_layout(); plt.savefig(os.path.join(outdir, "combined_uvot_white.png"), dpi=170); plt.close()


# ==================
# CLI + main
# ==================
def _parse_time_mask_str(mask_str: str):
    if not mask_str: return []
    out = []
    for chunk in re.split(r"[,\s]+", mask_str.strip()):
        if not chunk: continue
        if ":" in chunk:
            a,b = chunk.split(":",1)
            a = float(a) if a.strip()!="" else -np.inf
            b = float(b) if b.strip()!="" else  np.inf
        else:
            a = b = float(chunk)
        if b < a: a,b = b,a
        out.append((a,b))
    return out

def main():
    ap = argparse.ArgumentParser(description="Joint fitter: HLE (2SBPL) + FS + UVOT White")

    # Required
    ap.add_argument("--params", required=True, help="Path to params_combined.yaml")
    ap.add_argument("--outdir", required=True, help="Output directory")

    # XRT data + columns + QDP options
    ap.add_argument("--xrt", required=True, help="XRT light curve file (CSV/TSV/QDP)")
    ap.add_argument("--xrt-col-t",   required=True, help="Time column name")
    ap.add_argument("--xrt-col-y",   required=True, help="Y column name (flux or rate)")
    ap.add_argument("--xrt-col-err", default=None,  help="Symmetric error column (ignored if ypos/neg provided)")
    ap.add_argument("--xrt-col-ypos", default=None, help="Column for +ve Y error (Fluxpos/Ratepos)")
    ap.add_argument("--xrt-col-yneg", default=None, help="Column for -ve Y error (Fluxneg/Rateneg)")
    ap.add_argument("--xrt-col-tpos", default=None, help="Column for +ve time error (T+ve)")
    ap.add_argument("--xrt-col-tneg", default=None, help="Column for -ve time error (T-ve)")
    ap.add_argument("--xrt-err-sym", choices=["max","mean","pos","neg"], default="max",
                    help="How to symmetrize Y-errors when ypos/yneg are present")
    ap.add_argument("--xrt-qdp-set", type=str, default="ALL", choices=["WT","PC","ALL"],
                    help="For QDP: select dataset (WT=0, PC=1, ALL=both).")

    ap.add_argument("--gamma-data", default=None, help="Path to XRT Burst Analyser 'spectral evolution data' text file ""(e.g. spectral_evolution.txt). Plotted only (NOT used in fit).")

    # XRT band + fit space + extras
    ap.add_argument("--xrt-band", type=float, nargs=2, default=[0.3, 10.0],
                    help="XRT band [keV_min keV_max]")
    ap.add_argument("--xrt-logspace", action="store_true", help="Fit XRT in log10(y)")
    ap.add_argument("--xrt-extra-frac", type=float, default=0.0, help="Extra fractional error for XRT, added in quadrature")
    ap.add_argument("--xrt-tmin", type=float, default=None, help="Ignore XRT data earlier than tmin [s]")
    ap.add_argument("--xrt-tmax", type=float, default=None, help="Ignore XRT data later than tmax [s]")
    ap.add_argument("--xrt-mask", type=str, default="",
                    help='Exclude time ranges from XRT fit. Example: "80:130, 500:650"')

    # UVOT White data (optional)
    ap.add_argument("--uvot-white", default=None, help="UVOT White CSV (columns: t,Y,Yerr)")
    ap.add_argument("--uvot-col-t", default="t", help="UVOT column: time")
    ap.add_argument("--uvot-col-y", default="Y", help="UVOT column: value (counts/s)")
    ap.add_argument("--uvot-col-err", default="Yerr", help="UVOT column: error")
    ap.add_argument("--uvot-extra-frac", type=float, default=0.0, help="Extra fractional error for UVOT")
    ap.add_argument("--uvot-tmin", type=float, default=None, help="Ignore UVOT data earlier than tmin [s]")
    ap.add_argument("--uvot-tmax", type=float, default=None, help="Ignore UVOT data later than tmax [s]")
    ap.add_argument("--uvot-mask", type=str, default="", help="Exclude time ranges for UVOT")
    ap.add_argument("--uvot-effarea-dir", default="/swift_caldb/data/swift/uvota/bcf/effarea",
                    help="Path to UVOT CALDB effective area directory")
    ap.add_argument("--uvot-exclude-preprompt", action="store_true",
                    help="Exclude UVOT points with t_obs < t0_obs (if t0_obs is available).")
    ap.add_argument("--host-ebv", type=float, default=None,
                    help="Enable host-galaxy extinction in UVOT flux-mode by setting E(B-V)_host.")
    ap.add_argument("--host-rv", type=float, default=None,
                    help="R_V of the host extinction law (default 3.1).")
    ap.add_argument("--host-law", type=str, choices=["mw"], default=None,
                    help="Host extinction law. 'mw' = CCM89 + O'Donnell 94.")

    # UVOT FITS pipeline (preferred; uses predict_lightcurve_from_fit.py)
    ap.add_argument("--uvot-fits", default=None,
                    help="Path to the global UVOT catalog FITS (e.g. Catalogue_final_0708_2.fits)")
    ap.add_argument("--uvot-filter", default="WHITE",
                    choices=["WHITE","V","B","U","UVW1","UVM2","UVW2"],
                    help="UVOT filter to select from the FITS catalog")
    ap.add_argument("--grb-key", default=None,
                    help="GRB key used in the UVOT catalog (e.g. '061121'). If omitted, no UVOT FITS import is performed.")
    ap.add_argument("--trigger-met", type=float, default=None,
                    help="Override Swift BAT trigger MET [s]. If omitted, use DEFAULT_TRIGGER_MET in predict_* if available.")
    ap.add_argument("--uvot-mode", default=None, choices=["flux","counts"],
                    help="How to compare UVOT data to the model. 'flux' uses band-averaged dereddened Fν (predict_* style). 'counts' uses folded counts/s. Default: 'flux' if FITS pipeline is used, otherwise 'counts'.")
    ap.add_argument("--uvot-logspace", action="store_true",
                    help="Fit the UVOT block in log10-space (like XRT --xrt-logspace).")


    # Ronchini optical Rc dataset (optional; alternative to UVOT block)
    ap.add_argument("--ronchini-rc", default=None,
                    help="Ronchini optical Rc file (CSV with header: ts,fkmicJy,dfkmicJy). "
                         "If provided, it replaces the entire UVOT block.")
    ap.add_argument("--ronchini-nu", type=float, default=None,
                    help="Override the effective frequency [Hz] for the Ronchini Rc dataset. "
                         "If omitted, a default Rc effective frequency is used.")
    ap.add_argument("--ronchini-extra-frac", type=float, default=0.0,
                    help="Extra fractional error for Ronchini optical block, added in quadrature.")
    ap.add_argument("--ronchini-tmin", type=float, default=None, help="Ignore Ronchini data earlier than tmin [s]")
    ap.add_argument("--ronchini-tmax", type=float, default=None, help="Ignore Ronchini data later than tmax [s]")
    ap.add_argument("--ronchini-mask", type=str, default="", help="Exclude time ranges for Ronchini (same syntax as --uvot-mask)")
    ap.add_argument("--ronchini-logspace", action="store_true",
                    help="Fit the Ronchini optical block in log10-space (like --xrt-logspace).")

    # Targets (HLE observed breaks)
    ap.add_argument("--target-b1", action="append", default=[], help='Observed E_b1 targets "t,E[,frac]" (repeatable)')
    ap.add_argument("--target-b2", action="append", default=[], help='Observed E_b2 targets "t,E[,frac]" (repeatable)')

    # Loss / optimizer
    ap.add_argument("--robust-loss", default="linear", help="Least-squares loss (soft_l1 recommended)")
    ap.add_argument("--f-scale", type=float, default=1.0, help="f_scale for robust loss")
    ap.add_argument("--max-nfev", type=int, default=300, help="Max function evaluations")

    # Weights (per residual block)
    ap.add_argument("--w-xrt", type=float, default=1.0, help="Weight for XRT residual block")
    ap.add_argument("--w-uvot", type=float, default=1.0, help="Weight for UVOT residual block")
    ap.add_argument("--w-targets", type=float, default=1.0, help="Weight for targets residual block")

    # Apply t0 flags (override YAML if provided)
    ap.add_argument("--apply-t0-hle", type=int, choices=[0,1], default=None, help="Override YAML: apply t0 to HLE (1/0)")
    ap.add_argument("--apply-t0-fs",  type=int, choices=[0,1], default=None, help="Override YAML: apply t0 to FS (1/0)")

    # Fast grid override (used during fit only; final plots use YAML/native resolution)
    ap.add_argument("--fast", action="store_true", help="Use reduced grids during fit for speed")
    ap.add_argument("--fast-n-theta", type=int, default=96)
    ap.add_argument("--fast-n-phi",   type=int, default=96)
    ap.add_argument("--fast-n-bins",  type=int, default=80)
    ap.add_argument("--fast-n-nu",    type=int, default=8)
    ap.add_argument("--fast-min-grid",type=int, default=20)
    # Optional: subsample data during optimization
    ap.add_argument("--fast-max-xrt", type=int, default=200, help="Max XRT points used during fit when --fast")
    ap.add_argument("--fast-max-uvot", type=int, default=80, help="Max UVOT points used during fit when --fast")
    # Finite-difference step tuning
    ap.add_argument("--diff-step", type=float, default=1e-2, help="Finite-difference step for jacobian (2-point)")
    # Ultra-fast options
    ap.add_argument("--only-hle", action="store_true",
                    help="Disable FS component during optimization.")
    ap.add_argument("--only-fs", action="store_true",
                    help="Disable HLE component during optimization.")
    ap.add_argument("--ultra-fast", action="store_true",
                    help="Use extremely coarse model grids during optimization.")

    # Plotting options
    ap.add_argument("--plot-time-relative", action="store_true",
                help="Plot figures versus t_rel = t_obs - t0_obs instead of absolute t_obs.")

    ap.add_argument("--plot-abs-floor-s", type=float, default=None,
                    help="In absolute-time plots (no --plot-time-relative), force the left x-axis boundary to this value [s] (log-safe). Overrides fit.plot_abs_floor_s in YAML.")

    # Reporting helpers
    ap.add_argument("--chi2-xrt-only", action="store_true",
                help="Also report a clean XRT-only chi-square (statistical, unweighted) with its own dof/redchi2.")
    args = ap.parse_args()

    conf = load_yaml(args.params)
    # If provided in YAML, prefer UVOT CALDB effarea dir from config
    if isinstance(conf.get("uvot"), dict) and conf["uvot"].get("caldb_effarea_dir"):
        args.uvot_effarea_dir = conf["uvot"]["caldb_effarea_dir"]
    _mkdir(args.outdir)

    # Ensure effarea dir is always available from conf
    conf.setdefault("uvot", {})
    conf["uvot"].setdefault("caldb_effarea_dir", args.uvot_effarea_dir)

    # Inject CLI overrides into config (non-destructive)
    conf.setdefault("fit", {})
    conf["fit"]["xrt_band_keV"] = list(map(float, args.xrt_band))
    conf["fit"]["robust_loss"]  = args.robust_loss
    conf["fit"]["f_scale"]      = float(args.f_scale)
    conf["fit"]["max_nfev"]     = int(args.max_nfev)
    conf["fit"]["weight_blocks"] = {"xrt":float(args.w_xrt), "uvot_white":float(args.w_uvot), "targets":float(args.w_targets)}
    if args.apply_t0_hle is not None:
        conf["fit"]["apply_t0_to_hle"] = bool(args.apply_t0_hle)
    if args.apply_t0_fs  is not None:
        conf["fit"]["apply_t0_to_fs"]  = bool(args.apply_t0_fs)

    # Merge UVOT-related CLI overrides into config (non-destructive)
    conf.setdefault("uvot", {})
    uv = conf["uvot"]
    if args.uvot_fits is not None:
        uv["catalog_fits"] = args.uvot_fits
    if args.grb_key is not None:
        uv["grb_key"] = args.grb_key
    if args.uvot_filter is not None:
        uv["filter"] = args.uvot_filter
    if args.trigger_met is not None:
        uv["trigger_met"] = float(args.trigger_met)
    if args.uvot_mode is not None:
        uv["mode"] = args.uvot_mode
    # Default to 'flux' when a FITS catalog is provided, otherwise keep 'counts'
    uv.setdefault("mode", "flux" if uv.get("catalog_fits") else "counts")
    # Log-space for UVOT residuals (CLI override; default False)
    uv.setdefault("logspace", bool(args.uvot_logspace))

    # Optional coordinates file and dust settings (used by predict_* pipeline)
    uv.setdefault("coords_file", "data/grb_coords.txt")
    conf.setdefault("dust", {})
    conf["dust"].setdefault("rv", 3.1)
    conf["dust"].setdefault("use_galactic_extinction", True)   # true → deredden data in 'flux' mode
    conf["dust"].setdefault("ebv_mode", "map")                 # use SFD maps via predict_* helpers
    # Host extinction sub-block (optional)
    conf["dust"].setdefault("host", {})
    conf["dust"]["host"].setdefault("use_host_extinction", False)
    conf["dust"]["host"].setdefault("ebv_host", 0.0)
    conf["dust"]["host"].setdefault("rv_host", 3.1)
    conf["dust"]["host"].setdefault("law", "mw")

    # CLI overrides: set use_host_extinction=True if --host-ebv is provided
    if args.host_ebv is not None:
        conf["dust"]["host"]["use_host_extinction"] = True
        conf["dust"]["host"]["ebv_host"] = float(args.host_ebv)
    if args.host_rv is not None:
        conf["dust"]["host"]["rv_host"] = float(args.host_rv)
    if args.host_law is not None:
        conf["dust"]["host"]["law"] = str(args.host_law)

    # Store fast grid overrides (to be used during the fit only)
    conf["fit"]["fast"] = bool(args.fast)
    conf["fit"]["fast_n_theta"] = int(args.fast_n_theta)
    conf["fit"]["fast_n_phi"]   = int(args.fast_n_phi)
    conf["fit"]["fast_n_bins"]  = int(args.fast_n_bins)
    conf["fit"]["fast_n_nu"]    = int(args.fast_n_nu)
    conf["fit"]["fast_min_grid"]= int(args.fast_min_grid)
    # Expose CLI toggles to residuals (make_residual_fn sees only 'conf')
    conf["fit"]["only_hle"]   = bool(args.only_hle)
    conf["fit"]["only_fs"]    = bool(args.only_fs)
    conf["fit"]["ultra_fast"] = bool(args.ultra_fast)
    # Plot mode: absolute t_obs vs relative t_rel = t_obs - t0_obs
    conf["fit"]["plot_time_relative"] = bool(args.plot_time_relative)
    if args.plot_abs_floor_s is not None:
        conf["fit"]["plot_abs_floor_s"] = float(args.plot_abs_floor_s)

    def _log_subsample_indices(t, max_points):
        """Return indices that log-subsample t to at most max_points."""
        import numpy as np
        n = len(t)
        if n <= max_points:
            return np.arange(n, dtype=int)
        t = np.asarray(t, float)
        # guard: strictly positive for log binning; fallback to linear if needed
        tp = t[t > 0]
        if tp.size < 3:
            # linear fallback
            return np.linspace(0, n-1, max_points, dtype=int)
        bins = np.geomspace(tp.min(), tp.max(), max_points + 1)
        idx = []
        pos_idx = np.where(t > 0)[0]
        for i in range(max_points):
            lo, hi = bins[i], bins[i+1]
            m = pos_idx[(t[pos_idx] >= lo) & (t[pos_idx] < hi)]
            if m.size:
                idx.append(int(m[m.size // 2]))
        if not idx:
            return np.linspace(0, n-1, max_points, dtype=int)
        return np.unique(np.array(idx, dtype=int))

    # ---- XRT data load + masks
    t_x, y_x, yerr_x, tpos_x, tneg_x = read_table_auto(
        args.xrt, args.xrt_col_t, args.xrt_col_y, args.xrt_col_err,
        col_ypos=args.xrt_col_ypos, col_yneg=args.xrt_col_yneg,
        col_tpos=args.xrt_col_tpos, col_tneg=args.xrt_col_tneg,
        err_sym=args.xrt_err_sym, qdp_set=args.xrt_qdp_set
    )

    # Keep immutable RAW copies for later plotting (and for "excluded" points)
    t_x_raw   = t_x.copy()
    y_x_raw   = y_x.copy()
    yerr_x_raw= yerr_x.copy()

    # Build RAW-time masks (window + user-excluded ranges)
    m_valid = np.isfinite(t_x_raw) & np.isfinite(y_x_raw) & np.isfinite(yerr_x_raw)
    m_window = np.ones_like(m_valid, dtype=bool)
    if args.xrt_tmin is not None:
        m_window &= (t_x_raw >= float(args.xrt_tmin))
    if args.xrt_tmax is not None:
        m_window &= (t_x_raw <= float(args.xrt_tmax))
    xrt_mask_ranges = _parse_time_mask_str(args.xrt_mask)
    m_maskranges = np.zeros_like(m_valid, dtype=bool)
    for (a, b) in xrt_mask_ranges:
        m_maskranges |= (t_x_raw >= a) & (t_x_raw <= b)

    # Points actually used for FIT
    m_use = m_valid & m_window & (~m_maskranges)

    # Subset used by residuals from here on
    t_x, y_x, yerr_x = t_x_raw[m_use], y_x_raw[m_use], yerr_x_raw[m_use]

    # -----------------------------------------------------------------------------
    # Load UVOT data through the full pipeline defined in predict_lightcurve_from_fit.py
    # -----------------------------------------------------------------------------
    import os
    from pathlib import Path

    # Optical block input (UVOT White by default; can be replaced by Ronchini Rc)
    optical_kind = "none"
    optical_nu_hz = None
    # Default Rc effective frequency (Cousins R): lambda_eff ~ 6410 Å → nu ~ 4.68e14 Hz
    DEFAULT_R_C_NU_HZ = 4.55e14

    t_w, y_w, yerr_w = np.zeros(0,float), np.zeros(0,float), np.zeros(0,float)
    is_ul_w_raw = np.zeros(0, dtype=bool)

    if args.ronchini_rc is not None:
        if not os.path.exists(args.ronchini_rc):
            raise FileNotFoundError(f"[RONCHINI] File not found: {args.ronchini_rc}")
        optical_kind = "ronchini_rc"
        optical_nu_hz = float(args.ronchini_nu) if args.ronchini_nu is not None else DEFAULT_R_C_NU_HZ
        print(f"[RONCHINI] Using Ronchini Rc dataset: {args.ronchini_rc}")
        print(f"[RONCHINI] Effective frequency nu = {optical_nu_hz:.3e} Hz")
        t_w, y_w, yerr_w, is_ul_w = load_ronchini_rc_txt(args.ronchini_rc, to_cgs=True)
        is_ul_w_raw = is_ul_w.copy()
        print(f"[RONCHINI] Loaded {t_w.size} valid points (Fnu in cgs)")
    else:
        try:
            uvot_conf = conf.get("uvot", {})
            if uvot_conf.get("catalog_fits") and uvot_conf.get("grb_key"):
                # --- Inputs from YAML
                catalog_path = Path(uvot_conf["catalog_fits"])
                grb_key = uvot_conf["grb_key"]
                filt = uvot_conf.get("filter", "WHITE")
                coords_file = uvot_conf.get("coords_file", "data/grb_coords.txt")
                mode = uvot_conf.get("mode", "flux")  # 'flux' (Fnu) or 'rate'

                optical_kind = "uvot"
                print(f"[UVOT] Using FITS catalog pipeline from predict_lightcurve_from_fit.py")
                print(f"       GRB={grb_key}, filter={filt}, mode={mode}")

                # --- Read RA/DEC + trigger MET from coords_file (our helper already added earlier)
                coords_dict = load_grb_coords_and_triggers(coords_file)
                if grb_key not in coords_dict:
                    raise RuntimeError(f"[UVOT] GRB '{grb_key}' not found in {coords_file}")
                trigger_met = coords_dict[grb_key].get("trigger_met", None)
                if trigger_met is not None:
                    print(f"[UVOT] Using trigger MET = {trigger_met:.3f} s from {coords_file}")

                # --- Compute EBV from SFD maps through predict_* helpers (same as predict_*)
                #     1) get RA/DEC (prefer key 'GRB{key}', else plain key), 2) EBV, 3) WHITE band A_band
                try:
                    ra_deg, dec_deg = uvot_util.read_grb_coords(f"GRB{grb_key}", coords_file)
                except Exception:
                    ra_deg, dec_deg = uvot_util.read_grb_coords(grb_key, coords_file)
                EBV = uvot_util.get_ebv_sfd(ra_deg, dec_deg)

                wave_A, EA = uvot_util.load_white_effarea(args.uvot_effarea_dir)
                A_band_white = uvot_util.compute_A_band_from_EA(wave_A, EA, EBV=EBV, Rv=3.1)

                # --- Load UVOT measurements using the SAME signature as in predict_*
                #     Returns (t_obs, Fnu_obs, Ferr_obs) when to_cgs=True
                t_obs, Fnu_obs, Ferr_obs = uvot_util.load_uvot_from_fits(
                    str(catalog_path),
                    grb_key=grb_key,
                    uvot_filter=filt,
                    to_cgs=True,
                    trigger_met=trigger_met,
                    t0_obs=None,
                    strict=False,
                )
                print(f"[UVOT] Loaded {len(t_obs)} points for GRB {grb_key} / {filt}")
                print(f"[UVOT] Before dereddening: Fν range = [{np.nanmin(Fnu_obs):.3e}, {np.nanmax(Fnu_obs):.3e}]")

                # --- Apply Galactic dereddening EXACTLY as in predict_* (Fν, σFν)
                Fnu_corr, Ferr_corr = uvot_util.apply_dereddening(Fnu_obs, Ferr_obs, A_band_white)
                print(f"[UVOT] After dereddening:  Fν range = [{np.nanmin(Fnu_corr):.3e}, {np.nanmax(Fnu_corr):.3e}]")

                # --- Select output according to 'mode'
                #     - mode='flux' -> use dereddened Fν [cgs]
                #     - mode!='flux' -> fall back to raw rate columns is NOT supported by predict_* loader here
                #       (the catalog reader we call already outputs fluxes; 'rate' fallback stays in CSV branch)
                t_w = np.asarray(t_obs, float)
                is_ul_w_raw = np.zeros_like(t_w, dtype=bool)
                if mode.lower() == "flux":
                    y_w  = np.asarray(Fnu_corr, float)
                    yerr_w = np.asarray(Ferr_corr, float)
                else:
                    # NOTE: 'rate' is only supported via CSV fallback; keep interface but warn.
                    print("[UVOT] WARNING: 'mode=rate' not supported for FITS pipeline; using dereddened fluxes instead.")
                    y_w  = np.asarray(Fnu_corr, float)
                    yerr_w = np.asarray(Ferr_corr, float)

                print(f"[UVOT] Loaded {t_w.size} valid WHITE points from FITS catalog (pipeline-aligned)")

            elif args.uvot_white and os.path.exists(args.uvot_white):
                # --- CSV fallback (old path): keep behavior unchanged
                optical_kind = "uvot"
                print(f"[UVOT] No FITS catalog found → falling back to CSV loader")
                t_w_raw, y_w_raw, yerr_w_raw = load_uvot_white_csv(
                    args.uvot_white, args.uvot_col_t, args.uvot_col_y, args.uvot_col_err
                )
                m_w = np.isfinite(t_w_raw) & np.isfinite(y_w_raw) & np.isfinite(yerr_w_raw)
                t_w, y_w, yerr_w = t_w_raw[m_w], y_w_raw[m_w], yerr_w_raw[m_w]
                is_ul_w_raw = np.zeros_like(t_w, dtype=bool)
            else:
                print("[UVOT] No UVOT input provided")
        except Exception as e:
            print(f"[UVOT] Error during FITS pipeline load: {e}")

    # -------------------------------------------------------------------------
    # UVOT RAW-time masks (apply tmin/tmax/mask and optional pre-t0 exclusion)
    # -------------------------------------------------------------------------
    # Keep immutable RAW copies for plotting/summary (like XRT)
    t_w_raw    = t_w.copy()
    y_w_raw    = y_w.copy()
    yerr_w_raw = yerr_w.copy()

    # Build RAW-time validity mask
    m_valid_w = (
        np.isfinite(t_w_raw)
        & np.isfinite(y_w_raw)
        & (y_w_raw > 0)
        & (~is_ul_w_raw)
    )
    # Select CLI controls depending on the optical dataset type
    opt_tmin = args.ronchini_tmin if (optical_kind == "ronchini_rc") else args.uvot_tmin
    opt_tmax = args.ronchini_tmax if (optical_kind == "ronchini_rc") else args.uvot_tmax
    opt_mask_str = args.ronchini_mask if (optical_kind == "ronchini_rc") else args.uvot_mask
    opt_extra_frac = float(args.ronchini_extra_frac) if (optical_kind == "ronchini_rc") else float(args.uvot_extra_frac)
    opt_logspace = bool(args.ronchini_logspace) if (optical_kind == "ronchini_rc") else bool(conf.get("uvot", {}).get("logspace", False))
    opt_exclude_preprompt = bool(getattr(args, "uvot_exclude_preprompt", False)) if (optical_kind != "ronchini_rc") else False


    # Window mask from CLI
    m_window_w = np.ones_like(m_valid_w, dtype=bool)
    if opt_tmin is not None:
        m_window_w &= (t_w_raw >= float(opt_tmin))
    if opt_tmax is not None:
        m_window_w &= (t_w_raw <= float(opt_tmax))

    # Ranges to exclude from CLI (same parser used for XRT)
    m_maskranges_w = np.zeros_like(m_valid_w, dtype=bool)
    for (a, b) in _parse_time_mask_str(opt_mask_str):
        m_maskranges_w |= (t_w_raw >= a) & (t_w_raw <= b)

    # Optional: exclude pre-prompt times using shared.t0_obs if provided
    m_preprompt_w = np.zeros_like(m_valid_w, dtype=bool)
    if opt_exclude_preprompt:
        # Try to read t0_obs from YAML (works for scalar or {value:...})
        t0_yaml = _conf_get_numeric(conf, "shared", "t0_obs", None)
        if t0_yaml is not None and np.isfinite(t0_yaml):
            m_preprompt_w = (t_w_raw < float(t0_yaml))

    # Final selection of UVOT points used in the fit
    m_use_w = m_valid_w & m_window_w & (~m_maskranges_w) & (~m_preprompt_w)

    # Apply selection
    t_w, y_w, yerr_w = t_w_raw[m_use_w], y_w_raw[m_use_w], yerr_w_raw[m_use_w]

    # Build ParamVector
    # If only one component is being fitted, freeze the other block's parameters
    if bool(conf.get("fit", {}).get("only_hle", False)) and (not bool(conf.get("fit", {}).get("only_fs", False))):
        if isinstance(conf.get("fs", {}), dict):
            for k, entry in conf["fs"].items():
                if isinstance(entry, dict) and ("value" in entry):
                    entry["fixed"] = True

    if bool(conf.get("fit", {}).get("only_fs", False)) and (not bool(conf.get("fit", {}).get("only_hle", False))):
        if isinstance(conf.get("hle", {}), dict):
            for k, entry in conf["hle"].items():
                if isinstance(entry, dict) and ("value" in entry):
                    entry["fixed"] = True

    pvec = ParamVector(conf)
    active = pvec.active_mask()
    u0_free = pvec.u0[active]; lb_free = pvec.lb[active]; ub_free = pvec.ub[active]


    # UVOT folding helper (only needed when the optical dataset is UVOT/WHITE)
    uvw = None
    if (optical_kind == "uvot") and (t_w.size > 0):
        uvw = UVOTWhite(args.uvot_effarea_dir, conf.get("dust", {}), conf.get("shared", {}))

    # Compose residual function inputs
    # Compose residual function inputs
    xrt_block = {
        "t": t_x, "y": y_x, "yerr": yerr_x,
        "logspace": bool(args.xrt_logspace),
        "extra_frac": float(args.xrt_extra_frac),
        # For plotting: raw arrays and masks
        "t_raw": t_x_raw, "y_raw": y_x_raw, "yerr_raw": yerr_x_raw,
        "m_use_raw": m_use,              # True → used in fit; False → excluded
        "mask_ranges": xrt_mask_ranges,  # list[(tmin,tmax)] excluded from fit (plotting only)
        "tmin_cli": args.xrt_tmin,       # for optional markers
        "tmax_cli": args.xrt_tmax
    }

    uvw_block = {
        "t": t_w, "y": y_w, "yerr": yerr_w,
        # "extra_frac": float(args.uvot_extra_frac),  # superseded by opt_extra_frac
        "extra_frac": float(opt_extra_frac),
        "logspace": bool(opt_logspace),
        "kind": optical_kind,
        "nu_hz": optical_nu_hz,
        "targets_b2": parse_targets_list(args.target_b2),
        # For plotting/summary consistency (mirror of XRT keys)
        "t_raw": t_w_raw, "y_raw": y_w_raw, "yerr_raw": yerr_w_raw,
        "upper_limits": is_ul_w_raw,
        "m_use_raw": m_use_w,                 # True → used in fit; False → excluded
        "tmin_cli": opt_tmin,           # optional markers if you choose to add them to plots
        "tmax_cli": opt_tmax
    }

    # Build fast views for the optimizer when --fast is enabled
    if conf.get("fit", {}).get("fast", False):
        # XRT
        xrt_block_fast = dict(xrt_block)
        if xrt_block_fast["t"].size > args.fast_max_xrt:
            ii = _log_subsample_indices(xrt_block_fast["t"], args.fast_max_xrt)
            xrt_block_fast["t"]    = xrt_block_fast["t"][ii]
            xrt_block_fast["y"]    = xrt_block_fast["y"][ii]
            xrt_block_fast["yerr"] = xrt_block_fast["yerr"][ii]
        else:
            xrt_block_fast = xrt_block

        # UVOT
        uvw_block_fast = dict(uvw_block)
        if uvw_block_fast["t"].size > args.fast_max_uvot:
            jj = _log_subsample_indices(uvw_block_fast["t"], args.fast_max_uvot)
            uvw_block_fast["t"]    = uvw_block_fast["t"][jj]
            uvw_block_fast["y"]    = uvw_block_fast["y"][jj]
            uvw_block_fast["yerr"] = uvw_block_fast["yerr"][jj]
        else:
            uvw_block_fast = uvw_block
    else:
        xrt_block_fast = xrt_block
        uvw_block_fast = uvw_block

    fun = make_residual_fn(conf, xrt_block_fast, uvw_block_fast, uvw, pvec)

    res = least_squares(
        fun, u0_free,
        bounds=(lb_free, ub_free),
        loss=conf["fit"]["robust_loss"],
        f_scale=float(conf["fit"]["f_scale"]),
        method="trf",
        jac="2-point",
        x_scale="jac",
        diff_step=float(getattr(args, "diff_step", 1e-2)),
        max_nfev=int(conf["fit"]["max_nfev"]),
        verbose=2
    )

    # Compose best u vector
    u_best = pvec.u0.copy(); u_best[active] = res.x

    # Write best params (mirror YAML schema, updating only 'value' fields when present)
    blocks = pvec.u_to_blocks(u_best)
    best = {"shared":{}, "hle":{}, "fs":{}}
    for sect in ("shared","hle","fs"):
        best[sect] = {}
        for k,v in blocks[sect].items():
            templ = conf.get(sect,{}).get(k, None)
            if isinstance(templ, dict) and "value" in templ:
                e = dict(templ); e["value"]=v; best[sect][k]=e
            else:
                best[sect][k]=v
    dump_yaml(best, os.path.join(args.outdir, "best_params.yaml"))

    # Figures
    make_figures(args.outdir, conf, xrt_block, uvw_block, u_best, pvec, uvw, gamma_data_path=args.gamma_data)

    # --- Dense chi-square recompute at best-fit (independent of robust loss) ---
    # We compute UNWEIGHTED residuals (W=1) so that the χ² diagnostics reflect pure
    # residuals. Dataset weights from the fit are used only to include/exclude an
    # entire dataset. Therefore we: (i) recompute residuals with weights forced to 1,
    # (ii) zero-out the corresponding χ² blocks (and remove their points from the
    # dof count) whenever the user set that dataset weight to 0.
    was_fast = bool(conf["fit"].get("fast", False))
    conf["fit"]["fast"] = False  # force dense grids inside residuals

    # Keep track of user-requested dataset inclusion.
    wb_user = dict(conf.get("fit", {}).get("weight_blocks", {}))
    w_xrt_user = float(wb_user.get("xrt", 1.0))
    w_uvot_user = float(wb_user.get("uvot_white", 1.0))
    w_tgt_user = float(wb_user.get("targets", 1.0))
    include_xrt = (w_xrt_user != 0.0)
    include_uvot = (w_uvot_user != 0.0)
    include_tgt = (w_tgt_user != 0.0)

    conf_stat = deepcopy(conf)
    conf_stat["fit"]["weight_blocks"] = {"xrt": 1.0, "uvot_white": 1.0, "targets": 1.0}
    fun_dense = make_residual_fn(conf_stat, xrt_block, uvw_block, uvw, pvec)

    u_best_free = u_best[active]
    r_dense = fun_dense(u_best_free)  # concatenated residuals (XRT | UVOT | targets)

    conf["fit"]["fast"] = was_fast  # restore user choice

    # Split by block (sizes are known)
    nx = int(xrt_block["t"].size)
    nw = int(uvw_block["t"].size)
    r_x = r_dense[:nx]
    r_w = r_dense[nx:nx+nw]
    r_t = r_dense[nx+nw:]

    # Statistical χ²: only real data (XRT+UVOT).
    # Dataset weights are interpreted as include/exclude switches: if a dataset was
    # excluded from the fit (w=0), we report χ²=0 for that dataset and we do NOT
    # count its points in χ² / redχ².
    chi2_x = float(np.sum(r_x**2))
    chi2_w = float(np.sum(r_w**2))
    chi2_t = float(np.sum(r_t**2))   # penalty term (not statistical)

    chi2_x_eff = chi2_x if include_xrt else 0.0
    chi2_w_eff = chi2_w if include_uvot else 0.0
    chi2_t_eff = chi2_t if include_tgt else 0.0

    chi2_stat = chi2_x_eff + chi2_w_eff
    chi2_augmented = chi2_stat + chi2_t_eff

    # Effective number of free parameters actually used by this run
    only_hle = bool(conf["fit"].get("only_hle", False))
    only_fs  = bool(conf["fit"].get("only_fs",  False))
    used_blocks = {"shared"}
    if only_hle and not only_fs:
        used_blocks.add("hle")
    elif only_fs and not only_hle:
        used_blocks.add("fs")
    else:
        used_blocks.update({"hle","fs"})

    npar_eff = int(sum((not ps.fixed) and (ps.block in used_blocks) for ps in pvec.specs))

    # dof computed ONLY from INCLUDED data points (XRT + UVOT); targets do not count
    npts_stat = int((nx if include_xrt else 0) + (nw if include_uvot else 0))
    if npts_stat > 0:
        ndof = int(max(npts_stat - npar_eff, 1))
        chi2_red = chi2_stat / ndof
    else:
        ndof = 0
        chi2_red = float("nan")

    # Optional: XRT-only stats (clean)
    chi2_xrt_only = None
    ndof_xrt_only = None
    chi2_red_xrt  = None
    if bool(getattr(args, "chi2_xrt_only", False)):
        if include_xrt and nx > 0:
            ndof_xrt_only = int(max(nx - npar_eff, 1))
            chi2_xrt_only = chi2_x
            chi2_red_xrt = chi2_xrt_only / ndof_xrt_only
        else:
            # If XRT is excluded (or has no points), report 0.
            chi2_xrt_only = 0.0
            ndof_xrt_only = 0
            chi2_red_xrt = float("nan")
    else:
        ndof_xrt_only = 0
        chi2_xrt_only = 0.0
        chi2_red_xrt = float("nan")

    # ========== Human-readable summary ===========
    def _fmt(v):
        try:
            return f"{float(v):.6g}"
        except Exception:
            return str(v)

    # Helper: decide if a parameter should be printed (exclude grid/tech knobs)
    _EXCLUDE_PREFIX = ("n_theta","n_phi","n_bins","n_nu","n_mc","min_grid_counts","fast")
    _EXCLUDE_EXACT  = {"spectrum"}  # enforced internally for HLE
    def _is_phys(name: str) -> bool:
        if name in _EXCLUDE_EXACT: return False
        return not name.startswith(_EXCLUDE_PREFIX)

    # Current best-fit values per block
    # (re-use 'blocks' from best_params.yaml section above)
    shared_blk = blocks["shared"]
    hle_blk    = blocks["hle"]
    fs_blk     = blocks["fs"]

    # Determine fitted vs fixed from ParamVector specs
    # Build maps: {block: {name: fixed_bool}}
    fixed_map = {"shared": {}, "hle": {}, "fs": {}}
    for ps in pvec.specs:
        fixed_map[ps.block][ps.name] = bool(ps.fixed)

    # Separate Fitted / Fixed lists for HLE and FS (filtered by _is_phys)
    def _split_block(blk_name: str, blk_vals: dict):
        fitted, fixed = [], []
        fm = fixed_map.get(blk_name, {})
        for k, v in blk_vals.items():
            if not isinstance(v, (int, float)):     # skip non-numerical (e.g. strings)
                continue
            if not _is_phys(k):
                continue
            (fixed if fm.get(k, True) else fitted).append((k, v))
        # stable, name-sorted
        fitted.sort(key=lambda kv: kv[0]); fixed.sort(key=lambda kv: kv[0])
        return fitted, fixed

    sh_fit,  sh_fix  = _split_block("shared", shared_blk)
    hle_fit, hle_fix = _split_block("hle",    hle_blk)
    fs_fit,  fs_fix  = _split_block("fs",     fs_blk)

    # Compose text lines
    lines = []
    lines.append("=== Combined fit summary (HLE + FS) ===")
    lines.append(f"XRT file: {args.xrt}")
    lines.append(f"UVOT White: {args.uvot_white if args.uvot_white else 'None'}")
    lines.append(f"XRT band: [{conf['fit']['xrt_band_keV'][0]}, {conf['fit']['xrt_band_keV'][1]}] keV")
    lines.append(f"Weights: XRT={conf['fit']['weight_blocks'].get('xrt',1.0)}, "
                 f"UVOT={conf['fit']['weight_blocks'].get('uvot_white',1.0)}, "
                 f"Targets={conf['fit']['weight_blocks'].get('targets',1.0)}")
    lines.append(f"Loss: {conf['fit']['robust_loss']}   f_scale={_fmt(conf['fit']['f_scale'])}   "
                 f"max_nfev={int(conf['fit']['max_nfev'])}")
    lines.append("")
    # Report used vs total (useful to reproduce legacy fitter numbers)
    n_x_used = int(xrt_block["t"].size)
    n_x_total = int(np.asarray(xrt_block.get("t_raw", xrt_block["t"])).size)
    n_w_used = int(uvw_block["t"].size)
    n_w_total = int(np.asarray(uvw_block.get("t_raw", uvw_block["t"])).size)

    lines.append(f"Points: XRT used/total = {n_x_used}/{n_x_total}   UVOT used/total = {n_w_used}/{n_w_total}   "
                 f"Free params (effective) = {npar_eff}")
    lines.append(f"stat χ² (XRT+UVOT) = {_fmt(chi2_stat)}   dof={int(ndof)}   "
                 f"redχ²={_fmt(chi2_red)}   "
                 f"AIC(stat)={_fmt(chi2_stat + 2*npar_eff)}   "
                 f"BIC(stat)={_fmt(chi2_stat + npar_eff*np.log(max(npts_stat,2)))}")
    lines.append(f"augmented χ² (stat + targets) = {_fmt(chi2_augmented)}")

    xrt_block_str = f"χ²_XRT={_fmt(chi2_x_eff)}"
    if not include_xrt:
        xrt_block_str += " (excluded)"
    uvot_block_str = f"χ²_UVOT={_fmt(chi2_w_eff)}"
    if not include_uvot:
        uvot_block_str += " (excluded)"
    targ_block_str = f"penalties(targets)={_fmt(chi2_t_eff)}"
    if not include_tgt:
        targ_block_str += " (excluded)"

    lines.append(f"  blocks: {xrt_block_str}   {uvot_block_str}   {targ_block_str}")
    if chi2_xrt_only is not None:
        xrt_only_str = f"XRT-only: χ²={_fmt(chi2_xrt_only)}   dof={int(ndof_xrt_only)}   redχ²={_fmt(chi2_red_xrt)}"
        if not include_xrt:
            xrt_only_str += " (excluded)"
        lines.append(xrt_only_str)

    lines.append(f"XRT logspace: {bool(xrt_block.get('logspace', False))}   extra_frac: {float(xrt_block.get('extra_frac', 0.0))}")
    lines.append("")

    # ---- SHARED block (Fitted / Fixed) ----
    lines.append("SHARED – Fitted parameters:")
    if sh_fit:
        for k, v in sh_fit:
            lines.append(f"  {k:>18s} = {_fmt(v)}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("SHARED – Fixed parameters:")
    if sh_fix:
        for k, v in sh_fix:
            lines.append(f"  {k:>18s} = {_fmt(v)}")
    else:
        lines.append("  (none)")
    lines.append("")
    # Show reference times with the application flags for clarity
    if "t0_obs" in shared_blk:
        t0_hle = float(shared_blk.get("t0_obs", 0.0))
        t0_fs  = float(shared_blk.get("t0_fs_obs", t0_hle))
        lines.append(
            f"t0_hle_obs: {_fmt(t0_hle)} s   t0_fs_obs: {_fmt(t0_fs)} s "
            f"(apply_t0_to_hle={bool(conf['fit'].get('apply_t0_to_hle', True))}, "
            f"apply_t0_to_fs={bool(conf['fit'].get('apply_t0_to_fs', True))})"
        )
        lines.append("")

    # ---- HLE block (Fitted / Fixed) ----
    lines.append("HLE – Fitted parameters:")
    if hle_fit:
        for k, v in hle_fit:
            lines.append(f"  {k:>18s} = {_fmt(v)}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("HLE – Fixed parameters:")
    if hle_fix:
        for k, v in hle_fix:
            lines.append(f"  {k:>18s} = {_fmt(v)}")
    else:
        lines.append("  (none)")
    lines.append("")

    # ---- FS block (Fitted / Fixed) ----
    lines.append("FS – Fitted parameters:")
    if fs_fit:
        for k, v in fs_fit:
            lines.append(f"  {k:>18s} = {_fmt(v)}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("FS – Fixed parameters:")
    if fs_fix:
        for k, v in fs_fix:
            lines.append(f"  {k:>18s} = {_fmt(v)}")
    else:
        lines.append("  (none)")
    lines.append("")

    # ---- FS derived dynamical scales (diagnostics) ----
    try:
        # FS kinetic energy is deterministically forced from prompt Egamma & eta_gamma
        Eg = float(shared_blk.get("Egamma_iso_core", np.nan))
        eta = float(shared_blk.get("eta_gamma", np.nan))
        Eiso_kin = float(_derive_fs_Eiso_from_prompt(Eg, eta))

        # Medium can be stored as a non-numeric field in the FS block
        fs_medium = fs_blk.get("medium", conf.get("fs", {}).get("medium", {}).get("value", "ism"))
        fs_theta_edge = shared_blk.get("theta_edge_deg", None)

        diag = fs.compute_deceleration_scales(
            Gamma_c=float(shared_blk.get("Gamma_c", np.nan)),
            theta_j_deg=float(shared_blk.get("theta_j_deg", np.nan)),
            g=float(shared_blk.get("g", np.nan)),
            epsilon_c=float(shared_blk.get("epsilon_c", 1.0)),
            k_struct=float(shared_blk.get("k_struct", np.nan)),
            theta_v_deg=float(shared_blk.get("theta_v_deg", 0.0)),
            theta_edge_deg=fs_theta_edge,
            medium=str(fs_medium),
            n0=float(fs_blk.get("n0", 1.0)),
            Astar=float(fs_blk.get("Astar", 1.0)),
            Eiso_core=Eiso_kin,
            z=float(shared_blk.get("z", 0.0)),
            n_theta=4096,
            rmin_frac=1e-3,
        )

        lines.append("FS – Derived dynamical scales:")
        lines.append(f"       Eiso_core(kin) = {_fmt(Eiso_kin)} erg  (forced from Egamma_iso_core & eta_gamma)")
        lines.append(f"     Rdec_core(θ=0) = {_fmt(diag['Rdec_core_cm'])} cm")
        lines.append(f"          Rdec_min = {_fmt(diag['Rdec_min_cm'])} cm  at θ={_fmt(diag['theta_min_deg'])} deg")
        lines.append(f"         Rmin_eff = {_fmt(diag['R_min_eff_cm'])} cm  (=1e-3 * Rdec_min; overrides YAML R_min)")
        lines.append(f"  tdec_core (approx) = {_fmt(diag['tdec_core_s'])} s")
        lines.append(f"   tdec_min (approx) = {_fmt(diag['tdec_min_s'])} s")
        lines.append("")
    except Exception as e:
        lines.append("FS – Derived dynamical scales: (failed to compute)")
        lines.append(f"  reason: {e}")
        lines.append("")

    # Write summary text (same spirit/name as legacy fitter)
    with open(os.path.join(args.outdir, "fit_summary.txt"), "w", encoding="utf-8") as ftxt:
        ftxt.write("\n".join(lines))

    # Also dump model@data CSVs for quick diagnostics
    # XRT: recompute model on data times with best params (log-interp wrapper)
    shared = blocks["shared"]; hpars = blocks["hle"]; fpars = blocks["fs"]
    fpars_eff = _apply_fs_onset_coupling(shared, hpars, fpars, conf)
    t0_hle = float(shared.get("t0_obs", 0.0))
    t0_fs  = float(shared.get("t0_fs_obs", t0_hle))
    tx_h, Fx_h, _ = hle_band_flux_curve(_shared_to_hle_kwargs(shared), hpars, tuple(conf["fit"]["xrt_band_keV"]))
    tx_f, Fx_f, _ = fs_band_flux_curve(shared, fpars_eff, tuple(conf["fit"]["xrt_band_keV"]))
    Fx_best = _log_interp(tx_h, Fx_h, _shift_times(xrt_block["t"], t0_hle, bool(conf["fit"].get("apply_t0_to_hle", True)))) \
            + _log_interp(tx_f, Fx_f, _shift_times(xrt_block["t"], t0_fs, bool(conf["fit"].get("apply_t0_to_fs", True))))
    np.savetxt(os.path.join(args.outdir, "xrt_fit_at_data.csv"),
               np.column_stack([xrt_block["t"], xrt_block["y"], xrt_block["yerr"], Fx_best]),
               delimiter=",", header="t, y, yerr, model", comments="")
    if uvw_block["t"].size:
        kind = str(uvw_block.get("kind", "uvot")).lower()
        t_eval_h = _shift_times(uvw_block["t"], t0_hle, bool(conf["fit"].get("apply_t0_to_hle", True)))
        t_eval_f = _shift_times(uvw_block["t"], t0_fs,  bool(conf["fit"].get("apply_t0_to_fs", True)))

        if kind == "ronchini_rc":
            nu_hz = float(uvw_block.get("nu_hz", np.nan))
            Fw_hle, Fw_fs, Fw_tot = _monochromatic_Fnu_from_components(
                t_eval_hle=t_eval_h,
                t_eval_fs=t_eval_f,
                nu_hz=nu_hz,
                shared=shared,
                hle_pars=hpars,
                fs_pars=fpars_eff,
                return_components=True
            )
            np.savetxt(
                os.path.join(args.outdir, "rc_fit_at_data.csv"),
                np.column_stack([uvw_block["t"], uvw_block["y"], uvw_block["yerr"], Fw_hle, Fw_fs, Fw_tot]),
                delimiter=",",
                header="t, Fnu, Fnu_err, model_Fnu_hle, model_Fnu_fs, model_Fnu_tot",
                comments=""
            )

        else:
            uvot_mode = str(conf.get("uvot", {}).get("mode", "counts")).lower()
            uw = uvw
            if uw is None:
                uw = UVOTWhite(args.uvot_effarea_dir, conf.get("dust", {}), conf.get("shared", {}))

                if uvot_mode == "flux":
                    # Save dereddened flux-density comparison (consistent with residuals)
                    Fw_best = uw.band_flux_from_components(
                        t_eval_hle=t_eval_h, shared=shared, hle_pars=hpars, fs_pars=fpars_eff,
                        t_eval_fs=t_eval_f, apply_extinction=False
                    )
                    np.savetxt(os.path.join(args.outdir, "uvot_fit_at_data.csv"),
                               np.column_stack([uvw_block["t"], uvw_block["y"], uvw_block["yerr"], Fw_best]),
                               delimiter=",", header="t, Fnu_dered, Fnu_dered_err, model_Fnu", comments="")
                else:
                    # Save counts/s comparison
                    Cw_best = uw.counts_from_components(
                        t_eval_hle=t_eval_h, shared=shared, hle_pars=hpars, fs_pars=fpars_eff, t_eval_fs=t_eval_f
                    )
                    np.savetxt(os.path.join(args.outdir, "uvot_fit_at_data.csv"),
                               np.column_stack([uvw_block["t"], uvw_block["y"], uvw_block["yerr"], Cw_best]),
                               delimiter=",", header="t, rate, rate_err, model_counts", comments="")



            # Optional: save UVOT model in flux mode (if mode='flux')
            if uvw_block["t"].size and str(uvw_block.get("kind","uvot")).lower() == "uvot" and conf.get("uvot", {}).get("mode", "counts") == "flux":
                try:
                    wave_A, EA = uvot_util.load_white_effarea(conf["uvot"].get("caldb_effarea_dir"))
                    Fnu_best = uvot_util.predict_white_flux_from_model(
                        t_eval=_shift_times(uvw_block["t"], t0, bool(conf["fit"].get("apply_t0_to_hle", True))),
                        shared=shared,
                        hle_pars=hpars,
                        fs_pars=fpars_eff,
                        wave_A=wave_A,
                        EA=EA,
                        apply_extinction=False
                    )
                    np.savetxt(os.path.join(args.outdir, "uvot_fit_at_data_flux.csv"),
                               np.column_stack([uvw_block["t"], uvw_block["y"], uvw_block["yerr"], Fnu_best]),
                               delimiter=",", header="t, Fnu_dered, Fnu_dered_err, model_Fnu", comments="")
                    print("[UVOT] Saved additional flux comparison file → uvot_fit_at_data_flux.csv")
                except Exception as e:
                    print(f"[UVOT] Could not save flux-mode diagnostic: {e}")

    print("\n=== Fit finished ===")
    print(f"stat_chi2={chi2_stat:.6g}   dof={ndof}   redchi2={chi2_red:.6g}   (augmented={chi2_augmented:.6g})")
    print(f"blocks: {xrt_block_str}   {uvot_block_str}   {targ_block_str}")
    if chi2_xrt_only is not None:
        xrt_only_tag = " (excluded)" if (not include_xrt) else ""
        print(f"XRT-only: chi2={chi2_xrt_only:.6g}   dof={int(ndof_xrt_only)}   redchi2={chi2_red_xrt:.6g}{xrt_only_tag}")
    print(f"best_params.yaml → {os.path.join(args.outdir, 'best_params.yaml')}")
    print(f"figures          → {args.outdir}")

if __name__ == "__main__":
    main()
