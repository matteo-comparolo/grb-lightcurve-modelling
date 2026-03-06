"""
fit_hle_xrt.py -- HLE Fitter

Highlights:
• Reads CSV/TSV and QDP robustly (named columns; asymmetric errors; optional time-error inflation)
• Uses SciPy bounded least_squares (residual-level control; log-space option; extra fractional error)
• FIT PHASE: forces a lightweight model resolution + n_mc=0 for speed
• FINAL OUTPUT: recomputes a dense model curve at the YAML resolution once
• Time origin: observer time is the arrival of the first photon
• Saves: <out>_summary.txt, <out>_best_params.yaml, <out>_data.csv, <out>_model.csv,
         <out>_fit_at_data.csv (model@data times), <out>_plot.png

Requires:
    numpy, pandas, matplotlib, scipy, pyyaml
and 'hle_tophat.py' with compute_hle_lightcurve(**kwargs)
"""

import argparse, os, math, warnings, importlib, inspect, re, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy optimizer
from scipy.optimize import least_squares

# YAML (optional fallback to JSON)
HAVE_YAML = True
try:
    import yaml
except Exception:
    HAVE_YAML = False

# =========================
# Parameter Spec & I/O YAML
# =========================
@dataclass
class ParamSpec:
    name: str
    value: float
    min: float
    max: float
    fixed: bool = False
    transform: str = "lin"          # 'lin' or 'log'
    map_to: Optional[str] = None    # Key rename for hle.py

    def to_free(self, x: float) -> float:
        return np.log10(x) if self.transform == "log" else x

    def from_free(self, u: float) -> float:
        return 10.0**u if self.transform == "log" else u

    def bounds_free(self) -> Tuple[float, float]:
        lo = np.log10(self.min) if self.transform == "log" else self.min
        hi = np.log10(self.max) if self.transform == "log" else self.max
        return lo, hi


def load_params(path: str) -> List[ParamSpec]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            if not HAVE_YAML:
                raise RuntimeError("PyYAML not installed; cannot read YAML.")
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    specs: List[ParamSpec] = []
    for p in raw["params"]:
        specs.append(ParamSpec(
            name=p["name"],
            value=float(p["value"]),
            min=float(p["min"]),
            max=float(p["max"]),
            fixed=bool(p.get("fixed", False)),
            transform=p.get("transform", "lin"),
            map_to=p.get("map_to"),
        ))

    for s in specs:
        if s.transform == "log" and (s.min <= 0 or s.max <= 0):
            raise ValueError(f"{s.name}: log-param bounds must be > 0.")
        if not (s.min < s.max):
            raise ValueError(f"{s.name}: min must be < max.")
    return specs


def save_params(path: str, specs: List[ParamSpec]):
    payload = {"params": []}
    for s in specs:
        payload["params"].append({
            "name": s.name,
            "value": float(s.value),
            "min": float(s.min),
            "max": float(s.max),
            "fixed": bool(s.fixed),
            "transform": s.transform,
            "map_to": s.map_to,
        })
    if path.lower().endswith((".yaml", ".yml")) and HAVE_YAML:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

# ================
# Data file reader
# ================
def read_table_auto(path: str,
                    col_t: str, col_y: str, col_err: Optional[str] = None,
                    col_ypos: str = None, col_yneg: str = None,
                    col_tpos: str = None, col_tneg: str = None,
                    err_sym: str = "max",
                    qdp_set: str = "ALL"):
    """
    Robust CSV/TSV/QDP reader, including:
    - named columns
    - support for QDP WT/PC/ALL set selection
    - asymmetric y-errors with several symmetrizations
    - optional time errors (returned separately)
    """
    df = None
    p = str(path).lower()

    # Quick CSV/TSV attempt
    try:
        if p.endswith(".csv"):
            df = pd.read_csv(path, comment="#")
        elif p.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", comment="#")
    except Exception:
        df = None

    # QDP parsing
    if df is None and p.endswith(".qdp"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        header_idx, cols = None, None
        wanted_sets = [
            {"Time", "Flux", "Fluxpos", "Fluxneg"},
            {"Time", "Rate", "Ratepos", "Rateneg"},
            {"Time", "T+ve", "T-ve", "Flux", "Fluxpos", "Fluxneg"},
            {"Time", "T+ve", "T-ve", "Rate", "Ratepos", "Rateneg"},
        ]
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith(("!", "#")):
                continue
            toks = [t for t in re.split(r"\s+", s) if t]
            if any(ws.issubset(set(toks)) for ws in wanted_sets):
                header_idx, cols = i, toks
                break

        num_re = re.compile(r'^[\+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][\+\-]?\d+)?$')
        rows, set_ids, set_id = [], [], 0

        def is_sep(s: str) -> bool:
            tok = s.upper().split()
            return (len(tok) == 2 and tok[0] == "NO" and tok[1] == "NO")

        def is_directive(s: str) -> bool:
            u = s.upper()
            return u.startswith(("READ", "LABEL", "ERROR", "SERR", "TERR", "MARK", "NEW"))

        it = enumerate(lines[header_idx+1:] if header_idx is not None else lines)
        for _, line in it:
            s = line.strip()
            if not s:
                continue
            if is_sep(s):
                set_id += 1
                continue
            if s.startswith(("!", "#")) or is_directive(s):
                continue
            toks = [t for t in re.split(r"\s+", s) if t]
            nums = [float(t) for t in toks if num_re.match(t)]
            if nums:
                rows.append(nums)
                set_ids.append(set_id)
        if not rows:
            raise ValueError(f"No numeric data rows found in {path}.")

        lengths = pd.Series([len(r) for r in rows])
        L = int(lengths.mode().iloc[0])
        rows = [r for r in rows if len(r) == L]
        set_ids = [sid for (sid, r) in zip(set_ids, rows) if len(r) == L]

        if header_idx is not None and cols is not None:
            if len(cols) > L:
                cols = cols[:L]
            if L == 4 and ("Time" in cols):
                cols = ["Time", "Y", "Ypos", "Yneg"]
        else:
            cols = (["Time", "T+ve", "T-ve", "Flux", "Fluxpos", "Fluxneg"]
                    if L == 6 else ["Time", "Y", "Ypos", "Yneg"])
        df = pd.DataFrame(rows, columns=cols)
        df["set_id"] = np.array(set_ids, dtype=int)

        if qdp_set.upper() == "WT":
            df = df[df["set_id"] == 0]
        elif qdp_set.upper() == "PC":
            df = df[df["set_id"] == 1]
        if df.empty:
            raise ValueError(f"{path}: selection {qdp_set} produced an empty table. "
                             f"Available set_id values: {sorted(set(set_ids))}")

    # .dat-like whitespace tables (or CSV/TSV already read)
    if df is None:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        header_idx, cols = None, None
        wanted_sets = [
            {"Time", "Flux", "Fluxpos", "Fluxneg"},
            {"Time", "Rate", "Ratepos", "Rateneg"},
        ]
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            if s.startswith(("WT data", "PC data", "BINMODE", "BINMODE:", "Binning", "Binning:")):
                continue
            toks = [t for t in re.split(r"\s+", s) if t]
            lo = set(toks)
            if any(ws.issubset(lo) for ws in wanted_sets) and "Time" in lo:
                header_idx, cols = i, toks
                break
            if ("Time" in lo) and (("Flux" in lo) or ("Rate" in lo)):
                header_idx, cols = i, toks
                break

        if header_idx is not None and cols is not None:
            df = pd.read_csv(
                path, sep=r"\s+", engine="python", skiprows=header_idx+1,
                names=cols, comment="!", on_bad_lines="skip"
            )
        else:
            # find first numeric block (6 cols typical)
            first_data = None
            num6 = r"^[\d\.\+\-Ee]+\s+[\d\.\+\-Ee]+\s+[\d\.\+\-Ee]+\s+[\d\.\+\-Ee]+\s+[\d\.\+\-Ee]+\s+[\d\.\+\-Ee]+$"
            for i, line in enumerate(lines):
                s = line.strip()
                if not s or s.startswith("!"):
                    continue
                if re.match(num6, s):
                    first_data = i
                    break
            if first_data is None:
                raise ValueError(f"Could not locate header or numeric block inside {path}.")

            df = None
            for cand in (
                ["Time", "T+ve", "T-ve", "Flux", "Fluxpos", "Fluxneg"],
                ["Time", "T+ve", "T-ve", "Rate", "Ratepos", "Rateneg"],
            ):
                try:
                    df = pd.read_csv(
                        path, sep=r"\s+", engine="python", skiprows=first_data,
                        names=cand, comment="!", on_bad_lines="skip"
                    )
                    cols = cand
                    break
                except Exception:
                    df = None
            if df is None:
                raise ValueError(f"Failed to parse {path}: no header and fallback parse failed.")

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    df.replace({"NO": np.nan, "no": np.nan}, inplace=True)

    def _col_as_float(name: str):
        if name not in df.columns:
            raise ValueError(f"Required column '{name}' not found. Have: {list(df.columns)}")
        return pd.to_numeric(df[name], errors="coerce").to_numpy()

    if col_t not in df.columns or col_y not in df.columns:
        raise ValueError(f"Required columns not found. Have: {list(df.columns)}")

    t = _col_as_float(col_t)
    y = _col_as_float(col_y)

    # Asymmetric errors
    if col_ypos and col_yneg and (col_ypos in df.columns) and (col_yneg in df.columns):
        yp = np.abs(_col_as_float(col_ypos))
        yn = np.abs(_col_as_float(col_yneg))
        if   err_sym == "max":  yerr = np.maximum(yp, yn)
        elif err_sym == "mean": yerr = 0.5*(yp + yn)
        elif err_sym == "pos":  yerr = yp
        elif err_sym == "neg":  yerr = yn
        else: raise ValueError(f"Unknown err_sym={err_sym}")
    else:
        if col_err is None or (col_err not in df.columns):
            raise ValueError("No symmetric error column and no ypos/yneg provided.")
        yerr = np.abs(_col_as_float(col_err))

    tpos = tneg = None
    if (col_tpos and col_tpos in df.columns) and (col_tneg and col_tneg in df.columns):
        tpos = np.abs(_col_as_float(col_tpos))
        tneg = np.abs(_col_as_float(col_tneg))

    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if tpos is not None and tneg is not None:
        m &= np.isfinite(tpos) & np.isfinite(tneg)
        return t[m], y[m], yerr[m], tpos[m], tneg[m]
    else:
        return t[m], y[m], yerr[m], None, None




# ==========================================
# Spectral evolution (photon index) reader
# ==========================================
def read_spectral_evolution_gamma(path: str):
    """
    Read a Swift Burst Analyser-style 'spectral_evolution.txt' (BAT or XRT),
    returning:
        t, tpos, tneg, gamma, gpos, gneg
    where:
      - t is time since trigger [s]
      - tpos/tneg are +/− time errors [s]
      - gamma is the photon index Γ
      - gpos/gneg are +/− errors on Γ (asymmetric)
    The file can contain multiple blocks separated by 'NO NO ...' rows; those are skipped.
    Lines starting with '!' are skipped. Directives like 'READ ...' are skipped.
    This function does NOT try to interpret WT/PC separately; it just returns all numeric rows.
    """
    t_list, tp_list, tn_list = [], [], []
    g_list, gp_list, gn_list = [], [], []
    num_re = re.compile(r'^[\+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][\+\-]?\d+)?$')

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("!"):
                continue
            u = s.upper()
            if u.startswith(("READ", "LABEL", "ERROR", "SERR", "TERR", "MARK", "NEW", "SKIP")):
                continue
            # separators like "NO NO NO ..."
            if u.startswith("NO"):
                continue

            toks = [t for t in re.split(r"\s+", s) if t]
            # Keep only numeric tokens
            nums = [float(t) for t in toks if num_re.match(t)]
            if len(nums) < 15:
                continue

            # Empirically (Burst Analyser format): Time, +dT, -dT are columns 0..2;
            # Gamma, +err, -err are columns 12..14 (0-indexed) in the standard 21-col table.
            # If the table is longer, this still holds for the first 15 columns.
            t0, tpos, tneg = nums[0], nums[1], nums[2]
            gamma, gpos, gneg = nums[12], nums[13], nums[14]

            if not (np.isfinite(t0) and np.isfinite(gamma)):
                continue

            t_list.append(float(t0))
            tp_list.append(float(abs(tpos)))
            tn_list.append(float(abs(tneg)))
            g_list.append(float(gamma))
            gp_list.append(float(abs(gpos)))
            gn_list.append(float(abs(gneg)))

    if len(t_list) == 0:
        raise RuntimeError(f"No usable Γ(t) rows parsed from: {path}")

    t = np.asarray(t_list, float)
    tpos = np.asarray(tp_list, float)
    tneg = np.asarray(tn_list, float)
    gamma = np.asarray(g_list, float)
    gpos = np.asarray(gp_list, float)
    gneg = np.asarray(gn_list, float)

    m = np.isfinite(t) & np.isfinite(gamma)
    m &= np.isfinite(tpos) & np.isfinite(tneg) & np.isfinite(gpos) & np.isfinite(gneg)
    # Keep times > 0 for log-x plots; allow t==0 only if user wants linear (we always plot log-x)
    m &= (t > 0)

    return t[m], tpos[m], tneg[m], gamma[m], gpos[m], gneg[m]
# =============================
# HLE adapter (band/mono + fast)
# =============================
class HLEAdapter:
    """
    Wraps hle.compute_hle_lightcurve and interpolates the result to requested times.
    Supports either band-integrated (nu_min_keV..nu_max_keV) or monochromatic Emono [keV].
    A per-call 'res_override' can reduce grid during fits. n_mc can also be forced.
    """
    def __init__(self, emin: float, emax: float, z: float, units: str, Emono: Optional[float]):
        self.emin = float(emin)
        self.emax = float(emax)
        self.z = float(z)
        self.units = units
        self.Emono = Emono

        try:
            self.hle = importlib.import_module("hle_tophat")
        except Exception as e:
            raise ImportError("Could not import hle_tophat.py (keep in same folder or add to PYTHONPATH).") from e
        if not hasattr(self.hle, "compute_hle_lightcurve"):
            raise AttributeError("hle.py must define compute_hle_lightcurve(**kwargs).")
        self.func = getattr(self.hle, "compute_hle_lightcurve")
        self._allowed = set(inspect.signature(self.func).parameters.keys())
        self._E_HZ_PER_eV = getattr(self.hle, "E_HZ_PER_eV", 2.418e14)

    @staticmethod
    def _loglog_interp(x, xp, fp):
        y = np.full_like(x, np.nan, dtype=float)
        if xp is None or fp is None: return y
        good = np.isfinite(xp) & np.isfinite(fp) & (xp > 0) & (fp > 0)
        if not np.any(good): return y
        tiny = 1e-300
        xgood = (x > 0) & np.isfinite(x)
        if not np.any(xgood): return y
        y[xgood] = np.exp(np.interp(np.log(x[xgood]),
                                    np.log(xp[good]),
                                    np.log(np.clip(fp[good], tiny, None)),
                                    left=np.nan, right=np.nan))
        return y

    def _filter_kwargs(self, kwargs: Dict[str, float]) -> Dict[str, float]:
        return {k: v for k, v in kwargs.items() if k in self._allowed}

    def __call__(self, t_eval: np.ndarray, p_hle: Dict[str, float],
                 res_override: Optional[Dict[str, int]] = None,
                 force_n_mc_zero: bool = False) -> np.ndarray:
        """
        Evaluate model and return flux interpolated at t_eval.
        Optionally override resolution (n_theta, n_phi, n_bins, n_nu, min_grid_counts) and force n_mc=0.
        """
        kwargs = dict(p_hle)
        kwargs.update({"nu_min_keV": self.emin, "nu_max_keV": self.emax, "z": self.z})
        
        if res_override:
            for k in ("n_theta","n_phi","n_bins","n_nu","min_grid_counts"):
                if k in res_override and res_override[k] is not None:
                    kwargs[k] = int(res_override[k])
        if force_n_mc_zero:
            kwargs["n_mc"] = 0
        # Cst any resolution-like params to int
        for k in ("n_theta","n_phi","n_bins","n_nu","n_mc","min_grid_counts"):
            if k in kwargs:
                kwargs[k] = int(kwargs[k])

        # Run model
        ret = self.func(**self._filter_kwargs(kwargs))
        if not isinstance(ret, tuple):
            raise RuntimeError("hle.compute_hle_lightcurve should return a tuple (t, F, ...).")

        if len(ret) >= 6:
            t_model, F_band, _, _, F_tnu, nu_grid = ret[:6]
        elif len(ret) >= 2:
            t_model, F_band = ret[:2]
            F_tnu, nu_grid = None, None
        else:
            raise RuntimeError("hle.compute_hle_lightcurve returned an unexpected tuple shape.")

        # Choose band-integrated or monochromatic
        if (self.Emono is None) or (F_tnu is None) or (nu_grid is None):
            y_grid = F_band
        else:
            target_nu = self.Emono * 1e3 * self._E_HZ_PER_eV
            j = int(np.argmin(np.abs(nu_grid - target_nu)))
            y_grid = F_tnu[:, j]

        return self._loglog_interp(t_eval, t_model, y_grid)


# =====================
# Fitting util functions
# =====================
def assemble_param_vector(specs: List[ParamSpec]):
    free_names, free_init, b_lo, b_hi, fixed_map = [], [], [], [], {}
    for s in specs:
        if s.fixed:
            fixed_map[s.name] = float(s.value)
        else:
            free_names.append(s.name)
            free_init.append(s.to_free(s.value))
            lo, hi = s.bounds_free()
            b_lo.append(lo); b_hi.append(hi)
    return free_names, np.array(free_init, float), (np.array(b_lo), np.array(b_hi)), fixed_map


def merge_params(free_names, u, fixed_map, specs_by_name):
    p = dict(fixed_map)
    for name, ui in zip(free_names, u):
        p[name] = specs_by_name[name].from_free(ui)
    return p

def map_to_hle_names(p_lin: Dict[str, float], specs_by_name: Dict[str, ParamSpec]) -> Dict[str, float]:
    """
    Map fitter linear params to hle_tophat.py keyword args.
    - Drop 't0_obs' (fitter-only time shift).
    """
    out = {}
    for k, v in p_lin.items():
        # fitter-only knob, do not forward to model
        if k == "t0_obs":
            continue
        key = specs_by_name[k].map_to if specs_by_name[k].map_to else k
        # legacy frame switch: ignore entirely
        if key == "breaks_frame":
            continue
        out[key] = float(v)
    return out

def model_binavg(adapter: HLEAdapter,
                 t_obs: np.ndarray,
                 tpos: np.ndarray,
                 tneg: np.ndarray,
                 t0: float,
                 p_hle: Dict[str, float],
                 res_override: Optional[Dict[str, int]],
                 n_samples: int) -> np.ndarray:
    """
    Compute bin-averaged model flux for each data point, treating (t - tneg, t + tpos)
    as a time BIN. The model is averaged over log-spaced samples inside the bin,
    after applying the fitter's time shift t0_obs.

    All times are in observer seconds.
    """
    t_obs = np.asarray(t_obs, float)
    tpos = np.asarray(tpos, float)
    tneg = np.asarray(tneg, float)

    # Bin edges in "since first photon" time (apply t0 shift)
    t1 = t_obs - tneg - float(t0)
    t2 = t_obs + tpos - float(t0)

    # Enforce positive times for log interpolation
    # If t1 <= 0, clip to a small positive number.
    t1 = np.where(np.isfinite(t1), t1, np.nan)
    t2 = np.where(np.isfinite(t2), t2, np.nan)

    tiny = 1e-12
    t1c = np.where(t1 > tiny, t1, tiny)
    t2c = np.where(t2 > t1c, t2, t1c * (1.0 + 1e-6))

    # Build per-point log-spaced samples via broadcasting
    K = max(int(n_samples), 2)
    u = np.linspace(0.0, 1.0, K)[None, :]          # shape (1, K)
    logt1 = np.log(t1c)[:, None]                   # shape (N, 1)
    logt2 = np.log(t2c)[:, None]
    t_samp = np.exp(logt1 + u * (logt2 - logt1))   # shape (N, K)

    # Evaluate model in one vector call for speed
    y_flat = adapter(t_samp.reshape(-1), p_hle,
                     res_override=res_override,
                     force_n_mc_zero=True)
    y_mat = y_flat.reshape(t_samp.shape)

    # Average ignoring NaNs
    y_avg = np.nanmean(y_mat, axis=1)
    return y_avg

def residuals(u, free_names, fixed_map, specs_by_name,
              adapter_fit: HLEAdapter, t, y, yerr, logspace, extra_frac,
              res_override: Optional[Dict[str,int]], targets_b1, targets_b2,
              tpos=None, tneg=None, timeerr_interval: bool = False, timeerr_n_samples: int = 7):
    """
    Build residuals for least_squares. Applies t0_obs during the fit.
    """
    p_lin = merge_params(free_names, u, fixed_map, specs_by_name)
    t0 = float(p_lin.get("t0_obs", 0.0))
    p_hle = map_to_hle_names(p_lin, specs_by_name)

    # Evaluate model:
    # - default: point-evaluation at (t - t0_obs)
    # - if --timeerr-interval and time errors exist: BIN-AVERAGED model over [t-tneg, t+tpos]
    if timeerr_interval and (tpos is not None) and (tneg is not None):
        y_model = model_binavg(adapter_fit, t, tpos, tneg, t0, p_hle, res_override, timeerr_n_samples)
    else:
        t_eval = np.asarray(t, float) - t0
        # Disallow non-positive interpolation times (log-space interpolation)
        t_eval = np.where(np.isfinite(t_eval) & (t_eval > 0), t_eval, np.nan)
        y_model = adapter_fit(t_eval, p_hle, res_override=res_override, force_n_mc_zero=True)

    # ---------------------------
    # Soft constraints on breaks
    # ---------------------------
    def _log_interp(xx, xp, fp):
        # log–log interpolation (x>0, y>0); returns NaN outside/support
        y = np.full_like(xx, np.nan, dtype=float)
        good = np.isfinite(xp) & np.isfinite(fp) & (xp > 0) & (fp > 0)
        if not np.any(good): return y
        xgood = np.isfinite(xx) & (xx > 0)
        if not np.any(xgood): return y
        y[xgood] = np.exp(np.interp(np.log(xx[xgood]), np.log(xp[good]), np.log(fp[good]),
                                    left=np.nan, right=np.nan))
        return y

    expected_extra = len(targets_b1) + len(targets_b2)

    r_extra = []

    if targets_b1 or targets_b2:
        # Get hle.compute_hle_lightcurve dynamically and adapter z
        hle_mod = importlib.import_module("hle_tophat")
        f_compute = getattr(hle_mod, "compute_hle_lightcurve")
        z_for_adapter = adapter_fit.z

        # Build kwargs for a fast model call (we need observed breaks vs t)
        kwargs = dict(p_hle)
        kwargs.update({"nu_min_keV": adapter_fit.emin,
                       "nu_max_keV": adapter_fit.emax,
                       "z": z_for_adapter})
        if res_override:
            for k in ("n_theta","n_phi","n_bins","n_nu","min_grid_counts"):
                if k in res_override and res_override[k] is not None:
                    kwargs[k] = int(res_override[k])
        kwargs["n_mc"] = 0

        # Call hle directly to get observed breaks
        ret = f_compute(**{k: v for k, v in kwargs.items()
                           if k in inspect.signature(f_compute).parameters})
        t_model = ret[0]
        Eb1_obs = ret[6] if len(ret) >= 7 else None
        Eb2_obs = ret[7] if len(ret) >= 8 else None

        # Targets are ALREADY "since first photon": NO time shift
        def _to_model_time(t_obs):
            return max(float(t_obs), 1e-12)

        # Build penalties in log10(E)
        for (tt, EE, frac) in targets_b1:
            if isinstance(Eb1_obs, np.ndarray):
                Em = _log_interp(np.array([_to_model_time(tt)]), t_model, Eb1_obs)[0]
                if np.isfinite(Em) and Em > 0:
                    sigma_log = np.log10(1.0 + float(frac))
                    r_extra.append( (np.log10(Em) - np.log10(EE)) / max(sigma_log, 1e-6) )
        for (tt, EE, frac) in targets_b2:
            if isinstance(Eb2_obs, np.ndarray):
                Em = _log_interp(np.array([_to_model_time(tt)]), t_model, Eb2_obs)[0]
                if np.isfinite(Em) and Em > 0:
                    sigma_log = np.log10(1.0 + float(frac))
                    r_extra.append( (np.log10(Em) - np.log10(EE)) / max(sigma_log, 1e-6) )

    # Force r_extra to be fixed-length every time
    r_extra = np.asarray(r_extra, dtype=float)
    if r_extra.size < expected_extra:
        r_extra = np.pad(r_extra, (0, expected_extra - r_extra.size), constant_values=0.0)
    elif r_extra.size > expected_extra:
        r_extra = r_extra[:expected_extra]

    # Build uncertainties
    sigma = np.sqrt(yerr**2 + (extra_frac * y)**2)

    if logspace:
        m = (y > 0) & (y_model > 0) & (sigma > 0)
        r = np.zeros_like(y, dtype=float)
        # Sigma in log10-space: σ_log = σ_lin / (y ln10), plus extra_frac and r_extra (from targets)
        r[m] = (np.log10(y_model[m]) - np.log10(y[m])) / (sigma[m] / (y[m] * np.log(10.0)))
        return np.concatenate([r, r_extra])

    else:
        m = np.isfinite(y_model) & np.isfinite(y) & (sigma > 0)
        r = np.zeros_like(y, dtype=float)
        r[m] = (y_model[m] - y[m]) / sigma[m]
        return np.concatenate([r, r_extra])

def run_fit(adapter_fit: HLEAdapter, t, y, yerr, specs, logspace, extra_frac,
            res_override: Optional[Dict[str,int]], targets_b1=None, targets_b2=None,
            tpos=None, tneg=None, timeerr_interval: bool = False, timeerr_n_samples: int = 7):
    specs_by_name = {s.name: s for s in specs}
    free_names, u0, bounds, fixed_map = assemble_param_vector(specs)
    targets_b1 = targets_b1 or []
    targets_b2 = targets_b2 or []

    fun = lambda u: residuals(u, free_names, fixed_map, specs_by_name, adapter_fit,
                              t, y, yerr, logspace, extra_frac, res_override, targets_b1, targets_b2,
                              tpos=tpos, tneg=tneg,
                              timeerr_interval=timeerr_interval, timeerr_n_samples=timeerr_n_samples)

    res = least_squares(fun, u0, bounds=bounds,
                        method="trf", jac="2-point",
                        loss="soft_l1", f_scale=1.0,
                        max_nfev=30000)
    u_best = res.x

    p_best_lin = merge_params(free_names, u_best, fixed_map, specs_by_name)
    p_best_hle = map_to_hle_names(p_best_lin, specs_by_name)

    # Model evaluated at DATA times
    t0_best = float(p_best_lin.get("t0_obs", 0.0))
    if timeerr_interval and (tpos is not None) and (tneg is not None):
        y_best_at_data = model_binavg(adapter_fit, t, tpos, tneg, t0_best, p_best_hle, res_override, timeerr_n_samples)
    else:
        y_best_at_data = adapter_fit(np.clip(t - t0_best, 1e-12, None),
                                     p_best_hle, res_override=res_override, force_n_mc_zero=True)


    # Statistics
    sigma = np.sqrt(yerr**2 + (extra_frac * y)**2)
    if logspace:
        m = (y > 0) & (y_best_at_data > 0) & (sigma > 0)
        chi2 = np.sum(((np.log10(y_best_at_data[m]) - np.log10(y[m])) / (sigma[m] / (y[m]*np.log(10.0))))**2)
        npts = int(np.sum(m))
    else:
        m = np.isfinite(y_best_at_data) & np.isfinite(y) & (sigma > 0)
        chi2 = np.sum(((y_best_at_data[m] - y[m]) / sigma[m])**2)
        npts = int(np.sum(m))
    nfree = len([s for s in specs if not s.fixed])
    dof = max(npts - nfree, 1)
    AIC = chi2 + 2*nfree
    BIC = chi2 + nfree*np.log(max(npts, 2))

    return {"p_best_lin": p_best_lin, "p_best_hle": p_best_hle,
            "chi2": chi2, "dof": dof, "redchi2": chi2/dof, "AIC": AIC, "BIC": BIC}, y_best_at_data

def _parse_time_mask_str(mask_str: str):
    import re, numpy as np
    if not mask_str:
        return []
    out = []
    for chunk in re.split(r"[,\s]+", mask_str.strip()):
        if not chunk:
            continue
        if ":" in chunk:
            a, b = chunk.split(":", 1)
            a = float(a) if a.strip() != "" else -np.inf
            b = float(b) if b.strip() != "" else  np.inf
        else:
            a = float(chunk)
            b = a
        if b < a:
            a, b = b, a
        out.append((a, b))
    return out

def tailor_bounds_around_best(
    specs,
    orig_bounds: dict,
    frac_lin: float = 0.30,   # ±30% for linear params
    dex_log: float = 0.30,    # ±0.30 dex (~×2) for log params
    angle_deg: float = 1.0,   # ±1 deg for names containing 'theta' or ending with '_deg'
    t0_rel: float = 0.30,     # ±30% for t0_obs
    t0_abs_floor: float = 5.0 # but at least ±5 s
):
    tiny_lin = 1e-12
    for s in specs:
        # Original physics bounds (assumed valid: lo0 < hi0)
        lo0, hi0 = orig_bounds.get(s.name, (s.min, s.max))
        # Fixed params: leave bounds exactly as they were
        if s.fixed:
            # Do not modify s.min/s.max
            continue
        v = float(s.value)
        # Propose local bounds around the best value
        if s.transform == "log":
            # If value invalid for log, fall back to original bounds
            if v <= 0:
                new_lo, new_hi = lo0, hi0
            else:
                factor = 10.0 ** dex_log
                new_lo, new_hi = v / factor, v * factor
        else:
            name = s.name.lower()
            if name == "t0_obs":
                span = max(t0_rel * abs(v), t0_abs_floor)
                new_lo, new_hi = v - span, v + span
                new_lo = max(new_lo, 0.0)  # time ≥ 0
            elif ("theta" in name) or name.endswith("_deg"):
                span = angle_deg
                new_lo, new_hi = v - span, v + span
                new_lo = max(new_lo, 0.0)  # angles ≥ 0
            else:
                span = frac_lin * (abs(v) if abs(v) > tiny_lin else 1.0)
                new_lo, new_hi = v - span, v + span
        # Intersect with original physics bounds
        lo = max(new_lo, lo0)
        hi = min(new_hi, hi0)
        # Guarantee strict hi > lo
        if not (lo < hi):
            if s.transform == "log":
                # Try a tiny multiplicative span around v within original bounds
                if v > 0:
                    eps = 1.05
                    cand_lo = max(min(v / eps, hi0), max(lo0, tiny_lin))
                    cand_hi = min(max(v * eps, lo0), hi0)
                    if cand_lo < cand_hi:
                        lo, hi = cand_lo, cand_hi
                    else:
                        # Last resort: revert to original bounds (assumed valid)
                        lo, hi = lo0, hi0
                else:
                    lo, hi = lo0, hi0
            else:
                # Try a tiny additive span around v within original bounds
                eps = max(1e-6 * (abs(v) + 1.0), tiny_lin)
                cand_lo = max(min(v - eps, hi0 - tiny_lin), lo0)
                cand_hi = min(max(v + eps, lo0 + tiny_lin), hi0)
                if cand_lo < cand_hi:
                    lo, hi = cand_lo, cand_hi
                else:
                    # Last resort: revert to original bounds
                    lo, hi = lo0, hi0
        # Assign the safe bounds
        s.min, s.max = float(lo), float(hi)

# =========
# Main CLI
# =========
def main():
    ap = argparse.ArgumentParser(description="Fit Swift/XRT light curves with HLE (fast-pass fit + correct t0,obs).")
    ap.add_argument("data", help="CSV/TSV/QDP table")
    ap.add_argument("params", help="YAML/JSON parameter file (list-of-dicts schema)")

    # Column names for CSV/TSV/QDP
    ap.add_argument("--col-t",   required=True, help="Time column (s since trigger)")
    ap.add_argument("--col-y",   required=True, help="Y column (flux or rate)")
    ap.add_argument("--col-err", default=None,  help="Symmetric Y error (ignored if --col-ypos/--col-yneg provided)")
    ap.add_argument("--col-ypos", default=None, help="Column name for +ve flux error (Fluxpos/Ratepos)")
    ap.add_argument("--col-yneg", default=None, help="Column name for -ve flux error (Fluxneg/Rateneg)")
    ap.add_argument("--col-tpos", default=None, help="Column name for +ve time error (T+ve)")
    ap.add_argument("--col-tneg", default=None, help="Column name for -ve time error (T-ve)")
    ap.add_argument("--err-sym", choices=["max","mean","pos","neg"], default="max",
                    help="How to symmetrize y-errors when ypos/yneg are provided (default: max)")
    ap.add_argument("--qdp-set", type=str, default="ALL", choices=["WT","PC","ALL"],
                    help="For QDP: select WT (first dataset), PC (second), or ALL.")

    # Units / band / redshift
    ap.add_argument("--units", choices=["flux","rate"], default="flux", help="Match your data units")
    ap.add_argument("--emin", type=float, default=0.3, help="Band min [keV] if band-integrated")
    ap.add_argument("--emax", type=float, default=10.0, help="Band max [keV] if band-integrated")
    ap.add_argument("--Emono", type=float, default=None, help="Monochromatic energy [keV] (overrides band if set)")
    ap.add_argument("--z", type=float, default=None, help="Redshift (overrides YAML if given)")

    # Time range and error treatment
    ap.add_argument("--tmin", type=float, default=None, help="Ignore data earlier than tmin [s]")
    ap.add_argument("--tmax", type=float, default=None, help="Ignore data later than tmax [s]")
    ap.add_argument("--mask", type=str, default="",
                        help="Comma/space-separated time ranges to EXCLUDE from the fit. Examples: --mask 80:130, --mask 95:105")
    ap.add_argument("--use-time-errors", action="store_true",
                    help="Inflate y-errors using |dy/dt|*dt from time errors (adds in quadrature).")

    ap.add_argument("--timeerr-interval", action="store_true",
                    help="Treat asymmetric time errors as a TIME BIN per point: "
                         "compare data to the BIN-AVERAGED model flux over [t-tneg, t+tpos].")

    ap.add_argument("--timeerr-n-samples", type=int, default=7,
                    help="Number of log-spaced samples inside each time-error interval (used with --timeerr-interval).")
    # Fit options
    ap.add_argument("--logspace", action="store_true", help="Fit in log10(y)")
    ap.add_argument("--extra-frac", type=float, default=0.0, help="Extra fractional systematics added in quadrature.")

    # Break targets
    ap.add_argument("--target-b1", action="append", default=[],
                    help='Observed E_b1 targets as "t,E[,frac]"; repeatable')
    ap.add_argument("--target-b2", action="append", default=[],
                    help='Observed E_b2 targets as "t,E[,frac]"; repeatable')
    
    # Plot options
    ap.add_argument("--plot-all", action="store_true",
               help="Plot all data points even if they were excluded from the fit window.")
    ap.add_argument("--mark-fit-window", action="store_true",
               help="Draw vertical lines at tmin/tmax on the plot.")

    # Time axis for plots
    ap.add_argument("--plot-time-relative", action="store_true",
                    help="Plot time as t_rel = t_abs - t0_obs (since first HLE photon). "
                         "Default is absolute time since trigger.")

    # Optional: overlay XRT photon index (spectral evolution) without fitting it
    ap.add_argument("--xrt-photon-index", default=None,
                    help="Path to XRT spectral_evolution.txt (for overlay of Γ(t); not used in fit).")

    # Extend model past last available XRT data point (for showing the top-hat drop)
    ap.add_argument("--model-tail-factor", type=float, default=2.0,
                    help="Extend the model plotting range to factor * t_max(data_available).")

    # Suppress the very-early numerical spike in the HLE model by starting the *plotted* model at t_rel >= tmin.
    # NOTE: this affects ONLY plotting/output model grids, not the fit residuals.
    ap.add_argument("--model-tmin-rel", type=float, default=0.0,
                    help="Do not plot the HLE model for t_rel < this value [s]. Useful to hide early numerical spikes.")


    # Fast-pass controls (applied during FIT ONLY; final model uses YAML/native resolution)
    ap.add_argument("--fast", action="store_true", help="Use a faster model resolution during the fit (recommended).")
    ap.add_argument("--fast-n-theta", type=int, default=96)
    ap.add_argument("--fast-n-phi",   type=int, default=96)
    ap.add_argument("--fast-n-bins",  type=int, default=80)
    ap.add_argument("--fast-n-nu",    type=int, default=8)
    ap.add_argument("--fast-min-grid",type=int, default=20)

    # Output
    ap.add_argument("--out", default="fit", help="Output prefix (files written with this stem)")
    args = ap.parse_args()

    def _parse_targets(raw_list):
        out = []
        for s in (raw_list or []):
            try:
                parts = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x != ""]
                if len(parts) == 2:
                    t, E = parts; frac = 0.2  # default 20% tolerance
                elif len(parts) >= 3:
                    t, E, frac = parts[:3]
                else:
                    raise ValueError
                if t <= 0 or E <= 0 or frac <= 0:
                    continue
                out.append((t, E, frac))
            except Exception:
                warnings.warn(f"Could not parse target '{s}'. Expected 't,E[,frac]'. Skipped.")
        return out

    targets_b1 = _parse_targets(args.target_b1)
    targets_b2 = _parse_targets(args.target_b2)

    if args.units == "rate":
        warnings.warn("Units 'rate' selected, but model returns flux-like units. Ensure a free normalization (e.g. map_to='i_p_prime').")

    # Load data
    t, y, yerr, tpos, tneg = read_table_auto(
        args.data, args.col_t, args.col_y, args.col_err,
        col_ypos=args.col_ypos, col_yneg=args.col_yneg,
        col_tpos=args.col_tpos, col_tneg=args.col_tneg,
        err_sym=args.err_sym, qdp_set=args.qdp_set
    )
    
    
    # Optional: XRT photon index data for overlay (NOT used in fit)
    xrt_pi = None
    if args.xrt_photon_index:
        try:
            t_pi_abs, tpi_pos, tpi_neg, g_pi, gpi_pos, gpi_neg = read_spectral_evolution_gamma(args.xrt_photon_index)
            xrt_pi = {
                "t_abs": t_pi_abs,
                "tpos": tpi_pos,
                "tneg": tpi_neg,
                "gamma": g_pi,
                "gpos": gpi_pos,
                "gneg": gpi_neg,
            }
        except Exception as e:
            warnings.warn(f"Failed to read --xrt-photon-index={args.xrt_photon_index}: {e}")
            xrt_pi = None

# Optional time-error inflation σ_time ≈ |dy/dt|·Δt (first-order)
    if args.use_time_errors and (tpos is not None) and (tneg is not None):
        tt = np.asarray(t, float)
        yy = np.asarray(y, float)
        def local_abs_slope(i):
            j0 = max(i - 1, 0)
            j1 = min(i + 1, len(tt) - 1)
            dt = tt[j1] - tt[j0]
            if dt <= 0: return 0.0
            return abs((yy[j1] - yy[j0]) / dt)
        slope = np.array([local_abs_slope(i) for i in range(len(tt))])
        dt_sym = np.maximum(np.asarray(tpos, float), np.asarray(tneg, float))
        yerr = np.sqrt(yerr**2 + (slope * dt_sym)**2)

    # Keep immutable RAW copies for masking and for plotting later
    t_raw = t.copy()
    y_raw = y.copy()
    yerr_raw = yerr.copy()
    tpos_raw  = None if tpos is None else tpos.copy()
    tneg_raw  = None if tneg is None else tneg.copy()

    # Masks on RAW Swift/XRT times
    m_valid = np.isfinite(t_raw) & np.isfinite(y_raw) & np.isfinite(yerr_raw)
    # Window
    m_window = np.ones_like(m_valid, dtype=bool)
    if args.tmin is not None:
        m_window &= (t_raw >= float(args.tmin))
    if args.tmax is not None:
        m_window &= (t_raw <= float(args.tmax))
    # User exclusion ranges
    m_maskranges = np.zeros_like(m_valid, dtype=bool)
    for (a, b) in _parse_time_mask_str(getattr(args, "mask", "")):
        m_maskranges |= (t_raw >= a) & (t_raw <= b)

    # Points actually used for FIT
    m_use = m_valid & m_window & (~m_maskranges)

    # Subset used for FIT from here on
    t    = t_raw[m_use]
    y    = y_raw[m_use]
    yerr = yerr_raw[m_use]

    # Subset time errors used for FIT (if present)
    tpos = None if tpos_raw is None else tpos_raw[m_use]
    tneg = None if tneg_raw is None else tneg_raw[m_use]

    if t.size == 0:
        raise RuntimeError("No data left after applying tmin/tmax and --mask on RAW time.")

    # Params
    specs = load_params(args.params)
    orig_bounds = {s.name: (s.min, s.max) for s in specs}


    # Override z if given on CLI and fix it
    if args.z is not None:
        for s in specs:
            if s.name == "z" or (s.map_to and s.map_to == "z"):
                s.value = float(args.z); s.fixed = True; break

    # Build adapters
    z_for_adapter = next((s.value for s in specs if s.name == "z" or (s.map_to and s.map_to=="z")), args.z if args.z is not None else 0.0)
    adapter_fit   = HLEAdapter(args.emin, args.emax, z_for_adapter, args.units, args.Emono)
    adapter_dense = HLEAdapter(args.emin, args.emax, z_for_adapter, args.units, args.Emono)

    # Fast resolution override used during the fit
    res_override = None
    if args.fast:
        res_override = {
            "n_theta": args.fast_n_theta,
            "n_phi": args.fast_n_phi,
            "n_bins": args.fast_n_bins,
            "n_nu": args.fast_n_nu,
            "min_grid_counts": max(5, args.fast_min_grid)
        }

    # Fit
    results, y_best_at_data = run_fit(
        adapter_fit, t, y, yerr, specs, args.logspace, args.extra_frac, res_override,
        targets_b1, targets_b2,
        tpos=tpos, tneg=tneg,
        timeerr_interval=args.timeerr_interval,
        timeerr_n_samples=args.timeerr_n_samples
    )

    # Update spec values with best-fit (in physical space)
    for s in specs:
        if not s.fixed and s.name in results["p_best_lin"]:
            s.value = float(results["p_best_lin"][s.name])

    # Best-fit

    # Dense model recompute at YAML resolution
    # Map best linear params to hle names, filter kwargs to avoid surprises
    specs_by_name = {s.name: s for s in specs}
    p_lin_best = {s.name: float(s.value) for s in specs}
    p_hle_best = map_to_hle_names(p_lin_best, specs_by_name)

    # ============
    # SAVE PARAMS
    # ============ 

    out = args.out
    
    # Best-fit params YAML
    tailor_bounds_around_best(specs, orig_bounds)
    save_params(f"{out}_best_params.yaml", specs)

    # =========
    # SUMMARY
    # =========

    nfree = len([s for s in specs if not s.fixed])
    npts  = results["dof"] + nfree

    def _fmt(v):
        try:
            return f"{float(v):.6g}"
        except Exception:
            return str(v)

    specs_by_name = {s.name: s for s in specs}
    p_lin_best = {s.name: float(s.value) for s in specs}

    spectrum_val = int(p_lin_best.get("spectrum", specs_by_name.get("spectrum", type("o",(object,),{"value":2})) .value))

    # Define allow-lists (comoving-only)
    COMMON = {
        "Gamma_c","theta_j_deg","g","k","theta_v_deg","R","z",
        "t0_obs",              # fitter-only (not forwarded to model kwargs)
        "epsilon_c",
    }

    # Spectrum-specific sets
    PL = {"beta_pl"}

    # SBPL (spectrum=1), comoving-only
    SBPL_comov  = {"alpha0", "beta0", "nu_b0", "q", "sbpl_s", "nu0_prime", "i_p_prime"}

    # 2SBPL (spectrum=2), comoving-only
    SBPL2_comov = {"alpha_lo", "alpha_mid", "alpha_hi",
                   "nu_b1_0", "nu_b2_0", "s1", "s2", "q1", "q2",
                   "nu0_prime", "i_p_prime"}

    # Choose spectral allow-list according to spectrum (no frame switch anymore)
    if spectrum_val == 0:
        SPECT = PL
    elif spectrum_val == 1:
        SPECT = SBPL_comov
    else:
        SPECT = SBPL2_comov

    # Final allow-list for this run (note: NO A_scale here)
    ALLOW = {
        "Gamma_c","theta_j_deg","g","k","theta_v_deg","R","z",
        "t0_obs","epsilon_c",
    } | set(SPECT)

    # Helper: decide if a ParamSpec should be printed
    def _is_relevant(s):
        # never show numerics / MC / grid controls in the summary
        NON_PHYSICAL_PREFIX = ("n_theta","n_phi","n_bins","n_nu","n_mc","min_grid_counts","fast")
        if s.name.startswith(NON_PHYSICAL_PREFIX):
            return False
        # show only parameters in the allow-list
        return s.name in ALLOW

    used_specs = [s for s in specs if _is_relevant(s)]
    used_specs_free  = [s for s in used_specs if not s.fixed]
    used_specs_fixed = [s for s in used_specs if s.fixed]

    lines = []
    lines.append("=== HLE fit summary ===")
    lines.append(f"Data file: {args.data}")
    lines.append(f"Units: {args.units}   Band: [{args.emin}, {args.emax}] keV   "
                 f"z: {_fmt(p_lin_best.get('z', ''))}")
    lines.append(f"Fit window: tmin={args.tmin if args.tmin is not None else 'None'}, "
                 f"tmax={args.tmax if args.tmax is not None else 'None'} (s since trigger)")
    lines.append(f"logspace: {args.logspace}   extra_frac: {args.extra_frac}")
    lines.append("")
    lines.append(f"chi2={_fmt(results['chi2'])}   dof={results['dof']}   "
                 f"redchi2={_fmt(results['redchi2'])}   AIC={_fmt(results['AIC'])}   BIC={_fmt(results['BIC'])}")
    lines.append(f"npts used={npts}   nfree={nfree}")
    lines.append("")
    lines.append(f"spectrum={spectrum_val}")
    if targets_b1 or targets_b2:
        lines.append("Observed-break soft targets:")
        for (tt, EE, frac) in targets_b1:
            lines.append(f"  E_b1(t={tt:g}s) ≈ {EE:g} keV  (±{100*frac:.0f}% in log-space)")
        for (tt, EE, frac) in targets_b2:
            lines.append(f"  E_b2(t={tt:g}s) ≈ {EE:g} keV  (±{100*frac:.0f}% in log-space)")
        lines.append("")
    if "t0_obs" in p_lin_best:
        lines.append(f"t0_obs: {_fmt(p_lin_best['t0_obs'])} s")
    lines.append("")

    lines.append("Fitted parameters:")
    if used_specs_free:
        for s in used_specs_free:
            lines.append(f"  {s.name:>16s} = {_fmt(s.value)}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Fixed parameters:")
    if used_specs_fixed:
        for s in used_specs_fixed:
            lines.append(f"  {s.name:>16s} = {_fmt(s.value)}")
    else:
        lines.append("  (none)")
    lines.append("")

    with open(f"{out}_summary.txt", "w", encoding="utf-8") as fsum:
        fsum.write("\n".join(lines))

    # ======================
    # BEST MODEL EVALUATION
    # ======================

    model_at_native = adapter_dense
    hle_mod = importlib.import_module("hle")
    f_compute = getattr(hle_mod, "compute_hle_lightcurve")
    allowed = set(inspect.signature(f_compute).parameters.keys())

    # Build kwargs for dense recompute at YAML resolution
    kwargs_dense = {
        **p_hle_best,
        "nu_min_keV": args.emin,
        "nu_max_keV": args.emax,
        "z": z_for_adapter
    }

    # Cast resolution-like parameters to int
    for k in ("n_theta", "n_phi", "n_bins", "n_nu", "n_mc", "min_grid_counts"):
        if k in kwargs_dense:
            kwargs_dense[k] = int(kwargs_dense[k])

    # Enforce safe spectral resolution for Γ(t)
    if ("n_nu" not in kwargs_dense) or (kwargs_dense["n_nu"] < 32):
        kwargs_dense["n_nu"] = 32

    # Keep only allowed arguments
    kwargs_dense = {k: v for k, v in kwargs_dense.items() if k in allowed}

    # Run the dense model
    ret_dense = f_compute(**kwargs_dense)

    # Unpack dense HLE results robustly (handles SBPL and 2SBPL)
    t_dense        = ret_dense[0]
    F_dense        = ret_dense[1]
    counts_dense   = ret_dense[2] if len(ret_dense) > 2 else None
    beta_eff_dense = ret_dense[3] if len(ret_dense) > 3 else None
    F_tnu_dense    = ret_dense[4] if len(ret_dense) > 4 else None
    nu_grid_dense  = ret_dense[5] if len(ret_dense) > 5 else None

    E_b1_dense = E_b2_dense = E_b1p_dense = E_b2p_dense = None
    tprime_eff_dense = None
    
    if spectrum_val == 2:
        # 2SBPL
        if len(ret_dense) >= 7: E_b1_dense  = ret_dense[6]
        if len(ret_dense) >= 8: E_b2_dense  = ret_dense[7]
        if len(ret_dense) >= 9: E_b1p_dense = ret_dense[8]
        if len(ret_dense) >= 10: E_b2p_dense = ret_dense[9]
        if len(ret_dense) >= 11: tprime_eff_dense = ret_dense[10]
    else:
        # PL or SBPL
        if len(ret_dense) >= 7: E_b1_dense  = ret_dense[6]
        if len(ret_dense) >= 8: E_b1p_dense = ret_dense[7]
        if len(ret_dense) >= 9: tprime_eff_dense = ret_dense[8]

    # Coerce scalars to arrays for plotting
    def _as_array(x, ref):
        if isinstance(x, (int, float)) and np.isfinite(x):
            return np.full_like(ref, float(x), dtype=float)
        return x
    E_b1_dense  = _as_array(E_b1_dense,  t_dense)
    E_b2_dense  = _as_array(E_b2_dense,  t_dense)
    E_b1p_dense = _as_array(E_b1p_dense, t_dense)
    E_b2p_dense = _as_array(E_b2p_dense, t_dense)
    tprime_eff_dense = _as_array(tprime_eff_dense, t_dense)

    # --- Robust photon index from photon spectrum near 1 keV ---
    # We work with N_E ∝ F_nu / (h * nu). In log–log, slope s_N = d ln N_E / d ln E = d ln(F_nu/nu)/d ln nu.
    # Photon index is then Gamma = - s_N (since N_E ∝ E^{-Gamma}).
    alpha_eff_dense = None
    gamma_dense = None

    if isinstance(F_tnu_dense, np.ndarray) and isinstance(nu_grid_dense, np.ndarray):
        nu = np.asarray(nu_grid_dense, float)             # [Hz]
        Fi = np.asarray(F_tnu_dense, float)               # F_nu(t, nu) from hle.py  (not nu*F_nu)
        T, K = Fi.shape

        # pivot at 1 keV (in Hz)
        nu_piv = (1.0 * 1e3) * getattr(hle_mod, "E_HZ_PER_eV", 2.418e14)
        j_piv = int(np.argmin(np.abs(nu - nu_piv)))

        halfwin = 2  # use 5 points around the pivot when available

        def gamma_from_NE(row):
            Frow = Fi[row, :]
            good = np.isfinite(Frow) & (Frow > 0) & np.isfinite(nu) & (nu > 0)
            if good.sum() < 2:
                return np.nan
            j0 = max(0, j_piv - halfwin)
            j1 = min(K, j_piv + halfwin + 1)
            idx = np.arange(j0, j1)
            idx = idx[good[idx]]
            if idx.size < 2:
                # fallback: widest contiguous good span
                idx = np.where(good)[0]
                if idx.size < 2:
                    return np.nan
            x = np.log(nu[idx])
            # ln N_E = ln F_nu - ln nu + const
            y = np.log(Frow[idx]) - np.log(nu[idx])
            if idx.size >= 3:
                sN = np.polyfit(x, y, 1)[0]
            else:
                sN = (y[-1] - y[0]) / (x[-1] - x[0])
            return -sN  # Gamma

        gamma_dense = np.array([gamma_from_NE(i) for i in range(T)], dtype=float)
        # keep α for CSV/diagnostics: α = 1 - Γ
        alpha_eff_dense = 1.0 - gamma_dense

    # -----------------------
    # Masking for plotting
    # -----------------------
    # Model arrays to be plotted
    model_t_p = t_dense
    model_f   = F_dense.copy()
    t0_best   = float(results["p_best_lin"].get("t0_obs", 0.0))

    # -----------------------
    # Choose plotting time axis
    # -----------------------
    # Data times are stored as absolute "since trigger" (t_raw, t). The HLE model lives in t_rel (since first photon).
    # For absolute-time plots we shift the model by +t0_obs; for relative-time plots we shift the DATA by -t0_obs.
    plot_relative = bool(getattr(args, "plot_time_relative", False))
    if plot_relative:
        # Plot everything in t_rel
        t_fit_plot = (t - t0_best)
        t_all_plot = (t_raw - t0_best)
        model_t_plot = np.asarray(t_dense, float)
        xlab = r"$t_{\rm rel} = t_{\rm abs}-t_{0,\rm obs}$ [s]"
    else:
        # Plot everything in absolute time since trigger
        t_fit_plot = np.asarray(t, float)
        t_all_plot = np.asarray(t_raw, float)
        model_t_plot = np.asarray(t_dense, float) + t0_best
        xlab = r"$t_{\rm abs}$ since trigger [s]"

    # Fitted points: exactly the subset used in the fit (already windowed via RAW-time masks)
    m_fit_plot = (np.isfinite(t_fit_plot) & (t_fit_plot > 0) &
                  np.isfinite(y) & np.isfinite(yerr) & (y > 0))
    t_plot     = t_fit_plot[m_fit_plot]
    y_plot     = y[m_fit_plot]
    yerr_plot  = yerr[m_fit_plot]

    # ALL observed data for plotting (derive from RAW arrays)
    m_all_plot = (np.isfinite(t_all_plot) & (t_all_plot > 0) &
                  np.isfinite(y_raw) & (y_raw > 0) & np.isfinite(yerr_raw))
    # Everything valid but NOT used for the fit
    m_excluded_plot = m_all_plot & (~m_use)

    # Optional XRT photon index data for overlay
    # We split points into "used" vs "excluded" using the SAME tmin/tmax/mask logic as for the LC,
    # but applied to the photon-index time stamps (which are in absolute time since trigger).
    t_pi_plot = Gamma_pi = gpi_pos_plot = gpi_neg_plot = None
    t_pi_plot_excl = Gamma_pi_excl = gpi_pos_plot_excl = gpi_neg_plot_excl = None
    if xrt_pi is not None:
        tt_abs = np.asarray(xrt_pi["t_abs"], float)
        gg = np.asarray(xrt_pi["gamma"], float)
        gp = np.asarray(xrt_pi["gpos"], float)
        gn = np.asarray(xrt_pi["gneg"], float)

        m_pi_valid = np.isfinite(tt_abs) & (tt_abs > 0) & np.isfinite(gg) & np.isfinite(gp) & np.isfinite(gn)

        # Apply the same selection logic (in ABSOLUTE time) as for the LC
        m_pi_window = np.ones_like(m_pi_valid, dtype=bool)
        if args.tmin is not None:
            m_pi_window &= (tt_abs >= float(args.tmin))
        if args.tmax is not None:
            m_pi_window &= (tt_abs <= float(args.tmax))

        m_pi_maskranges = np.zeros_like(m_pi_valid, dtype=bool)
        for (a, b) in _parse_time_mask_str(getattr(args, "mask", "")):
            m_pi_maskranges |= (tt_abs >= a) & (tt_abs <= b)

        m_pi_use = m_pi_valid & m_pi_window & (~m_pi_maskranges)
        m_pi_excl = m_pi_valid & (~m_pi_use)

        # Convert to plot-time coordinate
        if plot_relative:
            tt_plot = tt_abs - t0_best
        else:
            tt_plot = tt_abs

        # Used points (red)
        m_use_plot = m_pi_use & np.isfinite(tt_plot) & (tt_plot > 0)
        if np.any(m_use_plot):
            t_pi_plot = tt_plot[m_use_plot]
            Gamma_pi = gg[m_use_plot]
            gpi_pos_plot = gp[m_use_plot]
            gpi_neg_plot = gn[m_use_plot]

        # Excluded points (grey) — only plotted when --plot-all is enabled
        m_excl_plot = m_pi_excl & np.isfinite(tt_plot) & (tt_plot > 0)
        if np.any(m_excl_plot):
            t_pi_plot_excl = tt_plot[m_excl_plot]
            Gamma_pi_excl = gg[m_excl_plot]
            gpi_pos_plot_excl = gp[m_excl_plot]
            gpi_neg_plot_excl = gn[m_excl_plot]

    # ---------------------------------------------------
    # Select model plotting domain (extend beyond data end)
    # ---------------------------------------------------
    # Work in the chosen plotting time axis (absolute or relative)
    model_t_p = np.asarray(model_t_plot, float)
    model_t_rel_full = np.asarray(t_dense, float)  # model-native time axis (since first HLE photon)
    model_f   = F_dense.copy()

    # Data span for plotting (use ALL available valid points, not only fit-used)
    if np.any(m_all_plot):
        tmin_data = float(np.nanmin(t_all_plot[m_all_plot]))
        tmax_data = float(np.nanmax(t_all_plot[m_all_plot]))
    else:
        # Fallback to fit-used points only
        tmin_data = float(np.nanmin(t_plot)) if t_plot.size else 0.0
        tmax_data = float(np.nanmax(t_plot)) if t_plot.size else np.inf

    # Extend model end-time to show the top-hat drop (default factor = 2)
    tail_fac = float(getattr(args, "model_tail_factor", 2.0))
    if (np.isfinite(tmax_data) and tmax_data > 0 and np.isfinite(tail_fac) and tail_fac > 1.0):
        tmax_plot = tmax_data * tail_fac
    else:
        tmax_plot = tmax_data

    # Do not clip the model earlier than its own start
    pos_model = model_t_p[np.isfinite(model_t_p) & (model_t_p > 0)]
    if pos_model.size:
        tmin_plot = min(tmin_data, float(np.nanmin(pos_model)))
    else:
        tmin_plot = tmin_data

    # Optional early-time clip: hide numerical spike by starting the plotted model at t_rel >= tmin
    tmin_rel_plot = float(getattr(args, "model_tmin_rel", 0.0))
    if not np.isfinite(tmin_rel_plot) or tmin_rel_plot < 0:
        tmin_rel_plot = 0.0


    # ---------------------------------------------------
    # If the top-hat emission turns exactly to zero, the log-y plot cannot show it.
    # To make the shutdown visually evident as a near-vertical drop, we replace ONLY
    # the first non-positive model flux after the last positive value with a tiny
    # positive number (in model-native time). This is a plotting-only tweak.
    # ---------------------------------------------------
    try:
        fpos_all = model_f[np.isfinite(model_f) & (model_f > 0)]
        if fpos_all.size:
            f_eps = float(np.nanmin(fpos_all)) * 1e-6
            if np.isfinite(f_eps) and f_eps > 0:
                # Find the first index where flux becomes non-positive within the plotting window
                in_win = (np.isfinite(model_t_p) & (model_t_p > 0) &
                          np.isfinite(model_t_rel_full) & (model_t_rel_full >= tmin_rel_plot) &
                          (model_t_p >= tmin_plot) & (model_t_p <= tmax_plot))
                # transition: previous >0, current <=0
                m_prev = np.r_[False, (model_f[:-1] > 0)]
                m_curr = (model_f <= 0)
                cand = np.where(in_win & m_prev & m_curr)[0]
                if cand.size:
                    model_f[int(cand[0])] = f_eps
    except Exception:
        pass

    m_model = (np.isfinite(model_t_p) & (model_t_p > 0) &
               np.isfinite(model_f) & (model_f > 0) &
               np.isfinite(model_t_rel_full) & (model_t_rel_full >= tmin_rel_plot) &
               (model_t_p >= tmin_plot) & (model_t_p <= tmax_plot))
# Observer-frame breaks to plot (use same mask as model)
    E_b1_plot = None if not isinstance(E_b1_dense, np.ndarray) else E_b1_dense[m_model]
    E_b2_plot = None if not isinstance(E_b2_dense, np.ndarray) else E_b2_dense[m_model]
    if E_b1_plot is not None and (not np.any(np.isfinite(E_b1_plot))): E_b1_plot = E_b1_dense
    if E_b2_plot is not None and (not np.any(np.isfinite(E_b2_plot))): E_b2_plot = E_b2_dense

        # Keep model times in both conventions for outputs
    model_t_rel = np.asarray(t_dense, float)[m_model]
    model_t_abs = model_t_rel + t0_best
# Mask model arrays
    model_t_p = model_t_p[m_model]
    model_f   = model_f[m_model]

    # Photon index masked to the same plotting domain
        # Photon index masked to the same plotting domain
    Gamma_t, t_gamma = None, None
    if isinstance(gamma_dense, np.ndarray):
        gmask = m_model & np.isfinite(gamma_dense)
        t_gamma_rel = np.asarray(t_dense, float)[gmask]
        Gamma_t = np.asarray(gamma_dense, float)[gmask]
        # Plot-time coordinate
        t_gamma = (t_gamma_rel if plot_relative else (t_gamma_rel + t0_best))

        # Save Γ CSV with both time conventions
        alpha_eff_out = (alpha_eff_dense[gmask] if isinstance(alpha_eff_dense, np.ndarray)
                         else np.full_like(Gamma_t, np.nan))
        t_abs_out = t_gamma_rel + t0_best
        np.savetxt(
            f"{out}_gamma_model.csv",
            np.column_stack([t_abs_out, t_gamma_rel, Gamma_t, alpha_eff_out]),
            delimiter=",",
            header="t_abs_s,t_rel_s,Gamma_model,beta_loc",
            comments=""
        )
# Model evaluated at DATA times (for the fit-at-data CSV)
    y_best_at_data_plot = y_best_at_data[m_fit_plot]

    # ======
    # CSVs
    # ======

    out = args.out
        # ======
    # CSVs (always save BOTH time conventions for downstream combined plots)
    # ======

    out = args.out

    # Data used in fit
    t_abs_used = np.asarray(t, float)[m_fit_plot]
    t_rel_used = t_abs_used - t0_best
    np.savetxt(
        f"{out}_data.csv",
        np.column_stack([t_abs_used, t_rel_used, y_plot, yerr_plot]),
        delimiter=",",
        header="t_abs_s,t_rel_s,flux,flux_err",
        comments=""
    )

    # Full model curve on dense grid
    np.savetxt(
        f"{out}_model.csv",
        np.column_stack([model_t_abs, model_t_rel, model_f]),
        delimiter=",",
        header="t_abs_s,t_rel_s,flux_model",
        comments=""
    )

    # Model evaluated at the data times (fast-fit resolution)
    y_best_at_data_plot = y_best_at_data[m_fit_plot]
    np.savetxt(
        f"{out}_fit_at_data.csv",
        np.column_stack([t_abs_used, t_rel_used, y_plot, yerr_plot, y_best_at_data_plot]),
        delimiter=",",
        header="t_abs_s,t_rel_s,flux,flux_err,flux_model_at_data_times (fast-fit resolution)",
        comments=""
    )

    # Optional: XRT photon-index data used for overlay (not fitted)
    if (t_pi_plot is not None) and (Gamma_pi is not None):
        t_pi_abs_out = np.asarray(xrt_pi["t_abs"], float)
        m_pi_out = np.isfinite(t_pi_abs_out) & np.isfinite(xrt_pi["gamma"]) & (t_pi_abs_out > 0)
        if np.any(m_pi_out):
            np.savetxt(
                f"{out}_gamma_xrt_data.csv",
                np.column_stack([
                    t_pi_abs_out[m_pi_out],
                    (t_pi_abs_out[m_pi_out] - t0_best),
                    np.asarray(xrt_pi["gamma"], float)[m_pi_out],
                    np.asarray(xrt_pi["gpos"], float)[m_pi_out],
                    np.asarray(xrt_pi["gneg"], float)[m_pi_out],
                ]),
                delimiter=",",
                header="t_abs_s,t_rel_s,Gamma_xrt,Gamma_poserr,Gamma_negerr",
                comments=""
            )
# Observed breaks (masked to the same plotting window)
    if (E_b1_plot is not None) or (E_b2_plot is not None):
        e1 = E_b1_plot if E_b1_plot is not None else np.full_like(model_t_p, np.nan)
        e2 = E_b2_plot if E_b2_plot is not None else np.full_like(model_t_p, np.nan)
        np.savetxt(
            f"{out}_breaks_observed.csv",
            np.column_stack([model_t_abs, model_t_rel, e1, e2]),
            delimiter=",",
            header="t_abs_s,t_rel_s,E_b1_keV,E_b2_keV",
            comments=""
        )

    # Comoving (intrinsic) breaks — save unmasked (full dense grid) if available
    if (E_b1p_dense is not None) or (E_b2p_dense is not None):
        e1p = E_b1p_dense if isinstance(E_b1p_dense, np.ndarray) else np.full_like(t_dense, np.nan)
        e2p = E_b2p_dense if isinstance(E_b2p_dense, np.ndarray) else np.full_like(t_dense, np.nan)
        np.savetxt(
            f"{out}_breaks_comoving.csv",
            np.column_stack([t_dense, e1p, e2p]),
            delimiter=",",
            header="t_obs_s,Eb1_prime_keV,Eb2_prime_keV",
            comments=""
        )

    if (tprime_eff_dense is not None) and ( (E_b1p_dense is not None) or (E_b2p_dense is not None) ):
        e1p = E_b1p_dense if isinstance(E_b1p_dense, np.ndarray) else np.full_like(tprime_eff_dense, np.nan)
        e2p = E_b2p_dense if isinstance(E_b2p_dense, np.ndarray) else np.full_like(tprime_eff_dense, np.nan)
        np.savetxt(
            f"{out}_breaks_comoving_vs_tprime.csv",
            np.column_stack([tprime_eff_dense, e1p, e2p]),
            delimiter=",",
            header="tprime_eff_s,Eb1_prime_keV,Eb2_prime_keV",
            comments=""
        )


    # ===============================================================
    # PLOTTING SECTION
    # ===============================================================
    import matplotlib.pyplot as plt

    def _asymm_xerr_for_logx(x, xneg, xpos, xmin=1e-12):
        """
        Build asymmetric x-errors for matplotlib.errorbar on a log-x axis.
        Ensures (x - xneg) stays > 0 by clipping the negative side.
        """
        x = np.asarray(x, float)
        xneg = np.asarray(xneg, float)
        xpos = np.asarray(xpos, float)
        
        # Replace non-finite/negative with 0
        xneg = np.where(np.isfinite(xneg) & (xneg > 0), xneg, 0.0)
        xpos = np.where(np.isfinite(xpos) & (xpos > 0), xpos, 0.0)

        # Clip negative error so that the left end remains > xmin (log-safe)
        xneg = np.minimum(xneg, np.maximum(x - xmin, 0.0))

        return np.vstack([xneg, xpos])

    # -------- Figure 1: Light curve + Photon Index  --------
    fig1, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.5, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08}
    )

    # --- Top: light curve (data + model) ---
    # Excluded points (grey), if requested
    if args.plot_all and np.any(m_excluded_plot):
        xerr_excl = None
        if (tpos_raw is not None) and (tneg_raw is not None):
            xerr_excl = _asymm_xerr_for_logx(
                t_all_plot[m_excluded_plot],
                tneg_raw[m_excluded_plot],
                tpos_raw[m_excluded_plot]
            )

        ax1.errorbar(
            t_all_plot[m_excluded_plot], y_raw[m_excluded_plot],
            xerr=xerr_excl,
            yerr=yerr_raw[m_excluded_plot] if 'yerr_raw' in locals() else None,
            fmt='o', ms=3, mfc='none', mec='0.5', ecolor='0.8', color='0.7',
            alpha=0.6, label='Excluded'
        )

    # Points used in the fit
    if t_plot.size:
        xerr_fit = None
        if (tpos is not None) and (tneg is not None):
            # tpos/tneg are already subset to m_use; now subset again to m_fit_plot
            tpos_plot = tpos[m_fit_plot]
            tneg_plot = tneg[m_fit_plot]
            xerr_fit = _asymm_xerr_for_logx(t_plot, tneg_plot, tpos_plot)

        ax1.errorbar(
            t_plot, y_plot,
            xerr=xerr_fit,
            yerr=yerr_plot if 'yerr_plot' in locals() else None,
            fmt='o', ms=3, alpha=0.85, mfc='none', mec='tab:red', ecolor='tab:red', color='tab:red', label='Data (used)'
        )

    # Model curve
    ax1.plot(model_t_p, model_f, lw=2.2, color='tab:orange', label='Best HLE model')

    # Fit window markers
    if args.mark_fit_window:
        if args.tmin is not None: ax1.axvline(args.tmin, ls='--', lw=1.0, color='0.6')
        if args.tmax is not None: ax1.axvline(args.tmax, ls='--', lw=1.0, color='0.6')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    # Set a physically correct y-label depending on band-integrated vs monochromatic
    is_mono = (args.Emono is not None)

    if args.units == "flux":
        if is_mono:
            # Monochromatic: F_nu (flux density)
            ax1.set_ylabel(r"Flux density $F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")
        else:
            # Band-integrated: F_band over [emin, emax]
            ax1.set_ylabel(r"Band-integrated flux $F_{\rm band}$ [erg cm$^{-2}$ s$^{-1}$]")
    elif args.units == "rate":
        # Data provided as (count) rate; keep neutral and unit-consistent
        ax1.set_ylabel(r"Count rate [s$^{-1}$]")
    else:
        # Fallback (should not happen with current CLI choices)
        ax1.set_ylabel("Flux")

    ax1.set_xlim(0.4, 300000)
    ax1.set_ylim(1e-13, 1e-6)
    ax1.grid(True, which="both", ls="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_title("Light curve fit and photon index evolution", fontsize=12, pad=6)

    # --- Bottom: photon index (local slope at 1 keV) ---
    if (t_gamma is not None) and (Gamma_t is not None) and t_gamma.size:
        ax2.semilogx(
            t_gamma, Gamma_t,
            lw=2.0, color='tab:orange',
            label=r"$\Gamma(t)$ (local photon index at $1\,\mathrm{keV}$)"
        )
 
        # Overlay XRT photon index data (not fitted)
        # Excluded points (grey) to match the LC behaviour
        if args.plot_all and (t_pi_plot_excl is not None) and (Gamma_pi_excl is not None):
            ax2.errorbar(
                t_pi_plot_excl, Gamma_pi_excl,
                yerr=np.vstack([gpi_neg_plot_excl, gpi_pos_plot_excl]) if (gpi_pos_plot_excl is not None and gpi_neg_plot_excl is not None) else None,
                fmt='o', ms=3, mfc='none', mec='0.5', ecolor='0.8', color='0.7',
                alpha=0.6
            )

        # Points used in the fit window/mask (red)
        if (t_pi_plot is not None) and (Gamma_pi is not None):
            ax2.errorbar(
                t_pi_plot, Gamma_pi,
                yerr=np.vstack([gpi_neg_plot, gpi_pos_plot]) if (gpi_pos_plot is not None and gpi_neg_plot is not None) else None,
                fmt='o', ms=3, mfc='none', mec='tab:red', ecolor='tab:red', color='tab:red',
                alpha=0.85
            )
    elif isinstance(beta_eff_dense, np.ndarray):
        ax2.semilogx(
            t_dense, 1.0 - beta_eff_dense,
            lw=2.0, color='tab:gray', ls='--', alpha=0.6,
            label=r"Fallback: $\Gamma(t)=1-\beta_{\rm eff}$ (band fit)"
        )

    ax2.set_xlabel(xlab)
    ax2.set_ylabel(r"Photon index $\Gamma$")
#    ax2.set_ylim(0.0, 3.0)
    ax2.grid(True, which="both", ls="--", alpha=0.35)

    fig1.tight_layout()
    fig1.savefig(f"{out}_LC_PI.png", dpi=180)
    plt.close(fig1)

    # -------- Figure 2: Observed breaks vs observer time --------
    fig_obs, axo = plt.subplots(figsize=(7.5, 4.8))
    
    # Shaded XRT band 0.3-10 keV (y-axis is energy)
    try:
        axo.axhspan(0.3, 10.0, color="gray", alpha=0.15, zorder=0, label="XRT band 0.3–10.0 keV")
    except Exception:
        pass

    # Observed breaks (vs observer time)
    if E_b1_plot is not None:
        axo.plot(model_t_p, E_b1_plot, color='tab:green', lw=2.0, label=r"$E^{\rm obs}_{b,1}$")
    if E_b2_plot is not None:
        axo.plot(model_t_p, E_b2_plot, color='tab:blue', lw=2.0, label=r"$E^{\rm obs}_{b,2}$")

    axo.set_xscale("log")
    axo.set_yscale("log")
    axo.set_xlabel(xlab)
    axo.set_ylabel("Observed break energy [keV]")
    axo.set_title("Observed break evolution")
    axo.set_xlim(0.3, None)
    axo.grid(True, which="both", ls=":", alpha=0.35)
    axo.legend(loc="best", frameon=True)

    fig_obs.tight_layout()
    fig_obs.savefig(f"{out}_breaks_observed.png", dpi=220)
    plt.close(fig_obs)

    # -------- Figure 3: Intrinsic (comoving) breaks vs comoving time --------
    fig_comov, axc = plt.subplots(figsize=(7.5, 4.8))
    
    # Comoving breaks (vs comoving time)
    if E_b1p_dense is not None and tprime_eff_dense is not None:
        axc.plot(tprime_eff_dense, E_b1p_dense, color='tab:green', lw=2.0, label=r"$E'_{b,1}$")
    if E_b2p_dense is not None and tprime_eff_dense is not None:
        axc.plot(tprime_eff_dense, E_b2p_dense, color='tab:blue', lw=2.0, label=r"$E'_{b,2}$")

    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlabel("Comoving time t′ [s]")
    axc.set_ylabel("Comoving break energy [keV′]")
    axc.set_title("Intrinsic comoving break evolution")
    axc.set_xlim(10.0, None)
    axc.grid(True, which="both", ls=":", alpha=0.35)
    axc.legend(loc="best", frameon=True)

    fig_comov.tight_layout()
    fig_comov.savefig(f"{out}_breaks_comoving.png", dpi=220)
    plt.close(fig_comov)

    print(f"Done. Wrote files with prefix '{out}'.")    

if __name__ == "__main__":
    main()
