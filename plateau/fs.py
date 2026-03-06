"""fs.py

Forward-shock afterglow emission from a *structured* power-law jet,
computed via full Equal Arrival Time Surface (EATS) integration.

The dynamics follow a Blandford-McKee–type relativistic blast wave
for each angular patch of a power-law structured jet, as in:

- Granot & Gill 2018, MNRAS, 481, 1597
- Beniamini, Granot & Gill 2020, MNRAS, 493, 3521

Radiation is computed with standard synchrotron microphysics:
gamma_m, gamma_c, nu'_m, nu'_c, B' and a broken power-law spectrum
following Sari, Piran & Narayan 1998. The observer-frame light curve
is obtained by integrating the comoving emissivity over the EATS in
(R, θ, φ).

The code is optimized for optically thin X-ray afterglows in the
relativistic regime (no self-absorption, no SSC cooling, no explicit
non-relativistic transition), but can be used for any narrow observer
band where these approximations hold.
"""

import numpy as np

# Physical constants [cgs]
c = 2.99792458e10       # Speed of light
H0_kms_Mpc = 67.7       # Hubble constant
Omega_m = 0.31          # Matter density parameter
E_HZ_PER_eV = 2.418e14  # 1 eV in Hz

D_MAX = 1e6
EPS_DEN = 1e-15

# Microphysical constants [cgs]
q_e = 4.8032068e-10         # Elementary charge
m_e = 9.10938356e-28        # Electron mass
m_p = 1.6726219e-24         # Proton mass
sigma_T = 6.6524587158e-25  # Thomson cross section
pi = np.pi

# Synchrotron normalization (Sari+98 order-of-magnitude)
# j'_{nu,max} ~ (sqrt(3) q_e^3 / (m_e c^2)) * n' B'
K_SYN = np.sqrt(3.0) * q_e**3 / (m_e * c**2)

# ----------------------------------------------------------------------
# Cosmology
# ----------------------------------------------------------------------
def luminosity_distance_flatlcdm(z, H0_kms_Mpc=H0_kms_Mpc, Omega_m=Omega_m):
    """
    Simple flat ΛCDM luminosity distance [cm].
    """
    if z <= 0.0:
        return 0.0

    MPC_CM = 3.085677581e24
    H0_s = H0_kms_Mpc / (3.085677581e19)  # km/s/Mpc -> 1/s

    zs = np.linspace(0.0, z, 4000)
    Ez = np.sqrt(Omega_m * (1.0 + zs) ** 3 + (1.0 - Omega_m))
    Dc = (c / H0_s) * np.trapz(1.0 / Ez, zs)  # comoving distance

    return (1.0 + z) * Dc  # luminosity distance

# ----------------------------------------------------------------------
# Microphysics: standard FS synchrotron (GG18-like)
# ----------------------------------------------------------------------
def synch_nu_prime(B_prime, gamma_e):
    """
    Characteristic synchrotron frequency in the comoving frame for a
    single electron with Lorentz factor gamma_e and magnetic field B'.

    Uses the standard approximation:
        nu'_syn ≈ (3 / 4π) * (q_e B' / (m_e c)) * gamma_e^2

    To avoid numerical overflows at extremely early/late times (where
    the contribution to the observed flux is negligible), we clip the
    electron Lorentz factor to a reasonable maximum value.
    """
    gamma_e = np.asarray(gamma_e, dtype=float)
    gamma_e = np.clip(gamma_e, 1.0, 1e8)  # prevent overflow in gamma_e**2

    return (3.0 / (4.0 * pi)) * (q_e * B_prime / (m_e * c)) * gamma_e**2

def microphysics_fs(G_loc, R, t_lab, k, A, p, eps_e, eps_B):
    """
    Compute local forward–shock microphysical quantities in the comoving
    frame, following the standard afterglow prescriptions (as in
    Sari+98 / Granot & Sari / Granot & Gill 2018):

        - External density: n_ext(R) = A R^{-k} / m_p
        - Post–shock density: n' ≈ 4 Γ n_ext
        - Internal energy: e' = (Γ-1) n' m_p c^2
        - Magnetic field: B'^2 / (8π) = eps_B e'
        - Electron distribution: power law with index p
          and minimum Lorentz factor
              gamma_m = [(p-1)/(p-2)] * eps_e * (m_p/m_e) * (Γ-1)
        - Cooling Lorentz factor (synchrotron cooling on t_lab):
              gamma_c = 6π m_e c / (σ_T B'^2 Γ t_lab)

    We then build the characteristic synchrotron frequencies:

        nu'_m = nu_syn'(B', gamma_m)
        nu'_c = nu_syn'(B', gamma_c)

    and a simple scaling for the peak specific power:

        P'_{nu,max} ∝ n' B'

    The overall numerical normalization is not important here, since the
    code is used to compute *relative* light-curve shapes; it can be
    absorbed into an overall flux normalization.

    Parameters
    ----------
    G_loc : array
        Local bulk Lorentz factor Γ(R,θ,φ).
    R : float
        Radius [cm].
    t_lab : array
        Lab-frame time at this radius for each angular cell [s].
    k : float
        External density power-law index (n_ext ∝ R^{-k}).
    A : float
        External density normalization.
    p : float
        Electron power-law index.
    eps_e : float
        Fraction of internal energy in electrons.
    eps_B : float
        Fraction of internal energy in magnetic fields.

    Returns
    -------
    nu_m_prime : array
        Synchrotron injection break frequency ν'_m [Hz].
    nu_c_prime : array
        Synchrotron cooling break frequency ν'_c [Hz].
    Pnu_max_prime : array
        Peak comoving specific power per unit volume [arb. units].
    n_prime : array
        Comoving post-shock number density n' [cm^{-3}].
    B_prime : array
        Comoving magnetic field B' [Gauss].
    """
    # External density and post-shock quantities
    rho_ext = A * R**(-k)                   # mass density [g cm^{-3}]
    n_ext = rho_ext / m_p                   # number density [cm^{-3}]
    n_prime = 4.0 * G_loc * n_ext           # post-shock comoving density
    e_prime = (G_loc - 1.0) * n_prime * m_p * c**2  # internal energy density

    # Magnetic field
    B_prime = np.sqrt(8.0 * pi * eps_B * e_prime + 1e-200)

    # Electron Lorentz factors
    # Guard against p <= 2 (unphysical in this simple prescription)
    p_eff = max(float(p), 2.01)
    gamma_m = ((p_eff - 1.0) / (p_eff - 2.0)) * eps_e * (m_p / m_e) * (G_loc - 1.0)
    gamma_m = np.maximum(gamma_m, 1.0)

    t_lab_safe = np.maximum(t_lab, 1e-30)

    # Comoving time: t' ≈ t_lab / Γ
    t_comoving = t_lab_safe / np.maximum(G_loc, 1.0)

    gamma_c = 6.0 * pi * m_e * c / (sigma_T * B_prime**2 * t_comoving)
    gamma_c = np.maximum(gamma_c, 1.0)

    # Break frequencies in the comoving frame
    nu_m_prime = synch_nu_prime(B_prime, gamma_m)
    nu_c_prime = synch_nu_prime(B_prime, gamma_c)

    # With physical units we include the synchrotron normalization constant:
    # j'_{nu,max} ≈ K_SYN * n' * B'  [erg s^-1 cm^-3 Hz^-1 sr^-1]
    Pnu_max_prime = K_SYN * n_prime * B_prime

    return nu_m_prime, nu_c_prime, Pnu_max_prime, n_prime, B_prime


def synch_spectrum_pl_segment(nu_prime, nu_m_prime, nu_c_prime, p):
    """Dimensionless optically-thin synchrotron spectrum shape.

    Returns F_nu / F_{nu,max} with correct continuity at nu_m and nu_c,
    using the standard Sari, Piran & Narayan (1998) broken power-law.

    Conventions
    -----------
    - Slow cooling (nu_m < nu_c): peak at nu_m
      * nu < nu_m:         F_nu ∝ nu^{1/3}
      * nu_m <= nu < nu_c: F_nu ∝ nu^{-(p-1)/2}
      * nu >= nu_c:        F_nu ∝ nu^{-p/2}

    - Fast cooling (nu_c < nu_m): peak at nu_c
      * nu < nu_c:         F_nu ∝ nu^{1/3}
      * nu_c <= nu < nu_m: F_nu ∝ nu^{-1/2}
      * nu >= nu_m:        F_nu ∝ nu^{-p/2}

    Notes
    -----
    This function returns only the *shape*; normalization is handled by
    P'_{nu,max} elsewhere.
    """

    nu = np.maximum(np.asarray(nu_prime, dtype=float), 1e-40)
    nu_m = np.maximum(np.asarray(nu_m_prime, dtype=float), 1e-40)
    nu_c = np.maximum(np.asarray(nu_c_prime, dtype=float), 1e-40)

    shape = np.zeros_like(nu, dtype=float)

    slow = nu_m <= nu_c
    fast = ~slow

    # ---- slow cooling: nu_m < nu_c ----
    if np.any(slow):
        ns = nu[slow]; nm = nu_m[slow]; nc = nu_c[slow]

        m1 = ns < nm
        m2 = (ns >= nm) & (ns < nc)
        m3 = ns >= nc

        shape_s = np.empty_like(ns)
        # nu < nu_m: anchored at nu_m
        shape_s[m1] = (ns[m1] / nm[m1])**(1.0 / 3.0)
        # nu_m <= nu < nu_c
        shape_s[m2] = (ns[m2] / nm[m2])**(-(p - 1.0) / 2.0)
        # nu >= nu_c: enforce continuity at nu_c
        shape_s[m3] = (nc[m3] / nm[m3])**(-(p - 1.0) / 2.0) * (ns[m3] / nc[m3])**(-p / 2.0)

        shape[slow] = shape_s

    # ---- fast cooling: nu_c < nu_m ----
    if np.any(fast):
        nf = nu[fast]; nm = nu_m[fast]; nc = nu_c[fast]

        m1 = nf < nc
        m2 = (nf >= nc) & (nf < nm)
        m3 = nf >= nm

        shape_f = np.empty_like(nf)
        # nu < nu_c: anchored at nu_c (peak at nu_c)
        shape_f[m1] = (nf[m1] / nc[m1])**(1.0 / 3.0)
        # nu_c <= nu < nu_m
        shape_f[m2] = (nf[m2] / nc[m2])**(-0.5)
        # nu >= nu_m: enforce continuity at nu_m
        shape_f[m3] = (nm[m3] / nc[m3])**(-0.5) * (nf[m3] / nm[m3])**(-p / 2.0)

        shape[fast] = shape_f

    return shape

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def beta_from_G(G):
    """
    Convert Lorentz factor to dimensionless velocity β = v/c.
    """
    G = np.asarray(G, dtype=float)
    G_safe = np.where(G > 1.0, G, 1.0)
    return np.sqrt(1.0 - 1.0 / (G_safe ** 2))


def Doppler(G, b, chi):
    """
    Doppler factor for angle χ between velocity and line of sight.
    """
    denom = G * np.maximum(1.0 - b * np.cos(chi), EPS_DEN)
    D = 1.0 / denom
    return np.minimum(D, D_MAX)

def compute_deceleration_scales(
    Gamma_c=300.0,
    theta_j_deg=6.0,
    g=2.0,
    epsilon_c=1.0,
    k_struct=4.0,
    theta_v_deg=3.0,
    theta_edge_deg=None,
    medium="ism",
    n0=1.0,
    Astar=1.0,
    Eiso_core=1e53,
    z=1.0,
    n_theta=4096,
    rmin_frac=1e-3,
):
    """
    Lightweight diagnostic: compute deceleration radius scales (R_dec) and the
    effective R_min used by the model (R_min_eff = rmin_frac * Rdec_min),
    without running the full EATS + radiation integration.

    Returns a dict with:
      - Rdec_core_cm
      - Rdec_min_cm
      - theta_min_deg
      - R_min_eff_cm
      - (optional) tdec_core_s, tdec_min_s (approximate)
    """

    # Geometry
    theta_c = np.deg2rad(theta_j_deg)
    theta_v = np.deg2rad(theta_v_deg)  # kept for completeness; not used here
    theta_edge = None if theta_edge_deg is None else np.deg2rad(theta_edge_deg)

    # Jet structure: same as in compute_fs_lightcurve
    def Gamma0_theta(th):
        th = np.asarray(th, dtype=float)
        tc = float(theta_c) if float(theta_c) > 0.0 else 1e-20
        x = th / tc
        x_safe = np.where(x <= 1.0, 1.0, x)
        outer = 1.0 + (Gamma_c - 1.0) * np.power(x_safe, -float(g))
        G = np.where(x <= 1.0, float(Gamma_c), np.maximum(outer, 1.0))
        return G

    def epsilon_theta(th):
        th = np.asarray(th, dtype=float)
        tc = float(theta_c) if float(theta_c) > 0.0 else 1e-20
        x = th / tc
        x_safe = np.where(x <= 1.0, 1.0, x)
        outer = float(epsilon_c) * np.power(x_safe, -float(k_struct))
        eps = np.where(x <= 1.0, float(epsilon_c), outer)
        if theta_edge is not None:
            eps = np.where(th > theta_edge, 0.0, eps)
        return eps

    # Theta grid (same spacing philosophy as the main code)
    theta_max = float(theta_edge) if theta_edge is not None else np.deg2rad(80.0)
    small = 1e-8
    th = np.linspace(np.sqrt(small), np.sqrt(theta_max), int(n_theta)) ** 2

    eps_th = epsilon_theta(th)
    active = eps_th > 0.0
    if not np.any(active):
        raise RuntimeError("No active angular cells inside the jet (epsilon_theta>0).")

    eps0 = float(epsilon_theta(0.0))
    if eps0 <= 0.0:
        eps0 = 1.0
    Eiso_th = float(Eiso_core) * (eps_th / eps0)

    # External medium normalization A in rho = A R^{-k}
    if str(medium).lower() == "ism":
        k = 0
    elif str(medium).lower() == "wind":
        k = 2
    else:
        raise ValueError("Only 'ism' or 'wind' are supported in this diagnostic.")

    if k == 0:
        m_p_local = 1.6726219e-24
        A = m_p_local * float(n0)
    else:
        A = 5e11 * float(Astar)

    # R_dec(theta)
    G0_th = Gamma0_theta(th)
    u0_th = np.sqrt(np.maximum(G0_th ** 2 - 1.0, 1.0))
    R_dec_th = ((3.0 - k) * Eiso_th /
                (4.0 * np.pi * A * c ** 2 * np.maximum(u0_th ** 2, 1e-30)))
    R_dec_th = np.power(np.maximum(R_dec_th, 1e-40), 1.0 / (3.0 - k))

    # Min over active cells
    Rdec_min = float(np.min(R_dec_th[active]))
    idx_min = int(np.argmin(np.where(active, R_dec_th, np.inf)))
    theta_min_deg = float(np.rad2deg(th[idx_min]))

    # Core scale (theta=0)
    u0_core = np.sqrt(max(float(Gamma_c) ** 2 - 1.0, 1.0))
    Rdec_core = ((3.0 - k) * float(Eiso_core) /
                 (4.0 * np.pi * A * c ** 2 * max(u0_core ** 2, 1e-30)))
    Rdec_core = float(np.power(max(Rdec_core, 1e-40), 1.0 / (3.0 - k)))

    # Effective R_min used by the model
    R_min_eff = float(rmin_frac) * Rdec_min

    # Optional: approximate observer t_dec (sanity check only)
    # Standard scaling: t_dec ~ (1+z) R_dec / (2 c Gamma^2) (on-axis approximation)
    tdec_core = (1.0 + float(z)) * Rdec_core / (2.0 * c * max(float(Gamma_c) ** 2, 1.0))
    tdec_min = (1.0 + float(z)) * Rdec_min / (2.0 * c * max(float(G0_th[idx_min]) ** 2, 1.0))

    return {
        "Rdec_core_cm": Rdec_core,
        "Rdec_min_cm": Rdec_min,
        "theta_min_deg": theta_min_deg,
        "R_min_eff_cm": R_min_eff,
        "tdec_core_s": float(tdec_core),
        "tdec_min_s": float(tdec_min),
    }

# ----------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------
def compute_fs_lightcurve(
    # Jet structure: PL jet as in Beniamini+2020 (eq. 1)
    Gamma_c=300.0,           # Core initial Lorentz factor Γ_c,0
    theta_j_deg=6.0,         # Core angle θ_c [deg]
    g=2.0,                   # b: slope of Γ_0(θ) (Lorentz structure)
    epsilon_c=1.0,           # ε_c: core kinetic energy per solid angle (arb. units)
    k_struct=4.0,            # a: slope of ε(θ) (energy structure)

    # Geometry
    theta_v_deg=3.0,         # Viewing angle θ_obs [deg]
    theta_edge_deg=None,     # Optional truncation of jet at θ_edge [deg]

    # External medium: ρ = A R^{-k}
    medium="ism",            # "ism" (k=0) or "wind" (k=2); medium fully determines k
    n0=1.0,                  # ISM number density [cm^-3] if k=0
    Astar=1.0,               # Wind parameter A_* if k=2 (ρ = 5e11 A_* R^{-2} g cm^-1)
    k=0,                     # External density index (0 for ISM, 2 for wind)

    # Energetics and redshift
    Eiso_core=1e53,          # Isotropic-equivalent energy along jet core [erg]
    z=1.0,                   # Redshift

    # Microphysics (only p is used here; ε_e, ε_B, Y_SSC are placeholders)
    p=2.2,
    eps_e=0.1,
    eps_B=1e-2,
    Y_SSC=0.0,

    # Frequency band in observer frame [keV]
    nu_min_keV=0.3,
    nu_max_keV=10.0,
    n_nu=8,

    # Resolution
    n_theta=192,
    n_phi=192,
    n_bins=120,
    n_R=200,

    # Radial domain [cm]
    R_min=1e15,
    R_max=1e19,

    # Options
    use_kdotn=False,         # If True, weight emissivity by n·k (projected area)
):
    """
    Compute forward-shock light curve for a structured PL jet with full
    synchrotron microphysics (nu_m, nu_c) and EATS integration in a
    relativistic Blandford-McKee–like regime. For typical X-ray GRB
    parameters the band often lies in the PLS G segment (nu_m < nu < nu_c).

    Returns:
        t_obs      : (n_bins,)   log-centered observer times [s]
        F_band     : (n_bins,)   band-integrated flux (arb. units)
        counts     : (n_bins,)   number of contributing cells per time bin
        beta_eff   : (n_bins,)   effective spectral index in band
        F_tnu      : (n_bins,n_nu) specific flux at each time and ν (arb. units)
        nu_obs_grid: (n_nu,)     observer-frame frequencies [Hz]
        Em_keV     : (n_bins,)   flux-weighted nu_m in observer frame [keV]
        Ec_keV     : (n_bins,)   flux-weighted nu_c in observer frame [keV]
    """

    # ------------------------------------------------------------------
    # Configuration for relativistic validity of the model
    # ------------------------------------------------------------------
    GAMMA_NR_CUTOFF = 1.05  # below this the flow is no longer relativistic; stop evolution

    # For the emissivity, we require a stricter relativistic condition.
    # Cells with Γ <= GAMMA_EMIT_CUTOFF are not trusted for synchrotron
    # emission in this BM-like model and are excluded from the flux.
    GAMMA_EMIT_CUTOFF = 1.05

    # ------------------------------------------------------------------
    # Basic geometry and frequency grid
    # ------------------------------------------------------------------
    theta_c = np.deg2rad(theta_j_deg)
    theta_v = np.deg2rad(theta_v_deg)
    theta_edge = None if theta_edge_deg is None else np.deg2rad(theta_edge_deg)

    # Frequency grid in Hz
    nu_min = nu_min_keV * 1e3 * E_HZ_PER_eV
    nu_max = nu_max_keV * 1e3 * E_HZ_PER_eV
    nu_obs_grid = np.geomspace(nu_min, nu_max, int(n_nu))
    ln_nu = np.log(nu_obs_grid)

    # ------------------------------------------------------------------
    # Debug: choose a central frequency index (≈ 1 keV) to monitor the
    # angle-integrated flux as a function of radius, F_nu(R).
    # ------------------------------------------------------------------
    nu_center_keV = 1.0
    nu_center = nu_center_keV * 1e3 * E_HZ_PER_eV
    debug_j = int(np.argmin(np.abs(nu_obs_grid - nu_center)))

    # Luminosity distance and global scaling (only for absolute units)
    D_L = luminosity_distance_flatlcdm(z)
    if (D_L <= 0.0) or (not np.isfinite(D_L)):
        # Work in arbitrary units if distance is not defined
        obs_scale = 1.0
    else:
        # Standard factor for isotropic comoving emissivity:
        # dF_nu = (1+z) / (4π D_L^2) * δ^3 * P'_{ν'} dV'
        obs_scale = (1.0 + z) / (4.0 * np.pi * D_L**2)

    # ------------------------------------------------------------------
    # Jet structure: Oganesyan-like top-hat core + power-law wings
    # ------------------------------------------------------------------
    def Gamma0_theta(th):
        th = np.asarray(th, dtype=float)
        tc = float(theta_c) if float(theta_c) > 0.0 else 1e-20  # avoid division by zero
        x = th / tc

        # Ensure x=1 inside the core, so the outer PL does not blow up
        x_safe = np.where(x <= 1.0, 1.0, x)

        outer = 1.0 + (Gamma_c - 1.0) * np.power(x_safe, -float(g))
        G = np.where(x <= 1.0, float(Gamma_c), np.maximum(outer, 1.0))
        return G

    def epsilon_theta(th):
        th = np.asarray(th, dtype=float)
        tc = float(theta_c) if float(theta_c) > 0.0 else 1e-20
        x = th / tc

        x_safe = np.where(x <= 1.0, 1.0, x)
        outer = float(epsilon_c) * np.power(x_safe, -float(k_struct))

        eps = np.where(x <= 1.0, float(epsilon_c), outer)
        
        # Optional truncation
        if theta_edge is not None:
            eps = np.where(th > theta_edge, 0.0, eps)
        
        return eps


    def cos_chi(th, ph):
        """Cosine of angle between local velocity and line of sight."""
        return (np.cos(th) * np.cos(theta_v) +
                np.sin(th) * np.sin(theta_v) * np.cos(ph))

    # ------------------------------------------------------------------
    # Angular grid (same construction as in hle.py)
    # ------------------------------------------------------------------
    # By default, integrate out to a wide angle (~80 deg) to capture the
    # wings of the structured jet. If theta_edge is provided, override
    # this and integrate up to that angle instead (useful for spherical
    # tests, where we want the full sphere / hemisphere).
    if theta_edge is not None:
        theta_max = float(theta_edge)
    else:
        theta_max = np.deg2rad(80.0)

    small = 1e-8

    th = np.linspace(np.sqrt(small), np.sqrt(theta_max), int(n_theta)) ** 2
    ph = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)

    dth = np.gradient(th)
    dph = (2.0 * np.pi) / int(n_phi)

    TH, PH = np.meshgrid(th, ph, indexing="ij")   # [n_theta, n_phi]
    dOmega = np.sin(TH) * dth[:, None] * dph      # solid angle of each cell

    cosc = np.clip(cos_chi(TH, PH), -1.0, 1.0)
    chi_vals = np.arccos(cosc)
    kdotn = np.clip(cosc, 0.0, 1.0)

    eps_grid = epsilon_theta(TH)
    G0_grid = Gamma0_theta(TH)
    beta0_grid = beta_from_G(G0_grid)

    # Active angular cells: inside the physical jet
    # For theta_edge is not None, epsilon_theta=0 outside the jet; those
    # cells should not contribute to the dynamics or flux.
    active = eps_grid > 0.0

    # Local isotropic equivalent energy per cell
    # Normalize so that ε(0) corresponds to Eiso_core.
    eps0 = epsilon_theta(0.0)
    if eps0 <= 0.0:
        eps0 = 1.0
    Eiso_grid = Eiso_core * (eps_grid / eps0)

    # ------------------------------------------------------------------
    # External medium normalization A in ρ = A R^{-k}
    # ------------------------------------------------------------------
    if medium.lower() == "ism":
        k = 0
    elif medium.lower() == "wind":
        k = 2

    if k == 0:
        # ρ = m_p n0 (constant)
        m_p = 1.6726219e-24
        A = m_p * float(n0)
    elif k == 2:
        # ρ = 5e11 A_* R^{-2} [g cm^-1] / (4π R^2) is often used;
        # here we simply use the usual 5e11 A_* prescription.
        A = 5e11 * float(Astar)
    else:
        raise ValueError("Only k=0 (ISM) or k=2 (wind) are supported.")

    # ------------------------------------------------------------------
    # Deceleration radius R_dec(θ) (eq. 3 in Beniamini+2020)
    # A more general expression with u0 = sqrt(Γ0^2 - 1) appears in GG18.
    # We follow eq. (1) of GG18 / eq. (3) of BGG20:
    #
    #   R_dec(θ) = [ (3-k) E_iso(θ) / (4π A c^2 u0(θ)^2) ]^{1/(3-k)}
    #
    # ------------------------------------------------------------------
    u0_grid = np.sqrt(np.maximum(G0_grid ** 2 - 1.0, 1.0))
    R_dec_grid = ((3.0 - k) * Eiso_grid /
                  (4.0 * np.pi * A * c ** 2 * np.maximum(u0_grid ** 2, 1e-30)))
    R_dec_grid = np.power(np.maximum(R_dec_grid, 1e-40), 1.0 / (3.0 - k))

    # ------------------------------------------------------------
    # Automatic choice of R_min based on the deceleration radius
    # ------------------------------------------------------------
    # Find the minimum deceleration radius over all *active* angular cells.
    # Cells outside the jet (eps_grid == 0) are excluded from this estimate.
    if not np.any(active):
        raise RuntimeError("No active angular cells inside the jet (eps_grid>0).")
    Rdec_min = float(np.min(R_dec_grid[active]))

    # Choose R_min as a fixed fraction of Rdec_min
    # This guarantees a sufficiently long coasting phase
    R_min_eff = 1e-3 * Rdec_min

    # Override the user-provided R_min
    R_min = R_min_eff

#    print("[FS] Automatic R_min:")
#    print(f"      Rdec_min = {Rdec_min:.3e} cm")
#    print(f"      R_min    = {R_min:.3e} cm  "
#          f"(R_min / Rdec_min = {R_min / Rdec_min:.3e})")

    # ------------------------------------------------------------------
    # Dynamics Γ(θ,R) from Panaitescu & Kumar 2000 / GG18 (eq. 4)
    #
    #   Γ(θ,R) = Γ0 + 0.5 ζ^{k-3} [ sqrt(1 + 4 Γ0/(Γ0+1) ζ^{3-k}
    #                                   + (2 ζ^{3-k}/(Γ0+1))^2 ) - 1 ]
    #
    # where ζ = R / R_dec(θ).
    # ------------------------------------------------------------------
    def Gamma_evo(R, R_dec, G0, k):
        """
        BM-like evolution of the bulk Lorentz factor.
        For each angular cell we define zeta = R / R_dec(θ) and impose:
        - Coasting phase for zeta < 1:
              Γ(R) = Γ0
        - Deceleration phase for zeta >= 1:
              Γ(R) = Γ0 * (1 + zeta^(3-k))^(-1/2)
        This reproduces the correct asymptotic scalings for an adiabatic
        relativistic blast wave:
        Γ ∝ R^0            for zeta << 1
        Γ ∝ R^{(k-3)/2}    for zeta >> 1
        """
        # Dimensionless radius in units of the local deceleration radius
        zeta = R / np.maximum(R_dec, 1e-40)

        # Start from pure coasting everywhere
        G = np.array(G0, copy=True)

        # Cells that have started to decelerate
        mask = zeta >= 1.0
        if np.any(mask):
            z = zeta[mask]
            G0_loc = G0[mask]

            # BM-like evolution
            G_dec = G0_loc * (1.0 + z ** (3.0 - k)) ** (-0.5)
            
            # Enforce Γ >= 1
            G[mask] = np.maximum(G_dec, 1.0)

        return G

    # ------------------------------------------------------------------
    # Radial grid
    # ------------------------------------------------------------------
    # Ensure that the number of radial cells is an integer
    n_R_int = max(int(n_R), 2)
    R_grid = np.geomspace(R_min, R_max, n_R_int)
    dR_grid = np.empty_like(R_grid)
    dR_grid[0] = 0.0
    dR_grid[1:] = np.diff(R_grid)

    # Lab-frame time initialization at R_min (coasting at Γ0)
    # Lab-frame time initialization at R_min (coasting at Γ0)
    R0 = float(R_min)
    t_lab_init = R0 / (np.maximum(beta0_grid, 1e-4) * c)

    # ------------------------------------------------------------------
    # Pre-pass over R: determine observer time range [t_min, t_max]
    # using the GG18 equal-arrival-time relation.
    #
    # We evolve both:
    #   - t_lab(R)   via dt_lab = dR / (β c)
    #   - t_obs(R,θ,φ) via dt_obs = (1+z) * (1/β - cosχ) * dR / c
    # ------------------------------------------------------------------
    t_lab_prev = t_lab_init.copy()
    t_obs_prev = np.zeros_like(G0_grid)

    t_obs_min = None
    t_obs_max = 0.0

    for iR, R in enumerate(R_grid):
        dR = dR_grid[iR]

        if dR <= 0.0:
            # First radius: no evolution yet
            G_loc = G0_grid
            beta_loc = beta0_grid
            dt_lab = 0.0
        else:
            # Dynamical evolution Γ(R,θ) (Panaitescu & Kumar / GG18)
            G_loc = Gamma_evo(R, R_dec_grid, G0_grid, k)

            # Stop once all active cells are mildly/non relativistic.
            # The model is only valid in the relativistic regime; continuing
            # the evolution would produce an unphysical late-time tail.
            if np.all(G_loc[active] <= GAMMA_NR_CUTOFF):
                break

        beta_loc = beta_from_G(G_loc)
        dt_lab = dR / (np.maximum(beta_loc, 1e-4) * c)

        # Lab-frame time at this radius (needed later for microphysics)
        t_lab = t_lab_prev + dt_lab
        t_lab_prev = t_lab

        # Observer time for each angular cell:
        #   dt_obs = (1+z) * (1/β - cosχ) * dR / c    [GG18 eq. (18)]
        inv_beta = 1.0 / np.maximum(beta_loc, 1e-6)
        dt_obs = (1.0 + z) * (inv_beta - cosc) * (dR / c)
        t_obs = t_obs_prev + dt_obs
        t_obs_prev = t_obs

        # Keep only positive & finite arrival times for active cells
        m = np.isfinite(t_obs) & (t_obs > 0.0) & active
        if not np.any(m):
            continue

        t_min_loc = np.min(t_obs[m])
        t_max_loc = np.max(t_obs[m])


        if t_obs_min is None:
            t_obs_min = t_min_loc
        else:
            t_obs_min = min(t_obs_min, t_min_loc)

        t_obs_max = max(t_obs_max, t_max_loc)

    if t_obs_min is None:
        raise RuntimeError("No positive observer times found in pre-pass.")

    # Add a small safety margin
    t_obs_min = max(t_obs_min * 0.8, 1e-12)
    t_obs_max = t_obs_max * 1.2

    # Time bins (log-spaced)
    tbins = np.geomspace(t_obs_min, t_obs_max, int(n_bins) + 1)
    t_centers = np.sqrt(tbins[:-1] * tbins[1:])

    # ------------------------------------------------------------------
    # Main EATS integration pass — physical microphysics (GG18-like)
    # ------------------------------------------------------------------

    flux_time_nu = np.zeros((int(n_bins), int(n_nu)), dtype=float)
    counts_binned = np.zeros(int(n_bins), dtype=int)

    # Flux-weighted averages of nu_m and nu_c in the observer frame.
    # We build them using the flux at ~1 keV (debug_j) as weight.
    Em_num = np.zeros(int(n_bins), dtype=float)
    Em_den = np.zeros(int(n_bins), dtype=float)
    Ec_num = np.zeros(int(n_bins), dtype=float)
    Ec_den = np.zeros(int(n_bins), dtype=float)
    
    # Debug: radial profile of the angle-integrated flux at the
    # central frequency (nu_center). This approximates F_nu(R).
    debug_F_R = np.zeros(len(R_grid), dtype=float)

    # Reset lab and observer times for the dynamical integration
    t_lab_prev = t_lab_init.copy()
    t_obs_prev = np.zeros_like(G0_grid)

    for iR, R in enumerate(R_grid):
        dR = dR_grid[iR]

        # --- Dynamical update (same as before) ---
        if dR <= 0.0:
            G_loc = G0_grid
            beta_loc = beta0_grid
            dt_lab = 0.0
        else:
            G_loc = Gamma_evo(R, R_dec_grid, G0_grid, k)

            # Stop once all active cells are mildly/non relativistic.
            # Beyond this point the BM-like prescription is not reliable and
            # integrating further leads to spurious late-time emission.
            if np.all(G_loc[active] <= GAMMA_NR_CUTOFF):
                break

        beta_loc = beta_from_G(G_loc)
        dt_lab = dR / (np.maximum(beta_loc, 1e-6) * c)

        # Lab-frame time at this radius
        t_lab = t_lab_prev + dt_lab
        t_lab_prev = t_lab

        # Observer time at this radius and angle:
        #   dt_obs = (1+z) * (1/β - cosχ) * dR / c
        inv_beta = 1.0 / np.maximum(beta_loc, 1e-6)
        dt_obs = (1.0 + z) * (inv_beta - cosc) * (dR / c)
        t_obs = t_obs_prev + dt_obs
        t_obs_prev = t_obs

        # Mask positive & finite arrival times for active cells only
        m = np.isfinite(t_obs) & (t_obs > 0.0) & active
        if not np.any(m):
            continue

        # Among those, keep only cells that are still safely relativistic
        # for the emissivity. This prevents mildly/non-relativistic cells
        # from contributing unphysical late-time emission.
        m_emit = m & (G_loc > GAMMA_EMIT_CUTOFF)
        if not np.any(m_emit):
            continue
        
        # Doppler factor for each angular cell
        D_loc = Doppler(G_loc, beta_loc, chi_vals)

        # --- Microphysics at this radius ---
        nu_m_prime, nu_c_prime, Pnu_max_prime, n_prime, B_prime = microphysics_fs(
            G_loc, R, t_lab, k, A, p, eps_e, eps_B
        )

        # --- Comoving shell thickness (GG18 eq. 16) ---
        Delta_prime = R / (4.0 * (3.0 - k) * np.maximum(G_loc, 1.0))

        # --- Comoving volume element per cell ---
        #    dV' = R^2 * Delta' * sinθ dθ dφ
        dV_prime = (R ** 2) * Delta_prime * dOmega

        # Loop over observer frequencies
        for j in range(int(n_nu)):
            nu_obs = nu_obs_grid[j]

            # Convert to comoving frequency for each cell
            nu_prime = nu_obs * (1.0 + z) / np.maximum(D_loc, 1e-30)

            # Spectral shape (dimensionless)
            shape = synch_spectrum_pl_segment(nu_prime, nu_m_prime, nu_c_prime, p)

            # Comoving specific luminosity per cell
            Pnu_prime = Pnu_max_prime * shape

            # Observed specific flux element:
            #
            #   dF_nu = (1+z) / (4π D_L^2) * δ^3 * P'_{ν'} dV'
            #
            # obs_scale already includes (1+z) / (4π D_L^2).
            flux_cell = obs_scale * (D_loc ** 3) * Pnu_prime * dV_prime

            # Debug: angle-integrated flux at the central frequency
            # as a function of radius. We sum over all active cells
            # (mask m) at this radius. This gives F_nu(R) ignoring
            # EATS time delays.
            if j == debug_j:
                debug_F_R[iR] = np.sum(flux_cell[m_emit])

            # Flatten and histogram in time bins.
            # Each cell contributes its instantaneous flux F_nu to the
            # appropriate observer-time bin, according to its t_obs.
            t_flat = t_obs[m_emit].ravel()
            f_flat = flux_cell[m_emit].ravel()

            idx = np.searchsorted(tbins, t_flat) - 1
            good = (idx >= 0) & (idx < int(n_bins))
            if np.any(good):
                f_sel = f_flat[good]
                idx_sel = idx[good]

                # Accumulate specific flux in the (time, frequency) grid
                np.add.at(flux_time_nu[:, j], idx_sel, f_sel)
                if j == 0:
                    # Count how many emitting cells contribute to each time bin
                    np.add.at(counts_binned, idx_sel, 1)

                # For j == debug_j (≈ 1 keV), use the flux at this
                # frequency as a weight to build flux-weighted averages
                # of nu_m and nu_c in the observer frame.
                if j == debug_j:
                    # Break frequencies and Doppler factors for the
                    # same emitting cells used above.
                    nu_m_emit = nu_m_prime[m_emit].ravel()[good]
                    nu_c_emit = nu_c_prime[m_emit].ravel()[good]
                    D_emit = D_loc[m_emit].ravel()[good]

                    # Observer-frame break frequencies for each cell:
                    # nu_obs = delta * nu_prime / (1+z)
                    nu_m_obs = nu_m_emit * D_emit / (1.0 + z)
                    nu_c_obs = nu_c_emit * D_emit / (1.0 + z)

                    # Flux-weighted sums in frequency space
                    np.add.at(Em_num, idx_sel, f_sel * nu_m_obs)
                    np.add.at(Em_den, idx_sel, f_sel)
                    np.add.at(Ec_num, idx_sel, f_sel * nu_c_obs)
                    np.add.at(Ec_den, idx_sel, f_sel)

    # ------------------------------------------------------------------
    # Band-integrated flux and effective spectral index
    # ------------------------------------------------------------------
    # At this point flux_time_nu[i,j] ≈ ∑_{cells in bin i} F_nu,j(cell),
    F_band = np.trapz(flux_time_nu, nu_obs_grid, axis=1)

    # Effective β_eff from a PL fit in log F vs log ν
    beta_eff = np.full(int(n_bins), np.nan)
    MIN_PTS = 3
    MIN_SPAN = np.log(5.0)  # at least 0.7 dex in ν

    for i in range(int(n_bins)):
        Fi = flux_time_nu[i, :]
        mF = np.isfinite(Fi) & (Fi > 0.0)
        if mF.sum() < MIN_PTS:
            continue
        if ln_nu[mF].ptp() < MIN_SPAN:
            continue
        slope, _ = np.polyfit(ln_nu[mF], np.log(Fi[mF]), 1)
        beta_eff[i] = slope  # should be ≈ -beta_nu

    # ------------------------------------------------------------------
    # Flux-weighted break energies (observer frame) in keV
    # ------------------------------------------------------------------
    Em_keV = np.full(int(n_bins), np.nan)
    Ec_keV = np.full(int(n_bins), np.nan)

    # Convert flux-weighted average frequencies to keV:
    # E[keV] = nu / (E_HZ_PER_eV * 1e3)
    m_Em = Em_den > 0.0
    if np.any(m_Em):
        Em_keV[m_Em] = (Em_num[m_Em] / Em_den[m_Em]) / (E_HZ_PER_eV * 1e3)

    m_Ec = Ec_den > 0.0
    if np.any(m_Ec):
        Ec_keV[m_Ec] = (Ec_num[m_Ec] / Ec_den[m_Ec]) / (E_HZ_PER_eV * 1e3)


    return t_centers, F_band, counts_binned, beta_eff, flux_time_nu, nu_obs_grid, Em_keV, Ec_keV
