"""Top-hat HLE model for GRB extended-emission light-curve calculations."""

import numpy as np

# Physical constants [cgs]
c = 2.99792458e10       # Speed of light 
E_HZ_PER_eV = 2.418e14  # eV to Hz conversion
H0_kms_Mpc = 67.7       # Hubble constant
Omega_m = 0.31          # Adimensional matter density parameter

def compute_hle_lightcurve(
    # Jet structure
    Gamma_c=300, theta_j_deg=6.0, g=4.5, epsilon_c=1.0, k=4.5,

    # Geometry
    theta_v_deg=3.00, R=1e15,

    # Spectral selector (0: PL; 1 SBPL; 2: 2SBPL)
    spectrum=2,
    two_breaks=None,            # (depcrecated) if provided 0->SBPL, 1->2SBPL

    # Reference time for break(s) evolution
    t0 = 1.0,                   

    # PL parameters
    beta_pl=1.0,                # PL slope
    nu0_prime=1.0,              # reference ν' for PL normalization (arbitrary)

    # SBPL legacy parameters
    nu_b0=1e19,                 # comoving break [Hz]
    q=1.0,                      # time-evolution index for nu_b0'
    i_p_prime=1.0,              # comoving amplitude at the (high) break
    sbpl_s=2.0,                 # SBPL smoothness
    alpha0=+1/3, beta0=-3/2,    # SBPL Fν slopes (low/high)

    # 2SBPL parameters
    nu_b1_0=None,               # low-energy comoving break [Hz] (≈ few keV in obs at t0)
    nu_b2_0=None,               # high-energy/peak comoving break [Hz]
    q1=0.0, q2=0.0,             # time-evolution indices for the two breaks (0 → no intrinsic evolution)
    alpha_lo=+1/3,              # Fν slope below nu_b1'
    alpha_mid=-1/2,             # Fν slope between nu_b1' and nu_b2'
    alpha_hi=-3/2,              # Fν slope above nu_b2'
    s1=2.0, s2=2.0,             # smoothness at the two joins

    # Observer-frame parametrization for breaks (optional)
    breaks_frame="comoving",    # "comoving" (default) or "observer"
    E_b_ref_keV=None,           # SBPL: observer-frame break at reference [keV]
    E_b1_ref_keV=None,          # 2SBPL: observer-frame low break at reference [keV]
    E_b2_ref_keV=None,          # 2SBPL: observer-frame high break (peak) at reference [keV]

    # Resolution
    n_theta=384, n_phi=384, n_bins=150, n_nu=12,

    # Instrument band (keV) and redshift
    nu_min_keV=0.3, nu_max_keV=10.0, z=3.0,

    # Patch Counting / MC
    n_mc=0, min_grid_counts=15,

    # Numerical guards
    D_MAX=1e6, EPS_DEN=1e-12,

    # Switch between numerical and analytical integration
    ring_mode="off",           # When = "auto" -> analytical integration if theta_v_deg == 0.0, else numerical

    # Normalization mode
        norm_mode="obs_pivot_eats", # "break", "obs_pivot" (LOS-only), obs_piv_eats (EATS-consistent)
        E_piv_keV=1.0,              # pivot energy if norm_mode = "obs_pivot"

    # Surface + edge controls
    use_kdotn=True,            # Projection factor for thin-shell surface
    theta_edge_deg=None,       # Hard edge: sets emissivity to zero for theta > theta_edge_deg

    tophat=1.0,   # 0 -> structured, 1 -> top-hat (force theta_edge=theta_j)

):
    
    """
    High-latitude emission (thin shell) with angular structure.
    two_breaks=0  → SBPL (default)
    two_breaks=1  → 2SBPL (two smooth breaks, three Fν slopes).
    i_p_prime is the comoving amplitude, normalized at the *high* break (nu_b' for SBPL; nu_b2' for 2SBPL).
    """
    
    # ===== PRELIMINARY WORK =====
    
    # Compatibility with 'two_breaks' flag
    if two_breaks is not None:
        spectrum = 1 if int(two_breaks) == 0 else 2

    # Force these to be integers
    n_theta = int(max(1, round(n_theta)))
    n_phi   = int(max(1, round(n_phi)))
    n_bins  = int(max(1, round(n_bins)))
    n_nu    = int(max(1, round(n_nu)))
    n_mc    = int(max(0, round(n_mc)))
    min_grid_counts = int(max(1, round(min_grid_counts)))

    # Convert angles
    theta_j = np.deg2rad(theta_j_deg)
    theta_v = np.deg2rad(theta_v_deg)
    theta_edge = None if theta_edge_deg is None else np.deg2rad(theta_edge_deg)

    # Force top-hat: hard edge always equals the core angle
    if float(tophat) >= 0.5:
        theta_edge = theta_j
        
    # Decide analytic vs numerical
    use_ring = (ring_mode == "on") or (ring_mode == "auto" and abs(theta_v_deg) <= 1e-3)

    # Frequency band [Hz]	
    nu_min = nu_min_keV * 1e3 * E_HZ_PER_eV
    nu_max = nu_max_keV * 1e3 * E_HZ_PER_eV

    # ===== COSMOLOGY =====
    
    MPC_CM = 3.085677581e24  # [cm]
    def luminosity_distance_flatlcdm(z, H0_kms_Mpc=67.7, Omega_m=0.31):
        if z <= 0:
            return 0.0
        H0_s = H0_kms_Mpc / (3.085677581e19)                      # Hubble function
        zs = np.linspace(0.0, z, 4000)                            # Redshift steps
        Ez = np.sqrt(Omega_m * (1.0 + zs)**3 + (1.0 - Omega_m))   # 
        Dc = (c / H0_s) * np.trapz(1.0 / Ez, zs)                  # Comoving distance [cm]
        return (1.0 + z) * Dc                                     # D_L = (1+z) Dc

    D_L_cm = luminosity_distance_flatlcdm(z, H0_kms_Mpc=H0_kms_Mpc, Omega_m=Omega_m)

    # Observer-frame normalization
    zfac = 1.0 / (1.0 + z)**3
    dl_fac = (1.0 / (D_L_cm**2)) if (np.isfinite(D_L_cm) and (D_L_cm > 0.0)) else 1.0
    obs_scale  = zfac * dl_fac
    area_scale = R**2

    # ===== JET STRUCTURE =====
    
    def Gamma(theta):
        th = np.asarray(theta, dtype=float)
        tj = float(theta_j) if float(theta_j) > 0.0 else 1e-20   # Avoid theta_j = 0
        x = th / tj
        x_safe = np.where(x <= 1.0, 1.0, x)   # Ensure x = 1 inside the core
        outer = 1.0 + (Gamma_c - 1.0) * np.power(x_safe, -g)
        return np.where(x <= 1.0, Gamma_c, np.maximum(outer, 1.0))

    def epsilon(theta):   # Same guards
        th = np.asarray(theta, dtype=float)
        tj = float(theta_j) if float(theta_j) > 0.0 else 1e-20
        x = th / tj
        x_safe = np.where(x <= 1.0, 1.0, x)
        outer = epsilon_c * np.power(x_safe, -k)
        emiss = np.where(x <= 1.0, epsilon_c, outer)
        if theta_edge is not None:
            emiss = np.where(th > theta_edge, 0.0, emiss)
        return emiss

    # ===== GEOMETRY FUNCTIONS =====
    
    def cos_chi(th, ph):
        return (np.cos(th)*np.cos(theta_v) +
                np.sin(th)*np.sin(theta_v)*np.cos(ph))

    def beta_from_G(G):
        G_safe = np.where(G > 1.0, G, 1.0)
        return np.sqrt(1. - 1./(G_safe**2))

    def Doppler(G, b, chi):
        denom = G * np.maximum(1.0 - b*np.cos(chi), EPS_DEN)
        D = 1.0 / denom
        return np.minimum(D, D_MAX)

    def t_obs_local(b, chi):
        return R/c*(1.0 - b*np.cos(chi))

    def dG_dtheta(th, G):
        return np.where(th <= theta_j, 0.0, -g * (G - 1.0) / np.maximum(th, 1e-30))

    # Line-of-sight (LOS) values
    G_ref = Gamma(theta_v)
    b_ref = beta_from_G(G_ref)
    D_ref = Doppler(G_ref, b_ref, 0.0)
    
    # ===== SPECTRA DEFINITIONS =====
    
    # Simple power-law
    def spectrum_pl(nu_prime, beta_pl=1.0, nu0_prime=1.0):
        return (np.maximum(nu_prime, 1e-300) / nu0_prime) ** (-beta_pl)

    # Single-break SBPL
    def sbpl_one(nu_p, nu_b_p, a_lo, a_hi, s):
        x = np.maximum(nu_p/np.maximum(nu_b_p, 1e-300), 1e-300)
        shape = x**a_lo * (1.0 + x**((a_lo - a_hi)*s))**(-1.0/s)
        return (2.0**(1.0/s)) * shape

    # Double smoothly broken PL (2SBPL): three slopes, two smooth joins
    # Normalized so that the multiplicative constants are 2^{1/s1} and 2^{1/s2} at the breaks
    def sbpl_two(nu_p, nu_b1_p, nu_b2_p, a1, a2, a3, s1, s2):
        x1 = np.maximum(nu_p/np.maximum(nu_b1_p, 1e-300), 1e-300)
        x2 = np.maximum(nu_p/np.maximum(nu_b2_p, 1e-300), 1e-300)
        term1 = x1**a1 * (1.0 + x1**((a1 - a2)*s1))**(-1.0/s1) * (2.0**(1.0/s1))
        term2 = (1.0 + x2**((a2 - a3)*s2))**(-1.0/s2) * (2.0**(1.0/s2))
        return term1 * term2

    def slope_sbpl_one(nu_p, nu_b_p, a_lo, a_hi, s):
        x  = np.maximum(nu_p/np.maximum(nu_b_p, 1e-300), 1e-300)
        da = (a_lo - a_hi)
        w  = (x**(da*s)) / (1.0 + x**(da*s))
        return a_lo - da * w

    def slope_sbpl_two(nu_p, nu_b1_p, nu_b2_p, a1, a2, a3, s1, s2):
        x1 = np.maximum(nu_p/np.maximum(nu_b1_p, 1e-300), 1e-300)
        x2 = np.maximum(nu_p/np.maximum(nu_b2_p, 1e-300), 1e-300)
        d1 = (a1 - a2)
        d2 = (a2 - a3)
        w1 = (x1**(d1*s1)) / (1.0 + x1**(d1*s1))
        w2 = (x2**(d2*s2)) / (1.0 + x2**(d2*s2))
        return a1 - d1*w1 - d2*w2

    # ===== GRID-RELATED UTILITIES =====
    
    # Grid definition
    theta_max = np.deg2rad(80.0)
    small = 1e-8                                                          # Avoid divisions by zero
    th = (np.linspace(np.sqrt(small), np.sqrt(theta_max), n_theta)**2)    # Non-uniform θ (finer near axis)
    ph = np.linspace(0.0, 2*np.pi, n_phi, endpoint=False)                 # Uniform φ without duplication

    # Cell sizes Δθ, Δφ for dΩ = sinθ Δθ Δφ
    dth = np.gradient(th)                                                 # For a non-uniform θ grid
    dph = (2*np.pi)/n_phi                                                 # For a uniform φ grid

    # Mesh (θ, φ)
    TH, PH = np.meshgrid(th, ph, indexing='ij')                           # [n_theta, n_phi]

    # Compute geometry with guards
    G_vals = Gamma(TH)
    b_vals = beta_from_G(G_vals)
    cosc = np.clip(cos_chi(TH, PH), -1.0, 1.0)                            # Force in [-1;1] to avoid NaN in arccos
    chi_vals = np.arccos(cosc)
    D_vals = Doppler(G_vals, b_vals, chi_vals)
    t_vals = t_obs_local(b_vals, chi_vals)
    eps = epsilon(TH)
    kdotn = np.clip(cosc, 0.0, 1.0)

    # On-axis values
    G0 = Gamma(0.0)
    b0 = beta_from_G(G0)

    # Radial factor
    Rfac = (b_vals / b0)**2
    
    # dΩ per cell
    dOmega = np.sin(TH) * (dth[:, None]) * dph                            

    # Define observer time bins (apply observer-frame shift)
    t_full = (t_vals.ravel() * (1.0 + z))
    sel = t_full > 0.0
    tf  = t_full[sel]
    if tf.size == 0:
        raise RuntimeError("All model times are <= 0; check geometry/parameters.")

    # Re-zero time to the first photon that can reach the observer
    t0_first = tf.min()
    tf = tf - t0_first

    # Avoid exact zeros in log bin edges
    tiny = max(1e-12, 1e-12 * t0_first)
    tf = np.clip(tf, tiny, None)

    tbins = np.logspace(np.log10(tf.min()), np.log10(tf.max()), n_bins+1)
    t_centers = np.exp(0.5 * (np.log(tbins[:-1]) + np.log(tbins[1:])))
    dt = np.diff(tbins)

    # Reference time for break evolution
    t_ref = float(t0)
    tiny_t = 1e-12  # to avoid exact zeros in power laws


    # Geometric weights for break handling
    w0 = (eps * (D_vals**2) * (kdotn if use_kdotn else 1.0) * Rfac * dOmega * area_scale * obs_scale * 1.0/(np.maximum(G_vals, 1e-30))).ravel()[sel]

    # Spectral time variable in the comoving frame
    t_engine = t_vals - (t0_first / max(1.0 + z, 1.0))
    t_engine = np.maximum(t_engine, 0.0)
    t_prime  = np.maximum(D_vals * t_engine, 1e-8 * t0) 

    # ===== COMPUTE BREAKS ONLY WHEN NEEDED =====

    KEV_TO_HZ = 1e3 * E_HZ_PER_eV
    
    # Deafult: PL spectrum => no breaks
    
    if spectrum == 1:
        # SBPL (single break)
        if (breaks_frame == "observer") and (E_b_ref_keV is not None):
            nu_b0_eff = (E_b_ref_keV * KEV_TO_HZ) * (1.0 + z) / D_ref
        else:
            nu_b0_eff = nu_b0
        nu_b_prime = nu_b0_eff * (t_prime / t0)**(-q)

    elif spectrum == 2:
        # 2SBPL (two breaks)
        if (breaks_frame == "observer"):
            if E_b1_ref_keV is not None:
                nu_b1_0 = (E_b1_ref_keV * KEV_TO_HZ) * (1.0 + z) / D_ref
            if E_b2_ref_keV is not None:
                nu_b2_0 = (E_b2_ref_keV * KEV_TO_HZ) * (1.0 + z) / D_ref
        if nu_b1_0 is None: nu_b1_0 = nu_b0 / 10.0
        if nu_b2_0 is None: nu_b2_0 = nu_b0
        if nu_b2_0 <= nu_b1_0:
            nu_b2_0 = max(nu_b1_0*(1.0+1e-9), nu_b1_0 + 1e-9)
        nu_b1_prime = nu_b1_0 * (t_prime / t0)**(-q1)
        nu_b2_prime = nu_b2_0 * (t_prime / t0)**(-q2)

    # Convert negative-convention slopes (alpha) to positive-convention slopes for 2SBPL
    # F_nu ∝ nu^(−alpha_i)  ⇒  beta_i = −alpha_i
    beta1 = -float(alpha_lo)    # low-energy slope of F_nu (β_low)
    beta2 = -float(alpha_mid)   # mid-energy slope of F_nu (β_mid)
    beta3 = -float(alpha_hi)    # high-energy slope of F_nu (β_high)

    # Observer-pivot normalization
    i_p_prime_eff = i_p_prime
    if (norm_mode == "obs_pivot") and (E_piv_keV is not None):
        # Observer pivot at the reference epoch; map to comoving with D_ref
        nu_piv_obs   = E_piv_keV * 1e3 * E_HZ_PER_eV
        nu_piv_prime = (1.0 + z) * nu_piv_obs / np.maximum(D_ref, 1e-300)

        if spectrum == 2:
            # Use the comoving breaks at t0
            S_piv = sbpl_two(
                nu_piv_prime, nu_b1_0, nu_b2_0,
                a1=beta1, a2=beta2, a3=beta3, s1=s1, s2=s2
            )
        elif spectrum == 1:
            # Use the comoving break at t0
            nu_b0_at_t0 = (
                (E_b_ref_keV * 1e3 * E_HZ_PER_eV) * (1.0 + z) / np.maximum(D_ref, 1e-300)
            ) if ((breaks_frame == "observer") and (E_b_ref_keV is not None)) else nu_b0
            S_piv = sbpl_one(
                nu_piv_prime, nu_b0_at_t0, a_lo=alpha0, a_hi=beta0, s=sbpl_s
            )
        else:
            # Pure PL
            S_piv = spectrum_pl(nu_piv_prime, beta_pl=beta_pl, nu0_prime=nu0_prime)

        i_p_prime_eff = i_p_prime / max(S_piv, 1e-30)
    
    # Frequency grid definition
    nu_obs_grid = np.geomspace(nu_min, nu_max, n_nu)
    flux_time_nu = np.zeros((n_bins, n_nu))  # Fν(t_i, ν_j)
 
    # Helper: spectral shape at the observer pivot mapped to comoving at t0
    def _S_piv_at_t0(nu_piv_prime_cell):
        """Return S'_piv evaluated at t0 using comoving breaks at t0."""
        if spectrum == 2:
            return sbpl_two(
                nu_piv_prime_cell, float(nu_b1_0), float(nu_b2_0),
                a1=beta1, a2=beta2, a3=beta3, s1=s1, s2=s2
            )
        elif spectrum == 1:
            nu_b0_at_t0 = float(nu_b0)
            return sbpl_one(
                nu_piv_prime_cell, nu_b0_at_t0, a_lo=alpha0, a_hi=beta0, s=sbpl_s
            )
        else:  # PL
            return spectrum_pl(nu_piv_prime_cell, beta_pl=beta_pl, nu0_prime=nu0_prime)


    # ===== COMPUTE FLUX PER FREQUENCY AND TIME BIN =====

    for j, nu_obs in enumerate(nu_obs_grid):

        # ANALYTIC BRANCH
        if use_ring:
            # Precompute factors for the shifted time map
            Tfac = (1.0 + z) * R / (c * b0)
            G0 = Gamma(0.0)
            b0 = beta_from_G(G0)
            # Time bin centers
            t_ctr = 0.5*(tbins[:-1] + tbins[1:])
            F_t = np.zeros_like(t_ctr)

            for i, ti in enumerate(t_ctr):
                # Solve ti = Tfac * (b0 - β(θ) cosθ)
                lo, hi = 0.0, np.deg2rad(89.9)
                for _ in range(60):
                    mid = 0.5*(lo+hi)
                    Gm = Gamma(mid); bm = beta_from_G(Gm)
                    f  = Tfac*(1.0 - bm*np.cos(mid)) - ti
                    lo, hi = (mid, hi) if f < 0.0 else (lo, mid)
                th = 0.5*(lo+hi)
                if (theta_edge is not None) and (th > theta_edge):
                    continue

                G = Gamma(th); b = beta_from_G(G)
                D = 1.0 / (G * (1.0 - b*np.cos(th)))

                tprime_loc = np.maximum(D * (ti) / (1.0 + z), 1e-12 * t0)  # t0 is the SBPL reference time

                # Breaks evolution

                if spectrum == 1:
                    if (breaks_frame == "observer") and (E_b_ref_keV is not None):
                        KEV_TO_HZ = 1e3 * E_HZ_PER_eV
                        nu_b_prime_loc = (E_b_ref_keV * KEV_TO_HZ) * (1.0 + z) / np.maximum(D, 1e-300)
                        if q != 0.0:
                            nu_b_prime_loc *= (ti / np.maximum(t_ref, 1e-30))**(-q)
                    else:
                        # COMOVING-frame
                        nu_b_prime_loc = nu_b0 * (tprime_loc / t0)**(-q)

                elif spectrum == 2:
                    if (breaks_frame == "observer"):
                        KEV_TO_HZ = 1e3 * E_HZ_PER_eV
                        if E_b1_ref_keV is not None:
                            nu_b1_prime_loc = (E_b1_ref_keV * KEV_TO_HZ) * (1.0 + z) / np.maximum(D, 1e-300)
                            if q1 != 0.0:
                                nu_b1_prime_loc *= (ti / np.maximum(t_ref, 1e-30))**(-q1)
                        else:
                            nu_b1_prime_loc = nu_b1_0 * (tprime_loc / t0)**(-q1)

                        if E_b2_ref_keV is not None:
                            nu_b2_prime_loc = (E_b2_ref_keV * KEV_TO_HZ) * (1.0 + z) / np.maximum(D, 1e-300)
                            if q2 != 0.0:
                                nu_b2_prime_loc *= (ti / np.maximum(t_ref, 1e-30))**(-q2)
                        else:
                            nu_b2_prime_loc = nu_b2_0 * (tprime_loc / t0)**(-q2)
                    else:
                        # COMOVING-frame
                        nu_b1_prime_loc = nu_b1_0 * (tprime_loc / t0)**(-q1)
                        nu_b2_prime_loc = nu_b2_0 * (tprime_loc / t0)**(-q2)

                else:
                    # PL: no breaks
                    pass

                # Comoving frequency for this ring
                nu_p = (1.0 + z) * nu_obs / D

                # EATS-consistent pivot normalization (per ring)
                if norm_mode == "obs_pivot_eats" and (E_piv_keV is not None):
                    nu_piv_obs   = E_piv_keV * 1e3 * E_HZ_PER_eV
                    nu_piv_prime = (1.0 + z) * nu_piv_obs / np.maximum(D, 1e-300)
                    S_piv_cell   = _S_piv_at_t0(nu_piv_prime)
                    i_p_prime_loc = i_p_prime / max(S_piv_cell, 1e-30)
                else:
                    i_p_prime_loc = i_p_prime_eff  # legacy LOS-pivot or "break"

                # Sprime selection
                if spectrum == 0:
                    Spr = i_p_prime_loc * spectrum_pl(nu_p, beta_pl=beta_pl, nu0_prime=nu0_prime)
                elif spectrum == 1:
                    Spr = i_p_prime_loc * sbpl_one(nu_p, nu_b_prime_loc, a_lo=alpha0, a_hi=beta0, s=sbpl_s)
                else:  # spectrum == 2
                    Spr = i_p_prime_loc * sbpl_two(nu_p, nu_b1_prime_loc, nu_b2_prime_loc,
                                               a1=beta1, a2=beta2, a3=beta3, s1=s1, s2=s2)

                # On axis ring formalism from Oganesyan et al. 2020
                eps = epsilon(th)
                dG  = dG_dtheta(th, G)
                db  = dG / (b * G**3 + 1e-30)
                dt_dth = Tfac * (b*np.sin(th) - db*np.cos(th))
                if dt_dth <= 0.0:
                    F_t[i] = 0.0
                    continue
                geom = (2.0*np.pi) * (R**2) * ((b*b)/max(G, 1e-30)) * (np.sin(th) * np.maximum(np.cos(th), 0.0)) / np.maximum(dt_dth, 1e-30)
                F_t[i] = eps * (D**2.0) * Spr * geom * obs_scale

            flux_time_nu[:, j] = F_t
            continue  # Skip the numerical EATS path for this ν


        # NUMERICAL BRANCH
        nu_p = (1.0 + z) * nu_obs / D_vals

        # OBSERVER-frame breaks
        if breaks_frame == "observer":
            KEV_TO_HZ = 1e3 * E_HZ_PER_eV
            invD = 1.0 / np.maximum(D_vals, 1e-300)

            # Choose reference observer time for the evolution (first bin center)

            if spectrum == 1:
                if E_b_ref_keV is not None:
                    # Build per-sample time factor on valid samples 'sel'
                    if q != 0.0:
                        fac_all = np.ones_like(D_vals)
                        flat = fac_all.ravel()
                        flat[sel] = (tf / np.maximum(t_ref, 1e-30))**(-q)
                        fac_all = flat.reshape(D_vals.shape)
                    else:
                        fac_all = 1.0
                    nu_b_prime_local = (E_b_ref_keV * KEV_TO_HZ) * (1.0 + z) * invD * fac_all
                else:
                    nu_b_prime_local = nu_b_prime  # fallback (not expected in observer mode)

            elif spectrum == 2:
                if E_b1_ref_keV is not None:
                    if q1 != 0.0:
                        fac1_all = np.ones_like(D_vals)
                        flat1 = fac1_all.ravel()
                        flat1[sel] = (tf / np.maximum(t_ref, 1e-30))**(-q1)
                        fac1_all = flat1.reshape(D_vals.shape)
                    else:
                        fac1_all = 1.0
                    nu_b1_prime_local = (E_b1_ref_keV * KEV_TO_HZ) * (1.0 + z) * invD * fac1_all
                else:
                    nu_b1_prime_local = nu_b1_prime

                if E_b2_ref_keV is not None:
                    if q2 != 0.0:
                        fac2_all = np.ones_like(D_vals)
                        flat2 = fac2_all.ravel()
                        flat2[sel] = (tf / np.maximum(t_ref, 1e-30))**(-q2)
                        fac2_all = flat2.reshape(D_vals.shape)
                    else:
                        fac2_all = 1.0
                    nu_b2_prime_local = (E_b2_ref_keV * KEV_TO_HZ) * (1.0 + z) * invD * fac2_all
                else:
                    nu_b2_prime_local = nu_b2_prime

        else:
            # COMOVING-frame breaks
            if spectrum == 1:
                nu_b_prime_local = nu_b_prime
            elif spectrum == 2:
                nu_b1_prime_local = nu_b1_prime
                nu_b2_prime_local = nu_b2_prime

        # EATS-consistent pivot normalization (per cell)
        if norm_mode == "obs_pivot_eats" and (E_piv_keV is not None):
            nu_piv_obs    = E_piv_keV * 1e3 * E_HZ_PER_eV
            nu_piv_primeA = (1.0 + z) * nu_piv_obs / np.maximum(D_vals, 1e-300)
            S_pivA        = _S_piv_at_t0(nu_piv_primeA)
            i_p_prime_cell = i_p_prime / np.maximum(S_pivA, 1e-30)
        else:
            i_p_prime_cell = i_p_prime_eff  # scalar (LOS-pivot) or "break"

        # Sprime selection
        if spectrum == 0:
            Sprime = i_p_prime_cell * spectrum_pl(nu_p, beta_pl=beta_pl, nu0_prime=nu0_prime)
        elif spectrum == 1:
            Sprime = i_p_prime_cell *  sbpl_one(nu_p, nu_b_prime_local, a_lo=alpha0, a_hi=beta0, s=sbpl_s)
        else:  # spectrum == 2
            Sprime = i_p_prime_cell * sbpl_two(nu_p, nu_b1_prime_local, nu_b2_prime_local,
                                          a1=beta1, a2=beta2, a3=beta3, s1=s1, s2=s2)

        contrib = eps * (D_vals**2) * Sprime * Rfac * dOmega * area_scale * obs_scale * (1.0 / np.maximum(G_vals, 1e-30))
        if use_kdotn:
            contrib = contrib * kdotn  # Surface projection factor
            
        # Histogram into time bins (robust against right-edge hits)
        w_all = contrib.ravel()[sel]
        t_all = np.maximum(tf, 1e-40)
        elog = np.log10(tbins)
        tlog = np.log10(t_all)
        # Use left-inclusion so exact edges fall into the lower bin
        idx = np.searchsorted(elog, tlog, side="left") - 1
        # Keep only indices that fall inside [0, n_bins-1]
        m = (idx >= 0) & (idx < n_bins)
        idx  = idx[m]
        tlog = tlog[m]
        w    = w_all[m]
        # Fractional position within the bin
        left  = elog[idx]
        right = elog[idx + 1]
        width = np.maximum(right - left, 1e-30)
        frac  = (tlog - left) / width
        # Avoid pathological 50/50 splits when the ring collapses
        edge_tol = 1e-6
        frac = np.where(frac < edge_tol, 0.0, np.where(frac > 1.0 - edge_tol, 1.0, frac))
        # If we’re in the last bin, deposit the "right" piece into the same bin
        last = (idx == (n_bins - 1))
        # Add left piece (and fold the right piece for last-bin samples into the same bin)
        np.add.at(flux_time_nu[:, j], idx, w * (1.0 - frac) + w * frac * last)
        # Add right piece only for non-last bins
        if np.any(~last):
            np.add.at(flux_time_nu[:, j], idx[~last] + 1, w[~last] * frac[~last])


    # Divide by bin widths to get average Fν in each time bin (NUMERICAL branch only)
    if not use_ring:
        flux_time_nu /= dt[:, None]

    # ANALYTIC OBSERVER-FRAME BREAK ENERGY/IES PER BIN
    t_for_bins = tf
    den = np.histogram(t_for_bins,  bins=tbins, weights=w0)[0]
    E_b1_keV = None; E_b2_keV = None

    if spectrum == 1:
        if breaks_frame == "observer":
            # Observer-parametrized SBPL
            if E_b_ref_keV is not None:
                val = float(E_b_ref_keV)
                E_b1_keV = np.full_like(t_centers, val, dtype=float) if q == 0.0 \
                    else val * (t_centers / max(t_ref, 1e-30))**(-q)
        else:
            # Comoving-parametrized SBPL: use D-weighted mapping
            num = np.histogram(t_for_bins, bins=tbins,
                               weights=w0 * (D_vals**(1.0 - q)).ravel()[sel])[0]
            D_eff_1mq = np.divide(num, np.maximum(den, 1e-300))
            E_b1_keV = (nu_b0_eff * (t0**q) * ((1.0 + z)**(q - 1.0)) *
                        D_eff_1mq * (t_centers**(-q))) / (1e3 * E_HZ_PER_eV)

    elif spectrum == 2:
        if breaks_frame == "observer":
            if E_b1_ref_keV is not None:
                v1 = float(E_b1_ref_keV)
                E_b1_keV = np.full_like(t_centers, v1, dtype=float) if q1 == 0.0 \
                    else v1 * (t_centers / max(t_ref, 1e-30))**(-q1)
            if E_b2_ref_keV is not None:
                v2 = float(E_b2_ref_keV)
                E_b2_keV = np.full_like(t_centers, v2, dtype=float) if q2 == 0.0 \
                    else v2 * (t_centers / max(t_ref, 1e-30))**(-q2)
        else:
            # Comoving-parametrized 2SBPL: D-weighted mapping
            num1 = np.histogram(t_for_bins, bins=tbins,
                                weights=w0 * (D_vals**(1.0 - q1)).ravel()[sel])[0]
            num2 = np.histogram(t_for_bins, bins=tbins,
                                weights=w0 * (D_vals**(1.0 - q2)).ravel()[sel])[0]
            D1 = np.divide(num1, np.maximum(den, 1e-300))
            D2 = np.divide(num2, np.maximum(den, 1e-300))
            E_b1_keV = (nu_b1_0 * (t0**q1) * ((1.0 + z)**(q1 - 1.0)) *
                        D1 * (t_centers**(-q1))) / (1e3 * E_HZ_PER_eV)
            E_b2_keV = (nu_b2_0 * (t0**q2) * ((1.0 + z)**(q2 - 1.0)) *
                        D2 * (t_centers**(-q2))) / (1e3 * E_HZ_PER_eV)

    # FLUX-WEIGHTED COMOVING (INTRINSIC) BREAK ENERGIES PER TIME BIN

    E_b1_prime_keV = None
    E_b2_prime_keV = None

    if 'den' not in locals():
        den = np.histogram(t_for_bins, bins=tbins, weights=w0)[0]

    def _safe_div(num, den):
        return np.divide(num, np.maximum(den, 1e-300))

    if breaks_frame == "observer":
        # Use weighted <1/D> to reconstruct comoving breaks from the imposed observed laws
        invD = (1.0 / np.maximum(D_vals, 1e-300)).ravel()[sel]
        num_invD = np.histogram(t_for_bins, bins=tbins, weights=w0 * invD)[0]
        invD_eff = _safe_div(num_invD, den)     # <1/D>_w per time bin

        if E_b1_keV is not None:
            E_b1_prime_keV = (1.0 + z) * E_b1_keV * invD_eff
        if (spectrum == 2) and (E_b2_keV is not None):
            E_b2_prime_keV = (1.0 + z) * E_b2_keV * invD_eff

    else:
        # breaks_frame == 'comoving': weight the actual comoving fields used by the model
        if ('nu_b1_prime' in locals()) and (nu_b1_prime is not None):
            num_b1p = np.histogram(
                t_for_bins, bins=tbins,
                weights=w0 * np.ravel(nu_b1_prime)[sel]
            )[0]
            nu_b1p_eff = _safe_div(num_b1p, den)  # Hz
            E_b1_prime_keV = nu_b1p_eff / (1e3 * E_HZ_PER_eV)

        if (spectrum == 2) and ('nu_b2_prime' in locals()) and (nu_b2_prime is not None):
            num_b2p = np.histogram(
                t_for_bins, bins=tbins,
                weights=w0 * np.ravel(nu_b2_prime)[sel]
            )[0]
            nu_b2p_eff = _safe_div(num_b2p, den)  # Hz
            E_b2_prime_keV = nu_b2p_eff / (1e3 * E_HZ_PER_eV)

    # Flux-weighted effective comoving time per observer bin, t'_eff(t_obs)
    tprime_eff = None
    if not use_ring:
        # Numerical EATS: full grid comoving time 't_prime'
        num_tprime = np.histogram(
            t_for_bins, bins=tbins,
            weights=w0 * np.ravel(t_prime)[sel]
        )[0]
        tprime_eff = np.divide(num_tprime, np.maximum(den, 1e-300))
    else:
        # Analytic ring: recompute ring Doppler at each bin center to build t'_eff deterministically
        tprime_eff = np.zeros_like(t_centers)
        Tfac = (1.0 + z) * R / (c * b0)
        for i, ti in enumerate(t_centers):
            lo, hi = 0.0, np.deg2rad(89.9)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                Gm = Gamma(mid); bm = beta_from_G(Gm)
                f  = Tfac * (1.0 - bm * np.cos(mid)) - ti
                lo, hi = (mid, hi) if f < 0.0 else (lo, mid)
            th = 0.5 * (lo + hi)
            if (theta_edge is not None) and (th > theta_edge):
                tprime_eff[i] = np.nan
                continue
            Gm = Gamma(th); bm = beta_from_G(Gm)
            Dm = 1.0 / (Gm * np.maximum(1.0 - bm * np.cos(th), EPS_DEN))
            tprime_eff[i] = np.maximum(Dm * ti / (1.0 + z), 1e-12 * t0)

    # ===== OUTPUTS =====
    
    # Band-integrated flux (0.3–10 keV)
    flux_binned = np.trapz(flux_time_nu, nu_obs_grid, axis=1)

    # Effective spectral index in the observed band β_eff
    ln_nu = np.log(np.clip(nu_obs_grid, 1e-300, None))
    beta_eff = np.full(n_bins, np.nan)

    # minimum requirements to fit the spectral index
    MIN_PTS  = 6
    MIN_SPAN = np.log(5.0)   # ≈ 0.7 dex

    # pivot per il fallback locale
    E_piv_eff_keV = E_piv_keV if (E_piv_keV is not None) else 1.0
    nu_piv = E_piv_eff_keV * 1e3 * E_HZ_PER_eV
    j_piv  = int(np.argmin(np.abs(nu_obs_grid - nu_piv)))
    halfwin = max(1, min(3, len(nu_obs_grid)//3))

    for i in range(n_bins):
        Fi = flux_time_nu[i, :]
        if not np.any(np.isfinite(Fi)) or np.nanmax(Fi) <= 0.0:
            continue

        # Mask: "good" only
        m = np.isfinite(Fi) & (Fi > 0)
        span = ln_nu[m].ptp() if m.sum() else 0.0

        if (m.sum() >= MIN_PTS) and (span >= MIN_SPAN):
            # global fit on the entire band
            beta_eff[i] = np.polyfit(ln_nu[m], np.log(Fi[m]), 1)[0]
        else:
            # fallback: local estimate at 1keV
            j0 = max(0, j_piv - halfwin)
            j1 = min(len(nu_obs_grid), j_piv + halfwin + 1)
            idx = np.arange(j0, j1)
            mm  = m[idx]
            if mm.sum() >= 2:
                idx = idx[mm]
                beta_eff[i] = np.polyfit(ln_nu[idx], np.log(Fi[idx]), 1)[0]
            else:
                # smooth floor to avoid holes
                Ffloor = max(1e-12 * np.nanmax(Fi), 1e-300)
                y = np.log(np.maximum(Fi, Ffloor))
                beta_eff[i] = np.polyfit(ln_nu, y, 1)[0]

    # Grid-based counts
    grid_counts = np.histogram(tf, bins=tbins)[0]
    counts_binned = grid_counts.copy()
    
    # HYBRID MC fallback for patch counts
    if n_mc > 0:
        mc_batch = int(min(20000, max(1000, n_mc)))
        remaining = int(n_mc)
        cos_th_max = np.cos(theta_max)
        omega_tot = 2.0 * np.pi * (1.0 - cos_th_max)
        mc_counts = np.zeros_like(counts_binned, dtype=int)

        while remaining > 0:
            nb = min(mc_batch, remaining)
            remaining -= nb

            # Uniform sampling in solid angle over 0 - theta_max
            u = np.random.rand(nb)
            cos_th_samp = 1.0 - u * (1.0 - cos_th_max)
            theta_samp = np.arccos(cos_th_samp)
            phi_samp = 2.0 * np.pi * np.random.rand(nb)

            # Compute arrival times for samples
            chi_samp = np.arccos(np.clip(cos_chi(theta_samp, phi_samp), -1.0, 1.0))
            Gs = Gamma(theta_samp)
            bs = beta_from_G(Gs)
            ts = t_obs_local(bs, chi_samp)
            ts_obs = ts * (1.0 +z)

            # Histogram this batch into the same time bins 'tbins'
            batch_hist = np.histogram(ts_obs, bins=tbins)[0]
            mc_counts += batch_hist

        # Use MC counts where the grid is too sparse
        use_mc = grid_counts < min_grid_counts
        counts_binned = np.where(use_mc, mc_counts, grid_counts)

    else:
        counts_binned = grid_counts

    if spectrum in (0, 1):
        # SBPL
        return (
            t_centers,           # 0
            flux_binned,         # 1
            counts_binned,       # 2
            beta_eff,            # 3
            flux_time_nu,        # 4
            nu_obs_grid,         # 5
            E_b1_keV,            # 6  (observer)
            E_b1_prime_keV,      # 7  (comoving)
            tprime_eff           # 8  
        )
    else:
        # 2SBPL
        return (
            t_centers,           # 0
            flux_binned,         # 1
            counts_binned,       # 2
            beta_eff,            # 3
            flux_time_nu,        # 4
            nu_obs_grid,         # 5
            E_b1_keV,            # 6  (observer)
            E_b2_keV,            # 7  (observer)
            E_b1_prime_keV,      # 8  (comoving)
            E_b2_prime_keV,      # 9  (comoving)
            tprime_eff           # 10
        )
