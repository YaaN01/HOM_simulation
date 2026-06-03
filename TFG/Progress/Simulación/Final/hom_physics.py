import numpy as np
import scipy as sp
import scipy.constants as const
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson


crystal_dict = {
    "BBO (I)": {
        "formula": "borate",
        "no_coeffs": [2.7359, 0.01878, 0.01822, 0.01354],
        "ne_coeffs": [2.3753, 0.01224, 0.01667, 0.01516],
        "is_pp": False,
        "std_pump": 405.0,
        "spdc_type": 1
    },
    "BBO (II)": {
        "formula": "borate",
        "no_coeffs": [2.7359, 0.01878, 0.01822, 0.01354],
        "ne_coeffs": [2.3753, 0.01224, 0.01667, 0.01516],
        "is_pp": False,
        "std_pump": 405.0,
        "spdc_type": 2
    },
    "KTP (II)": {
        "formula": "kato_ktp", 
        "nx_coeffs": [3.29100, 0.04140, 0.03978, 9.35522, 31.45571],
        "ny_coeffs": [3.45018, 0.04341, 0.04597, 16.98825, 39.43799],
        "nz_coeffs": [4.59423, 0.06206, 0.04763, 110.80672, 86.12171],
        "pm_plane": "XY",
        "is_pp": False,
        "std_pump": 532.0,
        "spdc_type": 2
    },
    "PPKTP (II)": {
        "formula": "kato_ktp", 
        "nx_coeffs": [3.29100, 0.04140, 0.03978, 9.35522, 31.45571],
        "ny_coeffs": [3.45018, 0.04341, 0.04597, 16.98825, 39.43799],
        "nz_coeffs": [4.59423, 0.06206, 0.04763, 110.80672, 86.12171],
        "pm_plane": "XY",
        "is_pp": True,
        "std_pump": 405.0,
        "spdc_type": 2
    },
    "LBO (I)": {
        "formula": "borate",
        "nx_coeffs": [2.4542, 0.01125, 0.01135, 0.01388],
        "ny_coeffs": [2.5390, 0.01277, 0.01189, 0.01848],
        "nz_coeffs": [2.5865, 0.01310, 0.01223, 0.01861],
        "pm_plane": "XY",
        "is_pp": False,
        "std_pump": 532.0,
        "spdc_type": 1
    }}


def pump_envelope(lambda_p, sigma_p):
    
    # Convert pump wavelength (nm) to central angular frequency (Hz)
    omega_p = 2 * np.pi * const.c / (lambda_p * 1e-9)
    
    # Inner function that accepts 2D frequency grids
    def alpha(omega_s, omega_i):
        
        # Calculate the Gaussian envelope based on energy mismatch
        # Peak probability is at (omega_i + omega_s - omega_p) == 0
        WS, WI = np.meshgrid(omega_s, omega_i)
        return np.exp(-(((WI + WS - omega_p)**2) / (sigma_p**2)))
    
    # Return the unexecuted function object (closure)
    return alpha


def get_refractive_index(lambda_x, crystal_name, polarization, theta_rad=0.0):
    
    # Fetch empirical data for the chosen crystal
    data = crystal_dict[crystal_name]
    
    # Sellmeier mathematical models strictly require wavelength in micrometers (um)
    lam_um = lambda_x / 1000.0
    
    # Helper to calculate the base index purely from the math formula
    def calc_n(coeffs, formula):
        if formula == "borate":
            return np.sqrt(coeffs[0] + coeffs[1]/(lam_um**2 - coeffs[2]) - coeffs[3]*lam_um**2)
        elif formula == "kato_ktp":
            return np.sqrt(coeffs[0] + coeffs[1]/(lam_um**2 - coeffs[2]) + coeffs[3]/(lam_um**2 - coeffs[4]))
        else:
            raise ValueError(f"Formula '{formula}' not implemented.")

    # --- UNIAXIAL CRYSTALS (BBO, LN, etc.) ---
    if "no_coeffs" in data:
        n_o = calc_n(data["no_coeffs"], data["formula"])
        
        if polarization == 'o':
            return n_o
        else:
            n_e_base = calc_n(data["ne_coeffs"], data["formula"])
            # Ellipse bounded by No and Ne
            inv_ne_sq = (np.cos(theta_rad)**2 / n_o**2) + (np.sin(theta_rad)**2 / n_e_base**2)
            return np.sqrt(1 / inv_ne_sq)
            
    # --- BIAXIAL CRYSTALS (KTP, BiBO, etc.) ---
    elif "nx_coeffs" in data:
        n_x = calc_n(data["nx_coeffs"], data["formula"])
        n_y = calc_n(data["ny_coeffs"], data["formula"])
        n_z = calc_n(data["nz_coeffs"], data["formula"])
        
        plane = data.get("pm_plane", "XY") # Default to XY plane if not specified
        
        if plane == "XY":
            if polarization == 'o':
                return n_z
            else:
                # Ellipse bounded by Nx and Ny
                inv_ne_sq = (np.sin(theta_rad)**2 / n_x**2) + (np.cos(theta_rad)**2 / n_y**2)
                return np.sqrt(1 / inv_ne_sq)
                
        elif plane == "XZ":
            if polarization == 'o':
                return n_y
            else:
                # Ellipse bounded by Nx and Nz
                inv_ne_sq = (np.cos(theta_rad)**2 / n_x**2) + (np.sin(theta_rad)**2 / n_z**2)
                return np.sqrt(1 / inv_ne_sq)
            
        elif plane == "YZ":
            if polarization == 'o':
                return n_x
            else:
                # Ellipse bounded by Ny and Nz
                inv_ne_sq = (np.cos(theta_rad)**2 / n_y**2) + (np.sin(theta_rad)**2 / n_z**2)
                return np.sqrt(1 / inv_ne_sq)
                
        

def find_phase_matching_angle(lambda_p_nm, crystal_name):
    data = crystal_dict[crystal_name]
    spdc_type = data["spdc_type"]

    if data.get("is_pp", False):
        return 90.0

    lambda_s_nm = 2 * lambda_p_nm  # degenerate case

    def abs_delta_k_at_theta(theta_deg):
        theta_rad = np.radians(theta_deg)

        if spdc_type == 1:
            # Type-I: pump extraordinary, both outputs ordinary
            n_p = get_refractive_index(lambda_p_nm, crystal_name, 'e', theta_rad)
            n_s = get_refractive_index(lambda_s_nm, crystal_name, 'o', theta_rad)
            n_i = get_refractive_index(lambda_s_nm, crystal_name, 'o', theta_rad)

        elif spdc_type == 2:
            # Type-II: pump extraordinary, signal extraordinary, idler ordinary
            # Pass theta_rad consistently — get_refractive_index ignores it
            # for 'o' polarisation anyway, but this is more correct and explicit
            n_p = get_refractive_index(lambda_p_nm, crystal_name, 'e', theta_rad)
            n_s = get_refractive_index(lambda_s_nm, crystal_name, 'e', theta_rad)
            n_i = get_refractive_index(lambda_s_nm, crystal_name, 'o', theta_rad)

        mismatch = (n_p / lambda_p_nm) - (n_s / lambda_s_nm) - (n_i / lambda_s_nm)
        return abs(mismatch)

    result = minimize_scalar(abs_delta_k_at_theta, bounds=(0, 90), method='bounded')
    if result.fun > 1e-3:   # tune threshold to your units
        import warnings
        warnings.warn(f"No good phase-matching angle found for {crystal_name} at {lambda_p_nm} nm "f"(min |Δk| = {result.fun:.3g}). Output may be unphysical.")
    return result.x


def phase_matching(lambda_p, crystal_name, L_um):

    data = crystal_dict[crystal_name]
    spdc_type = data["spdc_type"] # Extract it directly here too
    
    # Automatically find the optimal crystal tilt angle
    theta_deg = find_phase_matching_angle(lambda_p, crystal_name)
    theta_rad = np.radians(theta_deg)
    
    data = crystal_dict[crystal_name]

    # Assign polarizations based on SPDC type (0=Type 0, 1=Type I, 2=Type II)
    if spdc_type == 1:
        p_pol, s_pol, i_pol = 'e', 'o', 'o'
    elif spdc_type == 2:
        p_pol, s_pol, i_pol = 'e', 'e', 'o'
    
    # Pre-calculate pump wavevector (kp) ONCE
    omega_p = (2 * const.pi * const.c) / (lambda_p * 1e-9)
    n_p = get_refractive_index(lambda_p, crystal_name, p_pol, theta_rad)
    kp = (n_p * omega_p) / (const.c * 1e6)

    # --- Auto-Poling para cristales QPM y cálculo de periodo ---
    delta_k_qpm = 0.0
    poling_period_um = 0.0 # Guardará la Lambda mayúscula
    
    if data.get("is_pp", False):
        lambda_s_center = 2 * lambda_p
        n_s_center = get_refractive_index(lambda_s_center, crystal_name, s_pol, theta_rad)
        n_i_center = get_refractive_index(lambda_s_center, crystal_name, i_pol, theta_rad)
        omega_s_center = omega_p / 2.0
        ks_c = (n_s_center * omega_s_center) / (const.c * 1e6)
        ki_c = (n_i_center * omega_s_center) / (const.c * 1e6)
        delta_k_qpm = kp - ks_c - ki_c
        
        # Λ = 2π / Δk
        if delta_k_qpm != 0:
            poling_period_um = abs((2 * const.pi) / delta_k_qpm)

    # Inner function that computes the actual 2D state matrix
    def phi(omega_s, omega_i):
        WS, WI = np.meshgrid(omega_s, omega_i)
        lambda_s_grid_nm = (2 * const.pi * const.c / WS) * 1e9
        lambda_i_grid_nm = (2 * const.pi * const.c / WI) * 1e9
        
        n_s = get_refractive_index(lambda_s_grid_nm, crystal_name, s_pol, theta_rad)
        n_i = get_refractive_index(lambda_i_grid_nm, crystal_name, i_pol, theta_rad)
        
        ks = (n_s * WS) / (const.c * 1e6)
        ki = (n_i * WI) / (const.c * 1e6)
        
        delta_k = kp - ks - ki
        
        if data.get("is_pp", False):
            delta_k -= delta_k_qpm
            
        arg = (delta_k * L_um) / 2.0
        return np.sinc(arg / const.pi)
        
    # --- TRUCO PYTHON: Añadir los parámetros como atributos a la función ---
    phi.theta_deg = theta_deg
    phi.poling_period = poling_period_um
    
    return phi


def jsa_function(lambda_p, sigma_p, crystal_name, L_um, grid_size=1000, span_factor=5):
    omega_p = 2 * np.pi * const.c / (lambda_p * 1e-9)
    omega_center = omega_p / 2.0
    
    # Now you can adjust the window size from the UI based on the crystal length
    span = span_factor * sigma_p

    # Calculate central pump frequency and the degenerate emission center (wp/2)
    omega_p = 2 * np.pi * const.c / (lambda_p * 1e-9)
    omega_center = omega_p / 2.0
    
    # Build a high-resolution grid zoomed specifically on the active emission zone
    omega_s = np.linspace(omega_center - span, omega_center + span, grid_size)
    omega_i = np.linspace(omega_center - span, omega_center + span, grid_size)
    
    # Instantiate the two physical models using our factory functions
    phi_func = phase_matching(lambda_p, crystal_name, L_um)
    alpha_func = pump_envelope(lambda_p, sigma_p)
    
    # Multiply Momentum Conservation (phi) by Energy Conservation (alpha)
    jsa_matrix = alpha_func(omega_s, omega_i) * phi_func(omega_s, omega_i)
    
    # --- QUANTUM NORMALIZATION BLOCK ---
    # 1. Get the probability density (Joint Spectral Intensity)
    jsi_matrix = np.abs(jsa_matrix)**2
    
    # 2. Integrate over both axes to find the total probability volume.
    # simpson integrates along the rows (omega_s) first, then the columns (omega_i).
    norm_factor = simpson(simpson(jsi_matrix, x=omega_s), x=omega_i)
    
    # 3. Divide the amplitude matrix by the square root of the volume
    jsa_matrix = jsa_matrix / np.sqrt(norm_factor)
    
    # Return the normalized 2D matrix ALONG WITH its physical axes
    return jsa_matrix, omega_s, omega_i


def independent_envelope(lambda_a, lambda_b, sigma_a, sigma_b):
    """
    Generates a normalized 2D Joint Spectral Amplitude matrix for two independent
    photons modeled as Gaussian wavepackets.
    """
    # 1. Convert center wavelengths (nm) to angular frequencies
    omega_a0 = 2 * np.pi * const.c / (lambda_a * 1e-9)
    omega_b0 = 2 * np.pi * const.c / (lambda_b * 1e-9)
    
    # Inner function that matches the signature of your SPDC alpha function
    def alpha(omega_s, omega_i):
        # Create the 2D grid exactly like your SPDC code to prevent broadcasting errors
        WS, WI = np.meshgrid(omega_s, omega_i)
        
        # Calculate the 2D independent wavepackets
        # (Using the same unnormalized exponent format as your pump_envelope)
        phi_a = np.exp(-((WS - omega_a0)**2) / (sigma_a**2))
        phi_b = np.exp(-((WI - omega_b0)**2) / (sigma_b**2))
        
        # The joint state is the multiplication of the two grids (Tensor Product)
        return phi_a * phi_b
        
    return alpha


def independent_jsa_function(lambda_a, lambda_b, sigma_a, sigma_b, grid_size=1000, span_factor=5):
    
    sigma_a_pump = sigma_a
    sigma_b_pump = sigma_b

    # Setup the central frequencies to build the grid
    omega_a0 = 2 * np.pi * const.c / (lambda_a * 1e-9)
    omega_b0 = 2 * np.pi * const.c / (lambda_b * 1e-9)
    
    # Build a shared coordinate window
    w_center_avg = (omega_a0 + omega_b0) / 2.0
    bw_avg = (sigma_a + sigma_b) / 2.0
    span = span_factor * bw_avg
    
    omega_s = np.linspace(w_center_avg - span, w_center_avg + span, grid_size)
    omega_i = np.linspace(w_center_avg - span, w_center_avg + span, grid_size)
    
    # --- The Unified API Action ---
    # 1. Instantiate the closure exactly like you did with pump_envelope
    alpha_func = independent_envelope(lambda_a, lambda_b, sigma_a_pump, sigma_b_pump)
    
    # 2. Evaluate the 2D grid
    jsa_matrix = alpha_func(omega_s, omega_i)
    
    # --- QUANTUM NORMALIZATION BLOCK ---
    jsi_matrix = np.abs(jsa_matrix)**2
    norm_factor = simpson(simpson(jsi_matrix, x=omega_s), x=omega_i)
    jsa_matrix = jsa_matrix / np.sqrt(norm_factor)
    
    return jsa_matrix, omega_s, omega_i


def apply_filter(jsa_matrix, omega_s, omega_i,
                 center_s, width_s,
                 center_i, width_i,
                 shape='gaussian'):
    """
    Apply spectral filters to signal and idler arms of the JSA/JEA.
    Centers and widths are in rad/s.
    After filtering the JSA is renormalised to unit norm.

    Shapes available:
      gaussian     — smooth bell curve, most common experimentally
      rectangular  — ideal bandpass, hard cutoff
      lorentzian   — broad tails, models cavity/resonator filters
      sinc         — sinc^2 profile, models grating-based filters
      super_gaussian — flat top with steep edges (order n=4)
      triangular   — linear rolloff on each side
    """
    from scipy.integrate import simpson

    def make_filter(omega, center, width, shape):
        x = omega - center
        if shape == 'gaussian':
            return np.exp(-(x**2) / (2 * width**2))
        elif shape == 'rectangular':
            return (np.abs(x) < width / 2).astype(float)
        elif shape == 'lorentzian':
            return 1.0 / (1.0 + (x / (width / 2))**2)
        elif shape == 'sinc':
            arg = x / width
            return np.sinc(arg)**2          # sinc^2, zero at x = ±width
        elif shape == 'super_gaussian':
            return np.exp(-(x**2 / (2 * width**2))**4)  # order 4
        elif shape == 'triangular':
            return np.maximum(0.0, 1.0 - np.abs(x) / width)
        else:
            raise ValueError(f"Unknown filter shape: '{shape}'. "
                             f"Choose from: gaussian, rectangular, "
                             f"lorentzian, sinc, super_gaussian, triangular.")

    H_s = make_filter(omega_s, center_s, width_s, shape)   # shape (N_s,)
    H_i = make_filter(omega_i, center_i, width_i, shape)   # shape (N_i,)

    # Broadcast to 2D: jsa_matrix has shape (N_i, N_s)
    jsa_filtered = jsa_matrix * H_s[np.newaxis, :] * H_i[:, np.newaxis]

    norm = simpson(simpson(np.abs(jsa_filtered)**2, x=omega_s), x=omega_i)
    if norm < 1e-30:
        raise ValueError(
            "Filter is too narrow — the filtered JSA has negligible norm. "
            "Increase the filter bandwidth."
        )
    return jsa_filtered / np.sqrt(norm)


def get_marginal_spectra(jsa_matrix, omega_s, omega_i):
    """
    Returns the individual 1D spectra for the signal and idler photons.
    """
    jsi = np.abs(jsa_matrix)**2
    # Integrate out the 'other' axis to see what one photon looks like alone
    marginal_s = simpson(jsi, x=omega_i, axis=1) # Axis 1 is idler
    marginal_i = simpson(jsi, x=omega_s, axis=0) # Axis 0 is signal
    return marginal_s, marginal_i


def get_intrinsic_indistinguishability(jsa_matrix, omega_s, omega_i):
    """
    Calculates the 'swap' overlap: <f(ws,wi) | f(wi,ws)>.
    This is the theoretical maximum depth of your HOM dip.
    """
    # Create the 'swapped' version of the state (transpose)
    jsa_swapped_conj = np.conj(jsa_matrix.T)
    
    # Calculate the overlap volume
    overlap_integrand = np.real(jsa_matrix * jsa_swapped_conj)
    visibility = simpson(simpson(overlap_integrand, x=omega_s), x=omega_i)
    return visibility


def plot_2d_jsi(jsa_matrix, omega_s, omega_i):
    """
    Creates a heatmap of the Joint Spectral Intensity (JSI).
    Shows frequency correlations/entanglement.
    """
    fs_thz = omega_s / (2 * np.pi * 1e12)
    fi_thz = omega_i / (2 * np.pi * 1e12)
    jsi = np.abs(jsa_matrix)**2

    fig = go.Figure(data=go.Heatmap(
        z=jsi, x=fs_thz, y=fi_thz,
        colorscale='Viridis',
        colorbar=dict(title="Intensity")
    ))
    
    fig.update_layout(
        title="Joint Spectral Intensity (JSI)",
        xaxis_title="Signal Frequency (THz)",
        yaxis_title="Idler Frequency (THz)",
        template="plotly_white",
        width=500, height=500
    )
    return fig


def plot_1d_marginals(omega_s, omega_i, marginal_s, marginal_i):
    """
    Plots the two marginal spectra on the same 1D axis.
    Shows if the photons are spectrally matched.
    """
    fs_thz = omega_s / (2 * np.pi * 1e12)
    fi_thz = omega_i / (2 * np.pi * 1e12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fs_thz, y=marginal_s, name="Signal", line=dict(color='#636EFA')))
    fig.add_trace(go.Scatter(x=fi_thz, y=marginal_i, name="Idler", line=dict(color='#EF553B', dash='dash')))

    fig.update_layout(
        title="Marginal Spectra",
        xaxis_title="Frequency (THz)",
        yaxis_title="Intensity (a.u.)",
        template="plotly_white",
        width=600, height=400
    )
    return fig


def plot_swap_kernel(jsa_matrix, omega_s, omega_i):
    """
    Visualizes the swap kernel S(omega_s, omega_i) = f*(omega_s, omega_i) * f(omega_i, omega_s).

    The HOM coincidence rate at tau = 0 is determined by the integral of the
    REAL PART of this kernel: regions where Re(S) > 0 contribute to the dip,
    regions where Re(S) < 0 fight against it. Looking at this plot makes
    "intrinsic indistinguishability" tangible — for an ideal symmetric JSA
    the real part is everywhere positive and the imaginary part is everywhere zero,
    so the integral reaches its maximum value of 1.

    Returns a Plotly figure with two side-by-side heatmaps (Real, Imaginary).
    """
    from plotly.subplots import make_subplots

    fs_thz = omega_s / (2 * np.pi * 1e12)
    fi_thz = omega_i / (2 * np.pi * 1e12)

    # Build the swap kernel: element [i, s] = f*(ws, wi) * f(wi, ws)
    swap = np.conj(jsa_matrix) * jsa_matrix.T

    re_swap = np.real(swap)
    im_swap = np.imag(swap)

    # Symmetric color scale around 0 for both panels (so the diverging colormap is meaningful)
    re_max = np.max(np.abs(re_swap))
    im_max = np.max(np.abs(im_swap))

    # Compute the τ=0 visibility for the title
    dw_s = omega_s[1] - omega_s[0]
    dw_i = omega_i[1] - omega_i[0]
    V = float(np.sum(re_swap) * dw_s * dw_i)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Re[ S ]   →   ∫∫ Re[S] = {V:.3f}",
                        "Im[ S ]   (averages to 0)"),
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Heatmap(
            z=re_swap, x=fs_thz, y=fi_thz,
            colorscale='RdBu_r', zmid=0, zmin=-re_max, zmax=re_max,
            colorbar=dict(title="Re[S]", x=0.43),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=im_swap, x=fs_thz, y=fi_thz,
            colorscale='RdBu_r', zmid=0, zmin=-im_max, zmax=im_max,
            colorbar=dict(title="Im[S]", x=1.0),
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="ωs / 2π (THz)", row=1, col=1)
    fig.update_xaxes(title_text="ωs / 2π (THz)", row=1, col=2)
    fig.update_yaxes(title_text="ωi / 2π (THz)", row=1, col=1)
    fig.update_yaxes(title_text="ωi / 2π (THz)", row=1, col=2)

    fig.update_layout(
        title="Swap kernel  S(ωs, ωi) = f*(ωs, ωi) · f(ωi, ωs)",
        template="plotly_white",
        width=1000, height=450,
    )

    return fig


def plot_delayed_jsa(jsa_matrix, omega_s, omega_i, tau_values=None):
    """
    Shows the JSA *with the time-delay phase imprinted* at several values of tau.

    The HOM dip happens because, as |tau| grows, the phase exp(i*omega_s*tau)
    causes rapid oscillations along the signal axis that wash out the swap-overlap
    integral. Visualizing |JSA| (unchanged) alongside the real or imaginary
    part of (JSA × delay phase) makes that mechanism visible.

    Parameters
    ----------
    jsa_matrix, omega_s, omega_i : as produced by jsa_function or independent_jsa_function
    tau_values : iterable of floats, optional
        Delays to display. Defaults to [0, half-FWHM, full-FWHM] of the dip,
        estimated from the inverse bandwidth of omega_s.

    Returns
    -------
    Plotly figure with one row per tau, two columns (|JSA|, Re[delayed JSA]).
    """
    from plotly.subplots import make_subplots

    fs_thz = omega_s / (2 * np.pi * 1e12)
    fi_thz = omega_i / (2 * np.pi * 1e12)

    # Pick sensible default delays if user didn't specify any
    if tau_values is None:
        # Use the spectral width of the JSI's signal marginal as a coherence-time proxy
        jsi = np.abs(jsa_matrix)**2
        marg_s = simpson(jsi, x=omega_i, axis=0)
        # Approx FWHM of the marginal in rad/s
        peak = marg_s.max()
        above = np.where(marg_s > peak / 2)[0]
        if len(above) > 1:
            fwhm_omega = omega_s[above[-1]] - omega_s[above[0]]
        else:
            fwhm_omega = omega_s[-1] - omega_s[0]
        # Coherence time ~ 1 / FWHM (no factor 2π since we're already in rad/s)
        tau_c = 2 * np.pi / max(fwhm_omega, 1e6)
        tau_values = [0.0, tau_c / 2, tau_c]

    n_rows = len(tau_values)

    subplot_titles = []
    for tau in tau_values:
        subplot_titles += [
            f"|JSA|     (τ = {tau*1e12:.2f} ps)",
            f"Re[ JSA · e<sup>iωsτ</sup> ]  (τ = {tau*1e12:.2f} ps)",
        ]

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15, vertical_spacing=0.10,
    )

    abs_max = np.max(np.abs(jsa_matrix))

    for row_idx, tau in enumerate(tau_values, start=1):
        # Apply the delay phase along the signal axis (column axis of jsa_matrix)
        phase = np.exp(1j * omega_s * tau)
        delayed = jsa_matrix * phase[np.newaxis, :]

        # Left column: magnitude (unchanged by phase, but we re-show it for visual reference)
        fig.add_trace(
            go.Heatmap(
                z=np.abs(jsa_matrix), x=fs_thz, y=fi_thz,
                colorscale='Viridis', zmin=0, zmax=abs_max,
                showscale=(row_idx == 1),
                colorbar=dict(title="|JSA|", x=0.43) if row_idx == 1 else None,
            ),
            row=row_idx, col=1,
        )

        # Right column: real part of the delayed JSA — phase ripples appear here
        re_max = np.max(np.abs(np.real(delayed)))
        fig.add_trace(
            go.Heatmap(
                z=np.real(delayed), x=fs_thz, y=fi_thz,
                colorscale='RdBu_r', zmid=0, zmin=-re_max, zmax=re_max,
                showscale=(row_idx == 1),
                colorbar=dict(title="Re[delayed JSA]", x=1.0) if row_idx == 1 else None,
            ),
            row=row_idx, col=2,
        )

        fig.update_xaxes(title_text="ωs / 2π (THz)", row=row_idx, col=1)
        fig.update_xaxes(title_text="ωs / 2π (THz)", row=row_idx, col=2)
        fig.update_yaxes(title_text="ωi / 2π (THz)", row=row_idx, col=1)
        fig.update_yaxes(title_text="ωi / 2π (THz)", row=row_idx, col=2)

    fig.update_layout(
        title="JSA with applied time-delay phase  e<sup>iωsτ</sup>",
        template="plotly_white",
        width=1000, height=380 * n_rows,
    )

    return fig


def hom_coincidence_rate(jsa_matrix, omega_s, omega_i, tau_array,
                         R=0.5, V_pol=1.0, statistics='boson'):
    T = 1.0 - R
    dw_s = omega_s[1] - omega_s[0]
    dw_i = omega_i[1] - omega_i[0]
    swap_kernel = np.conj(jsa_matrix) * jsa_matrix.T
    OS = omega_s[np.newaxis, :]
    OI = omega_i[:, np.newaxis]
    sign = -1.0 if statistics == 'boson' else +1.0   # bosón: dip; fermión: pico
    rate = np.empty_like(tau_array, dtype=float)
    for k, tau in enumerate(tau_array):
        phase = np.exp(1j * (OI - OS) * tau)
        overlap = np.sum(swap_kernel * phase) * dw_s * dw_i
        rate[k] = (T**2 + R**2) + sign * V_pol * 2.0 * R * T * np.real(overlap)
    return rate


def leviton_envelope(varepsilon, varepsilon_F, varepsilon_0, tau_0):

    hbar = const.hbar
    phi  = np.where(
        varepsilon >= varepsilon_F,
        np.exp(-tau_0 * (varepsilon - varepsilon_F) / hbar),
        0.0
    ).astype(complex)
    return phi


def gaussian_electron(varepsilon, varepsilon_0, sigma):

    return np.exp(-(varepsilon - varepsilon_0)**2 / (4 * sigma**2)).astype(complex)


def lorentzian_electron(varepsilon, varepsilon_F, varepsilon_0, Gamma):

    phi = np.where(
        varepsilon >= varepsilon_F,
        1.0 / (varepsilon - varepsilon_0 + 1j * Gamma / 2),
        0.0
    ).astype(complex)
    return phi


def independent_jea_function(varepsilon_0a, varepsilon_0b, shape_a, shape_b, params_a, params_b, varepsilon_F=0.0, grid_size=500, span_factor=15):

    # Build energy grid
    bw_a = params_a.get('sigma', params_a.get('Gamma', const.hbar / params_a.get('tau_0', 1e-12)))
    bw_b = params_b.get('sigma', params_b.get('Gamma', const.hbar / params_b.get('tau_0', 1e-12)))
    bw_avg    = (bw_a + bw_b) / 2.0
    e_center  = (varepsilon_0a + varepsilon_0b) / 2.0
    span      = span_factor * bw_avg

    varepsilon = np.linspace(max(varepsilon_F, e_center - span), e_center + span, grid_size)

    def make_phi(shape, varepsilon_0, params):
        if shape == 'gaussian':
            phi = gaussian_electron(varepsilon, varepsilon_0, params['sigma'])
        elif shape == 'lorentzian':
            phi = lorentzian_electron(varepsilon, varepsilon_F, varepsilon_0, params['Gamma'])
        elif shape == 'leviton':
            phi = leviton_envelope(varepsilon, varepsilon_F, varepsilon_0, params['tau_0'])
            
        else:
            raise ValueError(f"Unknown shape: {shape}")
        # Normalise
        de   = varepsilon[1] - varepsilon[0]
        norm = np.sqrt(np.sum(np.abs(phi)**2) * de)
        return phi / norm

    phi_a = make_phi(shape_a, varepsilon_0a, params_a)
    phi_b = make_phi(shape_b, varepsilon_0b, params_b)

    # Amplitud PRODUCTO: los dos electrones entran por puertos distintos, así que
    # la amplitud conjunta es un producto (igual que los fotones independientes).
    # La antisimetría NO va aquí: entra como el signo del término de interferencia
    # en hom_coincidence_rate (la anticonmutación de los operadores c).
    jea_matrix = phi_a[np.newaxis, :] * phi_b[:, np.newaxis]   # shape (N_i, N_s)

    de = varepsilon[1] - varepsilon[0]
    norm = np.sqrt(np.sum(np.abs(jea_matrix)**2) * de**2)

    return jea_matrix / norm, varepsilon, varepsilon


