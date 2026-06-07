"""
hom_physics.py — Núcleo físico del simulador del efecto Hong-Ou-Mandel.

Convenciones unificadas en esta versión
---------------------------------------
* Ancho de banda gaussiano: en TODAS las amplitudes gaussianas (bombeo SPDC,
  fotones independientes y electrones gaussianos) la amplitud es
        phi(x) ∝ exp[-(x - x0)^2 / (4 sigma^2)]
  de modo que |phi|^2 ∝ exp[-(x-x0)^2 / (2 sigma^2)] y por tanto
        sigma  ==  desviación estándar del espectro de INTENSIDAD.
  Es decir, "sigma" significa lo mismo (ancho de banda en intensidad) en las
  dos pestañas del simulador.

* Función de phase-matching: se usa Phi = sinc(Delta k L / 2) REAL. La fase
  exp(i Delta k L / 2) (origen en la cara de entrada del cristal) se descarta:
  su parte lineal es un retardo de grupo trivialmente compensable que solo
  desplaza el dip en tau, no su profundidad. La visibilidad observable se
  reporta como el mínimo de la curva sobre tau (no el valor en tau=0), lo que
  recoge la asimetría real de tipo II sin contaminarla con ese retardo.

* Estadística fermiónica: la amplitud de dos partículas se modela como un
  ESTADO PRODUCTO phi_a ⊗ phi_b (igual que el bosónico). El signo fermiónico
  NO se introduce antisimetrizando la amplitud, sino como un cambio de signo
  global del término de interferencia (anticonmutación) en hom_coincidence_rate.
  Por tanto V >= 0 universalmente y el pico de antibunching proviene del signo +.
"""

import numpy as np
import scipy.constants as const
from scipy.optimize import minimize_scalar


crystal_dict = {
    "BBO (I)": {
        "formula": "borate",
        "no_coeffs": [2.7359, 0.01878, 0.01822, 0.01354],
        "ne_coeffs": [2.3753, 0.01224, 0.01667, 0.01516],
        "is_pp": False,
        "std_pump": 405.0,
        "spdc_type": 1,
    },
    "BBO (II)": {
        "formula": "borate",
        "no_coeffs": [2.7359, 0.01878, 0.01822, 0.01354],
        "ne_coeffs": [2.3753, 0.01224, 0.01667, 0.01516],
        "is_pp": False,
        "std_pump": 405.0,
        "spdc_type": 2,
    },
    "KTP (II)": {
        "formula": "kato_ktp",
        "nx_coeffs": [3.29100, 0.04140, 0.03978, 9.35522, 31.45571],
        "ny_coeffs": [3.45018, 0.04341, 0.04597, 16.98825, 39.43799],
        "nz_coeffs": [4.59423, 0.06206, 0.04763, 110.80672, 86.12171],
        "pm_plane": "XY",
        "is_pp": False,
        "std_pump": 532.0,
        "spdc_type": 2,
    },
    "PPKTP (II)": {
        "formula": "kato_ktp",
        "nx_coeffs": [3.29100, 0.04140, 0.03978, 9.35522, 31.45571],
        "ny_coeffs": [3.45018, 0.04341, 0.04597, 16.98825, 39.43799],
        "nz_coeffs": [4.59423, 0.06206, 0.04763, 110.80672, 86.12171],
        "pm_plane": "XY",
        "is_pp": True,
        "std_pump": 405.0,
        "spdc_type": 2,
    },
    "LBO (I)": {
        "formula": "borate",
        "nx_coeffs": [2.4542, 0.01125, 0.01135, 0.01388],
        "ny_coeffs": [2.5390, 0.01277, 0.01189, 0.01848],
        "nz_coeffs": [2.5865, 0.01310, 0.01223, 0.01861],
        "pm_plane": "XY",
        "is_pp": False,
        "std_pump": 532.0,
        "spdc_type": 1,
    },
}


# ──────────────────────────────────────────────────────────────────────────
#  BOSONES — SPDC
# ──────────────────────────────────────────────────────────────────────────

def pump_envelope(lambda_p, sigma_p):
    """Envolvente de bombeo (conservación de energía).

    sigma_p es la desviación estándar del espectro de INTENSIDAD del bombeo
    (rad/s): amplitud ∝ exp[-(w_s + w_i - w_p)^2 / (4 sigma_p^2)].
    """
    omega_p = 2 * np.pi * const.c / (lambda_p * 1e-9)

    def alpha(omega_s, omega_i):
        WS, WI = np.meshgrid(omega_s, omega_i)
        return np.exp(-((WI + WS - omega_p) ** 2) / (4 * sigma_p ** 2))

    return alpha


def get_refractive_index(lambda_x, crystal_name, polarization, theta_rad=0.0):
    """Índice de refracción vía Sellmeier; maneja cristales uniáxicos y biáxicos."""
    data = crystal_dict[crystal_name]
    lam_um = lambda_x / 1000.0   # Sellmeier en micras

    def calc_n(coeffs, formula):
        if formula == "borate":
            return np.sqrt(coeffs[0] + coeffs[1] / (lam_um ** 2 - coeffs[2]) - coeffs[3] * lam_um ** 2)
        elif formula == "kato_ktp":
            return np.sqrt(coeffs[0] + coeffs[1] / (lam_um ** 2 - coeffs[2]) + coeffs[3] / (lam_um ** 2 - coeffs[4]))
        else:
            raise ValueError(f"Formula '{formula}' no implementada.")

    # --- Uniáxicos (BBO, ...) ---
    if "no_coeffs" in data:
        n_o = calc_n(data["no_coeffs"], data["formula"])
        if polarization == 'o':
            return n_o
        n_e_base = calc_n(data["ne_coeffs"], data["formula"])
        inv_ne_sq = (np.cos(theta_rad) ** 2 / n_o ** 2) + (np.sin(theta_rad) ** 2 / n_e_base ** 2)
        return np.sqrt(1 / inv_ne_sq)

    # --- Biáxicos (KTP, LBO, ...) ---
    elif "nx_coeffs" in data:
        n_x = calc_n(data["nx_coeffs"], data["formula"])
        n_y = calc_n(data["ny_coeffs"], data["formula"])
        n_z = calc_n(data["nz_coeffs"], data["formula"])
        plane = data.get("pm_plane", "XY")

        if plane == "XY":
            if polarization == 'o':
                return n_z
            inv_ne_sq = (np.sin(theta_rad) ** 2 / n_x ** 2) + (np.cos(theta_rad) ** 2 / n_y ** 2)
            return np.sqrt(1 / inv_ne_sq)
        elif plane == "XZ":
            if polarization == 'o':
                return n_y
            inv_ne_sq = (np.cos(theta_rad) ** 2 / n_x ** 2) + (np.sin(theta_rad) ** 2 / n_z ** 2)
            return np.sqrt(1 / inv_ne_sq)
        elif plane == "YZ":
            if polarization == 'o':
                return n_x
            inv_ne_sq = (np.cos(theta_rad) ** 2 / n_y ** 2) + (np.sin(theta_rad) ** 2 / n_z ** 2)
            return np.sqrt(1 / inv_ne_sq)

    raise ValueError(f"Cristal '{crystal_name}' sin coeficientes Sellmeier válidos.")


def _pol_assignment(spdc_type):
    """Polarizaciones (bombeo, señal, idler) según el tipo de SPDC."""
    if spdc_type == 1:        # tipo I: bombeo e, ambos o
        return 'e', 'o', 'o'
    elif spdc_type == 2:      # tipo II: bombeo e, señal e, idler o
        return 'e', 'e', 'o'
    raise ValueError(f"spdc_type={spdc_type} no soportado (solo 1 y 2).")


def find_phase_matching_angle(lambda_p_nm, crystal_name):
    data = crystal_dict[crystal_name]
    spdc_type = data["spdc_type"]

    if data.get("is_pp", False):
        return 90.0   # los cristales periódicamente polados trabajan a 90° (QPM)

    p_pol, s_pol, i_pol = _pol_assignment(spdc_type)
    lambda_s_nm = 2 * lambda_p_nm   # degenerado

    def abs_delta_k_at_theta(theta_deg):
        theta_rad = np.radians(theta_deg)
        n_p = get_refractive_index(lambda_p_nm, crystal_name, p_pol, theta_rad)
        n_s = get_refractive_index(lambda_s_nm, crystal_name, s_pol, theta_rad)
        n_i = get_refractive_index(lambda_s_nm, crystal_name, i_pol, theta_rad)
        mismatch = (n_p / lambda_p_nm) - (n_s / lambda_s_nm) - (n_i / lambda_s_nm)
        return abs(mismatch)

    result = minimize_scalar(abs_delta_k_at_theta, bounds=(0, 90), method='bounded')
    if result.fun > 1e-3:
        import warnings
        warnings.warn(
            f"No se encontró buen ángulo de phase-matching para {crystal_name} "
            f"a {lambda_p_nm} nm (min |Δk|·λ = {result.fun:.3g}). La salida puede ser no física."
        )
    return result.x


def phase_matching(lambda_p, crystal_name, L_um):
    """Devuelve un closure phi(omega_s, omega_i) = sinc(Delta k L / 2) (real).

    La fase exp(i Delta k L/2) se descarta (ver cabecera del módulo).
    Atributos añadidos al closure: .theta_deg y .poling_period (um).
    """
    data = crystal_dict[crystal_name]
    spdc_type = data["spdc_type"]

    theta_deg = find_phase_matching_angle(lambda_p, crystal_name)
    theta_rad = np.radians(theta_deg)

    p_pol, s_pol, i_pol = _pol_assignment(spdc_type)

    omega_p = (2 * const.pi * const.c) / (lambda_p * 1e-9)
    n_p = get_refractive_index(lambda_p, crystal_name, p_pol, theta_rad)
    kp = (n_p * omega_p) / (const.c * 1e6)   # rad/um

    # --- Auto-poling para cristales QPM ---
    delta_k_qpm = 0.0
    poling_period_um = 0.0
    if data.get("is_pp", False):
        lambda_s_center = 2 * lambda_p
        n_s_center = get_refractive_index(lambda_s_center, crystal_name, s_pol, theta_rad)
        n_i_center = get_refractive_index(lambda_s_center, crystal_name, i_pol, theta_rad)
        omega_s_center = omega_p / 2.0
        ks_c = (n_s_center * omega_s_center) / (const.c * 1e6)
        ki_c = (n_i_center * omega_s_center) / (const.c * 1e6)
        delta_k_qpm = kp - ks_c - ki_c
        if delta_k_qpm != 0:
            poling_period_um = abs((2 * const.pi) / delta_k_qpm)

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
        return np.sinc(arg / const.pi)   # sinc real; se descarta la fase

    phi.theta_deg = theta_deg
    phi.poling_period = poling_period_um
    return phi


def _sinc_central_halfwidth(phi_func, omega_p, omega_center, sigma_p):
    """Semiancho (rad/s) del lóbulo central del sinc a lo largo de la antidiagonal
    (suma = omega_p). Mide hasta el primer punto donde |Phi| cae por debajo de 0.05,
    evitando perseguir las colas lentas del sinc.
    """
    probe = max(30.0 * sigma_p, 0.06 * omega_center)
    d = np.linspace(0.0, probe, 300)
    ws = omega_center + d
    wi = omega_p - ws
    Phi = np.abs(np.diag(phi_func(ws, wi)))
    if Phi[0] <= 0:
        return probe
    Phi /= Phi[0]
    below = np.where(Phi < 0.05)[0]
    return d[below[0]] if len(below) > 0 else probe


def _estimate_spdc_window(phi_func, omega_p, omega_center, sigma_p, spdc_type, grid_size):
    """Semiancho de ventana (rad/s) para la malla cuadrada (omega_s, omega_i).

    Tipo II: ventana = lóbulo central del sinc + ancho de bombeo (captura la
        asimetría que reduce V; el lóbulo es estrecho, así que el bombeo queda
        bien resuelto).
    Tipo I: el sinc es muy ancho y el estado es simétrico (V=1) — capturar todo el
        sinc submuestrearía el bombeo y daría artefactos. Se limita la ventana para
        mantener el bombeo resuelto (la profundidad del dip a 0 y V=1 se conservan).
    """
    d_fz = _sinc_central_halfwidth(phi_func, omega_p, omega_center, sigma_p)
    span = 1.3 * (d_fz + 3.0 * sigma_p)
    span = max(span, 6.0 * sigma_p)
    if spdc_type == 1:
        span = min(span, grid_size * sigma_p / 15.0)   # ~10 celdas por sigma_p (cuadratura rect. fiable)
    return span


def jsa_function(lambda_p, sigma_p, crystal_name, L_um, grid_size=400, span_factor=None):
    """JSA normalizada de SPDC y sus ejes (omega_s, omega_i en rad/s).

    La ventana se autodimensiona para contener toda la estructura (bombeo + sinc).
    span_factor se ignora salvo que se pase explícitamente (override manual).
    """
    omega_p = 2 * np.pi * const.c / (lambda_p * 1e-9)
    omega_center = omega_p / 2.0

    phi_func = phase_matching(lambda_p, crystal_name, L_um)
    alpha_func = pump_envelope(lambda_p, sigma_p)

    if span_factor is None:
        span = _estimate_spdc_window(phi_func, omega_p, omega_center, sigma_p,
                                     crystal_dict[crystal_name]["spdc_type"], grid_size)
    else:
        span = span_factor * sigma_p

    omega_s = np.linspace(omega_center - span, omega_center + span, grid_size)
    omega_i = np.linspace(omega_center - span, omega_center + span, grid_size)

    jsa_matrix = alpha_func(omega_s, omega_i) * phi_func(omega_s, omega_i)

    dw_s = omega_s[1] - omega_s[0]
    dw_i = omega_i[1] - omega_i[0]
    norm_factor = np.sum(np.abs(jsa_matrix) ** 2) * dw_s * dw_i   # cuadratura coherente con V
    jsa_matrix = jsa_matrix / np.sqrt(norm_factor)
    return jsa_matrix, omega_s, omega_i


# ──────────────────────────────────────────────────────────────────────────
#  BOSONES — fotones independientes
# ──────────────────────────────────────────────────────────────────────────

def independent_envelope(lambda_a, lambda_b, sigma_a, sigma_b):
    """Closure de la JSA producto de dos fotones gaussianos independientes.

    sigma_a, sigma_b: desviación estándar del espectro de intensidad (rad/s).
    Amplitud ∝ exp[-(w - w0)^2 / (4 sigma^2)].
    """
    omega_a0 = 2 * np.pi * const.c / (lambda_a * 1e-9)
    omega_b0 = 2 * np.pi * const.c / (lambda_b * 1e-9)

    def alpha(omega_s, omega_i):
        WS, WI = np.meshgrid(omega_s, omega_i)
        phi_a = np.exp(-((WS - omega_a0) ** 2) / (4 * sigma_a ** 2))
        phi_b = np.exp(-((WI - omega_b0) ** 2) / (4 * sigma_b ** 2))
        return phi_a * phi_b

    return alpha


def independent_jsa_function(lambda_a, lambda_b, sigma_a, sigma_b, grid_size=400, span_factor=6):
    omega_a0 = 2 * np.pi * const.c / (lambda_a * 1e-9)
    omega_b0 = 2 * np.pi * const.c / (lambda_b * 1e-9)

    w_center = (omega_a0 + omega_b0) / 2.0
    bw = (sigma_a + sigma_b) / 2.0
    detune = abs(omega_a0 - omega_b0)
    span = span_factor * bw + detune   # asegura que ambos picos entran en la ventana

    omega_s = np.linspace(w_center - span, w_center + span, grid_size)
    omega_i = np.linspace(w_center - span, w_center + span, grid_size)

    alpha_func = independent_envelope(lambda_a, lambda_b, sigma_a, sigma_b)
    jsa_matrix = alpha_func(omega_s, omega_i)

    dw_s = omega_s[1] - omega_s[0]
    dw_i = omega_i[1] - omega_i[0]
    norm_factor = np.sum(np.abs(jsa_matrix) ** 2) * dw_s * dw_i   # cuadratura coherente con V
    jsa_matrix = jsa_matrix / np.sqrt(norm_factor)
    return jsa_matrix, omega_s, omega_i


def gaussian_overlap_V(lambda_a, lambda_b, sigma_a, sigma_b):
    """V analítica = |<phi_a|phi_b>|^2 para dos gaussianas (convención sigma = std intensidad).

    sigma_a, sigma_b en rad/s. Sirve de comprobación cruzada del V_swap numérico.
        V = (2 sa sb / (sa^2 + sb^2)) * exp[-(dw)^2 / (2 (sa^2 + sb^2))]
    """
    wa = 2 * np.pi * const.c / (lambda_a * 1e-9)
    wb = 2 * np.pi * const.c / (lambda_b * 1e-9)
    dw = wa - wb
    pref = 2 * sigma_a * sigma_b / (sigma_a ** 2 + sigma_b ** 2)
    return pref * np.exp(-(dw ** 2) / (2 * (sigma_a ** 2 + sigma_b ** 2)))


# ──────────────────────────────────────────────────────────────────────────
#  Filtros espectrales
# ──────────────────────────────────────────────────────────────────────────

def apply_filter(jsa_matrix, omega_s, omega_i,
                 center_s, width_s, center_i, width_i, shape='gaussian'):
    """Aplica filtros espectrales a señal e idler y renormaliza. Anchuras en rad/s."""

    def make_filter(omega, center, width, shape):
        x = omega - center
        if shape == 'gaussian':
            return np.exp(-(x ** 2) / (2 * width ** 2))
        elif shape == 'rectangular':
            return (np.abs(x) < width / 2).astype(float)
        elif shape == 'lorentzian':
            return 1.0 / (1.0 + (x / (width / 2)) ** 2)
        elif shape == 'sinc':
            return np.sinc(x / width) ** 2
        elif shape == 'super_gaussian':
            return np.exp(-(x ** 2 / (2 * width ** 2)) ** 4)
        elif shape == 'triangular':
            return np.maximum(0.0, 1.0 - np.abs(x) / width)
        else:
            raise ValueError(f"Forma de filtro desconocida: '{shape}'.")

    H_s = make_filter(omega_s, center_s, width_s, shape)
    H_i = make_filter(omega_i, center_i, width_i, shape)
    jsa_filtered = jsa_matrix * H_s[np.newaxis, :] * H_i[:, np.newaxis]

    dw_s = omega_s[1] - omega_s[0]
    dw_i = omega_i[1] - omega_i[0]
    norm = np.sum(np.abs(jsa_filtered) ** 2) * dw_s * dw_i   # cuadratura coherente con V
    if norm < 1e-30:
        raise ValueError("El filtro es demasiado estrecho — la JSA filtrada tiene norma "
                         "despreciable. Aumenta el ancho del filtro.")
    return jsa_filtered / np.sqrt(norm)


# ──────────────────────────────────────────────────────────────────────────
#  Diagnósticos comunes (bosones y fermiones)
# ──────────────────────────────────────────────────────────────────────────

def get_marginal_spectra(jsa_matrix, omega_s, omega_i):
    """Espectros marginales 1D de señal e idler (o electrón A y B).

    Usa suma rectangular, no Simpson. Los pesos alternantes 4-2-4-2 de Simpson,
    aplicados a la cresta estrecha y antidiagonal de la JSI —cuyo pico cambia de
    paridad en cada columna—, inyectan un rizado a frecuencia de Nyquist en la
    marginal. La suma rectangular no tiene ese término alternante y, además, es
    la misma cuadratura que la normalización y V.
    """
    jsi = np.abs(jsa_matrix) ** 2
    dw_s = omega_s[1] - omega_s[0]
    dw_i = omega_i[1] - omega_i[0]
    marginal_s = np.sum(jsi, axis=0) * dw_i   # integra sobre idler (eje 0)
    marginal_i = np.sum(jsi, axis=1) * dw_s   # integra sobre señal (eje 1)
    return marginal_s, marginal_i


def get_intrinsic_indistinguishability(jsa_matrix, omega_s, omega_i):
    """Solapamiento de intercambio V = ∫∫ Re[f(ws,wi) f*(wi,ws)] (en tau=0).

    Para un estado simétrico/separable da 0 <= V <= 1. NO es necesariamente la
    profundidad máxima del dip cuando la JSA es asimétrica (tipo II): para eso se
    usa el mínimo de la curva sobre tau (ver hom_coincidence_rate). Usa la misma
    cuadratura (suma rectangular) que hom_coincidence_rate, de modo que V coincide
    exactamente con el valor de la curva en tau=0.
    """
    dw_s = omega_s[1] - omega_s[0]
    dw_i = omega_i[1] - omega_i[0]
    overlap_integrand = np.real(jsa_matrix * np.conj(jsa_matrix.T))
    return float(np.sum(overlap_integrand) * dw_s * dw_i)


def hom_coincidence_rate(jsa_matrix, omega_s, omega_i, tau_array,
                         R=0.5, V_pol=1.0, statistics='boson'):
    """Probabilidad de coincidencia frente al retardo tau.

    P_c(tau) = (T^2 + R^2) + s · V_pol · 2RT · Re[∫∫ f*(ws,wi) f(wi,ws) e^{i(wi-ws)tau}]
    con s = -1 (bosones, dip) o s = +1 (fermiones, pico de antibunching).

    El JSA/JEA de entrada es siempre un producto/estado simétrico; la estadística
    entra ÚNICAMENTE por el signo s (anticonmutación), no por antisimetrizar f.

    Para electrones: pasar tau_array ya dividido por hbar y los ejes en energía (J).

    Implementación: como omega_s y omega_i comparten la misma malla uniforme, el
    integral sobre tau es la transformada de Fourier 1D de las sumas diagonales del
    kernel de intercambio K[i,j] (wi-ws = (i-j)·dw constante en cada diagonal):
        overlap(tau) = dw^2 · Σ_m h[m] e^{i m dw tau},   h[m] = Σ_{i-j=m} K[i,j].
    Esto evita el bucle 2D por cada tau y hace el cálculo interactivo.
    """
    T = 1.0 - R
    dw = omega_s[1] - omega_s[0]   # misma malla en ambos ejes
    N = jsa_matrix.shape[0]

    K = np.conj(jsa_matrix) * jsa_matrix.T   # kernel de intercambio (N_i, N_s)

    # h[m] = suma de la diagonal con desplazamiento (i - j = m), m = -(N-1)..(N-1)
    offsets = np.arange(-(N - 1), N)
    h = np.array([np.trace(K, offset=-m) for m in offsets])   # complejo, longitud 2N-1

    # overlap(tau) = dw^2 * Σ_m h[m] exp(i m dw tau)  — vectorizado sobre tau
    phase = np.exp(1j * np.outer(offsets * dw, tau_array))     # (2N-1, n_tau)
    overlap = (h @ phase) * dw ** 2                            # (n_tau,)

    sign = -1.0 if statistics == 'boson' else +1.0
    rate = (T ** 2 + R ** 2) + sign * V_pol * 2.0 * R * T * np.real(overlap)

    # P_c es una probabilidad: recorta ruido numérico residual por debajo de 0.
    return np.maximum(rate, 0.0)


# ──────────────────────────────────────────────────────────────────────────
#  FERMIONES — paquetes de onda electrónicos (energías en J, tiempos en s)
# ──────────────────────────────────────────────────────────────────────────

def gaussian_electron(varepsilon, varepsilon_0, sigma):
    """Gaussiano: amplitud ∝ exp[-(e-e0)^2/(4 sigma^2)]; sigma = std de intensidad."""
    return np.exp(-(varepsilon - varepsilon_0) ** 2 / (4 * sigma ** 2)).astype(complex)


def lorentzian_electron(varepsilon, varepsilon_F, varepsilon_0, Gamma):
    """Lorentziana COMPLEJA (fuente tipo capacitor mesoscópico):
        phi(e) ∝ 1 / (e - e0 + i Gamma/2)   para e >= eF, 0 en otro caso.
    Gamma es la anchura a media altura (FWHM) del espectro de intensidad.
    """
    phi = np.where(
        varepsilon >= varepsilon_F,
        1.0 / (varepsilon - varepsilon_0 + 1j * Gamma / 2.0),
        0.0,
    ).astype(complex)
    return phi


def leviton_envelope(varepsilon, varepsilon_F, tau_0):
    """Levitón (excitación mínima): phi(e) ∝ exp[-tau_0 (e - eF)/hbar] para e >= eF.

    Anclado al nivel de Fermi (no tiene parámetro e0): su contenido energético
    queda fijado por tau_0 a través de la escala hbar/tau_0.
    """
    hbar = const.hbar
    phi = np.where(
        varepsilon >= varepsilon_F,
        np.exp(-tau_0 * (varepsilon - varepsilon_F) / hbar),
        0.0,
    ).astype(complex)
    return phi


def _electron_center(shape, varepsilon_0, varepsilon_F, params):
    """Centro energético efectivo para construir/centrar la malla."""
    if shape == 'leviton':
        return varepsilon_F + const.hbar / params['tau_0']
    return varepsilon_0


def _electron_bandwidth(shape, params):
    """Escala de ancho de banda en energía para dimensionar la malla."""
    if shape == 'gaussian':
        return params['sigma']
    if shape == 'lorentzian':
        return params['Gamma']
    if shape == 'leviton':
        return const.hbar / params['tau_0']
    raise ValueError(f"Forma desconocida: {shape}")


def independent_jea_function(varepsilon_0a, varepsilon_0b, shape_a, shape_b,
                             params_a, params_b, varepsilon_F=0.0,
                             grid_size=500, span_factor=15):
    """JEA producto normalizada de dos electrones y sus ejes (energía en J).

    Estado PRODUCTO phi_a ⊗ phi_b (no determinante de Slater): la antisimetría
    fermiónica se aplica como cambio de signo del término de interferencia en
    hom_coincidence_rate, no aquí.
    """
    bw_a = _electron_bandwidth(shape_a, params_a)
    bw_b = _electron_bandwidth(shape_b, params_b)
    c_a = _electron_center(shape_a, varepsilon_0a, varepsilon_F, params_a)
    c_b = _electron_center(shape_b, varepsilon_0b, varepsilon_F, params_b)

    bw_avg = (bw_a + bw_b) / 2.0
    e_center = (c_a + c_b) / 2.0
    span = span_factor * bw_avg + abs(c_a - c_b)

    varepsilon = np.linspace(max(varepsilon_F, e_center - span), e_center + span, grid_size)

    def make_phi(shape, varepsilon_0, params):
        if shape == 'gaussian':
            phi = gaussian_electron(varepsilon, varepsilon_0, params['sigma'])
        elif shape == 'lorentzian':
            phi = lorentzian_electron(varepsilon, varepsilon_F, varepsilon_0, params['Gamma'])
        elif shape == 'leviton':
            phi = leviton_envelope(varepsilon, varepsilon_F, params['tau_0'])
        else:
            raise ValueError(f"Forma desconocida: {shape}")
        de = varepsilon[1] - varepsilon[0]
        norm = np.sqrt(np.sum(np.abs(phi) ** 2) * de)
        if norm < 1e-300:
            raise ValueError("Paquete con norma nula (revisa energías/anchuras frente a eF).")
        return phi / norm

    phi_a = make_phi(shape_a, varepsilon_0a, params_a)
    phi_b = make_phi(shape_b, varepsilon_0b, params_b)

    jea_matrix = phi_a[np.newaxis, :] * phi_b[:, np.newaxis]   # (N_i, N_s) producto

    de = varepsilon[1] - varepsilon[0]
    norm = np.sqrt(np.sum(np.abs(jea_matrix) ** 2) * de ** 2)
    return jea_matrix / norm, varepsilon, varepsilon
