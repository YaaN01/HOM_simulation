"""
Hong-Ou-Mandel Effect Simulator — Streamlit app.

Características principales
--------------------------
* Parámetros físicos mediante number_input (sin sliders), con botón de reset.
* Eje de retardo τ con rango (mínimo y máximo) EDITABLE junto a la gráfica,
  por defecto de -1000 a 1000 (fs para fotones, ps para electrones). Es un
  ajuste de vista, no un parámetro físico, y por eso se sitúa junto a la curva.
* Escala electrónica realista: energías en μeV, tiempos en ps.
* Convención de σ unificada: σ = desviación estándar del espectro de intensidad.
* En el régimen fermiónico la forma del espectro energético es ÚNICA y común a
  los dos electrones (no se mezclan formas); sus energías y anchuras sí pueden
  diferir.
* JSI/JEI con ventana autodimensionada y zoom interactivo (Plotly).
* Visibilidad reportada como el MÍNIMO (dip) o MÁXIMO (pico) de la curva sobre τ.
* Todas las figuras se renderizan cuadradas.
"""
import io
import numpy as np
import scipy.constants as const
import plotly.graph_objects as go
import streamlit as st

from hom_physics import (
    crystal_dict,
    jsa_function,
    independent_jsa_function,
    independent_jea_function,
    apply_filter,
    hom_coincidence_rate,
    get_intrinsic_indistinguishability,
    get_marginal_spectra,
    gaussian_overlap_V,
    phase_matching,
)

# ──────────────────────────────────────────────────────────────────────
# Constantes globales
# ──────────────────────────────────────────────────────────────────────
TAU_DEFAULT_FS = 1000.0    # semiventana τ por defecto, fotones (fs)
TAU_DEFAULT_PS = 1000.0    # semiventana τ por defecto, electrones (ps)
TAU_BOUND      = 10000.0   # tope de los campos de rango τ
N_TAU          = 351       # nº de puntos del barrido en τ
GRID           = 500       # tamaño de la malla espectral del cálculo
FIG_SIZE       = 560       # lado (px) de las figuras; se fuerza cuadrado
HEATMAP_MAX    = 800       # máx. celdas/eje a Plotly (inerte a GRID=500; protege si se sube N)

# ──────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HOM Simulator", page_icon="🔬", layout="wide")
st.title("🔬 Hong–Ou–Mandel Effect Simulator")
st.caption("Two-particle interference at a beam splitter — Master's thesis simulation")


# ──────────────────────────────────────────────────────────────────────
# Widgets de parámetros (solo number_input + reset)
# ──────────────────────────────────────────────────────────────────────
def param_widget(label, key, min_val, max_val, default, step, help_text=""):
    """Campo numérico con botón de reset al valor por defecto.

    El valor canónico se guarda en st.session_state[key]; el widget usa la clave
    auxiliar f"{key}_input". No se pasa value= para evitar el aviso de Streamlit
    "default value + Session State API".
    """
    input_key = f"{key}_input"
    if key not in st.session_state:
        st.session_state[key] = default
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state[key]

    def on_input_change():
        st.session_state[key] = st.session_state[input_key]

    col_label, col_reset = st.columns([5, 1])
    col_label.markdown(f"**{label}**", help=help_text if help_text else None)
    if col_reset.button("↺", key=f"{key}_reset", help=f"Reset to default ({default})"):
        st.session_state[key] = default
        st.session_state[input_key] = default
        st.rerun()

    st.number_input(
        label, min_value=min_val, max_value=max_val, step=step,
        key=input_key, label_visibility="collapsed", on_change=on_input_change,
    )
    return st.session_state[key]


def reset_param(key, default):
    """Fija un parámetro (valor canónico y compañero _input) a `default`."""
    st.session_state[key] = default
    st.session_state[f"{key}_input"] = default


# ──────────────────────────────────────────────────────────────────────
# Exportación de resultados a CSV
# ──────────────────────────────────────────────────────────────────────
def csv_download_button(data_dict, filename, label="📥 Download as CSV", key=None):
    """Botón de descarga de un conjunto de columnas numéricas como CSV."""
    keys = list(data_dict.keys())
    arrays = [data_dict[k] for k in keys]
    n = len(arrays[0])
    buf = io.StringIO()
    buf.write(",".join(keys) + "\n")
    for i in range(n):
        buf.write(",".join(f"{arr[i]:.6e}" for arr in arrays) + "\n")
    st.download_button(label=label, data=buf.getvalue(),
                       file_name=filename, mime="text/csv", key=key)


# ──────────────────────────────────────────────────────────────────────
# Renderizado de figuras (siempre cuadradas)
# ──────────────────────────────────────────────────────────────────────
def show_plot(fig):
    """Renderiza una figura de Plotly con tamaño cuadrado fijo.

    Se usa width=height y use_container_width=False para que las figuras no
    aparezcan estiradas/aplastadas al ajustarse al ancho del contenedor. En los
    mapas de calor, además, scaleanchor fuerza una relación de aspecto 1:1 de los
    datos (ver heatmap helpers).
    """
    fig.update_layout(width=FIG_SIZE, height=FIG_SIZE, autosize=False)
    st.plotly_chart(fig, use_container_width=False)


# ──────────────────────────────────────────────────────────────────────
# Filtros espectrales (solo bosones)
# ──────────────────────────────────────────────────────────────────────
FILTER_SHAPES = ["gaussian", "rectangular", "lorentzian",
                 "sinc", "super_gaussian", "triangular"]
FILTER_HELP = {
    "gaussian": "Smooth bell curve. Most common in experiments.",
    "rectangular": "Hard bandpass cutoff. Ideal filter.",
    "lorentzian": "Broad tails. Models cavity/resonator filters.",
    "sinc": "Sinc² profile. Models grating-based filters.",
    "super_gaussian": "Flat top with steep edges. Order-4.",
    "triangular": "Linear rolloff on each side.",
}


def filter_ui(prefix, center_default_nm):
    """Panel de filtrado espectral. Devuelve (activo, función de filtrado | None)."""
    with st.expander("🔧 Spectral filters", expanded=False):
        use_filter = st.toggle(
            "Enable spectral filtering", key=f"{prefix}_filter_on",
            help="Apply a spectral filter to signal and idler arms. "
                 "Can increase HOM visibility for asymmetric sources.",
        )
        if not use_filter:
            return False, None

        filter_shape = st.selectbox(
            "Filter shape", FILTER_SHAPES, key=f"{prefix}_filter_shape",
            help="\n".join(f"**{k}**: {v}" for k, v in FILTER_HELP.items()),
        )

        st.markdown("**Signal arm**")
        c1, c2 = st.columns(2)
        with c1:
            fc_s_nm = param_widget("Center λ_s (nm)", f"{prefix}_fc_s",
                                   400.0, 1550.0, center_default_nm, 0.5)
        with c2:
            fw_s_THz = param_widget("Width (THz)", f"{prefix}_fw_s",
                                    0.01, 10.0, 2.0, 0.01)
        st.markdown("**Idler arm**")
        c3, c4 = st.columns(2)
        with c3:
            fc_i_nm = param_widget("Center λ_i (nm)", f"{prefix}_fc_i",
                                   400.0, 1550.0, center_default_nm, 0.5)
        with c4:
            fw_i_THz = param_widget("Width (THz)", f"{prefix}_fw_i",
                                    0.01, 10.0, 2.0, 0.01)

        # Conversión a rad/s (la malla interna trabaja en frecuencia angular).
        center_s = 2 * np.pi * const.c / (fc_s_nm * 1e-9)
        center_i = 2 * np.pi * const.c / (fc_i_nm * 1e-9)
        width_s = fw_s_THz * 1e12 * 2 * np.pi
        width_i = fw_i_THz * 1e12 * 2 * np.pi

        def filtered_jsa_fn(jsa, ws, wi):
            return apply_filter(jsa, ws, wi, center_s=center_s, width_s=width_s,
                                center_i=center_i, width_i=width_i, shape=filter_shape)
        return True, filtered_jsa_fn

    return False, None


# ──────────────────────────────────────────────────────────────────────
# Control del rango del eje de retardo τ (ajuste de vista, no parámetro físico)
# ──────────────────────────────────────────────────────────────────────
def delay_axis_range(prefix, unit_label):
    """Lee el rango τ [min, max] actual desde session_state, inicializándolo a
    [-1000, +1000] la primera vez. Se llama ANTES de calcular para fijar la
    ventana del barrido; los widgets que lo editan se dibujan junto a la gráfica
    con render_delay_axis_widgets(). Devuelve (lo, hi) en la unidad nativa.
    """
    lo_key, hi_key = f"{prefix}_tau_min", f"{prefix}_tau_max"
    if lo_key not in st.session_state:
        st.session_state[lo_key] = -TAU_DEFAULT_FS if prefix == "boson" else -TAU_DEFAULT_PS
    if hi_key not in st.session_state:
        st.session_state[hi_key] = TAU_DEFAULT_FS if prefix == "boson" else TAU_DEFAULT_PS
    lo = float(st.session_state[lo_key])
    hi = float(st.session_state[hi_key])
    if hi <= lo:                       # protección frente a rango degenerado
        hi = lo + 100.0
    return lo, hi


def render_delay_axis_widgets(prefix, unit_label):
    """Dibuja los campos de mínimo y máximo del eje τ, junto a la gráfica.

    Se distingue de los parámetros físicos (panel izquierdo) por su posición y
    por el rótulo: es un ajuste de la ventana de visualización.
    """
    st.caption(f"Delay-axis window ({unit_label}) — view setting, not a physical parameter")
    ca, cb = st.columns(2)
    with ca:
        st.number_input(f"τ min ({unit_label})", min_value=-TAU_BOUND, max_value=0.0,
                        step=100.0, key=f"{prefix}_tau_min")
    with cb:
        st.number_input(f"τ max ({unit_label})", min_value=0.0, max_value=TAU_BOUND,
                        step=100.0, key=f"{prefix}_tau_max")


# ──────────────────────────────────────────────────────────────────────
# Helpers de figuras
# ──────────────────────────────────────────────────────────────────────
def _downsample_heatmap(z, x, y):
    """Reduce la malla ENVIADA a Plotly (no la del cálculo) para que los mapas
    de calor sigan siendo fluidos cuando N es grande. Es solo visualización:
    V, dip y marginales se calculan con la malla completa.
    """
    step = max(1, int(np.ceil(z.shape[0] / HEATMAP_MAX)))
    if step > 1:
        return z[::step, ::step], x[::step], y[::step]
    return z, x, y


def hom_plot(x_values, p_coinc, x_label, title, baseline=0.5, x_range=None):
    """Curva de coincidencias frente al retardo τ."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=p_coinc, mode='lines',
        line=dict(width=3, color='#636EFA'), name='simulation',
    ))
    fig.add_hline(y=baseline, line_dash="dot", line_color="gray",
                  annotation_text="classical baseline", annotation_position="right")
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title="Coincidence probability",
        template="plotly_white", yaxis=dict(range=[-0.05, 1.10]),
        xaxis=dict(range=x_range) if x_range else None,
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def jsa_heatmap(jsa, ws, wi, title="Joint Spectral Intensity |JSA|²"):
    """Mapa |JSA|² sobre el plano de frecuencias (centrado en la degenerada)."""
    f_center = ws.mean() / (2 * np.pi * 1e12)
    fs = ws / (2 * np.pi * 1e12) - f_center
    fi = wi / (2 * np.pi * 1e12) - f_center
    z = np.abs(jsa) ** 2
    z, fs, fi = _downsample_heatmap(z, fs, fi)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=fs, y=fi, colorscale='Inferno',
        colorbar=dict(title="|JSA|²"), zsmooth='best',
    ))
    fig.update_layout(
        title=title + "  (drag to zoom)",
        xaxis_title="Δf_s (THz)", yaxis_title="Δf_i (THz)", template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)   # plano de datos cuadrado
    return fig


def marginals_plot(jsa, ws, wi):
    """Espectros marginales de señal e idler."""
    marginal_s, marginal_i = get_marginal_spectra(jsa, ws, wi)
    f_center = ws.mean() / (2 * np.pi * 1e12)
    fs = ws / (2 * np.pi * 1e12) - f_center
    fi = wi / (2 * np.pi * 1e12) - f_center
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fs, y=marginal_s / marginal_s.max(), mode='lines',
                             line=dict(width=2.5, color='#636EFA'), name='Signal'))
    fig.add_trace(go.Scatter(x=fi, y=marginal_i / marginal_i.max(), mode='lines',
                             line=dict(width=2.5, color='#EF553B', dash='dash'), name='Idler'))
    fig.update_layout(title="Marginal spectra", xaxis_title="Δf (THz)",
                      yaxis_title="Intensity (norm.)", template="plotly_white",
                      legend=dict(x=0.75, y=0.95))
    return fig


def jea_heatmap(jea, ea, eb, title="Joint Energy Amplitude |JEA|²"):
    """Mapa |JEA|² sobre el plano de energías (μeV)."""
    ueV = 1e-6 * const.e
    z = np.abs(jea) ** 2
    z, xa, yb = _downsample_heatmap(z, ea / ueV, eb / ueV)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=xa, y=yb, colorscale='Inferno',
        colorbar=dict(title="|JEA|²"), zsmooth='best',
    ))
    fig.update_layout(title=title + "  (drag to zoom)",
                      xaxis_title="ε_s (μeV)", yaxis_title="ε_i (μeV)", template="plotly_white")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def electron_marginals_plot(jea, ea, eb):
    """Espectros marginales de energía de los dos electrones."""
    ueV = 1e-6 * const.e
    marginal_s, marginal_i = get_marginal_spectra(jea, ea, eb)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ea / ueV, y=marginal_s / marginal_s.max(), mode='lines',
                             line=dict(width=2.5, color='#636EFA'), name='Electron A'))
    fig.add_trace(go.Scatter(x=eb / ueV, y=marginal_i / marginal_i.max(), mode='lines',
                             line=dict(width=2.5, color='#EF553B', dash='dash'), name='Electron B'))
    fig.update_layout(title="Marginal energy spectra", xaxis_title="ε (μeV)",
                      yaxis_title="Intensity (norm.)", template="plotly_white",
                      legend=dict(x=0.72, y=0.95))
    return fig


def swap_kernel_plot(jsa, ws, wi, axis_unit="THz"):
    """Mapa Re[S] = Re[f*(x_s,x_i)·f(x_i,x_s)], el integrando de la visibilidad V."""
    re_s = np.real(np.conj(jsa) * jsa.T)
    if axis_unit == "THz":
        center = ws.mean() / (2 * np.pi * 1e12)
        x = ws / (2 * np.pi * 1e12) - center
        y = wi / (2 * np.pi * 1e12) - center
        x_label, y_label = "Δf_s (THz)", "Δf_i (THz)"
    else:  # μeV para electrones
        ueV = 1e-6 * const.e
        x, y = ws / ueV, wi / ueV
        x_label, y_label = "ε_s (μeV)", "ε_i (μeV)"
    vmax = np.abs(re_s).max()
    z, x, y = _downsample_heatmap(re_s, x, y)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y, colorscale='RdBu_r', zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="Re[S]"), zsmooth='best',
    ))
    fig.update_layout(title="Swap kernel Re[S] — integrand of V  (drag to zoom)",
                      xaxis_title=x_label, yaxis_title=y_label, template="plotly_white")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Snapshots (superposición de varias curvas simuladas)
# ──────────────────────────────────────────────────────────────────────
def init_snapshots(prefix):
    if f"{prefix}_snapshots" not in st.session_state:
        st.session_state[f"{prefix}_snapshots"] = []


def save_snapshot(prefix, label, x_arr, y_arr):
    """Guarda la curva actual para poder superponerla con otras."""
    init_snapshots(prefix)
    st.session_state[f"{prefix}_snapshots"].append(
        {"label": label, "x": x_arr.copy(), "y": y_arr.copy()})


def render_snapshot_panel(prefix, x_label, y_label, title):
    """Dibuja todas las instantáneas guardadas superpuestas."""
    init_snapshots(prefix)
    snaps = st.session_state[f"{prefix}_snapshots"]
    if not snaps:
        st.info("No snapshots yet. Use **📸 Save snapshot** above to overlay curves.")
        return
    fig = go.Figure()
    palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    for k, snap in enumerate(snaps):
        fig.add_trace(go.Scatter(x=snap["x"], y=snap["y"], mode='lines',
                                 line=dict(width=2.5, color=palette[k % len(palette)]),
                                 name=snap["label"]))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                      template="plotly_white", yaxis=dict(range=[-0.05, 1.10]),
                      legend=dict(orientation="v", x=1.02, y=1.0))
    show_plot(fig)
    if st.button("🗑️ Clear snapshots", key=f"{prefix}_clear_snaps"):
        st.session_state[f"{prefix}_snapshots"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# BOSONIC tab
# ══════════════════════════════════════════════════════════════════════
def render_bosonic():
    col_params, col_plot = st.columns([1, 2], gap="large")

    # Rango τ (ventana de vista); se lee aquí para usarlo en el cálculo y los
    # widgets que lo editan se dibujan más abajo, junto a la gráfica.
    tau_lo, tau_hi = delay_axis_range("boson", "fs")

    with col_params:
        st.subheader("Source")
        source = st.radio(
            "Source type", ["SPDC pair", "Independent photons"],
            key="boson_source",
            help="SPDC: entangled photon pairs from a nonlinear crystal. "
                 "Independent: two separate single-photon sources.",
        )
        # Al cambiar de fuente, reinicia los parámetros propios de cada modo.
        if st.session_state.get("boson_source_prev") != source:
            st.session_state["boson_source_prev"] = source
            if source == "Independent photons":
                for k, v in [("boson_la", 800.0), ("boson_lb", 800.0),
                             ("boson_sa", 1.0), ("boson_sb", 1.0)]:
                    reset_param(k, v)
            else:
                crystal = st.session_state.get("boson_crystal", "BBO (I)")
                reset_param("boson_lp", float(crystal_dict[crystal]['std_pump']))
            st.rerun()

        st.divider()

        if source == "SPDC pair":
            st.subheader("Crystal")
            crystal_name = st.selectbox(
                "Crystal", list(crystal_dict.keys()), key="boson_crystal",
                help="Type-I (BBO-I, LBO-I): symmetric JSA (V→1). "
                     "Type-II (BBO-II, KTP-II, PPKTP-II): asymmetric (V<1).",
            )
            default_lp = float(crystal_dict[crystal_name]['std_pump'])
            # Al cambiar de cristal, reajusta la λ de bombeo a su valor típico.
            if st.session_state.get("boson_crystal_prev") != crystal_name:
                st.session_state["boson_crystal_prev"] = crystal_name
                reset_param("boson_lp", default_lp)
                st.rerun()

            st.subheader("Pump")
            lambda_p = param_widget("λ pump (nm)", "boson_lp", 300.0, 1064.0,
                                    default_lp, 0.5, help_text="Central pump wavelength.")
            sigma_p_THz = param_widget(
                "Pump bandwidth σ (THz)", "boson_sp", 0.05, 10.0, 1.0, 0.05,
                help_text="σ = std of the intensity spectrum. "
                          "0.05 THz ≈ quasi-CW, 1 THz ≈ ~1 ps pulse. "
                          "Very narrow pumps approach the grid resolution limit.")
            L_um = param_widget("Crystal length (μm)", "boson_L", 100.0, 20000.0,
                                1000.0, 100.0, help_text="Longer = narrower phase-matching sinc.")
            degenerate_nm = 2 * lambda_p
        else:
            st.subheader("Photon A")
            lambda_a = param_widget("λ A (nm)", "boson_la", 400.0, 1550.0, 800.0, 0.5)
            sigma_a_THz = param_widget("Bandwidth σ A (THz)", "boson_sa", 0.05, 10.0,
                                       1.0, 0.05, help_text="σ = std of the intensity spectrum.")
            st.divider()
            st.subheader("Photon B")
            lambda_b = param_widget("λ B (nm)", "boson_lb", 400.0, 1550.0, 800.0, 0.5)
            sigma_b_THz = param_widget("Bandwidth σ B (THz)", "boson_sb", 0.05, 10.0,
                                       1.0, 0.05, help_text="σ = std of the intensity spectrum.")
            degenerate_nm = (lambda_a + lambda_b) / 2

        st.divider()
        st.subheader("Beam splitter")
        R_bs = param_widget("Reflectivity R", "boson_R", 0.0, 1.0, 0.5, 0.01,
                            help_text="R=0.5 maximises HOM visibility.")
        V_pol = param_widget("Polarisation overlap |V|", "boson_V", 0.0, 1.0, 1.0, 0.01,
                            help_text="V=1: same polarisation.")

        st.divider()
        use_filter, filtered_jsa_fn = filter_ui("boson", degenerate_nm)

    # ── Cálculo ───────────────────────────────────────────────────────
    tau_array = np.linspace(tau_lo * 1e-15, tau_hi * 1e-15, N_TAU)

    if source == "SPDC pair":
        sigma_p = sigma_p_THz * 1e12 * 2 * np.pi
        try:
            jsa, ws, wi = jsa_function(lambda_p, sigma_p, crystal_name,
                                       int(L_um), grid_size=GRID)
            phi_func = phase_matching(lambda_p, crystal_name, int(L_um))
            theta_deg, poling_period = phi_func.theta_deg, phi_func.poling_period
        except Exception as e:
            with col_plot:
                st.error(f"JSA computation failed: {e}")
            return
        title = f"HOM dip — SPDC ({crystal_name}, λp={lambda_p:.0f} nm)"
        snap_label = (f"SPDC {crystal_name} | λp={lambda_p:.0f}nm | "
                      f"σp={sigma_p_THz:.2f}THz | L={L_um:.0f}μm | "
                      f"R={R_bs:.2f} | V={V_pol:.2f}")
    else:
        sigma_a = sigma_a_THz * 1e12 * 2 * np.pi
        sigma_b = sigma_b_THz * 1e12 * 2 * np.pi
        jsa, ws, wi = independent_jsa_function(lambda_a, lambda_b, sigma_a, sigma_b,
                                               grid_size=GRID)
        theta_deg, poling_period = None, None
        title = (f"HOM dip — independent photons "
                 f"(λA={lambda_a:.0f}, λB={lambda_b:.0f} nm)")
        snap_label = (f"Indep | λA={lambda_a:.0f}nm σA={sigma_a_THz:.2f}THz | "
                      f"λB={lambda_b:.0f}nm σB={sigma_b_THz:.2f}THz | R={R_bs:.2f}")

    if use_filter and filtered_jsa_fn is not None:
        try:
            jsa = filtered_jsa_fn(jsa, ws, wi)
            title += " [filtered]"
            snap_label += " [filtered]"
        except ValueError as e:
            with col_plot:
                st.warning(f"Filter warning: {e}")

    V_swap = get_intrinsic_indistinguishability(jsa, ws, wi)
    p_coinc = hom_coincidence_rate(jsa, ws, wi, tau_array, R=R_bs, V_pol=V_pol,
                                   statistics='boson')

    # ── Gráficas ──────────────────────────────────────────────────────
    with col_plot:
        T = 1 - R_bs
        baseline = T ** 2 + R_bs ** 2
        idx_min = int(np.argmin(p_coinc))
        p_min = float(p_coinc[idx_min])
        tau_dip_fs = tau_array[idx_min] * 1e15
        dip_vis = (baseline - p_min) / baseline if baseline > 0 else 0.0

        tab_dip, tab_jsa, tab_marg, tab_swap = st.tabs(
            ["📈 HOM dip", "🟦 JSI", "📊 Marginals", "🔄 Swap kernel"])

        with tab_dip:
            render_delay_axis_widgets("boson", "fs")
            show_plot(hom_plot(tau_array * 1e15, p_coinc, "τ (fs)", title,
                               baseline, x_range=[tau_lo, tau_hi]))
            c1, c2 = st.columns([1, 4])
            if c1.button("📸 Save snapshot", key="boson_save_snap"):
                save_snapshot("boson", snap_label, tau_array * 1e15, p_coinc)
                st.success(f"Saved: {snap_label[:80]}…")
            with c2:
                csv_download_button({"tau_fs": tau_array * 1e15, "p_coinc": p_coinc},
                                    "hom_dip.csv", "📥 Download HOM curve (CSV)",
                                    key="boson_csv_dip")

        with tab_jsa:
            show_plot(jsa_heatmap(jsa, ws, wi))

        with tab_marg:
            show_plot(marginals_plot(jsa, ws, wi))
            ms, mi = get_marginal_spectra(jsa, ws, wi)
            f_center = ws.mean() / (2 * np.pi * 1e12)
            csv_download_button(
                {"df_s_THz": ws / (2 * np.pi * 1e12) - f_center, "marginal_s": ms / ms.max(),
                 "df_i_THz": wi / (2 * np.pi * 1e12) - f_center, "marginal_i": mi / mi.max()},
                "marginals.csv", "📥 Download marginals (CSV)", key="boson_csv_marg")

        with tab_swap:
            show_plot(swap_kernel_plot(jsa, ws, wi, axis_unit="THz"))
            st.caption(
                "**What this shows.** The swap kernel "
                "Re[S(ω_s,ω_i)] = Re[f*(ω_s,ω_i)·f(ω_i,ω_s)] is the integrand of "
                "the indistinguishability V. Bright regions deepen the HOM dip; "
                "blue/negative regions shallow it. V is the integral of this map.")

        st.divider()
        with st.expander("📋 Numerical readouts", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("V (overlap, τ=0)", f"{V_swap:.4f}",
                      help="Intrinsic spectral indistinguishability at zero delay.")
            r2.metric("Dip visibility", f"{dip_vis:.4f}",
                      help="(baseline − min p)/baseline — the observable dip depth, "
                           "measured at the minimum of the curve over the τ window.")
            r3.metric("min p(τ)", f"{p_min:.4f}", help=f"at τ = {tau_dip_fs:.0f} fs")
            r4.metric("plateau (T²+R²)", f"{baseline:.4f}")

            if source == "SPDC pair":
                r5, r6, r7, r8 = st.columns(4)
                r5.metric("Phase-matching θ", f"{theta_deg:.2f}°")
                if poling_period > 0:
                    r6.metric("Poling period Λ", f"{poling_period:.2f} μm")
                else:
                    r6.metric("Poling period Λ", "N/A", help="Bulk crystal.")
                r7.metric("Walk-off ~ L·tan θ", f"{L_um * np.tan(np.radians(theta_deg)):.1f} μm")
                r8.metric("Pump τ_coh ≈ 1/σ", f"{1.0 / sigma_p_THz * 1000:.0f} fs")
            else:
                r5, r6, r7, _ = st.columns(4)
                r5.metric("τ_coh A", f"{1.0 / sigma_a_THz * 1000:.0f} fs")
                r6.metric("τ_coh B", f"{1.0 / sigma_b_THz * 1000:.0f} fs")
                V_analytic = gaussian_overlap_V(lambda_a, lambda_b, sigma_a, sigma_b)
                r7.metric("|⟨φA|φB⟩|² (analytic)", f"{V_analytic:.4f}",
                          help="Gaussian spectral overlap; matches V numerically.")

        st.divider()
        st.subheader("📸 Snapshots — overlay multiple curves")
        render_snapshot_panel("boson", "τ (fs)", "Coincidence probability",
                              "HOM dip — overlay")


# ══════════════════════════════════════════════════════════════════════
# FERMIONIC tab  (energías en μeV, tiempos en ps)
# ══════════════════════════════════════════════════════════════════════
def electron_params(suffix, shape, ueV):
    """Widgets de un electrón para la forma compartida `shape`.

    Devuelve (e0_ueV, valor_de_anchura, params_dict). El levitón no tiene ε₀
    (está anclado al nivel de Fermi); gaussiana usa σ y lorentziana usa Γ.
    """
    S = suffix.upper()
    if shape == "leviton":
        w = param_widget(f"τ₀ {S} (ps)", f"fermi_w{suffix}", 5.0, 1000.0, 50.0, 5.0,
                         help_text="Leviton width (pinned to ε_F; no ε₀).")
        return 0.0, w, {'tau_0': w * 1e-12}
    e0 = param_widget(f"ε₀ {S} (μeV)", f"fermi_e0_{suffix}", 0.0, 200.0, 30.0, 1.0,
                      help_text="Mean energy above the Fermi level.")
    if shape == "gaussian":
        w = param_widget(f"σ {S} (μeV)", f"fermi_w{suffix}", 1.0, 100.0, 10.0, 0.5,
                         help_text="σ = std of the intensity spectrum.")
        return e0, w, {'sigma': w * ueV}
    # lorentziana
    w = param_widget(f"Γ {S} (μeV)", f"fermi_w{suffix}", 1.0, 100.0, 10.0, 0.5,
                     help_text="FWHM linewidth.")
    return e0, w, {'Gamma': w * ueV}


def render_fermionic():
    col_params, col_plot = st.columns([1, 2], gap="large")
    ueV = 1e-6 * const.e
    hbar = const.hbar

    tau_lo, tau_hi = delay_axis_range("fermi", "ps")

    with col_params:
        st.subheader("Source — product state of two electrons")

        # Forma del espectro energético: ÚNICA y común a los dos electrones.
        shape = st.selectbox(
            "Energy-spectrum shape", ["gaussian", "lorentzian", "leviton"],
            key="fermi_shape",
            help="Applies to BOTH electrons (no mixing). Gaussian: bell. "
                 "Lorentzian: broad tails (mesoscopic capacitor). "
                 "Leviton: minimal excitation pinned to the Fermi level.")
        # Al cambiar la forma, reinicia las anchuras de ambos electrones.
        if st.session_state.get("fermi_shape_prev") != shape:
            st.session_state["fermi_shape_prev"] = shape
            default_w = 50.0 if shape == "leviton" else 10.0
            reset_param("fermi_wa", default_w)
            reset_param("fermi_wb", default_w)
            st.rerun()

        st.divider()
        st.subheader("Electron A")
        e0_a_ueV, w_a, params_a = electron_params("a", shape, ueV)

        st.divider()
        st.subheader("Electron B")
        e0_b_ueV, w_b, params_b = electron_params("b", shape, ueV)

        st.divider()
        eF_ueV = param_widget("Fermi energy ε_F (μeV)", "fermi_eF", 0.0, 100.0, 0.0, 1.0,
                              help_text="States below ε_F are forbidden.")

        st.divider()
        st.subheader("Beam splitter")
        R_bs = param_widget("Reflectivity R", "fermi_R", 0.0, 1.0, 0.5, 0.01)
        V_pol = param_widget("Spin overlap |V|", "fermi_V", 0.0, 1.0, 1.0, 0.01)

    # ── Cálculo ───────────────────────────────────────────────────────
    try:
        # Misma forma para los dos electrones (estado producto φ_a ⊗ φ_b).
        jea, ea, eb = independent_jea_function(
            e0_a_ueV * ueV, e0_b_ueV * ueV, shape, shape,
            params_a, params_b, varepsilon_F=eF_ueV * ueV, grid_size=GRID)
    except (ValueError, ZeroDivisionError) as e:
        with col_plot:
            st.error("JEA construction failed. Use distinct energies/widths.")
            st.exception(e)
        return

    V_swap = get_intrinsic_indistinguishability(jea, ea, eb)
    tau_array_s = np.linspace(tau_lo * 1e-12, tau_hi * 1e-12, N_TAU)
    # τ se pasa dividido por ħ (la fase es e^{i(ε_i−ε_s)τ/ħ}).
    p_coinc = hom_coincidence_rate(jea, ea, eb, tau_array_s / hbar,
                                   R=R_bs, V_pol=V_pol, statistics='fermion')

    snap_label = f"{shape} | w={w_a:.0f}/{w_b:.0f} | R={R_bs:.2f}"

    with col_plot:
        T = 1 - R_bs
        baseline = T ** 2 + R_bs ** 2
        idx_max = int(np.argmax(p_coinc))
        p_max = float(p_coinc[idx_max])
        tau_peak_ps = tau_array_s[idx_max] * 1e12
        peak_vis = (p_max - baseline) / baseline if baseline > 0 else 0.0
        title = f"Antibunching peak — {shape.capitalize()} electrons"

        tab_dip, tab_jea, tab_marg, tab_swap = st.tabs(
            ["📈 Antibunching peak", "🟦 JEI", "📊 Marginals", "🔄 Swap kernel"])

        with tab_dip:
            render_delay_axis_widgets("fermi", "ps")
            show_plot(hom_plot(tau_array_s * 1e12, p_coinc, "τ (ps)", title,
                               baseline, x_range=[tau_lo, tau_hi]))
            c1, c2 = st.columns([1, 4])
            if c1.button("📸 Save snapshot", key="fermi_save_snap"):
                save_snapshot("fermi", snap_label, tau_array_s * 1e12, p_coinc)
                st.success(f"Saved: {snap_label[:80]}…")
            with c2:
                csv_download_button({"tau_ps": tau_array_s * 1e12, "p_coinc": p_coinc},
                                    "hom_peak.csv", "📥 Download peak curve (CSV)",
                                    key="fermi_csv_dip")

        with tab_jea:
            show_plot(jea_heatmap(jea, ea, eb))

        with tab_marg:
            show_plot(electron_marginals_plot(jea, ea, eb))
            ms, mi = get_marginal_spectra(jea, ea, eb)
            csv_download_button(
                {"ea_ueV": ea / ueV, "marginal_s": ms / ms.max(),
                 "eb_ueV": eb / ueV, "marginal_i": mi / mi.max()},
                "electron_marginals.csv", "📥 Download marginals (CSV)", key="fermi_csv_marg")

        with tab_swap:
            show_plot(swap_kernel_plot(jea, ea, eb, axis_unit="ueV"))

        st.divider()
        with st.expander("📋 Numerical readouts", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("V (overlap, τ=0)", f"{V_swap:.4f}")
            r2.metric("Peak visibility", f"{peak_vis:.4f}",
                      help="(max p − baseline)/baseline — measured at the maximum "
                           "of the curve over the τ window.")
            r3.metric("max p(τ)", f"{p_max:.4f}", help=f"at τ = {tau_peak_ps:.0f} ps")
            r4.metric("plateau (T²+R²)", f"{baseline:.4f}")

            r5, r6, r7, _ = st.columns(4)

            def tcoh_ps(w):
                """Tiempo de coherencia en ps: τ₀ (levitón) o ħ/ancho (gauss/lorentz)."""
                if shape == "leviton":
                    return w                      # τ₀ ya en ps
                return (hbar / (w * ueV)) * 1e12  # ħ/σ o ħ/Γ en ps
            r5.metric("τ_coh A", f"{tcoh_ps(w_a):.0f} ps")
            r6.metric("τ_coh B", f"{tcoh_ps(w_b):.0f} ps")
            if shape != "leviton":
                r7.metric("Δε / max(width)",
                          f"{abs(e0_a_ueV - e0_b_ueV) / max(w_a, w_b):.2f}",
                          help="Energy detuning normalised to the larger width. "
                               "0 = aligned, ≫1 = distinguishable.")
            else:
                r7.metric("Δε / max(width)", "—",
                          help="Levitons are pinned to ε_F (no ε₀ detuning).")

        st.divider()
        st.subheader("📸 Snapshots — overlay multiple curves")
        render_snapshot_panel("fermi", "τ (ps)", "Coincidence probability",
                              "Antibunching peak — overlay")


# ══════════════════════════════════════════════════════════════════════
# Top-level layout
# ══════════════════════════════════════════════════════════════════════
tab_boson, tab_fermion = st.tabs(["🔵 Bosonic (photons)", "🟠 Fermionic (electrons)"])
with tab_boson:
    render_bosonic()
with tab_fermion:
    render_fermionic()
