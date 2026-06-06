"""
Hong-Ou-Mandel Effect Simulator — Streamlit app.

Cambios respecto a la versión anterior
---------------------------------------
* Parámetros solo por number_input (sin sliders).
* Ejes de retardo τ FIJOS y no editables, distintos por pestaña y en unidades
  físicas realistas: fotones τ_max = 1500 fs; electrones τ_max = 300 ps.
* Escala electrónica realista (energías en μeV, tiempos en ps), acorde con la
  óptica cuántica electrónica (paquetes de decenas de ps, energías de decenas
  de μeV).
* Convención σ unificada: σ = desviación estándar del espectro de intensidad.
* JSI: ventana autodimensionada (sin zoom forzado); el zoom es interactivo (Plotly).
* Visibilidad reportada como el MÍNIMO/MÁXIMO de la curva sobre τ (no en τ=0).
* Lectura de solapamiento de fotones independientes con la fórmula analítica
  correcta (coincide con V numérico).
* Caption fermiónica corregida: estado producto, V≥0, el pico viene del signo + .
* Registro de datos experimentales (activable) para superponer sobre la curva
  HOM y cargar sus parámetros en el simulador.
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
# Ejes de retardo fijos (no editables) y malla
# ──────────────────────────────────────────────────────────────────────
TAU_MAX_PHOTON_FS   = 1500.0    # eje τ de la pestaña de fotones
TAU_MAX_ELECTRON_PS = 300.0     # eje τ de la pestaña de electrones
N_TAU               = 351       # nº de puntos en τ (fijo)
GRID                = 500       # malla espectral para el cálculo

# ══════════════════════════════════════════════════════════════════════
# REGISTRO DE DATOS EXPERIMENTALES
# ----------------------------------------------------------------------
# Rellena estos diccionarios con tus medidas reales. Cada entrada lleva:
#   "params": valores que se cargarán en el simulador para comparar
#             (las claves deben ser las del estado de los widgets, ver más abajo).
#   "tau_fs" / "tau_ps": eje de retardo de los puntos experimentales.
#   "p_coinc": probabilidad/tasa de coincidencia normalizada de cada punto.
#
# Claves de "params" admitidas
#   Fotones SPDC:     boson_source="SPDC pair", boson_crystal, boson_lp (nm),
#                     boson_sp (THz), boson_L (μm), boson_R, boson_V
#   Fotones indep.:   boson_source="Independent photons", boson_la, boson_lb (nm),
#                     boson_sa, boson_sb (THz), boson_R, boson_V
#   Electrones:       fermi_shape_a/b ("gaussian"|"lorentzian"|"leviton"),
#                     fermi_e0_a/b (μeV; no aplica a leviton), fermi_wa/wb
#                     (μeV para gaussian/lorentzian, ps para leviton),
#                     fermi_eF (μeV), fermi_R, fermi_V
#
# Plantilla (descomenta y edita; deja el dict vacío si aún no hay datos):
# EXPERIMENTAL_DATASETS = {
#     "boson": {
#         "Mi medida BBO-II (2025)": {
#             "params": {
#                 "boson_source": "SPDC pair",
#                 "boson_crystal": "BBO (II)",
#                 "boson_lp": 405.0, "boson_sp": 1.0, "boson_L": 1000.0,
#                 "boson_R": 0.5, "boson_V": 1.0,
#             },
#             "tau_fs":  [-600, -400, -200, 0, 200, 400, 600],
#             "p_coinc": [ 0.50,  0.48,  0.43, 0.41, 0.43, 0.48, 0.50],
#         },
#     },
#     "fermion": {
#         "Mi medida levitones (2025)": {
#             "params": {
#                 "fermi_shape_a": "leviton", "fermi_shape_b": "leviton",
#                 "fermi_wa": 50.0, "fermi_wb": 50.0,
#                 "fermi_eF": 0.0, "fermi_R": 0.5, "fermi_V": 1.0,
#             },
#             "tau_ps":  [-200, -100, 0, 100, 200],
#             "p_coinc": [ 0.50,  0.62, 0.75, 0.62, 0.50],
#         },
#     },
# }
EXPERIMENTAL_DATASETS = {
    "boson": {},
    "fermion": {},
}


# ──────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HOM Simulator", page_icon="🔬", layout="wide")
st.title("🔬 Hong–Ou–Mandel Effect Simulator")
st.caption("Two-particle interference at a beam splitter — Master's thesis simulation")


# ──────────────────────────────────────────────────────────────────────
# Widgets (solo number_input + reset)
# ──────────────────────────────────────────────────────────────────────
def param_widget(label, key, min_val, max_val, default, step, help_text=""):
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

    # Sin value=: el widget toma su valor de session_state[input_key] (evita el
    # aviso de "default value + Session State API").
    st.number_input(
        label, min_value=min_val, max_value=max_val, step=step,
        key=input_key, label_visibility="collapsed", on_change=on_input_change,
    )
    return st.session_state[key]


def reset_param(key, default):
    st.session_state[key] = default
    st.session_state[f"{key}_input"] = default


# ──────────────────────────────────────────────────────────────────────
# Carga de parámetros de un experimento al estado del simulador
# ──────────────────────────────────────────────────────────────────────
_PREV_TRACKERS = {
    "boson_source": "boson_source_prev",
    "boson_crystal": "boson_crystal_prev",
    "fermi_shape_a": "fermi_shape_a_prev",
    "fermi_shape_b": "fermi_shape_b_prev",
}

def load_experiment_params(params):
    """Marca los parámetros del experimento para cargarlos en la PRÓXIMA ejecución.

    No se pueden modificar las claves de widgets ya instanciados en la ejecución
    en curso, así que se aplican al inicio del siguiente run (apply_pending_exp_load),
    antes de crear los widgets.
    """
    st.session_state["_pending_exp_load"] = dict(params)
    st.rerun()


def apply_pending_exp_load():
    """Aplica los parámetros pendientes (llamar antes de instanciar los widgets).

    Sincroniza el valor canónico, el compañero _input de number_input y los
    rastreadores *_prev de selectbox/radio para que sus detectores de cambio no
    reinicien los valores recién cargados.
    """
    params = st.session_state.pop("_pending_exp_load", None)
    if not params:
        return
    for k, v in params.items():
        st.session_state[k] = v
        st.session_state[f"{k}_input"] = v
        if k in _PREV_TRACKERS:
            st.session_state[_PREV_TRACKERS[k]] = v


def experimental_overlay_ui(domain):
    """UI del registro de datos experimentales. Devuelve lista de overlays
    [{x, y, label}] (en las unidades del eje τ de la pestaña) o lista vacía.
    """
    overlays = []
    datasets = EXPERIMENTAL_DATASETS.get(domain, {})

    with st.expander("🧪 Experimental data overlay", expanded=False):
        enabled = st.toggle(
            "Overlay experimental data",
            key=f"{domain}_exp_on",
            help="Compare the simulated curve against real measurements. "
                 "Define datasets in EXPERIMENTAL_DATASETS at the top of the file.",
        )
        if not enabled:
            return overlays

        tau_key = "tau_fs" if domain == "boson" else "tau_ps"

        if datasets:
            names = list(datasets.keys())
            chosen = st.multiselect(
                "Datasets to overlay", names, default=names[:1],
                key=f"{domain}_exp_pick",
            )
            # Cargar parámetros de UNO de ellos en el simulador
            if chosen:
                load_target = st.selectbox(
                    "Match simulator parameters to:", chosen,
                    key=f"{domain}_exp_loadsel",
                    help="Sets pump/crystal/etc. (or electron parameters) to the "
                         "values associated with this dataset, for a fair comparison.",
                )
                if st.button("📥 Load this experiment's parameters into the simulator",
                             key=f"{domain}_exp_load"):
                    params = datasets[load_target].get("params", {})
                    if params:
                        load_experiment_params(params)
                    else:
                        st.warning("This dataset has no 'params' block to load.")
            for name in chosen:
                ds = datasets[name]
                if tau_key in ds and "p_coinc" in ds:
                    overlays.append({
                        "x": np.asarray(ds[tau_key], dtype=float),
                        "y": np.asarray(ds["p_coinc"], dtype=float),
                        "label": name,
                    })
        else:
            st.info("No datasets defined yet. Add them to "
                    "`EXPERIMENTAL_DATASETS['" + domain + "']` in the source file.")

        # Opción cómoda: subir un CSV (columnas: tau, p) sin tocar el código
        up = st.file_uploader(
            f"…or upload a CSV (columns: {tau_key.replace('_', ' ')}, p_coinc)",
            type=["csv"], key=f"{domain}_exp_csv",
        )
        if up is not None:
            try:
                raw = np.genfromtxt(up, delimiter=",", names=True)
                xcol = raw.dtype.names[0]
                ycol = raw.dtype.names[1]
                overlays.append({
                    "x": np.asarray(raw[xcol], dtype=float),
                    "y": np.asarray(raw[ycol], dtype=float),
                    "label": f"CSV: {up.name}",
                })
            except Exception as e:
                st.warning(f"Could not parse CSV: {e}")

    return overlays


# ──────────────────────────────────────────────────────────────────────
# CSV download helper
# ──────────────────────────────────────────────────────────────────────
def csv_download_button(data_dict, filename, label="📥 Download as CSV", key=None):
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
# Plot helpers
# ──────────────────────────────────────────────────────────────────────
def hom_plot(x_values, p_coinc, x_label, title, baseline=0.5, overlays=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=p_coinc, mode='lines',
        line=dict(width=3, color='#636EFA'), name='simulation',
    ))
    if overlays:
        palette = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FF6692']
        for k, ov in enumerate(overlays):
            fig.add_trace(go.Scatter(
                x=ov["x"], y=ov["y"], mode='markers',
                marker=dict(size=8, color=palette[k % len(palette)],
                            line=dict(width=1, color='black')),
                name=ov["label"],
            ))
    fig.add_hline(y=baseline, line_dash="dot", line_color="gray",
                  annotation_text="classical baseline", annotation_position="right")
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title="Coincidence probability",
        template="plotly_white", height=500, yaxis=dict(range=[-0.05, 1.10]),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def jsa_heatmap(jsa, ws, wi, title="Joint Spectral Intensity |JSA|²"):
    f_center = ws.mean() / (2 * np.pi * 1e12)
    fs = ws / (2 * np.pi * 1e12) - f_center
    fi = wi / (2 * np.pi * 1e12) - f_center
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(jsa) ** 2, x=fs, y=fi, colorscale='Inferno',
        colorbar=dict(title="|JSA|²"), zsmooth='best',
    ))
    fig.update_layout(
        title=title + "  (drag to zoom)",
        xaxis_title="Δf_s (THz)", yaxis_title="Δf_i (THz)",
        template="plotly_white", height=500,
    )
    return fig


def marginals_plot(jsa, ws, wi):
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
                      height=500, legend=dict(x=0.75, y=0.95))
    return fig


def jea_heatmap(jea, ea, eb, title="Joint Energy Amplitude |JEA|²"):
    ueV = 1e-6 * const.e
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(jea) ** 2, x=ea / ueV, y=eb / ueV, colorscale='Inferno',
        colorbar=dict(title="|JEA|²"), zsmooth='best',
    ))
    fig.update_layout(title=title + "  (drag to zoom)",
                      xaxis_title="ε_s (μeV)", yaxis_title="ε_i (μeV)",
                      template="plotly_white", height=500)
    return fig


def electron_marginals_plot(jea, ea, eb):
    ueV = 1e-6 * const.e
    marginal_s, marginal_i = get_marginal_spectra(jea, ea, eb)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ea / ueV, y=marginal_s / marginal_s.max(), mode='lines',
                             line=dict(width=2.5, color='#636EFA'), name='Electron A'))
    fig.add_trace(go.Scatter(x=eb / ueV, y=marginal_i / marginal_i.max(), mode='lines',
                             line=dict(width=2.5, color='#EF553B', dash='dash'), name='Electron B'))
    fig.update_layout(title="Marginal energy spectra", xaxis_title="ε (μeV)",
                      yaxis_title="Intensity (norm.)", template="plotly_white",
                      height=500, legend=dict(x=0.72, y=0.95))
    return fig


def swap_kernel_plot(jsa, ws, wi, axis_unit="THz"):
    """Re[S(x_s,x_i)] = Re[f*(x_s,x_i)·f(x_i,x_s)], el integrando de V."""
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
    fig = go.Figure(data=go.Heatmap(
        z=re_s, x=x, y=y, colorscale='RdBu_r', zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="Re[S]"), zsmooth='best',
    ))
    fig.update_layout(title="Swap kernel Re[S] — integrand of V  (drag to zoom)",
                      xaxis_title=x_label, yaxis_title=y_label,
                      template="plotly_white", height=500)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Snapshots (overlay de varias curvas simuladas)
# ──────────────────────────────────────────────────────────────────────
def init_snapshots(prefix):
    if f"{prefix}_snapshots" not in st.session_state:
        st.session_state[f"{prefix}_snapshots"] = []


def save_snapshot(prefix, label, x_arr, y_arr):
    init_snapshots(prefix)
    st.session_state[f"{prefix}_snapshots"].append(
        {"label": label, "x": x_arr.copy(), "y": y_arr.copy()})


def render_snapshot_panel(prefix, x_label, y_label, title):
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
                      template="plotly_white", height=500, yaxis=dict(range=[-0.05, 1.10]),
                      legend=dict(orientation="v", x=1.02, y=1.0))
    st.plotly_chart(fig, use_container_width=True)
    if st.button("🗑️ Clear snapshots", key=f"{prefix}_clear_snaps"):
        st.session_state[f"{prefix}_snapshots"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# BOSONIC tab
# ══════════════════════════════════════════════════════════════════════
def render_bosonic():
    col_params, col_plot = st.columns([1, 2], gap="large")

    with col_params:
        st.subheader("Source")
        source = st.radio(
            "Source type", ["SPDC pair", "Independent photons"],
            key="boson_source",
            help="SPDC: entangled photon pairs from a nonlinear crystal. "
                 "Independent: two separate single-photon sources.",
        )
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
        st.caption(f"Delay axis τ fixed at ±{TAU_MAX_PHOTON_FS:.0f} fs, "
                   f"{N_TAU} points (not editable).")

        st.divider()
        use_filter, filtered_jsa_fn = filter_ui("boson", degenerate_nm)

        st.divider()
        overlays = experimental_overlay_ui("boson")

    # ── Compute ───────────────────────────────────────────────────────
    tau_array = np.linspace(-TAU_MAX_PHOTON_FS * 1e-15, TAU_MAX_PHOTON_FS * 1e-15, N_TAU)

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

    # ── Plot tabs ─────────────────────────────────────────────────────
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
            st.plotly_chart(
                hom_plot(tau_array * 1e15, p_coinc, "τ (fs)", title, baseline, overlays),
                use_container_width=True)
            c1, c2 = st.columns([1, 4])
            if c1.button("📸 Save snapshot", key="boson_save_snap"):
                save_snapshot("boson", snap_label, tau_array * 1e15, p_coinc)
                st.success(f"Saved: {snap_label[:80]}…")
            with c2:
                csv_download_button({"tau_fs": tau_array * 1e15, "p_coinc": p_coinc},
                                    "hom_dip.csv", "📥 Download HOM curve (CSV)",
                                    key="boson_csv_dip")

        with tab_jsa:
            st.plotly_chart(jsa_heatmap(jsa, ws, wi), use_container_width=True)

        with tab_marg:
            st.plotly_chart(marginals_plot(jsa, ws, wi), use_container_width=True)
            ms, mi = get_marginal_spectra(jsa, ws, wi)
            f_center = ws.mean() / (2 * np.pi * 1e12)
            csv_download_button(
                {"df_s_THz": ws / (2 * np.pi * 1e12) - f_center, "marginal_s": ms / ms.max(),
                 "df_i_THz": wi / (2 * np.pi * 1e12) - f_center, "marginal_i": mi / mi.max()},
                "marginals.csv", "📥 Download marginals (CSV)", key="boson_csv_marg")

        with tab_swap:
            st.plotly_chart(swap_kernel_plot(jsa, ws, wi, axis_unit="THz"),
                            use_container_width=True)
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
                           "measured at the minimum of the curve over τ.")
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
def render_fermionic():
    col_params, col_plot = st.columns([1, 2], gap="large")
    ueV = 1e-6 * const.e
    hbar = const.hbar

    with col_params:
        st.subheader("Source — product state of two electrons")
        st.caption("Two single-electron wave packets above the Fermi sea, collided "
                   "at a quantum point contact. Realistic electron-quantum-optics "
                   "scale: μeV energies, ps timescales.")

        st.subheader("Electron A")
        shape_a = st.selectbox(
            "Shape A", ["gaussian", "lorentzian", "leviton"], key="fermi_shape_a",
            help="Gaussian: bell. Lorentzian: broad tails (mesoscopic capacitor). "
                 "Leviton: minimal excitation pinned to the Fermi level.")
        if st.session_state.get("fermi_shape_a_prev") != shape_a:
            st.session_state["fermi_shape_a_prev"] = shape_a
            reset_param("fermi_wa", 50.0 if shape_a == "leviton" else 10.0)
            st.rerun()

        if shape_a == "leviton":
            w_a = param_widget("τ₀ A (ps)", "fermi_wa", 5.0, 1000.0, 50.0, 5.0,
                               help_text="Leviton width (pinned to ε_F; no ε₀).")
            params_a = {'tau_0': w_a * 1e-12}
            e0_a_ueV = 0.0
        else:
            e0_a_ueV = param_widget("ε₀ A (μeV)", "fermi_e0_a", 0.0, 200.0, 30.0, 1.0,
                                    help_text="Mean energy above the Fermi level.")
            if shape_a == "gaussian":
                w_a = param_widget("σ A (μeV)", "fermi_wa", 1.0, 100.0, 10.0, 0.5,
                                   help_text="σ = std of the intensity spectrum.")
                params_a = {'sigma': w_a * ueV}
            else:
                w_a = param_widget("Γ A (μeV)", "fermi_wa", 1.0, 100.0, 10.0, 0.5,
                                   help_text="FWHM linewidth.")
                params_a = {'Gamma': w_a * ueV}

        st.divider()
        st.subheader("Electron B")
        shape_b = st.selectbox("Shape B", ["gaussian", "lorentzian", "leviton"],
                               key="fermi_shape_b")
        if st.session_state.get("fermi_shape_b_prev") != shape_b:
            st.session_state["fermi_shape_b_prev"] = shape_b
            reset_param("fermi_wb", 50.0 if shape_b == "leviton" else 10.0)
            st.rerun()

        if shape_b == "leviton":
            w_b = param_widget("τ₀ B (ps)", "fermi_wb", 5.0, 1000.0, 50.0, 5.0,
                               help_text="Leviton width (pinned to ε_F; no ε₀).")
            params_b = {'tau_0': w_b * 1e-12}
            e0_b_ueV = 0.0
        else:
            e0_b_ueV = param_widget("ε₀ B (μeV)", "fermi_e0_b", 0.0, 200.0, 30.0, 1.0)
            if shape_b == "gaussian":
                w_b = param_widget("σ B (μeV)", "fermi_wb", 1.0, 100.0, 10.0, 0.5)
                params_b = {'sigma': w_b * ueV}
            else:
                w_b = param_widget("Γ B (μeV)", "fermi_wb", 1.0, 100.0, 10.0, 0.5)
                params_b = {'Gamma': w_b * ueV}

        st.divider()
        eF_ueV = param_widget("Fermi energy ε_F (μeV)", "fermi_eF", 0.0, 100.0, 0.0, 1.0,
                              help_text="States below ε_F are forbidden.")

        st.divider()
        st.subheader("Beam splitter")
        R_bs = param_widget("Reflectivity R", "fermi_R", 0.0, 1.0, 0.5, 0.01)
        V_pol = param_widget("Spin overlap |V|", "fermi_V", 0.0, 1.0, 1.0, 0.01)

        st.divider()
        st.caption(f"Delay axis τ fixed at ±{TAU_MAX_ELECTRON_PS:.0f} ps, "
                   f"{N_TAU} points (not editable).")

        st.divider()
        overlays = experimental_overlay_ui("fermion")

    # ── Compute ───────────────────────────────────────────────────────
    try:
        jea, ea, eb = independent_jea_function(
            e0_a_ueV * ueV, e0_b_ueV * ueV, shape_a, shape_b,
            params_a, params_b, varepsilon_F=eF_ueV * ueV, grid_size=GRID)
    except (ValueError, ZeroDivisionError) as e:
        with col_plot:
            st.error("JEA construction failed. Use distinct energies/widths.")
            st.exception(e)
        return

    V_swap = get_intrinsic_indistinguishability(jea, ea, eb)
    tau_array_s = np.linspace(-TAU_MAX_ELECTRON_PS * 1e-12,
                              TAU_MAX_ELECTRON_PS * 1e-12, N_TAU)
    p_coinc = hom_coincidence_rate(jea, ea, eb, tau_array_s / hbar,
                                   R=R_bs, V_pol=V_pol, statistics='fermionic')

    snap_label = (f"{shape_a}/{shape_b} | "
                  f"w={w_a:.0f}/{w_b:.0f} | R={R_bs:.2f}")

    with col_plot:
        T = 1 - R_bs
        baseline = T ** 2 + R_bs ** 2
        idx_max = int(np.argmax(p_coinc))
        p_max = float(p_coinc[idx_max])
        tau_peak_ps = tau_array_s[idx_max] * 1e12
        peak_vis = (p_max - baseline) / baseline if baseline > 0 else 0.0
        title = (f"Antibunching peak — "
                 f"{shape_a.capitalize()} + {shape_b.capitalize()} electrons")

        tab_dip, tab_jea, tab_marg, tab_swap = st.tabs(
            ["📈 Antibunching peak", "🟦 JEI", "📊 Marginals", "🔄 Swap kernel"])

        with tab_dip:
            st.plotly_chart(
                hom_plot(tau_array_s * 1e12, p_coinc, "τ (ps)", title, baseline, overlays),
                use_container_width=True)
            c1, c2 = st.columns([1, 4])
            if c1.button("📸 Save snapshot", key="fermi_save_snap"):
                save_snapshot("fermi", snap_label, tau_array_s * 1e12, p_coinc)
                st.success(f"Saved: {snap_label[:80]}…")
            with c2:
                csv_download_button({"tau_ps": tau_array_s * 1e12, "p_coinc": p_coinc},
                                    "hom_peak.csv", "📥 Download peak curve (CSV)",
                                    key="fermi_csv_dip")

        with tab_jea:
            st.plotly_chart(jea_heatmap(jea, ea, eb), use_container_width=True)

        with tab_marg:
            st.plotly_chart(electron_marginals_plot(jea, ea, eb), use_container_width=True)
            ms, mi = get_marginal_spectra(jea, ea, eb)
            csv_download_button(
                {"ea_ueV": ea / ueV, "marginal_s": ms / ms.max(),
                 "eb_ueV": eb / ueV, "marginal_i": mi / mi.max()},
                "electron_marginals.csv", "📥 Download marginals (CSV)", key="fermi_csv_marg")

        with tab_swap:
            st.plotly_chart(swap_kernel_plot(jea, ea, eb, axis_unit="ueV"),
                            use_container_width=True)
            st.caption(
                "**What this shows.** The swap kernel "
                "Re[S(ε_s,ε_i)] = Re[f*(ε_s,ε_i)·f(ε_i,ε_s)] is the integrand of the "
                "overlap V, and it is non-negative for this product state (V ≥ 0). "
                "The fermionic statistics enter not through the kernel's sign but "
                "through a global sign flip of the interference term (operator "
                "anticommutation): the **+** sign turns the bosonic dip into the "
                "antibunching **peak**.")

        st.divider()
        with st.expander("📋 Numerical readouts", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("V (overlap, τ=0)", f"{V_swap:.4f}")
            r2.metric("Peak visibility", f"{peak_vis:.4f}",
                      help="(max p − baseline)/baseline — measured at the maximum "
                           "of the curve over τ.")
            r3.metric("max p(τ)", f"{p_max:.4f}", help=f"at τ = {tau_peak_ps:.0f} ps")
            r4.metric("plateau (T²+R²)", f"{baseline:.4f}")

            r5, r6, r7, _ = st.columns(4)
            def tcoh_ps(shape, w):
                if shape == "leviton":
                    return w                      # τ₀ ya en ps
                return (hbar / (w * ueV)) * 1e12  # ℏ/σ o ℏ/Γ en ps
            r5.metric("τ_coh A", f"{tcoh_ps(shape_a, w_a):.0f} ps")
            r6.metric("τ_coh B", f"{tcoh_ps(shape_b, w_b):.0f} ps")
            if shape_a != "leviton" and shape_b != "leviton":
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
apply_pending_exp_load()   # aplica parámetros de experimento pendientes ANTES de los widgets

tab_boson, tab_fermion = st.tabs(["🔵 Bosonic (photons)", "🟠 Fermionic (electrons)"])
with tab_boson:
    render_bosonic()
with tab_fermion:
    render_fermionic()
