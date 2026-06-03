"""
Hong-Ou-Mandel Effect Simulator — Streamlit app

Stage 4: tabbed plots, CSV downloads, snapshot overlays,
        numerical readouts panel.
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
    phase_matching,
)


# ──────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HOM Simulator",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Hong–Ou–Mandel Effect Simulator")
st.caption("Two-particle interference at a beam splitter — Master's thesis simulation")


# ──────────────────────────────────────────────────────────────────────
# Shared widget helpers
# ──────────────────────────────────────────────────────────────────────

def param_widget(label, key, min_val, max_val, default, step, help_text=""):
    if key not in st.session_state:
        st.session_state[key] = default

    def on_input_change():
        st.session_state[key] = st.session_state[f"{key}_input"]

    def on_slider_change():
        st.session_state[key] = st.session_state[f"{key}_slider"]

    col_label, col_reset = st.columns([5, 1])
    col_label.markdown(f"**{label}**", help=help_text if help_text else None)
    if col_reset.button("↺", key=f"{key}_reset",
                        help=f"Reset to default ({default})"):
        st.session_state[key] = default
        st.session_state[f"{key}_input"] = default
        st.session_state[f"{key}_slider"] = default
        st.rerun()

    st.number_input(
        label, min_value=min_val, max_value=max_val,
        value=st.session_state[key], step=step,
        key=f"{key}_input", label_visibility="collapsed",
        on_change=on_input_change,
    )
    st.slider(
        "", min_value=min_val, max_value=max_val,
        value=st.session_state[key], step=step,
        key=f"{key}_slider", label_visibility="collapsed",
        on_change=on_slider_change,
    )
    return st.session_state[key]


def reset_param(key, default):
    st.session_state[key] = default
    st.session_state[f"{key}_input"] = default
    st.session_state[f"{key}_slider"] = default


# ──────────────────────────────────────────────────────────────────────
# CSV download helper
# ──────────────────────────────────────────────────────────────────────

def csv_download_button(data_dict, filename, label="📥 Download as CSV", key=None):
    """
    data_dict: {'col_name': np.ndarray, ...}
    All arrays must be the same length.
    """
    keys = list(data_dict.keys())
    arrays = [data_dict[k] for k in keys]
    n = len(arrays[0])

    buf = io.StringIO()
    buf.write(",".join(keys) + "\n")
    for i in range(n):
        row = ",".join(f"{arr[i]:.6e}" for arr in arrays)
        buf.write(row + "\n")

    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# ──────────────────────────────────────────────────────────────────────
# Filter UI (bosonic only)
# ──────────────────────────────────────────────────────────────────────

FILTER_SHAPES = [
    "gaussian", "rectangular", "lorentzian",
    "sinc", "super_gaussian", "triangular",
]

FILTER_HELP = {
    "gaussian":      "Smooth bell curve. Most common in experiments.",
    "rectangular":   "Hard bandpass cutoff. Ideal filter.",
    "lorentzian":    "Broad tails. Models cavity/resonator filters.",
    "sinc":          "Sinc² profile. Models grating-based filters.",
    "super_gaussian":"Flat top with steep edges. Order-4.",
    "triangular":    "Linear rolloff on each side.",
}


def filter_ui(prefix, center_default_nm, omega_center_default):
    with st.expander("🔧 Spectral filters", expanded=False):
        use_filter = st.toggle(
            "Enable spectral filtering",
            key=f"{prefix}_filter_on",
            help="Apply a spectral filter to signal and idler arms. "
                 "Can increase HOM visibility for asymmetric sources.",
        )
        if not use_filter:
            return False, None

        filter_shape = st.selectbox(
            "Filter shape", FILTER_SHAPES,
            key=f"{prefix}_filter_shape",
            help="\n".join(f"**{k}**: {v}" for k, v in FILTER_HELP.items()),
        )

        st.markdown("**Signal arm**")
        c1, c2 = st.columns(2)
        with c1:
            fc_s_nm = param_widget(
                "Center λ_s (nm)", f"{prefix}_fc_s",
                400.0, 1550.0, center_default_nm, 0.5,
                help_text="Central wavelength of the signal filter.",
            )
        with c2:
            fw_s_THz = param_widget(
                "Width (THz)", f"{prefix}_fw_s",
                0.01, 10.0, 2.0, 0.01,
                help_text="Filter bandwidth.",
            )

        st.markdown("**Idler arm**")
        c3, c4 = st.columns(2)
        with c3:
            fc_i_nm = param_widget(
                "Center λ_i (nm)", f"{prefix}_fc_i",
                400.0, 1550.0, center_default_nm, 0.5,
                help_text="Central wavelength of the idler filter.",
            )
        with c4:
            fw_i_THz = param_widget(
                "Width (THz)", f"{prefix}_fw_i",
                0.01, 10.0, 2.0, 0.01,
                help_text="Filter bandwidth.",
            )

        center_s = 2 * np.pi * const.c / (fc_s_nm * 1e-9)
        center_i = 2 * np.pi * const.c / (fc_i_nm * 1e-9)
        width_s  = fw_s_THz * 1e12 * 2 * np.pi
        width_i  = fw_i_THz * 1e12 * 2 * np.pi

        def filtered_jsa_fn(jsa, ws, wi):
            return apply_filter(
                jsa, ws, wi,
                center_s=center_s, width_s=width_s,
                center_i=center_i, width_i=width_i,
                shape=filter_shape,
            )
        return True, filtered_jsa_fn

    return False, None


# ──────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────

def hom_plot(x_values, p_coinc, x_label, title, baseline=0.5):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=p_coinc,
        mode='lines', line=dict(width=3, color='#636EFA'),
        name='p_coinc',
    ))
    fig.update_layout(
        title=title, xaxis_title=x_label,
        yaxis_title="Coincidence probability",
        template="plotly_white", height=500,
        yaxis=dict(range=[-0.05, 1.10]),
    )
    return fig


def jsa_heatmap(jsa, ws, wi, title="Joint Spectral Intensity"):
    f_center = ws.mean() / (2 * np.pi * 1e12)
    fs = ws / (2 * np.pi * 1e12) - f_center
    fi = wi / (2 * np.pi * 1e12) - f_center
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(jsa)**2, x=fs, y=fi,
        colorscale='Inferno',
        colorbar=dict(title="|JSA|²"),
        zsmooth='best',
    ))
    fig.update_layout(
        title=title,
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
    fig.add_trace(go.Scatter(
        x=fs, y=marginal_s / marginal_s.max(),
        mode='lines', line=dict(width=2.5, color='#636EFA'),
        name='Signal',
    ))
    fig.add_trace(go.Scatter(
        x=fi, y=marginal_i / marginal_i.max(),
        mode='lines', line=dict(width=2.5, color='#EF553B', dash='dash'),
        name='Idler',
    ))
    fig.update_layout(
        title="Marginal spectra",
        xaxis_title="Δf (THz)", yaxis_title="Intensity (norm.)",
        template="plotly_white", height=500,
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def jea_heatmap(jea, ea, eb, title="Joint Energy Amplitude |JEA|²"):
    eV = const.e
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(jea)**2,
        x=ea / (1e-3 * eV), y=eb / (1e-3 * eV),
        colorscale='Inferno', colorbar=dict(title="|JEA|²"),
        zsmooth='best',
    ))
    fig.update_layout(
        title=title,
        xaxis_title="ε_s (meV)", yaxis_title="ε_i (meV)",
        template="plotly_white", height=500,
    )
    return fig


def electron_marginals_plot(jea, ea, eb):
    eV = const.e
    marginal_s, marginal_i = get_marginal_spectra(jea, ea, eb)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ea / (1e-3 * eV), y=marginal_s / marginal_s.max(),
        mode='lines', line=dict(width=2.5, color='#636EFA'),
        name='Electron A',
    ))
    fig.add_trace(go.Scatter(
        x=eb / (1e-3 * eV), y=marginal_i / marginal_i.max(),
        mode='lines', line=dict(width=2.5, color='#EF553B', dash='dash'),
        name='Electron B',
    ))
    fig.update_layout(
        title="Marginal energy spectra",
        xaxis_title="ε (meV)", yaxis_title="Intensity (norm.)",
        template="plotly_white", height=500,
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def swap_kernel_plot(jsa, ws, wi, axis_unit="THz"):
    """
    Plots Re[S(ωs,ωi)] = Re[f*(ωs,ωi) · f(ωi,ωs)] as a single heatmap.
    Bright regions deepen the HOM dip; dark/negative regions shallow it.
    """
    swap = np.conj(jsa) * jsa.T
    re_s = np.real(swap)

    if axis_unit == "THz":
        center = ws.mean() / (2 * np.pi * 1e12)
        x = ws / (2 * np.pi * 1e12) - center
        y = wi / (2 * np.pi * 1e12) - center
        x_label, y_label = "Δf_s (THz)", "Δf_i (THz)"
    else:  # meV for electrons
        eV = const.e
        x = ws / (1e-3 * eV)
        y = wi / (1e-3 * eV)
        x_label, y_label = "ε_s (meV)", "ε_i (meV)"

    vmax = np.abs(re_s).max()
    fig = go.Figure(data=go.Heatmap(
        z=re_s, x=x, y=y,
        colorscale='RdBu_r', zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="Re[S]"),
        zsmooth='best',
    ))
    fig.update_layout(
        title="Swap kernel Re[S] — integrand of V",
        xaxis_title=x_label, yaxis_title=y_label,
        template="plotly_white", height=500,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Snapshot management
# ──────────────────────────────────────────────────────────────────────

def init_snapshots(prefix):
    if f"{prefix}_snapshots" not in st.session_state:
        st.session_state[f"{prefix}_snapshots"] = []


def save_snapshot(prefix, label, x_arr, y_arr):
    init_snapshots(prefix)
    st.session_state[f"{prefix}_snapshots"].append({
        "label": label, "x": x_arr.copy(), "y": y_arr.copy(),
    })


def render_snapshot_panel(prefix, x_label, y_label, title):
    init_snapshots(prefix)
    snaps = st.session_state[f"{prefix}_snapshots"]

    if not snaps:
        st.info("No snapshots yet. Use the **📸 Save snapshot** button "
                "above to add the current curve to this overlay.")
        return

    fig = go.Figure()
    palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    for k, snap in enumerate(snaps):
        fig.add_trace(go.Scatter(
            x=snap["x"], y=snap["y"],
            mode='lines', line=dict(width=2.5, color=palette[k % len(palette)]),
            name=snap["label"],
        ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        template="plotly_white", height=500,
        yaxis=dict(range=[-0.05, 1.10]),
        legend=dict(orientation="v", x=1.02, y=1.0),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([1, 5])
    if c1.button("🗑️ Clear snapshots", key=f"{prefix}_clear_snaps"):
        st.session_state[f"{prefix}_snapshots"] = []
        st.rerun()

    # CSV download of all snapshots
    buf = io.StringIO()
    buf.write("x," + ",".join(s["label"].replace(",", ";") for s in snaps) + "\n")
    n = max(len(s["x"]) for s in snaps)
    for i in range(n):
        row_vals = []
        if i < len(snaps[0]["x"]):
            row_vals.append(f"{snaps[0]['x'][i]:.6e}")
        else:
            row_vals.append("")
        for s in snaps:
            if i < len(s["y"]):
                row_vals.append(f"{s['y'][i]:.6e}")
            else:
                row_vals.append("")
        buf.write(",".join(row_vals) + "\n")
    c2.download_button(
        "📥 Download all snapshots (CSV)",
        data=buf.getvalue(),
        file_name=f"{prefix}_snapshots.csv",
        mime="text/csv",
        key=f"{prefix}_snap_csv",
    )


# ──────────────────────────────────────────────────────────────────────
# BOSONIC tab
# ──────────────────────────────────────────────────────────────────────

def render_bosonic():
    col_params, col_plot = st.columns([1, 2], gap="large")

    with col_params:
        st.subheader("Source")
        source = st.radio(
            "Source type",
            ["SPDC pair", "Independent photons"],
            key="boson_source",
            help="SPDC: entangled photon pairs from a nonlinear crystal. "
                 "Independent: two separate single-photon sources.",
        )

        if st.session_state.get("boson_source_prev") != source:
            st.session_state["boson_source_prev"] = source
            if source == "Independent photons":
                for k, v in [("boson_la", 800.0), ("boson_lb", 800.0),
                              ("boson_sa", 1.0),   ("boson_sb", 1.0)]:
                    reset_param(k, v)
            else:
                crystal = st.session_state.get("boson_crystal", "BBO (I)")
                lp = float(crystal_dict[crystal]['std_pump'])
                reset_param("boson_lp", lp)
            st.rerun()

        st.divider()

        if source == "SPDC pair":
            st.subheader("Crystal")
            crystal_name = st.selectbox(
                "Crystal", list(crystal_dict.keys()),
                key="boson_crystal",
                help="Type-I (BBO-I, LBO-I): symmetric JSA (V≈1). "
                     "Type-II (BBO-II, KTP-II, PPKTP-II): asymmetric (V<1).",
            )
            default_lp = float(crystal_dict[crystal_name]['std_pump'])

            if st.session_state.get("boson_crystal_prev") != crystal_name:
                st.session_state["boson_crystal_prev"] = crystal_name
                reset_param("boson_lp", default_lp)
                st.rerun()

            st.subheader("Pump")
            lambda_p = param_widget("λ pump (nm)", "boson_lp",
                                    300.0, 1064.0, default_lp, 0.5,
                                    help_text="Central pump wavelength.")
            sigma_p_THz = param_widget("Pump bandwidth σ (THz)", "boson_sp",
                                       0.01, 10.0, 1.0, 0.01,
                                       help_text="0.01 THz≈CW, 1 THz≈1ps pulse.")
            L_um = param_widget("Crystal length (μm)", "boson_L",
                                100.0, 20000.0, 1000.0, 100.0,
                                help_text="Longer = narrower phase-matching sinc.")
            degenerate_nm = 2 * lambda_p

        else:
            st.subheader("Photon A")
            lambda_a = param_widget("λ A (nm)", "boson_la",
                                    400.0, 1550.0, 800.0, 0.5,
                                    help_text="Wavelength of photon A.")
            sigma_a_THz = param_widget("Bandwidth σ A (THz)", "boson_sa",
                                       0.01, 10.0, 1.0, 0.01,
                                       help_text="Bandwidth of photon A.")
            st.divider()
            st.subheader("Photon B")
            lambda_b = param_widget("λ B (nm)", "boson_lb",
                                    400.0, 1550.0, 800.0, 0.5,
                                    help_text="Wavelength of photon B.")
            sigma_b_THz = param_widget("Bandwidth σ B (THz)", "boson_sb",
                                       0.01, 10.0, 1.0, 0.01,
                                       help_text="Bandwidth of photon B.")
            degenerate_nm = (lambda_a + lambda_b) / 2

        st.divider()
        st.subheader("Beam splitter")
        R_bs  = param_widget("Reflectivity R", "boson_R",
                             0.0, 1.0, 0.5, 0.01,
                             help_text="R=0.5 maximises HOM visibility.")
        V_pol = param_widget("Polarisation overlap |V|", "boson_V",
                             0.0, 1.0, 1.0, 0.01,
                             help_text="V=1: same polarisation.")

        st.divider()
        st.subheader("Delay axis τ")
        tau_max_fs = param_widget("τ max (fs)", "boson_tau",
                                  100.0, 5000.0, 2000.0, 100.0,
                                  help_text="Half-width of delay axis.")
        n_tau = st.slider("τ points", 51, 501, 201, 50, key="boson_ntau")

        st.divider()
        use_filter, filtered_jsa_fn = filter_ui(
            prefix="boson",
            center_default_nm=degenerate_nm,
            omega_center_default=2*np.pi*const.c/(degenerate_nm*1e-9),
        )

    # ── Compute ──────────────────────────────────────────────────────
    tau_array = np.linspace(-tau_max_fs * 1e-15, tau_max_fs * 1e-15, n_tau)

    if source == "SPDC pair":
        sigma_p = sigma_p_THz * 1e12 * 2 * np.pi
        try:
            jsa, ws, wi = jsa_function(lambda_p, sigma_p, crystal_name,
                                        int(L_um), grid_size=500)
            jsa_vis, ws_vis, wi_vis = jsa_function(
                lambda_p, sigma_p, crystal_name, int(L_um),
                grid_size=400, span_factor=3,
            )
            phi_func = phase_matching(lambda_p, crystal_name, int(L_um))
            theta_deg     = phi_func.theta_deg
            poling_period = phi_func.poling_period
        except Exception as e:
            with col_plot:
                st.error(f"JSA computation failed: {e}")
            return
        title = f"HOM dip — SPDC ({crystal_name}, λp={lambda_p:.0f} nm)"
        snap_label = (f"SPDC {crystal_name} | λp={lambda_p:.0f}nm | "
                      f"σp={sigma_p_THz:.2f}THz | L={L_um:.0f}μm | "
                      f"R={R_bs:.2f} | V_pol={V_pol:.2f}")
    else:
        sigma_a = sigma_a_THz * 1e12 * 2 * np.pi
        sigma_b = sigma_b_THz * 1e12 * 2 * np.pi
        jsa, ws, wi = independent_jsa_function(
            lambda_a, lambda_b, sigma_a, sigma_b, grid_size=500,
        )
        jsa_vis, ws_vis, wi_vis = independent_jsa_function(
            lambda_a, lambda_b, sigma_a, sigma_b,
            grid_size=400, span_factor=3,
        )
        theta_deg, poling_period = None, None
        title = (f"HOM dip — independent photons "
                 f"(λA={lambda_a:.0f}, λB={lambda_b:.0f} nm)")
        snap_label = (f"Indep | λA={lambda_a:.0f}nm σA={sigma_a_THz:.2f}THz | "
                      f"λB={lambda_b:.0f}nm σB={sigma_b_THz:.2f}THz | "
                      f"R={R_bs:.2f}")

    if use_filter and filtered_jsa_fn is not None:
        try:
            jsa     = filtered_jsa_fn(jsa, ws, wi)
            jsa_vis = filtered_jsa_fn(jsa_vis, ws_vis, wi_vis)
            title   += " [filtered]"
            snap_label += " [filtered]"
        except ValueError as e:
            with col_plot:
                st.warning(f"Filter warning: {e}")

    V_swap  = get_intrinsic_indistinguishability(jsa, ws, wi)
    p_coinc = hom_coincidence_rate(jsa, ws, wi, tau_array,
                                   R=R_bs, V_pol=V_pol)

    # ── Plot tabs ─────────────────────────────────────────────────────
    with col_plot:
        T        = 1 - R_bs
        baseline = T**2 + R_bs**2

        tab_dip, tab_jsa, tab_marg, tab_swap = st.tabs([
            "📈 HOM dip", "🟦 JSI", "📊 Marginals", "🔄 Swap kernel",
        ])

        with tab_dip:
            st.plotly_chart(
                hom_plot(tau_array * 1e15, p_coinc,
                         "τ (fs)", title, baseline),
                use_container_width=True,
            )
            c1, c2 = st.columns([1, 4])
            if c1.button("📸 Save snapshot", key="boson_save_snap"):
                save_snapshot("boson", snap_label,
                              tau_array * 1e15, p_coinc)
                st.success(f"Saved: {snap_label[:80]}…")
            with c2:
                csv_download_button(
                    {"tau_fs": tau_array * 1e15, "p_coinc": p_coinc},
                    filename="hom_dip.csv",
                    label="📥 Download HOM curve (CSV)",
                    key="boson_csv_dip",
                )

        with tab_jsa:
            st.plotly_chart(
                jsa_heatmap(jsa_vis, ws_vis, wi_vis),
                use_container_width=True,
            )

        with tab_marg:
            st.plotly_chart(
                marginals_plot(jsa_vis, ws_vis, wi_vis),
                use_container_width=True,
            )
            ms, mi = get_marginal_spectra(jsa_vis, ws_vis, wi_vis)
            f_center = ws_vis.mean() / (2 * np.pi * 1e12)
            fs = ws_vis / (2 * np.pi * 1e12) - f_center
            fi = wi_vis / (2 * np.pi * 1e12) - f_center
            csv_download_button(
                {"df_s_THz": fs, "marginal_s": ms / ms.max(),
                 "df_i_THz": fi, "marginal_i": mi / mi.max()},
                filename="marginals.csv",
                label="📥 Download marginals (CSV)",
                key="boson_csv_marg",
            )

        with tab_swap:
            st.plotly_chart(
                swap_kernel_plot(jsa, ws, wi, axis_unit="THz"),
                use_container_width=True,
            )
            st.caption(
                "**What this shows.** The swap kernel "
                "Re[S(ω_s,ω_i)] = Re[f*(ω_s,ω_i)·f(ω_i,ω_s)] is the "
                "integrand of the indistinguishability V. Bright regions "
                "deepen the HOM dip (constructive HOM interference); "
                "blue/negative regions shallow it. V is the integral of "
                "this whole 2D map."
            )

        # ── Numerical readouts ────────────────────────────────────────
        st.divider()
        with st.expander("📋 Numerical readouts", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Spectral indistinguishability V", f"{V_swap:.4f}",
                      help="Determines max dip depth.")
            r2.metric("p(τ=0)", f"{p_coinc[len(p_coinc)//2]:.4f}")
            r3.metric("min p(τ)", f"{p_coinc.min():.4f}")
            r4.metric("plateau (T²+R²)", f"{baseline:.4f}")

            if source == "SPDC pair":
                r5, r6, r7, r8 = st.columns(4)
                r5.metric("Phase-matching θ", f"{theta_deg:.2f}°",
                          help="Angle between propagation and crystal axis.")
                if poling_period > 0:
                    r6.metric("Poling period Λ", f"{poling_period:.2f} μm",
                              help="QPM period for periodically-poled crystals.")
                else:
                    r6.metric("Poling period Λ", "N/A",
                              help="Bulk crystal — no periodic poling.")
                walkoff_um = L_um * np.tan(np.radians(theta_deg))
                r7.metric("Walk-off ~ L·tan(θ)", f"{walkoff_um:.1f} μm",
                          help="Spatial separation of o- and e-rays at exit.")
                tau_coh_fs = 1.0 / sigma_p_THz * 1000
                r8.metric("Pump τ_coh ≈ 1/σ", f"{tau_coh_fs:.0f} fs",
                          help="Order-of-magnitude pump coherence time.")
            else:
                r5, r6, r7, _ = st.columns(4)
                tau_coh_a = 1.0 / sigma_a_THz * 1000
                tau_coh_b = 1.0 / sigma_b_THz * 1000
                sigma_a_rad = sigma_a_THz * 1e12 * 2 * np.pi
                sigma_b_rad = sigma_b_THz * 1e12 * 2 * np.pi
                domega = (2*np.pi*const.c/(lambda_a*1e-9) -
                          2*np.pi*const.c/(lambda_b*1e-9))
                overlap = ((2*sigma_a_rad*sigma_b_rad /
                            (sigma_a_rad**2 + sigma_b_rad**2)) *
                           np.exp(-domega**2 / (sigma_a_rad**2 + sigma_b_rad**2)))
                r5.metric("τ_coh A", f"{tau_coh_a:.0f} fs")
                r6.metric("τ_coh B", f"{tau_coh_b:.0f} fs")
                r7.metric("|⟨φA|φB⟩|²", f"{overlap**2:.4f}",
                          help="Single-photon spectral overlap (Gaussian formula).")

        # ── Snapshots overlay ─────────────────────────────────────────
        st.divider()
        st.subheader("📸 Snapshots — overlay multiple curves")
        render_snapshot_panel("boson",
                              x_label="τ (fs)",
                              y_label="Coincidence probability",
                              title="HOM dip — overlay")


# ──────────────────────────────────────────────────────────────────────
# FERMIONIC tab
# ──────────────────────────────────────────────────────────────────────

def render_fermionic():
    col_params, col_plot = st.columns([1, 2], gap="large")
    eV   = const.e
    hbar = const.hbar

    with col_params:
        st.subheader("Source — Slater determinant of two electrons")

        st.subheader("Electron A")
        shape_a = st.selectbox(
            "Shape A", ["gaussian", "lorentzian", "leviton"],
            key="fermi_shape_a",
            help="Gaussian: bell. Lorentzian: broad tails. "
                 "Leviton: minimal-excitation, one-sided exponential.",
        )
        if st.session_state.get("fermi_shape_a_prev") != shape_a:
            st.session_state["fermi_shape_a_prev"] = shape_a
            reset_param("fermi_wa", 30.0 if shape_a == "leviton" else 2.0)
            st.rerun()

        e0_a_meV = param_widget("ε₀ A (meV)", "fermi_e0_a",
                                0.0, 50.0, 10.0, 0.1,
                                help_text="Energy above Fermi level.")
        if shape_a == "gaussian":
            w_a = param_widget("σ A (meV)", "fermi_wa", 0.1, 30.0, 2.0, 0.1,
                               help_text="Energy std deviation.")
            params_a = {'sigma': w_a * 1e-3 * eV}
        elif shape_a == "lorentzian":
            w_a = param_widget("Γ A (meV)", "fermi_wa", 0.1, 30.0, 2.0, 0.1,
                               help_text="FWHM linewidth.")
            params_a = {'Gamma': w_a * 1e-3 * eV}
        else:
            w_a = param_widget("τ₀ A (fs)", "fermi_wa", 10.0, 2000.0, 30.0, 10.0,
                               help_text="Leviton decay time.")
            params_a = {'tau_0': w_a * 1e-15}

        st.divider()
        st.subheader("Electron B")
        shape_b = st.selectbox(
            "Shape B", ["gaussian", "lorentzian", "leviton"],
            key="fermi_shape_b",
        )
        if st.session_state.get("fermi_shape_b_prev") != shape_b:
            st.session_state["fermi_shape_b_prev"] = shape_b
            reset_param("fermi_wb", 60.0 if shape_b == "leviton" else 2.0)
            st.rerun()

        e0_b_meV = param_widget("ε₀ B (meV)", "fermi_e0_b",
                                0.0, 50.0, 11.0, 0.1)
        if shape_b == "gaussian":
            w_b = param_widget("σ B (meV)", "fermi_wb", 0.1, 30.0, 2.0, 0.1)
            params_b = {'sigma': w_b * 1e-3 * eV}
        elif shape_b == "lorentzian":
            w_b = param_widget("Γ B (meV)", "fermi_wb", 0.1, 30.0, 2.0, 0.1)
            params_b = {'Gamma': w_b * 1e-3 * eV}
        else:
            w_b = param_widget("τ₀ B (fs)", "fermi_wb",
                               10.0, 2000.0, 60.0, 10.0)
            params_b = {'tau_0': w_b * 1e-15}

        st.divider()
        eF_meV = param_widget("Fermi energy ε_F (meV)", "fermi_eF",
                              0.0, 30.0, 0.0, 0.1,
                              help_text="States below ε_F are forbidden.")

        st.divider()
        st.subheader("Beam splitter")
        R_bs  = param_widget("Reflectivity R", "fermi_R",
                             0.0, 1.0, 0.5, 0.01)
        V_pol = param_widget("Spin overlap |V|", "fermi_V",
                             0.0, 1.0, 1.0, 0.01)

        st.divider()
        st.subheader("Delay axis τ")
        tau_max_fs = param_widget("τ max (fs)", "fermi_tau",
                                  10.0, 1000.0, 100.0, 10.0)
        n_tau = st.slider("τ points", 51, 501, 201, 50, key="fermi_ntau")

    # ── Compute ──────────────────────────────────────────────────────
    try:
        jea, ea, eb = independent_jea_function(
            e0_a_meV * 1e-3 * eV, e0_b_meV * 1e-3 * eV,
            shape_a, shape_b, params_a, params_b,
            varepsilon_F=eF_meV * 1e-3 * eV,
        )
    except (ValueError, ZeroDivisionError) as e:
        with col_plot:
            st.error("JEA construction failed (likely Pauli exclusion). "
                     "Use distinct energies or widths.")
            st.exception(e)
        return

    V_swap      = get_intrinsic_indistinguishability(jea, ea, eb)
    tau_array_s = np.linspace(-tau_max_fs * 1e-15, tau_max_fs * 1e-15, n_tau)
    p_coinc     = hom_coincidence_rate(jea, ea, eb, tau_array_s / hbar,
                                        R=R_bs, V_pol=V_pol)

    snap_label = (f"{shape_a}/{shape_b} | ε₀={e0_a_meV:.1f}/{e0_b_meV:.1f}meV | "
                  f"w={w_a:.1f}/{w_b:.1f} | R={R_bs:.2f}")

    # ── Plot tabs ─────────────────────────────────────────────────────
    with col_plot:
        T        = 1 - R_bs
        baseline = T**2 + R_bs**2
        title    = (f"Antibunching peak — "
                    f"{shape_a.capitalize()} + {shape_b.capitalize()} electrons")

        tab_dip, tab_jea, tab_marg, tab_swap = st.tabs([
            "📈 Antibunching peak", "🟦 JEI", "📊 Marginals", "🔄 Swap kernel",
        ])

        with tab_dip:
            st.plotly_chart(
                hom_plot(tau_array_s * 1e15, p_coinc,
                         "τ (fs)", title, baseline),
                use_container_width=True,
            )
            c1, c2 = st.columns([1, 4])
            if c1.button("📸 Save snapshot", key="fermi_save_snap"):
                save_snapshot("fermi", snap_label,
                              tau_array_s * 1e15, p_coinc)
                st.success(f"Saved: {snap_label[:80]}…")
            with c2:
                csv_download_button(
                    {"tau_fs": tau_array_s * 1e15, "p_coinc": p_coinc},
                    filename="hom_peak.csv",
                    label="📥 Download peak curve (CSV)",
                    key="fermi_csv_dip",
                )

        with tab_jea:
            st.plotly_chart(jea_heatmap(jea, ea, eb),
                            use_container_width=True)

        with tab_marg:
            st.plotly_chart(electron_marginals_plot(jea, ea, eb),
                            use_container_width=True)
            ms, mi = get_marginal_spectra(jea, ea, eb)
            csv_download_button(
                {"ea_meV": ea / (1e-3 * eV), "marginal_s": ms / ms.max(),
                 "eb_meV": eb / (1e-3 * eV), "marginal_i": mi / mi.max()},
                filename="electron_marginals.csv",
                label="📥 Download marginals (CSV)",
                key="fermi_csv_marg",
            )

        with tab_swap:
            st.plotly_chart(
                swap_kernel_plot(jea, ea, eb, axis_unit="meV"),
                use_container_width=True,
            )
            st.caption(
                "**What this shows.** The swap kernel "
                "Re[S(ε_s,ε_i)] is the integrand of V. For fermions the "
                "JEA is antisymmetric, so S is overall **negative** — "
                "this is what produces V<0 and the antibunching peak "
                "(rather than a dip)."
            )

        # ── Numerical readouts ────────────────────────────────────────
        st.divider()
        with st.expander("📋 Numerical readouts", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Spectral indistinguishability V", f"{V_swap:.4f}")
            r2.metric("p(τ=0)", f"{p_coinc[len(p_coinc)//2]:.4f}")
            r3.metric("max p(τ)", f"{p_coinc.max():.4f}")
            r4.metric("plateau (T²+R²)", f"{baseline:.4f}")

            r5, r6, r7, _ = st.columns(4)
            if shape_a == "gaussian":
                tcoh_a_fs = (hbar / (w_a * 1e-3 * eV)) * 1e15
            elif shape_a == "lorentzian":
                tcoh_a_fs = (hbar / (w_a * 1e-3 * eV)) * 1e15
            else:
                tcoh_a_fs = w_a
            if shape_b == "gaussian":
                tcoh_b_fs = (hbar / (w_b * 1e-3 * eV)) * 1e15
            elif shape_b == "lorentzian":
                tcoh_b_fs = (hbar / (w_b * 1e-3 * eV)) * 1e15
            else:
                tcoh_b_fs = w_b
            r5.metric("τ_coh A", f"{tcoh_a_fs:.0f} fs",
                      help="Coherence time of electron A.")
            r6.metric("τ_coh B", f"{tcoh_b_fs:.0f} fs",
                      help="Coherence time of electron B.")
            r7.metric("Δε / max(σ_A, σ_B)",
                      f"{abs(e0_a_meV - e0_b_meV) / max(w_a, w_b):.2f}",
                      help="Energy detuning normalised to the larger width. "
                           "0 = identical, ≫1 = fully distinguishable.")

        # ── Snapshots overlay ─────────────────────────────────────────
        st.divider()
        st.subheader("📸 Snapshots — overlay multiple curves")
        render_snapshot_panel("fermi",
                              x_label="τ (fs)",
                              y_label="Coincidence probability",
                              title="Antibunching peak — overlay")


# ──────────────────────────────────────────────────────────────────────
# Top-level layout
# ──────────────────────────────────────────────────────────────────────

tab_boson, tab_fermion = st.tabs([
    "🔵 Bosonic (photons)",
    "🟠 Fermionic (electrons)",
])

with tab_boson:
    render_bosonic()

with tab_fermion:
    render_fermionic()