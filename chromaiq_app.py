"""
ChromaIQ — Plateforme d'Intelligence Analytique
SaaS démo pour laboratoires analytiques (LC-MS, GC-MS, HRMS, ICP-MS)

Run :  streamlit run chromaiq_app.py
Deps:  pip install streamlit plotly numpy pandas
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChromaIQ — Intelligence Analytique",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Professional scientific SaaS look
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  /* ── Root vars ── */
  :root {
    --bg: #F0F4F8;
    --surface: #FFFFFF;
    --sidebar: #0E1521;
    --primary: #0284C7;
    --success: #059669;
    --warning: #D97706;
    --danger: #DC2626;
    --text: #0F172A;
    --text2: #475569;
    --text3: #94A3B8;
    --border: #E2E8F0;
    --border2: #F1F5F9;
    --mono: 'IBM Plex Mono', 'Consolas', monospace;
    --sans: 'IBM Plex Sans', system-ui, sans-serif;
  }

  /* ── App background ── */
  .stApp { background: var(--bg) !important; font-family: var(--sans) !important; }
  .stApp * { font-family: var(--sans) !important; }

  /* ── Main block padding ── */
  .main .block-container { padding: 2rem 2.5rem 1rem 2.5rem !important; max-width: 1400px !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background: var(--sidebar) !important; }
  [data-testid="stSidebar"] * { color: #C8D6E8 !important; }
  [data-testid="stSidebar"] .stRadio label { color: #94A3B8 !important; font-size: 13px; }
  [data-testid="stSidebar"] hr { border-color: #1A2535 !important; }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #fff !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px 10px 0 0;
    border: 1px solid var(--border);
    border-bottom: none;
    padding: 4px 4px 0;
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text2);
    border-radius: 8px 8px 0 0;
    border: none;
    background: transparent;
  }
  .stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    background: var(--bg) !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--primary) !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0 10px 10px 10px;
    padding: 24px;
  }

  /* ── Metric cards ── */
  [data-testid="stMetricValue"] { font-family: var(--mono) !important; font-size: 28px !important; color: var(--text) !important; }
  [data-testid="stMetricLabel"] { font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text3) !important; font-family: var(--mono) !important; }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }
  [data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
  .stDataFrame table { font-family: var(--mono) !important; font-size: 12px !important; }
  .stDataFrame thead th { background: var(--border2) !important; color: var(--text2) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600 !important; }

  /* ── Buttons ── */
  .stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text2) !important;
  }
  .stButton button:hover { border-color: var(--primary) !important; color: var(--primary) !important; }

  /* ── Selectbox ── */
  [data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    font-size: 13px !important;
    background: var(--surface) !important;
  }

  /* ── Headings ── */
  h1 { font-size: 20px !important; font-weight: 700 !important; color: var(--text) !important; }
  h2 { font-size: 15px !important; font-weight: 600 !important; color: var(--text) !important; }
  h3 { font-size: 13px !important; font-weight: 600 !important; color: var(--text2) !important; }

  /* ── Custom components ── */
  .badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 9px; border-radius: 6px;
    font-size: 11px; font-weight: 700; font-family: var(--mono);
    white-space: nowrap; border: 1px solid transparent;
  }
  .badge-ok      { background: #ECFDF5; color: #065F46; border-color: #6EE7B7; }
  .badge-warning { background: #FFFBEB; color: #92400E; border-color: #FCD34D; }
  .badge-danger  { background: #FEF2F2; color: #991B1B; border-color: #FCA5A5; }
  .badge-info    { background: #EFF6FF; color: #1E40AF; border-color: #93C5FD; }

  .section-label {
    font-size: 11px; font-weight: 700; color: var(--text3);
    text-transform: uppercase; letter-spacing: 1px;
    font-family: var(--mono); margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
  }

  .alert-danger {
    background: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 16px;
  }
  .alert-warning {
    background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 16px;
  }
  .alert-success {
    background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 16px;
  }
  .alert-title { font-size: 14px; font-weight: 700; margin-bottom: 4px; }
  .alert-body  { font-size: 12px; line-height: 1.7; }

  .info-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .kv-row { display: flex; gap: 8px; margin-bottom: 6px; font-size: 12px; }
  .kv-key { color: var(--text3); font-family: var(--mono); min-width: 160px; font-size: 11px; }
  .kv-val { color: var(--text); }

  .rec-box {
    border-radius: 8px; padding: 12px 16px; margin-top: 12px;
    font-size: 12px; line-height: 1.9;
  }
  .rec-box-danger  { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }
  .rec-box-warning { background: #FFFBEB; border: 1px solid #FDE68A; color: #78350F; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCIENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

COMPOUNDS = [
    {"id": 0, "name": "Étalon interne (IS)", "code": "IS",   "rt_ref": 2.15,  "s0": 0.042, "a0": 85000,  "type": "std", "color": "#6366F1"},
    {"id": 1, "name": "Principe actif (PA)", "code": "PA",   "rt_ref": 4.82,  "s0": 0.060, "a0": 238000, "type": "api", "color": "#0284C7", "spec_lo": 95,     "spec_hi": 105},
    {"id": 2, "name": "Impureté A",          "code": "ImpA", "rt_ref": 6.24,  "s0": 0.035, "a0": 4500,   "type": "imp", "color": "#D97706", "spec_hi_pct": 0.50},
    {"id": 3, "name": "Dégradant B",         "code": "DegB", "rt_ref": 8.91,  "s0": 0.040, "a0": 2300,   "type": "deg", "color": "#DC2626", "spec_hi_pct": 0.20},
    {"id": 4, "name": "Impureté C",          "code": "ImpC", "rt_ref": 11.33, "s0": 0.032, "a0": 1800,   "type": "imp", "color": "#7C3AED", "spec_hi_pct": 0.20},
]


def _lcg(seed, n):
    s = int(seed) & 0xFFFFFFFF
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        out[i] = s / 4294967296.0
    return out


def gauss(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


@st.cache_data(show_spinner=False)
def build_signal(drift: float, seed: int):
    N, t_max = 3000, 15.0
    t = np.linspace(0, t_max, N)
    rng = _lcg(seed, N + 20)
    noise_scale = 110.0 * (1.0 + drift * 0.58)

    cmpds = []
    for i, c in enumerate(COMPOUNDS):
        cmpds.append({**c,
            "rt":  c["rt_ref"]  + drift * (0.09 + i * 0.043),
            "sig": c["s0"]      * (1.0 + drift * 0.14),
            "amp": c["a0"]      * (1.0 - drift * 0.068),
        })

    y = (190.0
         + np.sin(t * 0.35 + 0.8) * 55.0
         + np.sin(t * 1.9) * 18.0
         + (rng[:N] - 0.5) * noise_scale * 2.3)
    for c in cmpds:
        y += gauss(t, c["rt"], c["sig"], c["amp"])
    y = np.maximum(y, 0.0)

    return t, y, cmpds, noise_scale


@st.cache_data(show_spinner=False)
def analyze_peaks(t, y, cmpds, noise_scale):
    BL, n_std = 190.0, noise_scale * 0.63
    results = []
    for c in cmpds:
        mask    = np.abs(t - c["rt"]) < 0.6
        w_y     = np.where(mask, y, -np.inf)
        mi      = int(np.argmax(w_y))
        apex_t  = float(t[mi])
        apex_y  = float(y[mi])
        half_h  = (apex_y + BL) / 2.0

        li = mi
        while li > 0 and y[li] > half_h:
            li -= 1
        ri = mi
        while ri < len(y) - 1 and y[ri] > half_h:
            ri += 1
        w50   = float(t[ri] - t[li])
        margin = int((ri - li) * 0.9)
        a_s   = max(0, li - margin)
        a_e   = min(len(y) - 1, ri + margin)
        bl2   = (y[a_s] + y[a_e]) / 2.0
        area  = float(np.trapz(np.maximum(y[a_s:a_e + 1] - bl2, 0.0), t[a_s:a_e + 1]))
        sn    = round((apex_y - BL) / max(n_std, 1e-6), 1)
        Np    = round(5.545 * (apex_t / max(w50, 1e-6)) ** 2)
        right_w = float(t[ri]) - apex_t
        left_w  = apex_t - float(t[li])
        tf    = round((left_w + right_w) / (2.0 * max(left_w, 1e-6)), 2)
        results.append({**c,
            "apex_t": round(apex_t, 3), "apex_y": apex_y,
            "w50": round(w50, 3), "area": round(area),
            "sn": sn, "Np": int(Np), "tf": tf,
            "li": a_s, "ri": a_e,
        })
    return results


RUNS_CFG = [
    {"label": "Référence (J-30)", "drift": 0.00, "seed": 42,  "id": "REF",   "date": "17/03/2026", "seq": "SEQ-0801", "status": "ok"},
    {"label": "J-7",              "drift": 0.32, "seed": 77,  "id": "J7",    "date": "09/04/2026", "seq": "SEQ-0839", "status": "warning"},
    {"label": "Aujourd'hui",      "drift": 0.68, "seed": 99,  "id": "TODAY", "date": "16/04/2026", "seq": "SEQ-0847", "status": "danger"},
]


@st.cache_data(show_spinner=False)
def get_run(drift, seed):
    t, y, cmpds, ns = build_signal(drift, seed)
    peaks = analyze_peaks(t, y, cmpds, ns)
    return t, y, cmpds, ns, peaks


@st.cache_data(show_spinner=False)
def get_history():
    rows = []
    for i in range(20):
        d = 0.0 if i < 13 else (i - 12) * 0.10
        rng = _lcg(i * 137 + 7, 5)
        rows.append({
            "Séquence":  f"SEQ-{828 + i}",
            "Date":      f"{(i+1):02d}/04",
            "TR IS (min)": round(2.15 + d * 0.90 + (rng[0] - 0.5) * 0.01, 3),
            "TR PA (min)": round(4.82 + d * 1.15 + (rng[1] - 0.5) * 0.013, 3),
            "Bruit (×)":   round(1.00 + d * 0.58 + (rng[2] - 0.5) * 0.035, 3),
            "Nb. plateaux": int(11200 * (1 - d * 0.17)),
            "_drift": d,
        })
    return pd.DataFrame(rows)


# Pre-build all runs once
_runs_cache = {r["id"]: get_run(r["drift"], r["seed"]) for r in RUNS_CFG}
t_ref, y_ref, _, _, peaks_ref = _runs_cache["REF"]
HIST = get_history()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def badge_html(text, variant="ok"):
    return f'<span class="badge badge-{variant}">{text}</span>'


def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def make_chroma_fig(t, y, peaks, drift, ns, show_ref=True):
    fig = go.Figure()

    # Reference overlay
    if show_ref and drift > 0.05:
        fig.add_trace(go.Scatter(
            x=t_ref, y=y_ref,
            mode="lines", name="Référence (J-30)",
            line=dict(color="#CBD5E1", width=1, dash="dot"),
            hoverinfo="skip",
        ))

    # Peak fills
    for p in peaks:
        xs = np.concatenate([t[p["li"]:p["ri"]+1], t[p["li"]:p["ri"]+1][::-1]])
        ys_top = y[p["li"]:p["ri"]+1]
        ys_bot = np.full(len(ys_top), 190.0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([t[p["li"]:p["ri"]+1], t[p["li"]:p["ri"]+1][::-1]]),
            y=np.concatenate([ys_top, ys_bot[::-1]]),
            fill="toself", fillcolor=p["color"] + "28",
            line=dict(color="transparent"),
            showlegend=False, hoverinfo="skip",
        ))

    # Main signal
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines", name="Signal UV (mAU)",
        line=dict(color="#0284C7", width=2),
        hovertemplate="TR : %{x:.3f} min &nbsp;|&nbsp; Signal : %{y:.0f} mAU<extra></extra>",
    ))

    # Apex markers
    fig.add_trace(go.Scatter(
        x=[p["apex_t"] for p in peaks],
        y=[p["apex_y"] + peaks[0]["apex_y"] * 0.07 for p in peaks],
        mode="markers+text",
        marker=dict(color=[p["color"] for p in peaks], size=10, symbol="triangle-down"),
        text=[p["code"] for p in peaks],
        textposition="top center",
        textfont=dict(size=11, color=[p["color"] for p in peaks]),
        showlegend=False,
        hovertemplate="%{text} — TR : %{x:.3f} min<extra></extra>",
    ))

    # ΔRT annotations for drifted runs
    annotations = []
    if drift > 0.22:
        for i in [0, 1]:
            p, pr = peaks[i], peaks_ref[i]
            delta = round(p["apex_t"] - pr["apex_t"], 2)
            col = "#DC2626" if drift > 0.5 else "#D97706"
            annotations.append(dict(
                x=p["apex_t"], y=p["apex_y"] * 1.10,
                text=f"<b>Δ +{delta} min</b>", showarrow=False,
                font=dict(size=11, color=col, family="IBM Plex Mono, monospace"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=col, borderwidth=1, borderpad=3, xanchor="center",
            ))

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#FAFCFF",
        margin=dict(l=64, r=20, t=20, b=52), height=360,
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=11, color="#475569"),
        xaxis=dict(title=dict(text="Temps de rétention (min)", font=dict(size=11)),
                   gridcolor="#F1F5F9", zerolinecolor="#E2E8F0",
                   range=[0, 15], dtick=1),
        yaxis=dict(title=dict(text="Signal (mAU)", font=dict(size=11)),
                   gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", rangemode="tozero"),
        showlegend=drift > 0.05,
        legend=dict(orientation="h", y=-0.17, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest", annotations=annotations,
    )
    return fig


def make_drift_fig():
    x = list(range(len(HIST)))
    labels = HIST["Date"].tolist()

    # RT chart
    fig_rt = go.Figure()
    fig_rt.add_trace(go.Scatter(x=x, y=HIST["TR IS (min)"].tolist(), mode="lines+markers", name="TR IS", line=dict(color="#6366F1", width=2), marker=dict(size=5)))
    fig_rt.add_trace(go.Scatter(x=x, y=HIST["TR PA (min)"].tolist(), mode="lines+markers", name="TR PA", line=dict(color="#0284C7", width=2), marker=dict(size=5)))
    fig_rt.add_hline(y=2.30, line_dash="dash", line_color="#D97706", line_width=1.5, annotation_text="Seuil IS (+0.15)", annotation_position="top right", annotation_font_size=10, annotation_font_color="#D97706")
    fig_rt.update_layout(**_drift_layout("Temps de rétention (min)", labels, [2.0, 2.75]))

    # Noise chart
    fig_noise = go.Figure()
    fig_noise.add_trace(go.Scatter(x=x, y=HIST["Bruit (×)"].tolist(), mode="lines+markers", name="Bruit rel.", fill="tozeroy", fillcolor="#DC262618", line=dict(color="#DC2626", width=2), marker=dict(size=5)))
    fig_noise.add_hline(y=1.30, line_dash="dash", line_color="#DC2626", line_width=1.5, annotation_text="Seuil ×1.30", annotation_position="top right", annotation_font_size=10, annotation_font_color="#DC2626")
    fig_noise.update_layout(**_drift_layout("Bruit rel. (×référence)", labels, [0.85, 2.0]))

    # Plate count
    fig_np = go.Figure()
    fig_np.add_trace(go.Scatter(x=x, y=HIST["Nb. plateaux"].tolist(), mode="lines+markers", name="Nb. plateaux", line=dict(color="#059669", width=2), marker=dict(size=5)))
    fig_np.add_hline(y=10000, line_dash="dash", line_color="#D97706", line_width=1.5, annotation_text="Spec >10 000", annotation_position="bottom right", annotation_font_size=10, annotation_font_color="#D97706")
    fig_np.update_layout(**_drift_layout("Nb. plateaux théoriques (PA)", labels, [5500, 12500]))

    return fig_rt, fig_noise, fig_np


def _drift_layout(yaxis_title, labels, y_range):
    return dict(
        paper_bgcolor="white", plot_bgcolor="#FAFCFF",
        margin=dict(l=60, r=20, t=16, b=48), height=210,
        font=dict(family="IBM Plex Sans, system-ui", size=10, color="#475569"),
        xaxis=dict(tickvals=list(range(len(labels))), ticktext=labels, tickangle=-45, gridcolor="#F1F5F9", tickfont=dict(size=9)),
        yaxis=dict(title=dict(text=yaxis_title, font=dict(size=10)), gridcolor="#F1F5F9", range=y_range),
        showlegend=False, hovermode="x unified",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:8px 0 20px; border-bottom:1px solid #1A2535; margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="width:38px;height:38px;background:linear-gradient(135deg,#0284C7,#0EA5E9);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;">⌇</div>
        <div>
          <div style="font-size:17px;font-weight:800;color:#fff;letter-spacing:-0.5px;">Chroma<span style="color:#0EA5E9;">IQ</span></div>
          <div style="font-size:10px;color:#4B6080;font-family:monospace;">v2.1 · Labo QC-A</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Instruments
    st.markdown("**INSTRUMENTS**")
    instruments = [
        ("Waters Acquity UPLC #1", "LC-MS/MS", "ok",      "#059669"),
        ("Waters Acquity UPLC #2", "LC-MS/MS", "⚠ Dérive","#DC2626"),
        ("Agilent 7890B GC",       "GC-MS",    "ok",      "#059669"),
        ("Thermo Q Exactive",      "HRMS",     "attention","#D97706"),
    ]
    for name, itype, status, col in instruments:
        icon = "●" if status == "ok" else "●"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #1A2535;">
          <span style="color:{col};font-size:10px;">●</span>
          <div>
            <div style="color:#C8D6E8;font-size:12px;font-weight:500;">{name}</div>
            <div style="color:#4B6080;font-size:10px;font-family:monospace;">{itype} · {status}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Run selector (persisted in session state)
    st.markdown("**SÉQUENCE ACTIVE**")
    run_labels = [r["label"] for r in RUNS_CFG]
    if "run_idx" not in st.session_state:
        st.session_state.run_idx = 2  # Default: aujourd'hui (drifted)
    selected_label = st.radio("", run_labels, index=st.session_state.run_idx, label_visibility="collapsed")
    st.session_state.run_idx = run_labels.index(selected_label)
    run_cfg = RUNS_CFG[st.session_state.run_idx]

    # Run info
    st.markdown(f"""
    <div style="background:#0A1020;border-radius:8px;padding:12px;margin-top:8px;font-family:monospace;font-size:11px;line-height:1.9;">
      <div style="color:#4B6080;">Séquence</div>
      <div style="color:#22D3EE;">{run_cfg['seq']}</div>
      <div style="color:#4B6080;margin-top:4px;">Date</div>
      <div style="color:#C8D6E8;">{run_cfg['date']}</div>
      <div style="color:#4B6080;margin-top:4px;">Statut SST</div>
      <div style="color:{'#F87171' if run_cfg['status']=='danger' else '#FCD34D' if run_cfg['status']=='warning' else '#34D399'};">
        {'✗ Non conforme' if run_cfg['status']=='danger' else '⚠ Attention' if run_cfg['status']=='warning' else '✓ Conforme'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#4B6080;text-align:center;">
      M. Benali · Resp. QC<br>
      <span style="color:#1D4ED8;">●</span> En ligne
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Load active run data
run_cfg = RUNS_CFG[st.session_state.run_idx]
t_cur, y_cur, cmpds_cur, ns_cur, peaks_cur = _runs_cache[run_cfg["id"]]
drift = run_cfg["drift"]
total_area = sum(p["area"] for p in peaks_cur)

# Page header
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #E2E8F0;">
  <div>
    <h1 style="margin:0 0 4px 0;">ChromaIQ — Intelligence Analytique</h1>
    <div style="font-size:12px;color:#94A3B8;font-family:monospace;">
      Waters Acquity UPLC #2 · LC-MS/MS · Méthode MET-QC-042 v3.1 · {run_cfg['date']}
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    {badge_html("1 alerte active", "danger")}
    <span style="font-size:12px;color:#94A3B8;font-family:monospace;">
      {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chroma, tab_drift, tab_report = st.tabs([
    "🏠  Tableau de bord",
    "📈  Chromatogramme",
    "📉  Suivi dérive",
    "📄  Rapport",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:

    # Alert banner
    st.markdown("""
    <div class="alert-danger">
      <div class="alert-title" style="color:#991B1B;">⚠️  Dérive instrumentale détectée — Waters Acquity UPLC #2</div>
      <div class="alert-body" style="color:#7F1D1D;">
        Séquence <strong>SEQ-0847</strong> · Décalage TR étalon interne : <strong>+0.38 min</strong> (seuil ±0.15 min)
        · Bruit ×1.62 · Nb. plateaux PA : 8 420 (spec &gt;10 000)
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Runs aujourd'hui", "12", "+3 vs. hier")
    with k2: st.metric("Séquences PASS", "91.7 %", "-8.3 %", delta_color="inverse")
    with k3: st.metric("Instruments actifs", "4 / 4", "1 en dérive", delta_color="inverse")
    with k4: st.metric("Alertes ouvertes", "1", "Depuis 14h15", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 0.9])

    # Instrument status table
    with col_left:
        section_label("État du parc instrumental")
        inst_df = pd.DataFrame([
            {"Instrument": "Waters Acquity UPLC #1", "Type": "LC-MS/MS", "Inj. total": "847",  "Dernier run": "14:32", "Statut": "✓ Nominal"},
            {"Instrument": "Waters Acquity UPLC #2", "Type": "LC-MS/MS", "Inj. total": "1 203","Dernier run": "11:15", "Statut": "✗ Dérive"},
            {"Instrument": "Agilent 7890B GC",       "Type": "GC-MS",   "Inj. total": "456",  "Dernier run": "09:45", "Statut": "✓ Nominal"},
            {"Instrument": "Thermo Q Exactive",      "Type": "HRMS",    "Inj. total": "289",  "Dernier run": "16:08", "Statut": "⚠ Attention"},
        ])

        def color_status(val):
            if "✗" in val:   return "color:#DC2626;font-weight:700"
            if "⚠" in val:   return "color:#D97706;font-weight:700"
            return "color:#059669;font-weight:600"

        st.dataframe(
            inst_df.style.map(color_status, subset=["Statut"]),
            use_container_width=True, hide_index=True, height=185,
        )

    # Recent runs
    with col_right:
        section_label("Runs récents — UPLC #2")
        hist_display = HIST.drop(columns=["_drift"]).tail(8).iloc[::-1].copy()

        def style_tr_is(val):
            try:
                return "color:#DC2626;font-weight:700" if abs(float(val) - 2.15) > 0.15 else "color:#059669"
            except: return ""
        def style_tr_pa(val):
            try:
                return "color:#DC2626;font-weight:700" if abs(float(val) - 4.82) > 0.20 else "color:#059669"
            except: return ""
        def style_np(val):
            try:
                return "color:#DC2626;font-weight:700" if int(val) < 10000 else "color:#059669"
            except: return ""

        st.dataframe(
            hist_display.style
                .map(style_tr_is,  subset=["TR IS (min)"])
                .map(style_tr_pa,  subset=["TR PA (min)"])
                .map(style_np,     subset=["Nb. plateaux"]),
            use_container_width=True, hide_index=True, height=290,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CHROMATOGRAM
# ─────────────────────────────────────────────────────────────────────────────
with tab_chroma:

    # Alert banners
    rt_drift_IS = round(peaks_cur[0]["apex_t"] - peaks_ref[0]["apex_t"], 3)
    if drift > 0.5:
        st.markdown(f"""
        <div class="alert-danger">
          <div class="alert-title" style="color:#991B1B;">🔴  Dérive critique — Séquence {run_cfg['seq']}</div>
          <div class="alert-body" style="color:#7F1D1D;">
            Décalage TR IS : <strong>+{rt_drift_IS} min</strong> (seuil ±0.15 min) ·
            TR PA : +{round(peaks_cur[1]['apex_t']-peaks_ref[1]['apex_t'],3)} min ·
            Bruit ×{round(ns_cur/110,2)} · Nb. plateaux {peaks_cur[1]['Np']:,} (spec &gt;10 000)
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif drift > 0.2:
        st.markdown(f"""
        <div class="alert-warning">
          <div class="alert-title" style="color:#78350F;">⚠️  Début de dérive — Séquence {run_cfg['seq']}</div>
          <div class="alert-body" style="color:#92400E;">
            Décalage TR IS : <strong>+{rt_drift_IS} min</strong> — tendance à surveiller
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Legend pills
    legend_html = "".join([
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px;font-size:12px;color:#475569;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{p["color"]};display:inline-block;"></span>'
        f'{p["code"]} — {p["name"]}</span>'
        for p in peaks_cur
    ])
    if drift > 0.05:
        legend_html += '<span style="font-size:11px;color:#94A3B8;">— · — Référence J-30</span>'
    st.markdown(f'<div style="margin-bottom:10px;">{legend_html}</div>', unsafe_allow_html=True)

    # Chromatogram
    fig_chroma = make_chroma_fig(t_cur, y_cur, peaks_cur, drift, ns_cur, show_ref=drift > 0.05)
    st.plotly_chart(fig_chroma, use_container_width=True)

    # Bottom: table + SST + analysis
    col_table, col_side = st.columns([1.7, 1.0])

    with col_table:
        section_label("Résultats — Identification & Quantification")

        rows = []
        for p in peaks_cur:
            pr    = next(r for r in peaks_ref if r["id"] == p["id"])
            dRT   = round(p["apex_t"] - pr["apex_t"], 3)
            pct   = round(p["area"] / total_area * 100, 2)
            rt_ok = abs(dRT) <= (0.20 if p["id"] == 1 else 0.15)
            pct_ok = "spec_hi_pct" not in p or pct <= p["spec_hi_pct"]
            rows.append({
                "Composé":     f'{p["code"]} — {p["name"]}',
                "TR (min)":    p["apex_t"],
                "ΔTR réf.":   f'+{dRT:.3f}' if dRT >= 0 else f'{dRT:.3f}',
                "Aire":        p["area"],
                "% Aire":      f'{pct:.2f}',
                "S/B":         p["sn"],
                "N plateaux":  p["Np"] if p["id"] <= 1 else "—",
                "TF":          p["tf"],
                "_rt_ok":      rt_ok,
                "_pct_ok":     pct_ok,
                "_status":     "ok" if rt_ok and pct_ok else "fail",
            })

        pk_df = pd.DataFrame(rows)
        display_df = pk_df.drop(columns=["_rt_ok", "_pct_ok", "_status"])

        def style_dRT(val):
            try:
                v = float(val.replace("+", ""))
                return "color:#DC2626;font-weight:700" if abs(v) > 0.20 else ("color:#D97706;font-weight:600" if abs(v) > 0.15 else "color:#059669")
            except: return ""
        def style_pct(val):
            try: return "color:#DC2626;font-weight:700" if float(val) > 0.5 else ("color:#D97706" if float(val) > 0.35 else "")
            except: return ""

        st.dataframe(
            display_df.style
                .map(style_dRT, subset=["ΔTR réf."])
                .map(style_pct, subset=["% Aire"]),
            use_container_width=True, hide_index=True,
        )

    with col_side:
        # SST metrics
        section_label("Adéquation du système (SST)")
        p0, p1 = peaks_cur[0], peaks_cur[1]
        res = round(2 * (p1["apex_t"] - p0["apex_t"]) / max(p0["w50"] + p1["w50"], 0.001), 2)
        sst_rows = [
            ("Résolution IS/PA",   res,                   "> 2.0",    res >= 2.0),
            ("Efficacité N (PA)",  f"{p1['Np']:,}",       "> 10 000", p1["Np"] >= 10000),
            ("Facteur traîné IS",  p0["tf"],              "0.8–1.5",  0.8 <= p0["tf"] <= 1.5),
            ("S/B (PA)",           p1["sn"],              "> 50",     p1["sn"] >= 50),
            ("Bruit relatif",      f'{round(ns_cur/110,2)}×',"< 1.30×", ns_cur/110 < 1.30),
        ]
        for name, val, spec, ok in sst_rows:
            c1, c2, c3 = st.columns([2, 1.2, 0.4])
            c1.markdown(f'<span style="font-size:12px;color:#475569;">{name}</span>', unsafe_allow_html=True)
            col = "#059669" if ok else "#DC2626"
            c2.markdown(f'<span style="font-size:12px;font-family:monospace;color:{col};font-weight:{"700" if not ok else "400"};">{val}</span><span style="font-size:10px;color:#94A3B8;"> ({spec})</span>', unsafe_allow_html=True)
            c3.markdown(f'<span style="font-size:16px;color:{col};">{"✓" if ok else "✗"}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Auto-analysis
        section_label("Analyse automatique — ChromaIQ Engine v2.1")

        if drift == 0:
            st.markdown("""<div class="alert-success">
              <div class="alert-title" style="color:#065F46;">✓ Séquence conforme</div>
              <div class="alert-body" style="color:#047857;">Tous les critères SST dans les limites. TR stables (Δ &lt;0.01 min vs. référence). Aucune action corrective requise.</div>
            </div>""", unsafe_allow_html=True)
        else:
            sev_cls = "alert-danger" if drift > 0.5 else "alert-warning"
            sev_title_col = "#991B1B" if drift > 0.5 else "#78350F"
            sev_body_col  = "#7F1D1D" if drift > 0.5 else "#92400E"
            rt_pa = round(peaks_cur[1]["apex_t"] - peaks_ref[1]["apex_t"], 3)
            imp_a_pct = round(peaks_cur[2]["area"] / total_area * 100, 2)

            analysis_html = f"""
            <div class="{sev_cls}">
              <div class="alert-title" style="color:{sev_title_col};">
                {'✗ Anomalie critique — intervention requise' if drift > 0.5 else '⚠ Début de dérive — surveillance renforcée'}
              </div>
              <div class="alert-body" style="color:{sev_body_col};line-height:2.0;">
                <strong>① TR hors spécification</strong><br>
                IS : <strong>+{rt_drift_IS} min</strong> (seuil ±0.15) · PA : <strong>+{rt_pa} min</strong> (seuil ±0.20)<br>
                → {'Pattern monotone sur 7 jours — encrassement progressif de la colonne' if drift>0.4 else 'Tendance débutante à surveiller'}<br><br>
                {'<strong>② Bruit de fond élevé</strong><br>' + f'Niveau actuel : <strong>×{round(ns_cur/110,2)}</strong> (seuil ×1.30)<br>→ Contamination source ESI probable<br><br>' if ns_cur/110 > 1.20 else ''}
                {f'<strong>③ Impureté A proche du seuil</strong><br>ImpA = <strong>{imp_a_pct}%</strong> / limite 0.50% — marge {round(0.50-imp_a_pct,2)}%<br><br>' if imp_a_pct > 0.35 else ''}
              </div>
            </div>
            """
            st.markdown(analysis_html, unsafe_allow_html=True)

            # Recommendation box
            rec_cls = "rec-box-danger" if drift > 0.5 else "rec-box-warning"
            if drift > 0.4:
                rec_txt = "1. Rinçage colonne immédiat (90% ACN, 10 CV)<br>2. Si persistance → remplacement colonne HSS T3<br>3. Nettoyage source ESI (capillaire + cône)<br>4. Invalider SEQ-0844 à SEQ-0847"
            else:
                rec_txt = "1. Rinçage préventif en fin de journée<br>2. Renforcer le contrôle SST (toutes les 20 inj.)<br>3. Planifier le remplacement colonne"
            st.markdown(f"""
            <div class="rec-box {rec_cls}">
              <strong>Recommandation — Priorité {'haute' if drift>0.4 else 'modérée'}</strong><br>
              {rec_txt}
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DRIFT MONITORING
# ─────────────────────────────────────────────────────────────────────────────
with tab_drift:

    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Drift TR IS (J-30→J0)", "+0.38 min", "⬆ Tendance monotone 7 j", delta_color="inverse")
    with k2: st.metric("Niveau de bruit rel.", "×1.62", "⬆ +62% vs. référence",      delta_color="inverse")
    with k3: st.metric("Nb. plateaux PA",      "8 420",  "⬇ Spec > 10 000",           delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Suivi longitudinal — 20 dernières séquences · Waters Acquity UPLC #2")

    fig_rt, fig_noise, fig_np = make_drift_fig()

    st.plotly_chart(fig_rt, use_container_width=True)

    col_n, col_np = st.columns(2)
    with col_n:  st.plotly_chart(fig_noise, use_container_width=True)
    with col_np: st.plotly_chart(fig_np,    use_container_width=True)

    st.markdown("""
    <div class="alert-danger" style="margin-top:8px;">
      <div class="alert-title" style="color:#991B1B;">🔴  Interprétation automatique — Analyse de tendance (7 derniers runs)</div>
      <div class="alert-body" style="color:#7F1D1D;line-height:2.0;">
        <strong>Pattern observé :</strong> Augmentation monotone des TR depuis J-7 (corrélation r² = 0.97, tendance linéaire).<br>
        <strong>Cause probable :</strong> Encrassement progressif de la tête de colonne — accumulation de résidus matriciels non élués.<br>
        <strong>Corrélation :</strong> Augmentation simultanée du bruit (×1.62) et perte de 25% d'efficacité — cohérent avec un blocage mécanique.<br>
        <strong>Risque si non traité :</strong> Résolution IS/PA &lt; 2.0 estimée dans <strong>3 à 5 runs</strong> — invalidation de séquence probable.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:

    sev     = "danger" if drift > 0.5 else ("warning" if drift > 0.2 else "ok")
    sev_col = {"danger":"#DC2626","warning":"#D97706","ok":"#059669"}[sev]

    # Header row
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.markdown(f"""
        <h1 style="margin:0 0 4px 0;">Rapport d'analyse — {run_cfg['seq']}</h1>
        <div style="font-size:12px;color:#94A3B8;font-family:monospace;">
          Généré automatiquement par ChromaIQ Engine v2.1 · {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
        </div>
        """, unsafe_allow_html=True)
    with hcol2:
        c1, c2 = st.columns(2)
        c1.button("⬇ Exporter PDF")
        c2.button("📨 Envoyer LIMS", type="primary")

    st.markdown("<br>", unsafe_allow_html=True)

    # Info grid
    section_label("Informations de séquence")
    g1, g2, g3 = st.columns(3)
    info = {
        "Instrument":     "Waters Acquity UPLC #2",
        "Séquence":       run_cfg["seq"],
        "Méthode":        "MET-QC-042 v3.1",
        "Date / Heure":   f"{run_cfg['date']} · 11:15:32",
        "Opérateur":      "M. Benali — Resp. QC",
        "Colonne":        "HSS T3 1.8µm 50×2.1mm",
        "Débit":          "0.400 mL/min · 40°C",
        "Phase mobile":   "H₂O/ACN + 0.1% AF",
        "Vol. injection": "2.0 µL",
    }
    items = list(info.items())
    for col, chunk in zip([g1, g2, g3], [items[:3], items[3:6], items[6:]]):
        with col:
            for k, v in chunk:
                st.markdown(f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Résultats quantitatifs")

    rep_rows = []
    for p in peaks_cur:
        pct  = round(p["area"] / total_area * 100, 2)
        spec = ("Standard" if p["id"] == 0 else
                "95.0 – 105.0 %" if p["id"] == 1 else
                f"≤ {p['spec_hi_pct']} %")
        ok   = (True if p["id"] == 0 else
                (95 <= pct <= 105) if p["id"] == 1 else
                pct <= p["spec_hi_pct"])
        rep_rows.append({
            "Composé":       f'{p["name"]} ({p["code"]})',
            "TR (min)":      p["apex_t"],
            "Aire":          f'{p["area"]:,}',
            "% Teneur":      f'{pct:.2f} %',
            "Spécification": spec,
            "Résultat":      "✓ Conforme" if ok else "✗ Non conforme",
            "_ok":           ok,
        })

    rep_df = pd.DataFrame(rep_rows)
    display_rep = rep_df.drop(columns=["_ok"])

    def style_result(val):
        if "Non" in str(val): return "color:#DC2626;font-weight:700"
        return "color:#059669;font-weight:600"

    st.dataframe(
        display_rep.style.map(style_result, subset=["Résultat"]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Conclusion qualité")

    conc_map = {
        "ok":      ("✓ Séquence conforme — Résultats validés",
                    "Tous les critères SST sont dans les limites d'acceptation. Résultats conformes à la méthode MET-QC-042 v3.1."),
        "warning": ("⚠ Conforme avec réserve — Maintenance préventive recommandée",
                    "Les résultats de quantification sont conformes. Une tendance de dérive SST est observée. Suivi renforcé recommandé. Planifier une maintenance préventive."),
        "danger":  ("✗ Non conforme — Invalidation de séquence recommandée",
                    "Séquence non conforme sur les critères SST (TR hors spécification). Les résultats ne peuvent pas être validés. Intervention maintenance requise avant reprise. Invalider les séquences SEQ-0844 à SEQ-0847."),
    }
    conc_title, conc_body = conc_map[sev]
    cls = f"alert-{sev}" if sev != "ok" else "alert-success"
    st.markdown(f"""
    <div class="{cls}">
      <div class="alert-title" style="color:{sev_col};font-size:15px;">{conc_title}</div>
      <div class="alert-body" style="color:{sev_col};margin-top:4px;">{conc_body}</div>
    </div>
    """, unsafe_allow_html=True)

    # Signature line
    st.markdown(f"""
    <div style="margin-top:24px;padding:14px 18px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;font-size:11px;color:#94A3B8;font-family:monospace;line-height:2.0;">
      Rapport généré automatiquement par ChromaIQ Engine v2.1 · Données brutes archivées sous {run_cfg['seq']}-RAW<br>
      Méthode MET-QC-042 v3.1 (validée le 20/01/2026) · Système en cours de qualification GAMP5 Cat. 4<br>
      Ce rapport ne remplace pas la revue qualité par un analyste qualifié · Conforme aux BPL/GLP
    </div>
    """, unsafe_allow_html=True)
