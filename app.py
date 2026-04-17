#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromatoIA — Application Vitrine v1.0
SaaS IA d'analyse chromatographique (couche intelligence sur CDS existants)
Mode advisory only — aucune décision automatique, validation humaine systématique.

Lancement : streamlit run app.py
Dépendances : streamlit, plotly, numpy, pandas  (scipy non requis)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io

# Remplacement de scipy.stats.linregress par une implémentation numpy pure
# Évite la dépendance scipy sur Streamlit Cloud
def linregress(x, y):
    """
    Régression linéaire simple (y = slope*x + intercept).
    Retourne (slope, intercept, r, p, se) — interface identique à scipy.stats.linregress.
    Uniquement slope, r et r² sont utilisés dans detect_anomalies().
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    ss_xx = ((x - xm) ** 2).sum()
    ss_xy = ((x - xm) * (y - ym)).sum()
    ss_yy = ((y - ym) ** 2).sum()
    slope     = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = ym - slope * xm
    r         = ss_xy / np.sqrt(ss_xx * ss_yy) if (ss_xx * ss_yy) > 0 else 0.0
    return slope, intercept, r, 0.0, 0.0

# Seed fixe — toutes les données simulées sont reproductibles
np.random.seed(42)

# ============================================================
# CALENDRIER DE RUNS SIMULÉ
# ============================================================

def generate_run_calendar():
    """
    Génère un calendrier de 30 runs sur ~3 semaines (jours ouvrés).
    Retourne un OrderedDict : date_label -> [(run_num, time_label), ...]
    Permet la navigation par date plutôt que par numéro de run brut.
    """
    from collections import OrderedDict
    base = datetime(2026, 3, 17, 7, 30)
    cal = OrderedDict()
    run_num = 1
    day_offset = 0
    # Planning réaliste : 1 ou 2 séquences par jour ouvré
    daily_runs = [2,1,2,2,1, 2,1,2,1,2, 1,2,2,1,2, 2,1,2,1,1]
    day_idx = 0
    while run_num <= 30 and day_idx < len(daily_runs):
        d = base + timedelta(days=day_offset)
        if d.weekday() >= 5:          # week-end → passer
            day_offset += 1
            continue
        n_runs = daily_runs[day_idx]
        date_label = d.strftime("%A %d/%m/%Y").capitalize()
        times = ["07:30", "13:30"] if n_runs == 2 else ["08:00"]
        runs_today = []
        for t in times:
            if run_num > 30:
                break
            runs_today.append((run_num, t))
            run_num += 1
        cal[date_label] = runs_today
        day_idx += 1
        day_offset += 1
    return cal



# ============================================================
# CONFIG PAGE — PREMIER APPEL STREAMLIT OBLIGATOIRE
# ============================================================
st.set_page_config(
    page_title="ChromatoIA — Intelligence Analytique",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS GLOBAL (design system Notion + Linear + Stripe)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.main { background-color: #F8F9FC; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1200px; }
[data-testid="stSidebar"] { background-color: #F4F6FB; border-right: 1px solid #E8ECF4; }
h1 { font-size: 22px !important; font-weight: 700 !important; color: #1E293B !important; margin-bottom: 4px !important; }
h2 { font-size: 17px !important; font-weight: 600 !important; color: #1E293B !important; }
h3 { font-size: 11px !important; font-weight: 600 !important; color: #64748B !important;
     text-transform: uppercase !important; letter-spacing: 0.07em !important; }
.metric-card {
    background: #FFFFFF; border: 1px solid #E8ECF4; border-radius: 12px;
    padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); height: 100%;
}
.metric-label { font-size: 11px; font-weight: 600; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
.metric-value { font-size: 30px; font-weight: 700; color: #1E293B; line-height: 1.1; }
.metric-delta-warn { font-size: 11px; color: #F59E0B; font-weight: 600; margin-top: 6px; }
.metric-delta-ok   { font-size: 11px; color: #10B981; font-weight: 600; margin-top: 6px; }
.metric-delta-crit { font-size: 11px; color: #EF4444; font-weight: 600; margin-top: 6px; }
.badge-ok       { background:#D1FAE5; color:#065F46; padding:4px 11px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.badge-warning  { background:#FEF3C7; color:#92400E; padding:4px 11px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.badge-critical { background:#FEE2E2; color:#991B1B; padding:4px 11px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.alert-ok       { background:#F0FDF4; border-left:4px solid #10B981; border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0; }
.alert-warning  { background:#FFFBEB; border-left:4px solid #F59E0B; border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0; }
.alert-critical { background:#FFF5F5; border-left:4px solid #EF4444; border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0; }
.alert-title    { font-weight: 700; font-size: 13px; margin-bottom: 4px; }
.alert-body     { font-size: 12px; color: #374151; line-height: 1.5; }
.section-card   { background:#FFFFFF; border:1px solid #E8ECF4; border-radius:12px;
    padding:22px 24px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.page-header    { margin-bottom: 24px; }
.page-subtitle  { font-size: 13px; color: #64748B; margin-top: 4px; }
.tech-badge     { background:#EFF6FF; color:#1D4ED8; padding:5px 13px; border-radius:8px;
    font-size:12px; font-weight:600; display:inline-block; margin-bottom:20px; }
.advisory-banner { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
    padding:12px 18px; font-size:12px; color:#1E40AF; margin-bottom:20px; }
.stButton > button { background:#2563EB !important; color:white !important; border:none !important;
    border-radius:8px !important; padding:9px 22px !important; font-weight:600 !important; font-size:13px !important; }
.stButton > button:hover { background:#1D4ED8 !important; }
.stSelectbox label, .stSlider label, .stRadio label {
    font-size:11px !important; font-weight:600 !important; color:#64748B !important;
    text-transform:uppercase !important; letter-spacing:0.05em !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PARAMÈTRES PAR TECHNIQUE — source unique de vérité
# ============================================================
TECHNIQUES = {
    "LC / HPLC  (UV / DAD)": {
        "family": "LC", "type": "classical", "unit": "mAU", "unit_label": "Absorbance (mAU)",
        "time_range": (0.0, 30.0), "n_points": 3000,
        "rts":     [3.2,   6.7,    9.4,   13.1,  17.8,  22.3],
        "heights": [85000, 142000, 67000, 98000, 54000, 31000],
        "widths":  [0.18,  0.22,   0.19,  0.25,  0.21,  0.28],
        "noise_base": 300, "baseline_slope": 90,
        "peak_names": ["Pic 1", "Pic 2", "Pic 3", "Pic 4", "Pic 5", "Pic 6"],
        "drift_start": 15, "drift_rt_run": 0.004, "drift_w_run": 0.003,
        "alert_msg": "Dérive colonne C18 — perte d'efficacité progressive détectée sur 5 runs consécutifs",
        "alert_level": "warning",
        "maint_msg": "• Rinçage colonne recommandé (MeOH/H₂O gradient inverse, 10 CV)\n• Contrôle pression à ΔP nominal\n• Remplacement colonne prévu dans ~8 runs",
        "instrument": "Agilent 1290 Infinity II", "column": "Zorbax Eclipse Plus C18, 2.1×100mm, 1.8µm",
        "qc_res_min": 1.5, "qc_tail_max": 1.5, "qc_N_min": 5000,
    },
    "LC-MS  (ESI, quadrupôle)": {
        "family": "LC", "type": "ms", "unit": "counts", "unit_label": "Intensité (counts)",
        "time_range": (0.0, 20.0), "n_points": 2000,
        "rts":     [2.8,    5.4,    9.1,    14.6],
        "heights": [320000, 890000, 540000, 210000],
        "widths":  [0.12,   0.15,   0.13,   0.17],
        "noise_base": 800, "noise_run": 220,
        "peak_names": ["Composé A", "Composé B", "Composé C", "Composé D"],
        "drift_start": 12, "drift_intensity_run": -0.020, "drift_rt_run": 0.002,
        "alert_msg": "Suppression ionique progressive détectée — vérifier propreté source ESI",
        "alert_level": "warning",
        "maint_msg": "• Nettoyage source ESI (dépose + nettoyage optique)\n• Dilution supplémentaire des matrices complexes\n• Contrôle courant capillaire (attendu : 0.5–5 µA)",
        "instrument": "Waters Xevo TQS micro", "column": "Acquity BEH C18, 2.1×50mm, 1.7µm",
        "qc_rt_window": 0.10, "qc_rsd_max": 15.0,
    },
    "LC-MS/MS  (MRM, triple quadrupôle)": {
        "family": "LC", "type": "mrm", "unit": "counts", "unit_label": "Intensité MRM (counts)",
        "time_range": (0.0, 15.0), "n_points": 1500,
        "rts":     [3.1,    5.8,     8.4,          11.2],
        "heights": [180000, 310000,  240000,        150000],
        "widths":  [0.12,   0.14,    0.13,          0.15],
        "noise_base": 500,
        "compounds":    ["Amoxicilline", "Ciprofloxacine", "Tétracycline", "Érythromycine"],
        "trans_q":  ["365>208", "332>231", "445>427", "734>576"],
        "trans_qq": ["365>160", "332>288", "445>410", "734>158"],
        "qq_nom":   [0.45, 0.62, 0.38, 0.71],
        "qq_window": 0.20,
        "drift_start": 12, "drift_ratio_run": 0.012, "drift_rt_run": 0.002,
        "alert_msg": "Ratio qualifier/quantifier hors fenêtre ±20% sur Ciprofloxacine — vérifier suppression ionique",
        "alert_level": "warning",
        "maint_msg": "• Contrôle suppression ionique sur matrice réelle\n• Vérification étalon interne deutéré (IS recovery)\n• Rinçage source avant prochaine séquence",
        "instrument": "AB Sciex QTRAP 6500+", "column": "Phenomenex Kinetex C18, 2.1×100mm, 1.7µm",
        "qc_rt_window": 0.10, "qc_ratio_window": 0.20,
    },
    "LC-HRMS  (Orbitrap / Q-TOF)": {
        "family": "LC", "type": "hrms", "unit": "counts", "unit_label": "Intensité HR (counts)",
        "time_range": (0.0, 18.0), "n_points": 1800,
        "rts":     [2.4,    5.1,    8.7,    12.3,   15.8],
        "heights": [450000, 820000, 340000, 610000, 280000],
        "widths":  [0.08,   0.10,   0.09,   0.11,   0.10],
        "noise_base": 1200,
        "compounds":   ["C12H16N2O3", "C24H32N2O6", "C9H12N2O3", "C16H24N2O5", "C32H44N2O8"],
        "comp_names":  ["Composé 1", "Composé 2", "Composé 3", "Composé 4", "Composé 5"],
        "masses":      [234.1127, 456.2341, 178.0865, 312.1592, 589.3047],
        "ppm_base":    [0.8,  1.2,  0.6,  1.1,  0.9],
        "drift_start": 18, "drift_ppm_run": 0.30, "ppm_thresh": 5.0,
        "alert_msg": "Erreur de masse > 5 ppm sur Composé 3 — recalibration instrument recommandée",
        "alert_level": "critical",
        "maint_msg": "• Recalibration interne obligatoire avant prochaine séquence\n• Vérifier solution lock mass (leucine encephalin 100 fmol/µL)\n• Contrôle résolution orbitrap (attendu R > 60 000 FWHM @ 200 Da)",
        "instrument": "Thermo Orbitrap Exploris 480", "column": "Acquity BEH C18, 2.1×100mm, 1.7µm",
        "qc_ppm_max": 5.0, "qc_rt_window": 0.05,
    },
    "GC / FID": {
        "family": "GC", "type": "classical", "unit": "pA", "unit_label": "Signal FID (pA)",
        "time_range": (2.0, 40.0), "n_points": 3800,
        "rts":     [4.2,   8.7,   12.3,   16.8,  21.4,  26.9,  31.2,  36.7],
        "heights": [45000, 89000, 123000, 67000, 94000, 52000, 38000, 71000],
        "widths":  [0.25,  0.28,  0.24,   0.30,  0.27,  0.32,  0.29,  0.35],
        "noise_base": 150,
        "peak_names": [f"Pic {i+1}" for i in range(8)],
        "parasite_rts":     [1.85, 2.90],
        "parasite_heights": [8000, 5200],
        "drift_start": 20, "drift_rt_run": 0.006, "drift_w_run": 0.004,
        "alert_msg": "Contamination injecteur suspectée — pics parasites détectés en début de run",
        "alert_level": "critical",
        "maint_msg": "• Remplacement liner (splitless) immédiat recommandé\n• Remplacement septum (température programme > 280°C)\n• Nettoyage injecteur avant prochaine injection",
        "instrument": "Shimadzu GC-2030", "column": "Rtx-5, 30m × 0.25mm × 0.25µm",
        "qc_res_min": 1.5, "qc_tail_max": 1.5, "qc_N_min": 3000,
    },
    "GC-MS  (TIC, quadrupôle)": {
        "family": "GC", "type": "ms", "unit": "counts", "unit_label": "TIC (counts)",
        "time_range": (5.0, 45.0), "n_points": 4000,
        "rts":     [7.3,    13.6,    19.2,   25.8,   33.4,   41.1],
        "heights": [560000, 1240000, 890000, 430000, 670000, 320000],
        "widths":  [0.30,   0.35,    0.32,   0.38,   0.34,   0.40],
        "noise_base": 2000,
        "peak_names": [f"Composé {i+1}" for i in range(6)],
        "parasite_rts":     [5.85, 6.50],
        "parasite_heights": [45000, 28000],
        "drift_start": 18, "drift_rt_run": 0.008,
        "alert_msg": "Pics parasites détectés en début de run — suspicion contamination septum ou liner",
        "alert_level": "warning",
        "maint_msg": "• Remplacement septum préventif\n• Conditionnement colonne 1h à T_max avant séquence\n• Vérifier propreté des solvants d'extraction",
        "instrument": "Agilent 7890B / 5977B MSD", "column": "HP-5MS, 30m × 0.25mm × 0.25µm",
        "qc_snr_min": 10.0, "qc_rt_window": 0.10,
    },
    "GC-MS/MS  (MRM, triple quadrupôle)": {
        "family": "GC", "type": "mrm", "unit": "counts", "unit_label": "Intensité MRM (counts)",
        "time_range": (5.0, 42.0), "n_points": 3700,
        "rts":     [8.4,   14.7,   21.3,  28.6,   35.9],
        "heights": [95000, 140000, 78000, 110000, 62000],
        "widths":  [0.28,  0.32,   0.30,  0.35,   0.33],
        "noise_base": 800,
        "compounds":   ["Aldrine", "Dieldrine", "Endrine", "HCB", "Lindane"],
        "trans_q":  ["263>193", "277>241", "263>193", "284>214", "181>145"],
        "trans_qq": ["263>228", "277>207", "263>147", "284>249", "181>109"],
        "qq_nom":   [0.52, 0.41, 0.68, 0.35, 0.59],
        "qq_window": 0.20,
        "drift_start": 10, "drift_ratio_run": 0.015, "drift_rt_run": 0.005,
        "alert_msg": "Instabilité ratio MRM détectée sur HCB — vérifier pression gaz vecteur",
        "alert_level": "warning",
        "maint_msg": "• Vérifier régulateur pression hélium (débit : 1.2 mL/min)\n• Contrôler étanchéité connexions colonne (fuite = perte sensibilité)\n• Reconditionner la colonne si RT shift > 0.1 min",
        "instrument": "Waters Xevo TQ-GC", "column": "Rxi-5Sil MS, 15m × 0.25mm × 0.25µm",
        "qc_rt_window": 0.10, "qc_ratio_window": 0.20,
    },
    "GC-HRMS  (Q-TOF / Orbitrap GC)": {
        "family": "GC", "type": "hrms", "unit": "counts", "unit_label": "Intensité HR (counts)",
        "time_range": (5.0, 40.0), "n_points": 3500,
        "rts":     [9.2,    17.8,   26.4,   34.1],
        "heights": [280000, 520000, 190000, 340000],
        "widths":  [0.20,   0.25,   0.22,   0.28],
        "noise_base": 1800,
        "compounds":  ["PCDD/F 1", "PCDD/F 2", "PCB 126", "PCB 169"],
        "comp_names": ["Analyte HR1", "Analyte HR2", "Analyte HR3", "Analyte HR4"],
        "masses":     [128.0626, 256.1307, 384.1988, 512.2669],
        "formulas":   ["C₈H₈O₂", "C₁₆H₁₆N₂O₂", "C₂₀H₂₈N₄O₄", "C₂₈H₄₀N₄O₆"],
        "ppm_base":   [1.2, 1.5, 0.9, 1.8],
        "drift_start": 15, "drift_ppm_run": 0.40, "ppm_thresh": 5.0,
        "alert_msg": "Dérive masse exacte progressive détectée — recalibration recommandée sous 10 runs",
        "alert_level": "warning",
        "maint_msg": "• Recalibration PFTBA/PFK avant prochaine séquence\n• Vérifier stabilité source EI (70 eV, tension extraction)\n• Contrôle résolution spectrale (attendu R > 25 000)",
        "instrument": "Leco Pegasus BT 4D", "column": "Rxi-5Sil MS, 30m × 0.25mm × 0.25µm",
        "qc_ppm_max": 5.0, "qc_rt_window": 0.05,
    },
}

TECH_LIST = list(TECHNIQUES.keys())

# ============================================================
# COULEURS ET STYLE PLOTLY
# ============================================================
PC = {
    "primary":   "#2563EB",
    "secondary": "#6366F1",
    "accent":    "#0EA5E9",
    "ok":        "#10B981",
    "warning":   "#F59E0B",
    "critical":  "#EF4444",
    "grid":      "#F1F5F9",
    "paper":     "#FFFFFF",
    "plot":      "#FAFBFF",
    "text":      "#1E293B",
    "muted":     "#94A3B8",
}

def apply_fig_style(fig, height=380, legend=True):
    """Applique le design system uniformement a tous les graphiques Plotly.
    Note : dans les dict() imbriques, Plotly n accepte PAS le magic-underscore
    (font_size, tickfont_size). Il faut des sous-dicts explicites.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor=PC["paper"],
        plot_bgcolor=PC["plot"],
        font=dict(family="Inter, sans-serif", size=11, color=PC["text"]),
        margin=dict(l=50, r=20, t=40, b=50),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PC["text"],
            font=dict(color="white", size=11),
            bordercolor=PC["text"],
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
        ) if legend else dict(visible=False),
    )
    # update_xaxes/yaxes fonctionne sur tous les axes du subplot
    fig.update_xaxes(
        gridcolor=PC["grid"], showgrid=True, linecolor="#E2E8F0",
        tickfont=dict(size=10), title_font=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor=PC["grid"], showgrid=True, linecolor="#E2E8F0",
        tickfont=dict(size=10), title_font=dict(size=11),
    )
    return fig


# ============================================================
# FONCTIONS DE GÉNÉRATION DE DONNÉES
# ============================================================

def _asym_gaussian(t, rt, height, width, asym=1.08):
    """
    Pic gaussien asymétrique — modèle réaliste de pic chromatographique.
    asym > 1 : tailing côté droit (physiquement justifié : interactions résiduelles
    avec la phase stationnaire ou colonne en fin de vie).
    width : FWHM (largeur à mi-hauteur) en unités de temps.
    sigma = FWHM / 2.355 (relation gaussienne exacte)
    """
    sigma = width / 2.355
    dt = t - rt
    sig_eff = np.where(dt >= 0, sigma * asym, sigma)
    return height * np.exp(-0.5 * (dt / sig_eff) ** 2)


def _add_realistic_noise(n, noise_level, run_seed):
    """
    Bruit de fond analytique réaliste :
    mélange bruit blanc (shot noise détecteur) + composante lente (derive thermique).
    """
    rng = np.random.default_rng(run_seed)
    white = rng.normal(0, noise_level * 0.80, n)
    # Dérive lente : random walk filtré (simule fluctuation thermique ou pression)
    drift = np.cumsum(rng.normal(0, noise_level * 0.012, n))
    drift -= np.linspace(drift[0], drift[-1], n)  # tendance linéaire retirée
    return white + drift


def generate_chromatogram(tech_name, run_number):
    """
    Génère (time, signal, meta) pour la technique et le numéro de run donnés.
    La dérive est appliquée progressivement après drift_start.
    meta contient les paramètres effectifs (RT, hauteurs, largeurs).
    """
    p = TECHNIQUES[tech_name]
    seed = 42 + run_number * 31 + hash(tech_name) % 1000
    t = np.linspace(p["time_range"][0], p["time_range"][1], p["n_points"])

    # ---- Calcul dérive ----
    d_run = max(0, run_number - p["drift_start"])  # runs depuis début dérive

    rts     = np.array(p["rts"],     dtype=float)
    heights = np.array(p["heights"], dtype=float)
    widths  = np.array(p["widths"],  dtype=float)

    if "drift_rt_run" in p:
        rts += d_run * p["drift_rt_run"]
    if "drift_w_run" in p:
        widths += d_run * p["drift_w_run"]
    if "drift_intensity_run" in p:
        # Suppression ionique : décroissance géométrique de l'intensité
        heights *= (1 + p["drift_intensity_run"]) ** d_run

    # ---- Construction du signal ----
    sig = np.zeros(len(t))
    asym_factor = 1.08 + d_run * 0.006  # asymétrie croissante avec dérive

    for rt, h, w in zip(rts, heights, widths):
        if p["time_range"][0] <= rt <= p["time_range"][1]:
            sig += _asym_gaussian(t, rt, h, w, asym=asym_factor)

    # ---- Pics parasites (contamination injecteur/septum) ----
    if "parasite_rts" in p and d_run > 0:
        intensity_factor = min(1.0, d_run / 6.0)
        for para_rt, para_h in zip(p["parasite_rts"], p["parasite_heights"]):
            sig += _asym_gaussian(t, para_rt, para_h * intensity_factor, 0.15, 1.6)

    # ---- Baseline inclinée (LC surtout) ----
    if p.get("baseline_slope", 0) > 0:
        rng_bl = np.random.default_rng(seed + 1)
        bl = p["baseline_slope"] * (t - t[0]) / (t[-1] - t[0])
        bl_walk = np.cumsum(rng_bl.normal(0, 4.0, len(t)))
        bl_walk -= np.linspace(bl_walk[0], bl_walk[-1], len(t))
        sig += np.maximum(0, bl + bl_walk)

    # ---- Bruit ----
    noise_lvl = p["noise_base"]
    if "noise_run" in p:
        noise_lvl += d_run * p["noise_run"]
    sig += _add_realistic_noise(len(t), noise_lvl, seed + 2)
    sig = np.maximum(0.0, sig)

    meta = {
        "rts": rts, "heights": heights, "widths": widths,
        "noise": noise_lvl, "d_run": d_run, "asym": asym_factor,
    }
    return t, sig, meta


def detect_peaks(t, sig, tech_name, meta):
    """
    Calcule les propriétés des pics à partir des paramètres effectifs (post-dérive).
    Retourne un DataFrame avec : nom, RT, hauteur, aire, FWHM, facteur de tailing, N plateaux, statut.

    Formules analytiques standards :
    - N plateaux théoriques : N = 5.54 × (RT/FWHM)²  [USP]
    - Facteur de tailing (asymétrie) : TF = (a + b) / (2a)  avec a = demi-largeur gauche
      Pour notre modèle gaussien asymétrique : TF ≈ (1 + asym) / 2
    - Résolution : Rs = 1.18 × (RT₂ − RT₁) / (FWHM₁ + FWHM₂)  [résolution chromatographique]
    """
    p = TECHNIQUES[tech_name]
    rts     = meta["rts"]
    heights = meta["heights"]
    widths  = meta["widths"]
    asym    = meta.get("asym", 1.08)
    names   = p.get("peak_names", p.get("compounds", [f"Pic {i+1}" for i in range(len(rts))]))

    rows = []
    for i, (rt, h, w) in enumerate(zip(rts, heights, widths)):
        name = names[i] if i < len(names) else f"Pic {i+1}"
        # Hauteur réelle dans le signal (avec bruit)
        idx = np.argmin(np.abs(t - rt))
        h_real = float(sig[idx]) if 0 <= idx < len(sig) else h

        # Aire = intégrale numérique sur fenêtre ±3×sigma autour du pic
        sigma = w / 2.355
        mask = (t >= rt - 4 * sigma * asym) & (t <= rt + 4 * sigma)
        dt = (t[-1] - t[0]) / (len(t) - 1)
        area = float(np.sum(sig[mask]) * dt)

        # N plateaux théoriques
        N = 5.54 * (rt / w) ** 2 if w > 0 else 0

        # Facteur de tailing (USP) basé sur le modèle asymétrique
        tf = (1 + asym) / 2.0

        # Statut
        qc_ok = True
        if p["type"] == "classical":
            if tf > p.get("qc_tail_max", 1.5): qc_ok = False
            if N  < p.get("qc_N_min",    5000): qc_ok = False

        rows.append({
            "Pic": name, "TR (min)": round(rt, 3), "Hauteur": int(h_real),
            "Aire": int(area), "FWHM (min)": round(w, 4),
            "Symétrie (TF)": round(tf, 3),
            "N plateaux": int(N),
            "Statut": "✅ OK" if qc_ok else "⚠️ Hors spec",
        })

    df = pd.DataFrame(rows)

    # ---- Résolution entre pics adjacents (LC/GC classique) ----
    if p["type"] == "classical" and len(rows) > 1:
        res_list = ["—"]
        for i in range(1, len(rts)):
            rs = 1.18 * (rts[i] - rts[i-1]) / (widths[i] + widths[i-1])
            res_list.append(round(rs, 2))
        df["Rs"] = res_list

    return df


def _mrm_qq_at_run(tech_name, run_number):
    """
    Calcule les ratios Q/q effectifs pour chaque composé MRM à un run donné.
    La dérive est appliquée avec une direction alternée par composé (réalisme analytique :
    certains composés sont plus sensibles à la suppression ionique que d'autres).
    """
    p = TECHNIQUES[tech_name]
    d_run = max(0, run_number - p["drift_start"])
    rng = np.random.default_rng(42 + run_number * 7)
    ratios = []
    for i, nom in enumerate(p["qq_nom"]):
        # Direction de dérive alternée selon l'indice du composé
        direction = 1 if i % 2 == 0 else -1
        drift = direction * d_run * p["drift_ratio_run"]
        noise = rng.normal(0, 0.008)  # variabilité analytique run-to-run
        ratios.append(round(nom + drift + noise, 4))
    return ratios


def _ppm_at_run(tech_name, run_number):
    """
    Calcule l'erreur de masse (ppm) pour chaque composé HRMS à un run donné.
    Dérive progressive après drift_start.
    """
    p = TECHNIQUES[tech_name]
    d_run = max(0, run_number - p["drift_start"])
    rng = np.random.default_rng(42 + run_number * 13)
    ppms = []
    for base_ppm in p["ppm_base"]:
        drift = d_run * p["drift_ppm_run"]
        noise = rng.normal(0, 0.15)
        ppms.append(round(base_ppm + drift + noise, 2))
    return ppms


def calculate_drift_history(tech_name):
    """
    Calcule les métriques de dérive sur 30 runs pour la technique donnée.
    Retourne un DataFrame une ligne par run avec tous les indicateurs de trend.
    """
    p = TECHNIQUES[tech_name]
    N_RUNS = 30
    rows = []

    for run in range(1, N_RUNS + 1):
        d_run = max(0, run - p["drift_start"])
        rng = np.random.default_rng(42 + run * 19)

        # RT shift (décalage moyen par rapport au run 1)
        rt_shift = d_run * p.get("drift_rt_run", 0.002) + rng.normal(0, 0.002)

        # Largeur FWHM moyenne
        w_base = float(np.mean(p["widths"]))
        w_mean = w_base + d_run * p.get("drift_w_run", 0.001) + rng.normal(0, 0.003)

        # Bruit de fond
        noise = p["noise_base"] + d_run * p.get("noise_run", 0) + rng.normal(0, p["noise_base"] * 0.04)

        # Asymétrie
        asym = 1.08 + d_run * 0.006 + rng.normal(0, 0.008)

        row = {
            "Run": run, "RT_shift_min": round(rt_shift, 4),
            "Width_mean_min": round(w_mean, 4),
            "Noise": round(noise, 1), "Asymetrie": round(asym, 3),
        }

        if p["type"] == "mrm":
            ratios = _mrm_qq_at_run(tech_name, run)
            for i, (cmp, r) in enumerate(zip(p["compounds"], ratios)):
                row[f"Ratio_{i}"] = r

        if p["type"] == "hrms":
            ppms = _ppm_at_run(tech_name, run)
            for i, pp in enumerate(ppms):
                row[f"ppm_{i}"] = pp

        if p["type"] == "ms" and "drift_intensity_run" in p:
            intensity_factor = (1 + p["drift_intensity_run"]) ** d_run
            row["Intensity_factor"] = round(intensity_factor, 4)

        rows.append(row)

    return pd.DataFrame(rows)


def detect_anomalies(drift_df, tech_name):
    """
    Applique 3 règles de détection de dérive sur l'historique 30 runs.

    Règle 1 — Dépassement seuil absolu : la métrique dépasse un seuil prédéfini.
    Règle 2 — Tendance monotone sur 5 runs consécutifs : pente significative sur
              fenêtre glissante → alerte prédictive avant dépassement de seuil.
    Règle 3 — Accélération de la dérive : comparaison pente runs 1-15 vs 16-30.
    """
    p  = TECHNIQUES[tech_name]
    df = drift_df.copy()
    anomalies = []

    def check_trend_5(series):
        """Retourne True si tendance monotone significative sur les 5 derniers points."""
        if len(series) < 5:
            return False
        last5 = series.iloc[-5:].values
        slope, _, r, _, _ = linregress(range(5), last5)
        # R² > 0.80 = tendance linéaire forte, pas juste du bruit
        return (r ** 2 > 0.80) and (abs(slope) > 0.001)

    def check_acceleration(series):
        """Compare pente première moitié vs deuxième moitié → accélération."""
        if len(series) < 10:
            return False, 0
        mid = len(series) // 2
        s1, *_ = linregress(range(mid), series.iloc[:mid].values)
        s2, *_ = linregress(range(len(series) - mid), series.iloc[mid:].values)
        return abs(s2) > 2 * abs(s1) and abs(s2) > 0.001, round(s2 / max(abs(s1), 1e-9), 1)

    # --- RT shift ---
    rt_series = df["RT_shift_min"]
    rt_thresh = {"classical": 0.05, "ms": 0.04, "mrm": 0.05, "hrms": 0.03}.get(p["type"], 0.05)
    if rt_series.iloc[-1] > rt_thresh:
        anomalies.append({
            "rule": 1, "level": "warning",
            "metric": "Décalage TR moyen",
            "detail": f"TR shift actuel : {rt_series.iloc[-1]:.3f} min (seuil : {rt_thresh} min)",
        })
    elif check_trend_5(rt_series):
        anomalies.append({
            "rule": 2, "level": "warning",
            "metric": "Décalage TR moyen — tendance",
            "detail": f"Tendance monotone détectée sur 5 runs. Projection : dépassement seuil dans ~{max(1, int((rt_thresh - rt_series.iloc[-1]) / max(abs(rt_series.diff().mean()), 1e-6)))} runs.",
        })

    # --- Largeur de pic ---
    w_series = df["Width_mean_min"]
    w_base = float(np.mean(p["widths"]))
    w_thresh = w_base * 1.20
    if w_series.iloc[-1] > w_thresh:
        anomalies.append({
            "rule": 1, "level": "warning",
            "metric": "Élargissement de pic",
            "detail": f"FWHM moyen actuel : {w_series.iloc[-1]:.4f} min (+{100*(w_series.iloc[-1]/w_base-1):.1f}% vs nominal)",
        })
    acc, ratio = check_acceleration(w_series)
    if acc:
        anomalies.append({
            "rule": 3, "level": "critical",
            "metric": "Accélération dérive largeur",
            "detail": f"La vitesse d'élargissement a été multipliée par {ratio:.1f}× sur la 2ème moitié des runs. Action immédiate recommandée.",
        })

    # --- MRM : ratio Q/q ---
    if p["type"] == "mrm":
        for i, (cmp, nom) in enumerate(zip(p["compounds"], p["qq_nom"])):
            col = f"Ratio_{i}"
            if col not in df.columns:
                continue
            r_series = df[col]
            lo = nom * (1 - p["qq_window"])
            hi = nom * (1 + p["qq_window"])
            if not (lo <= r_series.iloc[-1] <= hi):
                anomalies.append({
                    "rule": 1, "level": "warning",
                    "metric": f"Ratio Q/q — {cmp}",
                    "detail": f"Ratio actuel : {r_series.iloc[-1]:.3f} | Fenêtre ±20% : [{lo:.3f}, {hi:.3f}]",
                })
            elif check_trend_5(r_series):
                anomalies.append({
                    "rule": 2, "level": "warning",
                    "metric": f"Ratio Q/q — {cmp} — tendance",
                    "detail": f"Tendance de dérive sur 5 runs. Sortie de fenêtre prévue dans ~3–5 runs.",
                })

    # --- HRMS : erreur ppm ---
    if p["type"] == "hrms":
        thresh = p["ppm_thresh"]
        comp_names = p.get("comp_names", [f"Composé {i+1}" for i in range(len(p["masses"]))])
        for i, cmp in enumerate(comp_names):
            col = f"ppm_{i}"
            if col not in df.columns:
                continue
            ppm_series = df[col]
            if ppm_series.iloc[-1] > thresh:
                anomalies.append({
                    "rule": 1, "level": "critical",
                    "metric": f"Erreur masse — {cmp}",
                    "detail": f"Erreur actuelle : {ppm_series.iloc[-1]:.2f} ppm (seuil : {thresh} ppm). Recalibration requise.",
                })
            elif check_trend_5(ppm_series):
                anomalies.append({
                    "rule": 2, "level": "warning",
                    "metric": f"Erreur masse — {cmp} — tendance",
                    "detail": f"Dérive progressive : +{ppm_series.diff().mean():.2f} ppm/run. Recalibration préventive recommandée.",
                })

    # --- LC-MS : intensité ---
    if p["type"] == "ms" and "Intensity_factor" in df.columns:
        if df["Intensity_factor"].iloc[-1] < 0.80:
            anomalies.append({
                "rule": 1, "level": "warning",
                "metric": "Perte de réponse ionique",
                "detail": f"Intensité résiduelle : {100*df['Intensity_factor'].iloc[-1]:.0f}% vs run 1. Seuil d'alerte : 80%.",
            })

    return anomalies


def generate_report_data(tech_name, run_number):
    """
    Agrège toutes les données pour le rapport QC.
    Retourne un dictionnaire structuré.
    """
    p           = TECHNIQUES[tech_name]
    t, sig, meta = generate_chromatogram(tech_name, run_number)
    drift_df    = calculate_drift_history(tech_name)
    anomalies   = detect_anomalies(drift_df.head(run_number), tech_name)

    n_crit = sum(1 for a in anomalies if a["level"] == "critical")
    n_warn = sum(1 for a in anomalies if a["level"] == "warning")

    if n_crit > 0:
        global_status = "NON CONFORME"
    elif n_warn > 0:
        global_status = "ATTENTION"
    else:
        global_status = "CONFORME"

    # Résumé exécutif auto-généré
    if global_status == "CONFORME":
        summary = (
            f"L'instrument {p['instrument']} présente des performances nominales sur "
            f"les {run_number} derniers runs analysés. Aucune dérive significative détectée. "
            "Maintenance préventive à programmer selon le plan calendaire standard."
        )
    elif global_status == "ATTENTION":
        summary = (
            f"Des signaux précoces de dérive ont été détectés sur {n_warn} indicateur(s). "
            "Aucun critère d'acceptation n'est encore dépassé, mais une intervention préventive "
            "est recommandée dans les 5 à 10 prochains runs pour éviter une invalidation de série."
        )
    else:
        summary = (
            f"{n_crit} critère(s) d'acceptation dépassé(s). Intervention corrective requise avant "
            "toute nouvelle séquence analytique. Les résultats des dernières séries doivent être "
            "réexaminés par le Responsable Qualité."
        )

    return {
        "tech": tech_name, "run": run_number,
        "instrument": p["instrument"],
        "column": p.get("column", "—"),
        "status": global_status,
        "summary": summary,
        "anomalies": anomalies,
        "n_crit": n_crit, "n_warn": n_warn,
        "drift_df": drift_df,
        "maintenance": p["maint_msg"],
        "alert_msg": p["alert_msg"],
        "alert_level": p["alert_level"],
    }


# ============================================================
# FONCTIONS GRAPHIQUES PLOTLY
# ============================================================

def plot_chromatogram(t, sig, tech_name, meta, run_number):
    """
    Chromatogramme interactif avec annotations des pics, seuil de bruit,
    et marqueurs de pics détectés. Adapté à la technique (unités, plage temporelle).
    """
    p = TECHNIQUES[tech_name]

    fig = go.Figure()

    # Trace principale — remplissage sous la courbe
    fig.add_trace(go.Scatter(
        x=t, y=sig,
        mode="lines",
        name=p["unit_label"],
        line=dict(color=PC["primary"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.05)",
        hovertemplate="t = %{x:.3f} min<br>Signal = %{y:,.0f} " + p["unit"] + "<extra></extra>",
    ))

    # Marqueurs des pics (triangles verts)
    names = p.get("peak_names", p.get("compounds", []))
    for i, (rt, h, w) in enumerate(zip(meta["rts"], meta["heights"], meta["widths"])):
        label = names[i] if i < len(names) else f"Pic {i+1}"
        # Position du marqueur légèrement au-dessus du pic
        idx = np.argmin(np.abs(t - rt))
        h_real = float(sig[idx]) if 0 <= idx < len(sig) else h
        fig.add_trace(go.Scatter(
            x=[rt], y=[h_real * 1.04],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=9, color=PC["ok"]),
            text=[label], textposition="top center",
            textfont=dict(size=9, color=PC["ok"]),
            name=label, showlegend=False,
            hovertemplate=f"<b>{label}</b><br>TR = {rt:.3f} min<br>H ≈ {h_real:,.0f} {p['unit']}<extra></extra>",
        ))

    # Seuil de bruit (ligne pointillée)
    noise_thresh = meta["noise"] * 3
    fig.add_hline(
        y=noise_thresh, line_dash="dot", line_color=PC["warning"], line_width=1.2,
        annotation_text=f"3×σ bruit ({noise_thresh:,.0f} {p['unit']})",
        annotation_font_size=9, annotation_font_color=PC["warning"],
    )

    # Pics parasites : zone rouge si présents
    d_run = meta["d_run"]
    if "parasite_rts" in p and d_run > 0:
        t0, t1 = p["time_range"]
        para_end = max(p["parasite_rts"]) + 1.0
        fig.add_vrect(
            x0=t0, x1=para_end,
            fillcolor="rgba(239,68,68,0.07)", line_width=0,
            annotation_text="Zone contamination", annotation_font_size=9,
            annotation_font_color=PC["critical"], annotation_position="top left",
        )

    apply_fig_style(fig, height=400)
    fig.update_layout(
        title=dict(text=f"<b>Chromatogramme — {tech_name}</b> | Run #{run_number}", font=dict(size=13)),
        xaxis_title="Temps de rétention (min)",
        yaxis_title=p["unit_label"],
        showlegend=False,
    )
    return fig


def plot_mrm_detail(tech_name, run_number):
    """
    Affiche 4 mini-chromatogrammes MRM (Q + q) pour chaque composé.
    Visuel clé pour LC-MS/MS et GC-MS/MS.
    """
    p = TECHNIQUES[tech_name]
    compounds = p["compounds"]
    n = len(compounds)
    ratios = _mrm_qq_at_run(tech_name, run_number)
    t = np.linspace(p["time_range"][0], p["time_range"][1], p["n_points"])

    fig = make_subplots(
        rows=2, cols=2 if n <= 4 else 3,
        subplot_titles=[f"{c}" for c in compounds[:4]],
        shared_xaxes=False, vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    positions = [(1,1), (1,2), (2,1), (2,2)]
    np.random.seed(42 + run_number * 5)

    for i, (cmp, rt, h, w) in enumerate(zip(
        compounds[:4], p["rts"][:4], p["heights"][:4], p["widths"][:4]
    )):
        if i >= len(positions):
            break
        row, col = positions[i]
        nom = p["qq_nom"][i]
        r_actual = ratios[i]
        status_ok = abs(r_actual - nom) / nom <= p["qq_window"]

        # Trace quantifier (Q)
        sig_q = _asym_gaussian(t, rt, h, w, 1.08)
        sig_q += np.random.normal(0, p["noise_base"] * 0.5, len(t))
        sig_q = np.maximum(0, sig_q)

        # Trace qualifier (q) — amplitude proportionnelle au ratio
        sig_qq = _asym_gaussian(t, rt, h * r_actual, w * 0.98, 1.08)
        sig_qq += np.random.normal(0, p["noise_base"] * 0.5, len(t))
        sig_qq = np.maximum(0, sig_qq)

        color_q  = PC["primary"]
        color_qq = PC["ok"] if status_ok else PC["critical"]

        fig.add_trace(go.Scatter(
            x=t, y=sig_q, mode="lines", name=f"Q ({p['trans_q'][i]})",
            line=dict(color=color_q, width=1.5), showlegend=(i == 0),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=t, y=sig_qq, mode="lines", name=f"q ({p['trans_qq'][i]})",
            line=dict(color=color_qq, width=1.3, dash="dash"), showlegend=(i == 0),
        ), row=row, col=col)

        # Annotation ratio
        ratio_txt = f"Q/q = {r_actual:.3f}" + (" ✓" if status_ok else " ✗")
        fig.add_annotation(
            x=rt + 1.5, y=h * 0.85, text=ratio_txt,
            font=dict(size=9, color=PC["ok"] if status_ok else PC["critical"]),
            showarrow=False, row=row, col=col,
        )

    apply_fig_style(fig, height=420)
    fig.update_layout(
        title=dict(text=f"<b>Transitions MRM — {tech_name}</b> | Run #{run_number}", font=dict(size=13)),
    )
    for i in range(1, 5):
        try:
            fig.update_xaxes(title_text="TR (min)", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
            fig.update_yaxes(title_text="counts",   row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        except Exception:
            pass
    return fig


def plot_hrms_ppm(tech_name, run_number, drift_df):
    """
    Graphique erreur de masse (ppm) par composé sur 30 runs.
    Seuil critique (5 ppm) affiché en rouge.
    """
    p = TECHNIQUES[tech_name]
    comp_names = p.get("comp_names", [f"Composé {i+1}" for i in range(len(p["masses"]))])
    thresh = p["ppm_thresh"]
    colors_list = [PC["primary"], PC["ok"], PC["secondary"], PC["accent"], PC["warning"]]

    fig = go.Figure()

    for i, cmp in enumerate(comp_names):
        col = f"ppm_{i}"
        if col not in drift_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=drift_df["Run"], y=drift_df[col],
            mode="lines+markers",
            name=cmp,
            line=dict(color=colors_list[i % len(colors_list)], width=1.5),
            marker=dict(size=4),
            hovertemplate="Run %{x}<br>Erreur = %{y:.2f} ppm<extra>" + cmp + "</extra>",
        ))

    # Seuil critique
    fig.add_hline(y=thresh, line_dash="dash", line_color=PC["critical"], line_width=1.5,
                  annotation_text=f"Seuil critique {thresh} ppm", annotation_font_size=9,
                  annotation_font_color=PC["critical"])

    # Seuil warning
    fig.add_hline(y=thresh * 0.7, line_dash="dot", line_color=PC["warning"], line_width=1.0,
                  annotation_text=f"Seuil warning {thresh*0.7:.1f} ppm", annotation_font_size=9,
                  annotation_font_color=PC["warning"])

    # Ligne de run courant
    fig.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"], line_width=1.2,
                  annotation_text=f"Run actuel ({run_number})", annotation_font_size=9)

    apply_fig_style(fig, height=360)
    fig.update_layout(
        title=dict(text="<b>Évolution erreur de masse (ppm)</b> — Historique 30 runs", font=dict(size=13)),
        xaxis_title="Numéro de run",
        yaxis_title="Erreur de masse (ppm)",
    )
    return fig


def plot_drift_trends(drift_df, tech_name, run_number):
    """
    4 graphiques de dérive : RT shift, largeur pic, bruit, asymétrie.
    Avec ligne de run actuel et zones d'alerte.
    """
    p = TECHNIQUES[tech_name]
    df = drift_df.copy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Décalage TR moyen (min)", "Largeur de pic FWHM (min)",
            "Niveau de bruit de fond", "Facteur d'asymétrie (TF)",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    def add_trend_trace(series, row, col, color, thresh_warn=None, thresh_crit=None):
        fig.add_trace(go.Scatter(
            x=df["Run"], y=series,
            mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(size=3.5),
            showlegend=False,
            hovertemplate="Run %{x}<br>%{y:.4f}<extra></extra>",
        ), row=row, col=col)
        if thresh_warn:
            fig.add_hline(y=thresh_warn, line_dash="dot", line_color=PC["warning"],
                          line_width=1.0, row=row, col=col)
        if thresh_crit:
            fig.add_hline(y=thresh_crit, line_dash="dash", line_color=PC["critical"],
                          line_width=1.2, row=row, col=col)
        fig.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"],
                      line_width=1.0, row=row, col=col)
        # Zone de dérive (après drift_start)
        fig.add_vrect(x0=p["drift_start"], x1=30, fillcolor="rgba(239,68,68,0.04)",
                      line_width=0, row=row, col=col)

    add_trend_trace(df["RT_shift_min"],    1, 1, PC["primary"],  0.035, 0.06)
    add_trend_trace(df["Width_mean_min"],  1, 2, PC["secondary"],
                    float(np.mean(p["widths"])) * 1.12, float(np.mean(p["widths"])) * 1.25)
    add_trend_trace(df["Noise"],           2, 1, PC["accent"],
                    p["noise_base"] * 1.5, p["noise_base"] * 2.5)
    add_trend_trace(df["Asymetrie"],       2, 2, PC["ok"], 1.30, 1.50)

    apply_fig_style(fig, height=460)
    fig.update_layout(
        title=dict(text="<b>Métriques de dérive instrumentale — 30 runs</b>", font=dict(size=13)),
    )
    return fig


def plot_mrm_ratios(drift_df, tech_name, run_number):
    """
    Évolution des ratios Q/q sur 30 runs par composé.
    Fenêtres ±20% affichées comme zones de tolérance.
    """
    p = TECHNIQUES[tech_name]
    compounds = p["compounds"]
    colors_list = [PC["primary"], PC["ok"], PC["secondary"], PC["accent"], PC["warning"]]

    fig = go.Figure()

    for i, (cmp, nom) in enumerate(zip(compounds, p["qq_nom"])):
        col = f"Ratio_{i}"
        if col not in drift_df.columns:
            continue

        # Zone de tolérance ±20%
        if i == 0:  # une seule fois pour la légende
            fig.add_trace(go.Scatter(
                x=list(drift_df["Run"]) + list(drift_df["Run"])[::-1],
                y=[nom * (1 + p["qq_window"])] * 30 + [nom * (1 - p["qq_window"])] * 30,
                fill="toself", fillcolor="rgba(16,185,129,0.07)",
                line=dict(color="rgba(0,0,0,0)"), name="Fenêtre ±20%",
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=drift_df["Run"], y=drift_df[col],
            mode="lines+markers",
            name=cmp,
            line=dict(color=colors_list[i % len(colors_list)], width=1.5),
            marker=dict(size=4),
            hovertemplate="Run %{x}<br>Ratio = %{y:.3f}<extra>" + cmp + "</extra>",
        ))

        # Limites ±20%
        fig.add_hline(y=nom * (1 + p["qq_window"]), line_dash="dot",
                      line_color=PC["warning"], line_width=0.8)
        fig.add_hline(y=nom * (1 - p["qq_window"]), line_dash="dot",
                      line_color=PC["warning"], line_width=0.8)

    fig.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"],
                  line_width=1.2, annotation_text=f"Run {run_number}", annotation_font_size=9)
    fig.add_vline(x=p["drift_start"], line_dash="dot", line_color=PC["critical"],
                  line_width=0.8, annotation_text="Début dérive", annotation_font_size=8)

    apply_fig_style(fig, height=360)
    fig.update_layout(
        title=dict(text="<b>Évolution ratios Q/q par composé — Historique 30 runs</b>", font=dict(size=13)),
        xaxis_title="Numéro de run",
        yaxis_title="Ratio qualifier / quantifier",
    )
    return fig


def plot_dashboard_overview(drift_df, tech_name, run_number):
    """
    Graphiques dashboard : RT shift + bruit côte à côte, puis largeur de pic.
    """
    p   = TECHNIQUES[tech_name]
    df  = drift_df.copy()

    # ---- Graphique 1 : RT shift ----
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["Run"], y=df["RT_shift_min"],
        mode="lines+markers", name="RT shift",
        line=dict(color=PC["primary"], width=1.5), marker=dict(size=3.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="Run %{x}<br>ΔRT = %{y:.3f} min<extra></extra>",
    ))
    fig1.add_hline(y=0.05, line_dash="dot", line_color=PC["warning"], line_width=1.0)
    fig1.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"], line_width=1.2)
    apply_fig_style(fig1, height=280, legend=False)
    fig1.update_layout(
        title=dict(text="<b>RT Shift moyen (min)</b>", font=dict(size=12)),
        xaxis_title="Run", yaxis_title="ΔRT (min)",
        margin=dict(l=45, r=15, t=35, b=40),
    )

    # ---- Graphique 2 : Bruit ----
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["Run"], y=df["Noise"],
        mode="lines+markers", name="Bruit",
        line=dict(color=PC["accent"], width=1.5), marker=dict(size=3.5),
        fill="tozeroy", fillcolor="rgba(14,165,233,0.07)",
        hovertemplate="Run %{x}<br>Bruit = %{y:.1f} " + p["unit"] + "<extra></extra>",
    ))
    fig2.add_hline(y=p["noise_base"] * 2.5, line_dash="dot", line_color=PC["warning"], line_width=1.0)
    fig2.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"], line_width=1.2)
    apply_fig_style(fig2, height=280, legend=False)
    fig2.update_layout(
        title=dict(text=f"<b>Bruit de fond ({p['unit']})</b>", font=dict(size=12)),
        xaxis_title="Run", yaxis_title=p["unit"],
        margin=dict(l=45, r=15, t=35, b=40),
    )

    # ---- Graphique 3 : Largeur pic (ou ppm pour HRMS) ----
    if p["type"] == "hrms":
        col0 = "ppm_0"
        fig3 = go.Figure()
        if col0 in df.columns:
            fig3.add_trace(go.Scatter(
                x=df["Run"], y=df[col0],
                mode="lines+markers",
                line=dict(color=PC["secondary"], width=1.5), marker=dict(size=3.5),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
                hovertemplate="Run %{x}<br>%{y:.2f} ppm<extra></extra>",
            ))
            fig3.add_hline(y=p["ppm_thresh"], line_dash="dash", line_color=PC["critical"], line_width=1.2)
        apply_fig_style(fig3, height=280, legend=False)
        fig3.update_layout(
            title=dict(text="<b>Erreur masse (ppm) — Composé 1</b>", font=dict(size=12)),
            xaxis_title="Run", yaxis_title="ppm",
            margin=dict(l=45, r=15, t=35, b=40),
        )
    else:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["Run"], y=df["Width_mean_min"],
            mode="lines+markers",
            line=dict(color=PC["ok"], width=1.5), marker=dict(size=3.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
            hovertemplate="Run %{x}<br>FWHM = %{y:.4f} min<extra></extra>",
        ))
        w_base = float(np.mean(p["widths"]))
        fig3.add_hline(y=w_base * 1.20, line_dash="dot", line_color=PC["warning"], line_width=1.0)
        fig3.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"], line_width=1.2)
        apply_fig_style(fig3, height=280, legend=False)
        fig3.update_layout(
            title=dict(text="<b>Largeur de pic FWHM (min)</b>", font=dict(size=12)),
            xaxis_title="Run", yaxis_title="FWHM (min)",
            margin=dict(l=45, r=15, t=35, b=40),
        )

    return fig1, fig2, fig3


# ============================================================
# COMPOSANTS UI RÉUTILISABLES
# ============================================================

def badge_html(text, level="ok"):
    return f'<span class="badge-{level}">{text}</span>'

def alert_html(title, body, level="warning"):
    icon = {"ok": "✅", "warning": "⚠️", "critical": "🔴"}.get(level, "ℹ️")
    return f"""
    <div class="alert-{level}">
        <div class="alert-title">{icon} {title}</div>
        <div class="alert-body">{body}</div>
    </div>"""

def metric_card_html(label, value, delta_text="", delta_level="ok"):
    delta_html = f'<div class="metric-delta-{delta_level}">{delta_text}</div>' if delta_text else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""

def advisory_banner():
    st.markdown("""
    <div class="advisory-banner">
        🛡️ <strong>Mode Advisory Only</strong> — ChromatoIA suggère, le technicien décide.
        Toutes les alertes sont indicatives. La validation humaine est systématique.
        Audit trail complet disponible.
    </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 1 : DASHBOARD
# ============================================================

def page_dashboard(tech_name, run_number):
    p = TECHNIQUES[tech_name]

    st.markdown('<div class="page-header">', unsafe_allow_html=True)
    st.markdown("# 🏠 Dashboard")
    st.markdown(f'<p class="page-subtitle">Vue d\'ensemble — {tech_name} | {p["instrument"]}</p>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    advisory_banner()

    drift_df  = calculate_drift_history(tech_name)
    anomalies = detect_anomalies(drift_df.head(run_number), tech_name)
    n_crit    = sum(1 for a in anomalies if a["level"] == "critical")
    n_warn    = sum(1 for a in anomalies if a["level"] == "warning")

    # ---- Statut instrument ----
    if n_crit > 0:
        instr_status, instr_level, instr_delta = "🔴 CRITIQUE", "critical", "Intervention immédiate requise"
    elif n_warn > 0:
        instr_status, instr_level, instr_delta = "⚠️ WARNING", "warning", f"{n_warn} indicateur(s) en dérive"
    else:
        instr_status, instr_level, instr_delta = "✅ NOMINAL", "ok", "Performances dans les spécifications"

    # ---- Maintenance ----
    d_run = max(0, run_number - p["drift_start"])
    maint_in = max(1, 25 - run_number + p["drift_start"])

    # ---- Métriques en cartes ----
    st.markdown("### Métriques clés")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card_html("Runs analysés (session)", str(run_number),
                    f"+{run_number} depuis démarrage", "ok"), unsafe_allow_html=True)
    with c2:
        n_anom = n_crit + n_warn
        dlevel = "critical" if n_crit > 0 else ("warning" if n_warn > 0 else "ok")
        st.markdown(metric_card_html("Anomalies détectées",
                    str(n_anom),
                    f"{n_crit} critique(s) · {n_warn} warning(s)",
                    dlevel), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("Statut instrument",
                    instr_status, instr_delta, instr_level), unsafe_allow_html=True)
    with c4:
        maint_level = "warning" if maint_in <= 5 else "ok"
        st.markdown(metric_card_html("Maintenance préventive",
                    f"~{maint_in} runs",
                    "⚠️ Bientôt" if maint_in <= 5 else "✅ Planifiée",
                    maint_level), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Graphiques de tendance ----
    st.markdown("### Tendances instrumentales")
    fig1, fig2, fig3 = plot_dashboard_overview(drift_df, tech_name, run_number)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ---- Tableau des 10 dernières analyses ----
    st.markdown("### Historique récent")
    base_date = datetime(2026, 4, 1, 7, 0)
    table_runs = list(range(max(1, run_number - 9), run_number + 1))
    drift_df_h = drift_df.head(run_number)

    table_rows = []
    for r in table_runs:
        d_r = max(0, r - p["drift_start"])
        rng_t = np.random.default_rng(42 + r * 11)
        dt_run = base_date + timedelta(hours=(r - 1) * 5 + int(rng_t.integers(0, 60)))
        anom_r = detect_anomalies(drift_df.head(r), tech_name)
        n_a = len(anom_r)
        status_r = "🔴 CRITIQUE" if any(a["level"]=="critical" for a in anom_r) else (
            "⚠️ WARNING" if n_a > 0 else "✅ OK"
        )
        detail = anom_r[0]["metric"] if anom_r else "—"
        table_rows.append({
            "Run": r, "Date/Heure": dt_run.strftime("%d/%m %H:%M"),
            "Technique": tech_name.split("(")[0].strip(),
            "Statut": status_r, "Anomalie": detail,
        })

    df_table = pd.DataFrame(table_rows[::-1])  # plus récent en premier
    st.dataframe(df_table, use_container_width=True, hide_index=True,
                 column_config={
                     "Run": st.column_config.NumberColumn("Run #", width="small"),
                     "Statut": st.column_config.TextColumn("Statut", width="medium"),
                 })


# ============================================================
# PAGE 2 : ANALYSE CHROMATOGRAMME
# ============================================================

def page_chromatogram(tech_name, run_number):
    p = TECHNIQUES[tech_name]

    st.markdown("# 📊 Analyse Chromatogramme")
    st.markdown(f'<p class="page-subtitle">{tech_name} · Run #{run_number} · {p["instrument"]}</p>',
                unsafe_allow_html=True)
    advisory_banner()

    t, sig, meta = generate_chromatogram(tech_name, run_number)

    # ---- Chromatogramme principal ----
    if p["type"] in ["mrm"]:
        # Pour MRM : chromatogramme TIC + détail transitions
        fig_chrom = plot_chromatogram(t, sig, tech_name, meta, run_number)
        st.plotly_chart(fig_chrom, use_container_width=True)
        st.markdown("#### Détail des transitions MRM")
        fig_mrm = plot_mrm_detail(tech_name, run_number)
        st.plotly_chart(fig_mrm, use_container_width=True)
    else:
        fig_chrom = plot_chromatogram(t, sig, tech_name, meta, run_number)
        st.plotly_chart(fig_chrom, use_container_width=True)

    # ---- Tableau de résultats adapté à la technique ----
    st.markdown("#### Résultats des pics")

    if p["type"] == "classical":
        df_peaks = detect_peaks(t, sig, tech_name, meta)
        st.dataframe(df_peaks, use_container_width=True, hide_index=True)

        # Indicateurs QC agrégés
        st.markdown("#### Indicateurs QC")
        ci1, ci2, ci3 = st.columns(3)
        n_ok    = (df_peaks["Statut"] == "✅ OK").sum()
        n_total = len(df_peaks)
        mean_tf = df_peaks["Symétrie (TF)"].mean()
        mean_N  = df_peaks["N plateaux"].mean()
        with ci1:
            st.markdown(metric_card_html("Pics conformes", f"{n_ok}/{n_total}",
                        "✅ Tous OK" if n_ok == n_total else f"⚠️ {n_total-n_ok} hors spec",
                        "ok" if n_ok == n_total else "warning"), unsafe_allow_html=True)
        with ci2:
            tail_ok = mean_tf <= p.get("qc_tail_max", 1.5)
            st.markdown(metric_card_html("Facteur de tailing moyen", f"{mean_tf:.3f}",
                        f"Seuil max : {p.get('qc_tail_max', 1.5):.1f}",
                        "ok" if tail_ok else "warning"), unsafe_allow_html=True)
        with ci3:
            n_ok_plates = mean_N >= p.get("qc_N_min", 5000)
            st.markdown(metric_card_html("N plateaux moyen", f"{int(mean_N):,}",
                        f"Seuil min : {p.get('qc_N_min', 5000):,}",
                        "ok" if n_ok_plates else "warning"), unsafe_allow_html=True)

        # Résolution si disponible
        if "Rs" in df_peaks.columns:
            rs_vals = pd.to_numeric(df_peaks["Rs"], errors="coerce").dropna()
            if len(rs_vals) > 0 and rs_vals.min() < p.get("qc_res_min", 1.5):
                st.markdown(alert_html("Résolution insuffisante",
                    f"Rs minimum = {rs_vals.min():.2f} (seuil requis : {p.get('qc_res_min', 1.5):.1f}). "
                    "Risque de co-élution sur les pics concernés.", "warning"),
                    unsafe_allow_html=True)

    elif p["type"] == "mrm":
        ratios = _mrm_qq_at_run(tech_name, run_number)
        d_run  = max(0, run_number - p["drift_start"])
        rts_eff = np.array(p["rts"]) + d_run * p.get("drift_rt_run", 0.002)
        rows = []
        for i, (cmp, rt, tq, tqq, nom, rat) in enumerate(zip(
            p["compounds"], rts_eff[:len(p["compounds"])],
            p["trans_q"], p["trans_qq"], p["qq_nom"], ratios,
        )):
            lo = nom * (1 - p["qq_window"])
            hi = nom * (1 + p["qq_window"])
            in_window = lo <= rat <= hi
            rows.append({
                "Composé": cmp, "TR (min)": round(rt, 3),
                "Transition Q": tq, "Transition q (qualifier)": tqq,
                "Ratio Q/q nominal": round(nom, 3), "Ratio Q/q mesuré": round(rat, 3),
                "Fenêtre ±20%": f"[{lo:.3f}, {hi:.3f}]",
                "Statut": "✅ OK" if in_window else "❌ Hors fenêtre",
            })
        df_mrm = pd.DataFrame(rows)
        st.dataframe(df_mrm, use_container_width=True, hide_index=True)

        n_ok = (df_mrm["Statut"] == "✅ OK").sum()
        pct  = 100 * n_ok / len(df_mrm)
        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown(metric_card_html("Ratios dans fenêtre", f"{n_ok}/{len(df_mrm)}",
                f"{pct:.0f}% conformes", "ok" if pct == 100 else "warning"), unsafe_allow_html=True)
        with ci2:
            rt_shift_mrm = d_run * p.get("drift_rt_run", 0.002)
            rt_ok = rt_shift_mrm <= p.get("qc_rt_window", 0.10)
            st.markdown(metric_card_html("RT shift maximal", f"{rt_shift_mrm:.3f} min",
                f"Seuil : ±{p.get('qc_rt_window',0.1):.2f} min",
                "ok" if rt_ok else "warning"), unsafe_allow_html=True)

    elif p["type"] == "hrms":
        ppms = _ppm_at_run(tech_name, run_number)
        d_run  = max(0, run_number - p["drift_start"])
        rts_eff = np.array(p["rts"]) + d_run * p.get("drift_rt_run", 0.001)
        comp_names = p.get("comp_names", [f"Composé {i+1}" for i in range(len(p["masses"]))])
        rows = []
        for i, (cmp, rt, m, f, ppm) in enumerate(zip(
            comp_names, rts_eff[:len(p["masses"])],
            p["masses"], p.get("formulas", [""] * len(p["masses"])), ppms,
        )):
            m_meas = round(m + m * ppm * 1e-6, 4)
            rows.append({
                "Composé": cmp, "Formule": f, "TR (min)": round(rt, 3),
                "Masse théorique (Da)": m, "Masse mesurée (Da)": m_meas,
                "Erreur (ppm)": round(ppm, 2),
                "Statut": "✅ OK" if ppm < p["ppm_thresh"] else "❌ > seuil",
            })
        df_hrms = pd.DataFrame(rows)
        st.dataframe(df_hrms, use_container_width=True, hide_index=True)
        ppm_mean = float(np.mean(ppms))
        pct_ok   = 100 * sum(1 for pp in ppms if pp < p["ppm_thresh"]) / len(ppms)
        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown(metric_card_html("Erreur masse moyenne", f"{ppm_mean:.2f} ppm",
                f"Seuil critique : {p['ppm_thresh']} ppm",
                "ok" if ppm_mean < p["ppm_thresh"] else "critical"), unsafe_allow_html=True)
        with ci2:
            st.markdown(metric_card_html("Composés < seuil ppm", f"{pct_ok:.0f}%",
                "✅ Conformes" if pct_ok == 100 else "⚠️ Attention calibration",
                "ok" if pct_ok == 100 else "warning"), unsafe_allow_html=True)

    elif p["type"] == "ms":
        d_run = max(0, run_number - p["drift_start"])
        int_factor = (1 + p.get("drift_intensity_run", 0)) ** d_run
        rts_eff = np.array(p["rts"]) + d_run * p.get("drift_rt_run", 0.002)
        rows = []
        for i, (name, rt, h) in enumerate(zip(
            p.get("peak_names", p.get("compounds", [])),
            rts_eff[:len(p["rts"])], np.array(p["heights"]) * int_factor
        )):
            idx = np.argmin(np.abs(t - rt))
            h_real = float(sig[idx])
            snr = h_real / max(meta["noise"], 1)
            rows.append({
                "Composé": name, "TR (min)": round(rt, 3),
                "Intensité (counts)": int(h_real),
                "S/N": round(snr, 1),
                "Réponse vs Run 1 (%)": round(100 * int_factor, 1),
                "Statut": "✅ OK" if int_factor >= 0.80 else "⚠️ Perte réponse",
            })
        df_ms = pd.DataFrame(rows)
        st.dataframe(df_ms, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3 : DÉTECTION DE DÉRIVE
# ============================================================

def page_drift(tech_name, run_number):
    p = TECHNIQUES[tech_name]

    st.markdown("# 📈 Détection de Dérive")
    st.markdown(f'<p class="page-subtitle">Algorithme CUSUM + analyse de tendance · {tech_name}</p>',
                unsafe_allow_html=True)
    advisory_banner()

    # ---- Description algorithme ----
    with st.expander("ℹ️ Algorithme de détection — Comprendre les règles", expanded=False):
        st.markdown("""
<style>
.algo-table { width:100%; border-collapse:collapse; font-size:12px; }
.algo-table th { background:#1E293B; color:white; padding:8px 10px; text-align:left; }
.algo-table td { padding:8px 10px; border-bottom:1px solid #E8ECF4; vertical-align:top; }
.algo-table tr:nth-child(even) td { background:#F8F9FC; }
</style>
<b>ChromatoIA applique 3 règles de détection complémentaires :</b>
<br><br>
<table class="algo-table">
<tr><th>Règle</th><th>Déclencheur</th><th>Niveau</th><th>Logique</th></tr>
<tr><td><b>1 — Seuil absolu</b></td><td>Métrique &gt; seuil prédéfini</td><td>⚠️ Warning</td><td>Réactive : détecte une dérive déjà présente</td></tr>
<tr><td><b>2 — Tendance 5 runs</b></td><td>Pente monotone sur 5 runs consécutifs (R² &gt; 0.80)</td><td>⚠️ Prédictif</td><td>Détecte une dérive <b>avant</b> le dépassement de seuil</td></tr>
<tr><td><b>3 — Accélération</b></td><td>Vitesse de dérive ×2 entre 1ère et 2ème moitié</td><td>🔴 Critique</td><td>Détecte une dégradation soudaine</td></tr>
</table>
<br>
<b>Métrique principale par technique :</b> LC/GC classique : RT shift, FWHM, asymétrie &nbsp;|&nbsp;
LC-MS / GC-MS : intensité, bruit &nbsp;|&nbsp; MRM : ratio Q/q &nbsp;|&nbsp; HRMS : erreur de masse (ppm)
""", unsafe_allow_html=True)

    drift_df  = calculate_drift_history(tech_name)
    anomalies = detect_anomalies(drift_df.head(run_number), tech_name)

    # ---- Graphiques de dérive ----
    if p["type"] in ["classical", "ms"]:
        fig_trend = plot_drift_trends(drift_df, tech_name, run_number)
        st.plotly_chart(fig_trend, use_container_width=True)
    elif p["type"] == "mrm":
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_ratio = plot_mrm_ratios(drift_df, tech_name, run_number)
            st.plotly_chart(fig_ratio, use_container_width=True)
        with col_b:
            fig_rt = go.Figure()
            fig_rt.add_trace(go.Scatter(
                x=drift_df["Run"], y=drift_df["RT_shift_min"],
                mode="lines+markers", line=dict(color=PC["primary"], width=1.5),
                marker=dict(size=3.5), showlegend=False,
                hovertemplate="Run %{x}<br>ΔRT = %{y:.3f} min<extra></extra>",
            ))
            fig_rt.add_hline(y=p.get("qc_rt_window", 0.1), line_dash="dot",
                             line_color=PC["warning"], line_width=1.0)
            fig_rt.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"])
            apply_fig_style(fig_rt, height=320, legend=False)
            fig_rt.update_layout(title=dict(text="<b>RT Shift moyen</b>", font=dict(size=12)),
                                 xaxis_title="Run", yaxis_title="ΔRT (min)")
            st.plotly_chart(fig_rt, use_container_width=True)
    elif p["type"] == "hrms":
        fig_ppm = plot_hrms_ppm(tech_name, run_number, drift_df)
        st.plotly_chart(fig_ppm, use_container_width=True)
        fig_rt2 = go.Figure()
        fig_rt2.add_trace(go.Scatter(
            x=drift_df["Run"], y=drift_df["RT_shift_min"],
            mode="lines+markers", line=dict(color=PC["primary"], width=1.5),
            marker=dict(size=3.5), showlegend=False,
        ))
        fig_rt2.add_vline(x=run_number, line_dash="solid", line_color=PC["secondary"])
        apply_fig_style(fig_rt2, height=260, legend=False)
        fig_rt2.update_layout(title=dict(text="<b>RT Shift moyen</b>", font=dict(size=12)),
                              xaxis_title="Run", yaxis_title="ΔRT (min)")
        st.plotly_chart(fig_rt2, use_container_width=True)

    # ---- Alertes et anomalies ----
    st.markdown("#### Alertes détectées")

    if not anomalies:
        st.markdown(alert_html(
            "Aucune anomalie détectée",
            f"Toutes les métriques sont dans les spécifications jusqu'au run #{run_number}. "
            "Instrument en fonctionnement nominal.",
            "ok"
        ), unsafe_allow_html=True)
    else:
        for anom in anomalies:
            level = anom["level"]
            rule_txt = {1: "Règle 1 — Seuil dépassé", 2: "Règle 2 — Tendance prédictive", 3: "Règle 3 — Accélération dérive"}.get(anom["rule"], "")
            st.markdown(alert_html(
                f"[{rule_txt}] {anom['metric']}",
                anom["detail"],
                level,
            ), unsafe_allow_html=True)

    # ---- Alerte principale technique ----
    st.markdown("#### Alerte instrumentale principale")
    d_run = max(0, run_number - p["drift_start"])
    if d_run > 0:
        st.markdown(alert_html(
            p["alert_msg"],
            f"Détectée à partir du run #{p['drift_start']}. "
            f"Dérive cumulée sur {d_run} run(s). "
            "Action recommandée avant la prochaine séquence analytique.",
            p["alert_level"]
        ), unsafe_allow_html=True)

        st.markdown("#### Recommandation maintenance")
        maint_lines = p["maint_msg"].split("\n")
        for i, line in enumerate(maint_lines):
            if line.strip():
                st.markdown(f"**{i+1}.** {line.strip().lstrip('•').strip()}")
    else:
        st.markdown(alert_html(
            "Aucune dérive détectée sur ce run",
            f"La dérive instrumentale simulée démarre au run #{p['drift_start']}. "
            "Utilisez le slider dans la sidebar pour avancer dans les runs.",
            "ok"
        ), unsafe_allow_html=True)


# ============================================================
# PAGE 4 : RAPPORT QC
# ============================================================

def page_report(tech_name, run_number):
    p = TECHNIQUES[tech_name]

    st.markdown("# 📋 Rapport QC")
    st.markdown(f'<p class="page-subtitle">Rapport automatisé · Mode advisory · Validation humaine requise</p>',
                unsafe_allow_html=True)

    report = generate_report_data(tech_name, run_number)

    # ---- En-tête rapport ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Instrument :** {report['instrument']}
**Colonne / Phase :** {report['column']}
**Technique :** {tech_name}
""")
    with c2:
        now = datetime.now()
        op_list = ["M. Dupont (Resp. Qualité)", "Mme Martin (Technicienne)", "M. Bernard (Analyste)"]
        op = op_list[run_number % 3]
        st.markdown(f"""
**Date :** {now.strftime('%d/%m/%Y %H:%M')}
**Opérateur :** {op}
**Run #:** {run_number} | **Série :** SER-2026-{run_number:04d}
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Statut global ----
    status = report["status"]
    status_colors = {"CONFORME": "#065F46", "ATTENTION": "#92400E", "NON CONFORME": "#991B1B"}
    status_bg     = {"CONFORME": "#D1FAE5", "ATTENTION": "#FEF3C7", "NON CONFORME": "#FEE2E2"}
    status_icon   = {"CONFORME": "✅", "ATTENTION": "⚠️", "NON CONFORME": "🔴"}

    st.markdown(f"""
    <div style="text-align:center; padding:28px; background:{status_bg[status]};
                border-radius:14px; margin:20px 0;">
        <div style="font-size:42px; margin-bottom:8px;">{status_icon[status]}</div>
        <div style="font-size:24px; font-weight:700; color:{status_colors[status]};">{status}</div>
        <div style="font-size:12px; color:{status_colors[status]}; margin-top:6px; opacity:0.8;">
            {report['n_crit']} critique(s) · {report['n_warn']} warning(s)
        </div>
    </div>""", unsafe_allow_html=True)

    # ---- Résumé exécutif ----
    st.markdown("#### Résumé exécutif")
    st.markdown(f'<div class="section-card"><p style="font-size:13px; line-height:1.6; color:#374151;">{report["summary"]}</p></div>',
                unsafe_allow_html=True)

    # ---- Tableau récapitulatif ----
    st.markdown("#### Tableau récapitulatif métriques QC")
    drift_df = report["drift_df"]
    row_run  = drift_df[drift_df["Run"] == run_number]
    if len(row_run):
        r = row_run.iloc[0]
        recaps = []
        rt_thresh = {"classical": 0.05, "ms": 0.04, "mrm": 0.05, "hrms": 0.03}.get(p["type"], 0.05)
        recaps.append({
            "Paramètre": "RT shift moyen", "Valeur": f"{r['RT_shift_min']:.4f} min",
            "Seuil": f"< {rt_thresh} min",
            "Résultat": "✅ Conforme" if r['RT_shift_min'] < rt_thresh else "⚠️ Déviation",
        })
        w_base = float(np.mean(p["widths"]))
        recaps.append({
            "Paramètre": "FWHM moyen", "Valeur": f"{r['Width_mean_min']:.4f} min",
            "Seuil": f"< {w_base*1.20:.4f} min (+20%)",
            "Résultat": "✅ Conforme" if r['Width_mean_min'] < w_base * 1.20 else "⚠️ Déviation",
        })
        recaps.append({
            "Paramètre": "Bruit de fond", "Valeur": f"{r['Noise']:.0f} {p['unit']}",
            "Seuil": f"< {p['noise_base'] * 2.5:.0f} {p['unit']}",
            "Résultat": "✅ Conforme" if r['Noise'] < p["noise_base"] * 2.5 else "⚠️ Déviation",
        })
        recaps.append({
            "Paramètre": "Facteur d'asymétrie", "Valeur": f"{r['Asymetrie']:.3f}",
            "Seuil": "< 1.50",
            "Résultat": "✅ Conforme" if r['Asymetrie'] < 1.50 else "⚠️ Déviation",
        })

        if p["type"] == "hrms" and "ppm_0" in r:
            ppm_v = r["ppm_0"]
            recaps.append({
                "Paramètre": "Erreur masse (Composé 1)", "Valeur": f"{ppm_v:.2f} ppm",
                "Seuil": f"< {p['ppm_thresh']} ppm",
                "Résultat": "✅ Conforme" if ppm_v < p["ppm_thresh"] else "🔴 Hors spec",
            })

        if p["type"] == "ms" and "Intensity_factor" in r:
            iv = r["Intensity_factor"]
            recaps.append({
                "Paramètre": "Réponse ionique relative", "Valeur": f"{100*iv:.1f}%",
                "Seuil": "> 80%",
                "Résultat": "✅ Conforme" if iv >= 0.80 else "⚠️ Perte réponse",
            })

        st.dataframe(pd.DataFrame(recaps), use_container_width=True, hide_index=True)

    # ---- Anomalies ----
    st.markdown("#### Anomalies détectées")
    anomalies = report["anomalies"]
    if not anomalies:
        st.markdown(alert_html("Aucune anomalie", "Toutes les métriques dans les spécifications.", "ok"),
                    unsafe_allow_html=True)
    else:
        for i, a in enumerate(anomalies, 1):
            level = a["level"]
            rule_label = {1:"Seuil abs.", 2:"Tendance 5R", 3:"Accélération"}.get(a["rule"], "")
            st.markdown(alert_html(
                f"#{i} [{rule_label}] {a['metric']}",
                a["detail"], level,
            ), unsafe_allow_html=True)

    # ---- Recommandations ----
    st.markdown("#### Recommandations (par ordre de priorité)")
    reco_lines = [l.strip() for l in report["maintenance"].split("\n") if l.strip()]
    for i, line in enumerate(reco_lines, 1):
        priority = "🔴" if i == 1 else ("⚠️" if i == 2 else "ℹ️")
        st.markdown(f"{priority} **Priorité {i}** — {line.lstrip('•').strip()}")

    # ---- Mention légale advisory ----
    st.markdown("""
    <div style="margin-top:24px; padding:14px 18px; background:#F8F9FC;
                border:1px solid #E8ECF4; border-radius:8px; font-size:11px; color:#64748B;">
        <strong>🛡️ Avis ChromatoIA — Mode Advisory Only</strong><br>
        Ce rapport est généré automatiquement à titre indicatif uniquement.
        Toutes les alertes et recommandations doivent être revues et validées par
        un opérateur qualifié avant toute action corrective.
        ChromatoIA ne prend aucune décision autonome sur les données analytiques.
        Audit trail disponible sur demande.
    </div>""", unsafe_allow_html=True)

    # ---- Bouton export CSV ----
    st.markdown("<br>", unsafe_allow_html=True)
    drift_export = drift_df.head(run_number).copy()
    drift_export.insert(0, "Technique", tech_name)
    drift_export.insert(0, "Instrument", p["instrument"])
    csv_buf = io.StringIO()
    drift_export.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Télécharger l'historique QC (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"chromatoIA_QC_{tech_name.split()[0]}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ============================================================
# PAGE 5 : À PROPOS
# ============================================================

def page_about():
    st.markdown("# ℹ️ À Propos de ChromatoIA")
    st.markdown('<p class="page-subtitle">SaaS IA d\'analyse chromatographique — Couche intelligence non-invasive sur CDS existants</p>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
### Notre mission
ChromatoIA est une couche d'intelligence analytique qui se connecte aux exports
de vos logiciels CDS existants (Empower, OpenLab, Chromeleon, LabSolutions)
pour détecter les dérives instrumentales **avant** qu'elles génèrent des OOS,
des reruns coûteux, ou des pannes non anticipées.

**Nous ne remplaçons pas votre CDS. Nous le rendons intelligent.**

### Pourquoi ChromatoIA ?

Les logiciels CDS actuels collectent vos données mais ne les analysent pas de
façon prédictive. Un technicien HPLC passe en moyenne **2 à 4 heures par semaine**
à investiguer manuellement des dérives qui auraient pu être détectées 10 à 15
runs plus tôt avec un algorithme de trending adapté.

**Coût d'un rerun LC-MS/MS en pharma GMP : 850 – 4 500 €.**
ChromatoIA vise à en éviter 3 à 5 par an par instrument.
        """)
    with col2:
        st.markdown("### Techniques supportées")
        tech_table = [
            {"Famille": "LC", "Technique": "LC / HPLC (UV/DAD)", "Détection dérive": "RT, FWHM, asymétrie"},
            {"Famille": "LC", "Technique": "LC-MS (ESI)", "Détection dérive": "Suppression ionique, bruit"},
            {"Famille": "LC", "Technique": "LC-MS/MS (MRM)", "Détection dérive": "Ratio Q/q, RT window"},
            {"Famille": "LC", "Technique": "LC-HRMS", "Détection dérive": "Erreur masse (ppm)"},
            {"Famille": "GC", "Technique": "GC / FID", "Détection dérive": "RT, contamination liner"},
            {"Famille": "GC", "Technique": "GC-MS (TIC)", "Détection dérive": "Septum, RT drift"},
            {"Famille": "GC", "Technique": "GC-MS/MS (MRM)", "Détection dérive": "Ratio Q/q, gaz vecteur"},
            {"Famille": "GC", "Technique": "GC-HRMS", "Détection dérive": "Erreur masse (ppm)"},
        ]
        st.dataframe(pd.DataFrame(tech_table), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Positionnement — Advisory Only")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(alert_html("Ce que ChromatoIA fait",
            "• Détecte les dérives instrumentales\n"
            "• Génère des alertes documentées\n"
            "• Produit des rapports QC traçables\n"
            "• Recommande des actions maintenance\n"
            "• Fournit un audit trail complet", "ok"), unsafe_allow_html=True)
    with col_b:
        st.markdown(alert_html("Ce que ChromatoIA ne fait PAS",
            "• Ne modifie jamais les données sources\n"
            "• Ne remplace pas le CDS existant\n"
            "• Ne prend aucune décision automatique\n"
            "• Ne gère pas le LIMS\n"
            "• Ne valide pas les séries à la place de l'opérateur", "warning"), unsafe_allow_html=True)
    with col_c:
        st.markdown(alert_html("Conformité réglementaire",
            "• Audit trail 21 CFR Part 11\n"
            "• Compatible GxP / ISO 17025\n"
            "• Compatible COFRAC\n"
            "• Hébergement OVH France (HDS)\n"
            "• Déploiement on-premise disponible", "ok"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:32px; background:linear-gradient(135deg, #EFF6FF, #F5F3FF);
                border-radius:16px; margin-top:16px;">
        <div style="font-size:22px; font-weight:700; color:#1E293B; margin-bottom:8px;">
            "On vous prévient avant que votre colonne commence à dériver — et on vous montre pourquoi."
        </div>
        <div style="font-size:13px; color:#64748B;">
            ChromatoIA — SaaS IA d'analyse chromatographique · Version démo 1.0 · 2026
        </div>
    </div>""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================


# ============================================================
# PAGE 0 : PARC ANALYTIQUE
# ============================================================

def page_parc_analytique():
    """
    Vue d'ensemble de tous les instruments du parc analytique.
    Chaque carte est cliquable pour naviguer vers l'instrument.
    Le statut est calculé automatiquement sur les 30 runs simulés.
    """
    st.markdown("# 🏭 Parc Analytique")
    st.markdown('<p class="page-subtitle">Vue d\'ensemble — Statut en temps réel de tous les instruments</p>',
                unsafe_allow_html=True)
    advisory_banner()

    # ---- Calcul statut de tous les instruments ----
    # Chaque instrument est à un stade différent de son cycle de maintenance —
    # réalisme : certains viennent d'être entretenus, d'autres sont en milieu de
    # cycle, 2-3 seulement sont effectivement en dérive détectable.
    CURRENT_RUNS = {
        "LC / HPLC  (UV / DAD)":            22,   # dérive colonne en cours
        "LC-MS  (ESI, quadrupôle)":           8,   # entretenu récemment — nominal
        "LC-MS/MS  (MRM, triple quadrupôle)": 16,  # début dérive ratio Q/q
        "LC-HRMS  (Orbitrap / Q-TOF)":        5,   # post-calibration — nominal
        "GC / FID":                           25,  # contamination injecteur détectée
        "GC-MS  (TIC, quadrupôle)":           12,  # nominal, proche du seuil
        "GC-MS/MS  (MRM, triple quadrupôle)": 20,  # dérive ratio MRM active
        "GC-HRMS  (Q-TOF / Orbitrap GC)":     9,  # entretenu — nominal
    }

    park_data = []
    for tech_name, p in TECHNIQUES.items():
        current_run = CURRENT_RUNS.get(tech_name, 15)
        drift_df = calculate_drift_history(tech_name)
        anomalies = detect_anomalies(drift_df.head(current_run), tech_name)
        n_crit = sum(1 for a in anomalies if a["level"] == "critical")
        n_warn = sum(1 for a in anomalies if a["level"] == "warning")

        d_run = max(0, current_run - p["drift_start"])
        if n_crit > 0:
            status = "CRITIQUE"
        elif n_warn > 0:
            status = "WARNING"
        else:
            status = "OK"

        rt_shift = drift_df.loc[current_run-1, "RT_shift_min"] if current_run <= len(drift_df) else 0
        park_data.append({
            "tech": tech_name,
            "instrument": p["instrument"],
            "family": p["family"],
            "type": p["type"],
            "status": status,
            "n_crit": n_crit,
            "n_warn": n_warn,
            "drift_runs": d_run,
            "drift_start": p["drift_start"],
            "current_run": current_run,
            "rt_shift": rt_shift,
            "alert_msg": p["alert_msg"],
        })

    # ---- Résumé parc ----
    total = len(park_data)
    n_ok   = sum(1 for d in park_data if d["status"] == "OK")
    n_warn = sum(1 for d in park_data if d["status"] == "WARNING")
    n_crit = sum(1 for d in park_data if d["status"] == "CRITIQUE")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card_html("Total instruments", str(total), "", "ok"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("Nominaux", str(n_ok),
                    "✅ Fonctionnement normal", "ok"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("En dérive", str(n_warn),
                    "⚠️ Action recommandée", "warning" if n_warn else "ok"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html("Critiques", str(n_crit),
                    "🔴 Intervention immédiate" if n_crit else "Aucun", "critical" if n_crit else "ok"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Cartes instruments (2 colonnes) ----
    st.markdown("### Instruments")

    STATUS_CONFIG = {
        "OK":       {"border": "#10B981", "bg": "#F0FDF4", "badge_bg": "#D1FAE5",
                     "badge_txt": "#065F46", "icon": "✅", "label": "NOMINAL"},
        "WARNING":  {"border": "#F59E0B", "bg": "#FFFBEB", "badge_bg": "#FEF3C7",
                     "badge_txt": "#92400E", "icon": "⚠️", "label": "DÉRIVE"},
        "CRITIQUE": {"border": "#EF4444", "bg": "#FFF5F5", "badge_bg": "#FEE2E2",
                     "badge_txt": "#991B1B", "icon": "🔴", "label": "CRITIQUE"},
    }

    # Trier : critiques en premier, warnings, puis OK
    order = {"CRITIQUE": 0, "WARNING": 1, "OK": 2}
    park_data.sort(key=lambda x: order[x["status"]])

    # Grille 2 colonnes
    for i in range(0, len(park_data), 2):
        col_left, col_right = st.columns(2)
        for j, col in enumerate([col_left, col_right]):
            idx = i + j
            if idx >= len(park_data):
                break
            d = park_data[idx]
            cfg = STATUS_CONFIG[d["status"]]
            tech_short = d["tech"].split("(")[0].strip()

            with col:
                # Carte HTML
                drift_info = (f"Dérive depuis {d['drift_runs']} run(s)"
                              if d["drift_runs"] > 0 else "Aucune dérive détectée")
                anomaly_txt = d["alert_msg"][:65] + "..." if len(d["alert_msg"]) > 65 else d["alert_msg"]
                if d["status"] == "OK":
                    anomaly_txt = "Toutes métriques dans les spécifications."

                # Barre de progression du cycle de maintenance
                pct = min(100, int(d["current_run"] / 30 * 100))
                bar_color = cfg["border"]

                card_html = f"""
<div style="background:{cfg['bg']}; border:2px solid {cfg['border']};
     border-radius:14px; padding:18px 20px; margin-bottom:14px;
     box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
    <div>
      <div style="font-size:13px; font-weight:700; color:#1E293B;">{tech_short}</div>
      <div style="font-size:10px; color:#64748B; margin-top:2px;">{d['instrument']}</div>
    </div>
    <span style="background:{cfg['badge_bg']}; color:{cfg['badge_txt']};
      padding:4px 10px; border-radius:20px; font-size:11px; font-weight:700;
      white-space:nowrap; flex-shrink:0; margin-left:8px;">
      {cfg['icon']} {cfg['label']}
    </span>
  </div>
  <div style="font-size:11px; color:#374151; line-height:1.5; margin-bottom:10px;">
    {anomaly_txt}
  </div>
  <div style="margin-bottom:8px;">
    <div style="display:flex; justify-content:space-between; font-size:9px; color:#94A3B8; margin-bottom:3px;">
      <span>Cycle maintenance</span><span>Run #{d['current_run']} / 30</span>
    </div>
    <div style="background:#E8ECF4; border-radius:4px; height:5px; overflow:hidden;">
      <div style="background:{bar_color}; width:{pct}%; height:100%; border-radius:4px;
                  transition:width 0.3s;"></div>
    </div>
  </div>
  <div style="display:flex; gap:10px; font-size:10px; color:#64748B; flex-wrap:wrap;">
    <span>📊 {drift_info}</span>
    <span>⚠️ {d['n_crit']} crit · {d['n_warn']} warn</span>
  </div>
</div>"""
                st.markdown(card_html, unsafe_allow_html=True)

                # Bouton cliquable pour naviguer vers cet instrument
                btn_label = f"📈 Analyser {tech_short}"
                if st.button(btn_label, key=f"btn_park_{idx}", use_container_width=True):
                    st.session_state["selected_tech"] = d["tech"]
                    st.session_state["selected_page"] = "📊 Analyse Chromatogramme"
                    st.rerun()

    # ---- Tableau récapitulatif ----
    st.markdown("### Tableau récapitulatif")
    rows = []
    for d in park_data:
        cfg = STATUS_CONFIG[d["status"]]
        rows.append({
            "Instrument": d["instrument"],
            "Technique": d["tech"].split("(")[0].strip(),
            "Statut": f"{cfg['icon']} {d['status']}",
            "Dérives": f"{d['n_crit']} critique · {d['n_warn']} warning",
            "RT shift": f"{d['rt_shift']:.3f} min",
            "Run actuel": f"#{d['current_run']}",
            "Dérive depuis": f"{d['drift_runs']} runs" if d["drift_runs"] > 0 else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_sidebar():
    # Session state pour navigation par clic depuis le parc analytique
    if "selected_tech" not in st.session_state:
        st.session_state["selected_tech"] = TECH_LIST[0]
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "🏭 Parc Analytique"

    cal = generate_run_calendar()
    date_labels = list(cal.keys())

    with st.sidebar:
        # Logo / titre
        st.markdown("""
        <div style="padding:16px 0 8px 0;">
            <div style="font-size:20px; font-weight:700; color:#1E293B;">🔬 ChromatoIA</div>
            <div style="font-size:11px; color:#64748B; margin-top:2px;">Intelligence Analytique</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Sélection technique
        st.markdown("##### INSTRUMENT / TECHNIQUE")
        tech_idx = TECH_LIST.index(st.session_state["selected_tech"])                    if st.session_state["selected_tech"] in TECH_LIST else 0
        tech_name = st.selectbox(
            "Technique analytique",
            TECH_LIST,
            index=tech_idx,
            label_visibility="collapsed",
        )
        st.session_state["selected_tech"] = tech_name

        # Badge technique
        fam = TECHNIQUES[tech_name]["family"]
        badge_color = "#DBEAFE" if fam == "LC" else "#D1FAE5"
        badge_text_color = "#1D4ED8" if fam == "LC" else "#065F46"
        st.markdown(
            f'<div style="background:{badge_color}; color:{badge_text_color}; '
            f'padding:5px 12px; border-radius:8px; font-size:11px; font-weight:600; '
            f'display:inline-block; margin-bottom:8px;">{fam} · {TECHNIQUES[tech_name]["type"].upper()}</div>',
            unsafe_allow_html=True
        )

        # ---- Sélecteur DATE ----
        st.markdown("##### DATE DE SÉQUENCE")
        # Pré-sélection : date qui contient run 20 (milieu de l'historique)
        default_date_idx = min(12, len(date_labels) - 1)
        selected_date = st.selectbox(
            "Date",
            date_labels,
            index=default_date_idx,
            label_visibility="collapsed",
        )

        # ---- Sélecteur RUN dans la date ----
        runs_this_day = cal[selected_date]
        run_options = [f"Run #{r}  —  {t}" for r, t in runs_this_day]
        st.markdown("##### SÉQUENCE")
        selected_run_label = st.selectbox(
            "Run",
            run_options,
            index=len(run_options) - 1,   # dernier run de la journée par défaut
            label_visibility="collapsed",
        )
        run_idx = run_options.index(selected_run_label)
        run_number = runs_this_day[run_idx][0]

        # Indicateur drift visuel
        drift_start = TECHNIQUES[tech_name]["drift_start"]
        if run_number >= drift_start:
            d_run = run_number - drift_start
            st.markdown(
                f'<div style="background:#FEF3C7; color:#92400E; padding:8px 12px; '
                f'border-radius:8px; font-size:11px; font-weight:500; margin-top:4px;">'
                f'⚠️ Dérive active depuis run #{drift_start} ({d_run} run(s))</div>',
                unsafe_allow_html=True
            )
        else:
            runs_to_drift = drift_start - run_number
            st.markdown(
                f'<div style="background:#D1FAE5; color:#065F46; padding:8px 12px; '
                f'border-radius:8px; font-size:11px; font-weight:500; margin-top:4px;">'
                f'✅ Nominal — dérive dans ~{runs_to_drift} runs</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Navigation
        st.markdown("##### NAVIGATION")
        page_options = [
            "🏭 Parc Analytique",
            "🏠 Dashboard",
            "📊 Analyse Chromatogramme",
            "📈 Détection de Dérive",
            "📋 Rapport QC",
            "ℹ️ À Propos",
        ]
        # Synchro session state → radio
        page_default = st.session_state.get("selected_page", "🏭 Parc Analytique")
        if page_default not in page_options:
            page_default = "🏭 Parc Analytique"
        page = st.radio(
            "Page",
            page_options,
            index=page_options.index(page_default),
            label_visibility="collapsed",
        )
        st.session_state["selected_page"] = page

        st.markdown("---")

        # Info instrument
        instr = TECHNIQUES[tech_name]["instrument"]
        col_inst = TECHNIQUES[tech_name].get("column", "—")
        st.markdown(f"""
        <div style="font-size:10px; color:#94A3B8; line-height:1.6;">
            <strong>Instrument</strong><br>{instr}<br>
            <strong>Colonne / Phase</strong><br>{col_inst}<br>
            <strong>Run sélectionné</strong><br>#{run_number} — {selected_date}
        </div>""", unsafe_allow_html=True)

    return tech_name, run_number, page


# ============================================================
# MAIN — ROUTING
# ============================================================

def main():
    tech_name, run_number, page = render_sidebar()

    if   "Parc"            in page: page_parc_analytique()
    elif "Dashboard"       in page: page_dashboard(tech_name, run_number)
    elif "Chromatogramme"  in page: page_chromatogram(tech_name, run_number)
    elif "Dérive"          in page: page_drift(tech_name, run_number)
    elif "Rapport"         in page: page_report(tech_name, run_number)
    elif "À Propos"        in page: page_about()


if __name__ == "__main__":
    main()
