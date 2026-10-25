import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import json
import math
from datetime import datetime

# Ensure src/ is on the Python path (needed for Streamlit Cloud deployment)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_manager import (
    init_db, save_patient_record, get_all_records, seed_sample_patients,
    delete_record, clear_all_records
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Virtual Kidney Biopsy Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CLEAN MEDICAL CSS ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #f1f5f9 !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 1400px !important;
    }

    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Cards */
    .med-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .med-card-header {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 12px;
    }

    /* Top Nav Bar */
    .top-bar {
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 14px 28px;
        margin: -1.5rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .top-bar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
    }
    .top-bar-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-left: 8px;
    }
    .top-bar-icon {
        width: 36px;
        height: 36px;
        background: #2563eb;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* Damage Level Badges */
    .damage-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }
    .damage-0 { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .damage-1 { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .damage-2 { background: #ffedd5; color: #9a3412; border: 1px solid #fdba74; }
    .damage-3 { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

    /* Circular Gauge */
    .gauge-container {
        position: relative;
        width: 160px;
        height: 160px;
        margin: 0 auto;
    }
    .gauge-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }
    .gauge-percent {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    .gauge-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Treatment Cards */
    .tx-card {
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .tx-disease {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #22c55e;
    }
    .tx-supportive {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
    }
    .tx-warning {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #ef4444;
    }
    .tx-title {
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .tx-disease .tx-title { color: #166534; }
    .tx-supportive .tx-title { color: #92400e; }
    .tx-warning .tx-title { color: #991b1b; }
    .tx-body {
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.55;
    }
    .tx-body ul {
        margin: 6px 0 0 0;
        padding-left: 18px;
    }
    .tx-body li { margin-bottom: 4px; }

    /* Simulator Card */
    .sim-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 18px 20px;
    }
    .sim-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 10px;
    }

    /* Patient info bar */
    .patient-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 18px;
        display: flex;
        gap: 24px;
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: #475569;
    }
    .patient-bar strong { color: #0f172a; }

    /* Section Headers */
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Override Streamlit inputs for cleaner look */
    .stNumberInput label, .stSelectbox label, .stTextInput label, .stTextArea label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- FILE PATHS ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'outputs/models/xgboost_biopsy_model.json')
SCALER_PATH = os.path.join(BASE_PATH, 'outputs/models/scaler_params.json')
CAUSAL_CSV_PATH = os.path.join(BASE_PATH, 'outputs/causal_results/causal_effects_20features.csv')
CAUSAL_GRAPH_IMG = os.path.join(BASE_PATH, 'outputs/causal_results/causal_graph_20features.png')

init_db()

@st.cache_resource
def load_resources():
    mdl = xgb.XGBClassifier()
    mdl.load_model(MODEL_PATH)
    with open(SCALER_PATH) as f:
        sp = json.load(f)
    cdf = pd.read_csv(CAUSAL_CSV_PATH) if os.path.exists(CAUSAL_CSV_PATH) else pd.DataFrame()
    return mdl, sp, cdf

model, scaler_params, causal_df = load_resources()

FEATURE_ORDER = [
    'gender_encoded', 'has_hypertension', 'albumin_mean', 'albumin_min',
    'potassium_mean', 'potassium_max', 'potassium_std', 'sodium_std',
    'calcium_min', 'phosphate_mean', 'phosphate_min', 'phosphate_max',
    'phosphate_std', 'glucose_min', 'bicarbonate_mean', 'bicarbonate_min',
    'bicarbonate_std', 'low_albumin', 'hyperkalemia', 'avg_los'
]

# --- PRESETS (ALL 20 FEATURES CONFIGURED) ---
PRESETS = {
    "➕ New Patient (Enter Custom Data)": {
        'id': 'PAT-NEW', 'name': '', 'age': 55, 'gender': 'Male',
        'htn': False, 'glucose_min': 95.0, 'avg_los': 3.0,
        'phosphate_mean': 3.5, 'phosphate_min': 3.0, 'phosphate_max': 4.0, 'phosphate_std': 0.3,
        'potassium_mean': 4.2, 'potassium_max': 4.5, 'potassium_std': 0.2, 'hyperkalemia': False, 'sodium_std': 1.5,
        'bicarbonate_mean': 24.0, 'bicarbonate_min': 22.0, 'bicarbonate_std': 1.2, 'calcium_min': 9.2,
        'albumin_mean': 4.0, 'albumin_min': 3.8, 'low_albumin': False,
        'notes': ''
    },
    "Case 1: Diabetic Nephropathy (Critical)": {
        'id': 'PAT-9021', 'name': 'Eleanor Vance', 'age': 67, 'gender': 'Female',
        'htn': True, 'glucose_min': 185.0, 'avg_los': 14.0,
        'phosphate_mean': 6.2, 'phosphate_min': 5.4, 'phosphate_max': 7.4, 'phosphate_std': 1.2,
        'potassium_mean': 5.6, 'potassium_max': 6.2, 'potassium_std': 0.8, 'hyperkalemia': True, 'sodium_std': 3.2,
        'bicarbonate_mean': 16.0, 'bicarbonate_min': 13.0, 'bicarbonate_std': 3.5, 'calcium_min': 7.1,
        'albumin_mean': 2.4, 'albumin_min': 2.0, 'low_albumin': True,
        'notes': 'Type 2 DM 15 yrs, severe HTN, peripheral edema.'
    },
    "Case 2: Hypertensive Nephrosclerosis (Severe)": {
        'id': 'PAT-5821', 'name': 'Robert Sterling', 'age': 59, 'gender': 'Male',
        'htn': True, 'glucose_min': 110.0, 'avg_los': 7.0,
        'phosphate_mean': 4.8, 'phosphate_min': 4.2, 'phosphate_max': 5.6, 'phosphate_std': 0.7,
        'potassium_mean': 5.1, 'potassium_max': 5.6, 'potassium_std': 0.5, 'hyperkalemia': True, 'sodium_std': 2.5,
        'bicarbonate_mean': 21.0, 'bicarbonate_min': 18.5, 'bicarbonate_std': 2.2, 'calcium_min': 7.8,
        'albumin_mean': 3.1, 'albumin_min': 2.7, 'low_albumin': True,
        'notes': 'HTN 12 yrs, intermittent medication compliance.'
    },
    "Case 3: Early Tubular Stress (Moderate)": {
        'id': 'PAT-8910', 'name': 'Sarah Jenkins', 'age': 44, 'gender': 'Female',
        'htn': False, 'glucose_min': 140.0, 'avg_los': 5.0,
        'phosphate_mean': 3.9, 'phosphate_min': 3.4, 'phosphate_max': 4.3, 'phosphate_std': 0.4,
        'potassium_mean': 4.4, 'potassium_max': 4.8, 'potassium_std': 0.3, 'hyperkalemia': False, 'sodium_std': 1.8,
        'bicarbonate_mean': 23.0, 'bicarbonate_min': 21.0, 'bicarbonate_std': 1.6, 'calcium_min': 8.5,
        'albumin_mean': 3.5, 'albumin_min': 3.2, 'low_albumin': False,
        'notes': 'Pre-diabetes, mild microalbuminuria.'
    },
    "Case 4: Healthy Baseline (Normal)": {
        'id': 'PAT-1029', 'name': 'Arthur Pendelton', 'age': 38, 'gender': 'Male',
        'htn': False, 'glucose_min': 85.0, 'avg_los': 3.0,
        'phosphate_mean': 3.2, 'phosphate_min': 2.8, 'phosphate_max': 3.6, 'phosphate_std': 0.3,
        'potassium_mean': 4.1, 'potassium_max': 4.4, 'potassium_std': 0.2, 'hyperkalemia': False, 'sodium_std': 1.5,
        'bicarbonate_mean': 26.0, 'bicarbonate_min': 24.5, 'bicarbonate_std': 1.2, 'calcium_min': 9.2,
        'albumin_mean': 4.2, 'albumin_min': 3.9, 'low_albumin': False,
        'notes': 'Routine checkup, no chronic disease.'
    }
}

# --- SVG GAUGE GENERATOR ---
def make_gauge_svg(percent, color):
    r = 65
    cx, cy = 80, 80
    circumference = 2 * math.pi * r
    offset = circumference * (1 - percent / 100)
    return f"""
    <div style="display:flex;justify-content:center;margin:8px 0;">
    <svg width="160" height="160" viewBox="0 0 160 160">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e2e8f0" stroke-width="12"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="12"
            stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
            stroke-linecap="round" transform="rotate(-90 {cx} {cy})"
            style="transition: stroke-dashoffset 0.8s ease;"/>
        <text x="{cx}" y="{cy-6}" text-anchor="middle" font-size="28" font-weight="700" fill="#0f172a">{percent:.1f}%</text>
        <text x="{cx}" y="{cy+16}" text-anchor="middle" font-size="10" fill="#64748b" font-weight="500" letter-spacing="0.08em">RISK SCORE</text>
    </svg>
    </div>
    """

# ====================================================
# TOP NAV BAR
# ====================================================
st.markdown("""
<div class="top-bar">
    <div class="top-bar-icon">VB</div>
    <span class="top-bar-title">Virtual Kidney Biopsy Portal</span>
    <span class="top-bar-subtitle">Clinical Decision Support System &mdash; Non-Invasive Histopathology Assessment</span>
</div>
""", unsafe_allow_html=True)

# ====================================================
# MAIN LAYOUT
# ====================================================
nav_tabs = ["Virtual Biopsy", "Patient Registry", "Research Evidence"]
if 'nav_tab' not in st.session_state:
    st.session_state['nav_tab'] = "Virtual Biopsy"
if st.session_state['nav_tab'] not in nav_tabs:
    st.session_state['nav_tab'] = "Virtual Biopsy"

def _sync_nav():
    st.session_state['nav_tab'] = st.session_state['main_nav_radio']

nav = st.radio(
    "", nav_tabs,
    index=nav_tabs.index(st.session_state['nav_tab']),
    horizontal=True, label_visibility="collapsed",
    key="main_nav_radio",
    on_change=_sync_nav
)

if nav == "Virtual Biopsy":
    left_col, spacer, right_col = st.columns([4.0, 0.2, 5.8])

    # ===================== LEFT: PATIENT INTAKE =====================
    with left_col:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        
        hdr_c1, hdr_c2 = st.columns([2.6, 1.4])
        with hdr_c1:
            st.markdown('<div class="med-card-header">Patient Intake Form</div>', unsafe_allow_html=True)
        with hdr_c2:
            if st.button("➕ Blank Form", use_container_width=True):
                st.session_state['selected_preset'] = "➕ New Patient (Enter Custom Data)"
                st.rerun()

        preset_keys = list(PRESETS.keys())
        default_preset = st.session_state.get('selected_preset', preset_keys[0])
        default_idx = preset_keys.index(default_preset) if default_preset in preset_keys else 0

        def _sync_preset():
            st.session_state['selected_preset'] = st.session_state['preset_selectbox']

        preset_choice = st.selectbox(
            "Select Patient Profile or Custom Entry",
            preset_keys,
            index=default_idx,
            label_visibility="collapsed",
            key="preset_selectbox",
            on_change=_sync_preset
        )
        p = PRESETS[preset_choice]

        st.markdown('<div class="section-header">1. Demographics & Hospital Context</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            patient_id = st.text_input("Patient ID / MRN", value=p['id'] if p and p['id'] != 'PAT-NEW' else "PAT-0042", placeholder="e.g. PAT-1049")
        with c2:
            patient_name = st.text_input("Name", value=p['name'] if p else "", placeholder="e.g. Eleanor Vance")
        c3, c4, c5 = st.columns([1.2, 1.3, 1.5])
        with c3:
            age = st.number_input("Age", min_value=18, max_value=99, value=int(p['age']) if p else 55)
        with c4:
            gender = st.selectbox("Sex", ["Male", "Female"], index=0 if (p and p['gender'] == 'Male') else 1)
        with c5:
            avg_los = st.number_input("Stay (Days)", min_value=1.0, max_value=60.0, value=float(p['avg_los']) if p else 4.0, step=1.0)

        st.markdown('<div class="section-header">2. Etiologic Root Causes (Causal Drivers)</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            has_htn = st.toggle("Hypertension History", value=p['htn'] if p else False)
        with rc2:
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=40.0, max_value=500.0,
                                              value=float(p['glucose_min']) if p else 100.0, step=5.0)

        st.markdown('<div class="section-header">3. Routine Blood Chemistry (Lab Slip Inputs)</div>', unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            phosphate = st.number_input("Phosphate (mg/dL)", min_value=1.0, max_value=15.0, value=float(p['phosphate_mean']) if p else 3.5, step=0.1)
            potassium = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=9.0, value=float(p['potassium_mean']) if p else 4.3, step=0.1)
            albumin = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=float(p['albumin_mean']) if p else 3.8, step=0.1)
        with bc2:
            bicarbonate = st.number_input("Bicarbonate (mEq/L)", min_value=5.0, max_value=45.0, value=float(p['bicarbonate_mean']) if p else 24.0, step=0.5)
            calcium = st.number_input("Calcium (mg/dL)", min_value=3.0, max_value=15.0, value=float(p['calcium_min']) if p else 8.8, step=0.1)
            sodium = st.number_input("Sodium (mEq/L)", min_value=110.0, max_value=160.0, value=140.0, step=1.0)

        # Automatic Physiological Derivations
        auto_ph_min = float(p['phosphate_min']) if p else max(0.5, round(phosphate - 0.5, 2))
        auto_ph_max = float(p['phosphate_max']) if p else round(phosphate + 0.8, 2)
        auto_ph_std = float(p['phosphate_std']) if p else (0.5 if phosphate <= 4.5 else 1.1)

        auto_k_max = float(p['potassium_max']) if p else round(potassium + 0.6, 2)
        auto_k_std = float(p['potassium_std']) if p else (0.4 if potassium <= 5.0 else 0.8)
        auto_hyperk = bool(p['hyperkalemia']) if p else (potassium > 5.0)

        auto_bc_min = float(p['bicarbonate_min']) if p else max(5.0, round(bicarbonate - 3.0, 2))
        auto_bc_std = float(p['bicarbonate_std']) if p else (2.0 if bicarbonate >= 22.0 else 3.5)

        auto_alb_min = float(p['albumin_min']) if p else max(0.5, round(albumin - 0.4, 2))
        auto_low_alb = bool(p['low_albumin']) if p else (albumin < 3.5)
        auto_na_std = float(p['sodium_std']) if p else 2.5

        # Visible Real-Time Auto-Derived 20-Feature Inspector
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;padding:12px 14px;margin:12px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-size:0.78rem;font-weight:700;color:#1e293b;text-transform:uppercase;letter-spacing:0.05em;">
                    ⚡ Auto-Derived Model Features (20 Total)
                </span>
                <span style="font-size:0.7rem;background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:12px;font-weight:600;">Live Auto-Calculated</span>
            </div>
            <div style="font-size:0.8rem;color:#475569;line-height:1.7;">
                <div><strong>Phosphate:</strong> Mean: <code>{ph:.2f}</code> | Min: <code>{ph_min:.2f}</code> | Max: <code>{ph_max:.2f}</code> | Std: <code>{ph_std:.2f}</code></div>
                <div><strong>Potassium:</strong> Mean: <code>{k:.2f}</code> | Max: <code>{k_max:.2f}</code> | Std: <code>{k_std:.2f}</code> | Hyperkalemia: <span style="color:{k_col};font-weight:600;">{k_flag}</span></div>
                <div><strong>Acid-Base:</strong> HCO3 Mean: <code>{bc:.1f}</code> | Min: <code>{bc_min:.1f}</code> | Std: <code>{bc_std:.2f}</code> | Calcium Min: <code>{ca:.2f}</code></div>
                <div><strong>Albumin:</strong> Mean: <code>{alb:.2f}</code> | Min: <code>{alb_min:.2f}</code> | Low Albumin: <span style="color:{alb_col};font-weight:600;">{alb_flag}</span></div>
                <div><strong>Etiology & Dynamics:</strong> Glucose Min: <code>{glu:.1f}</code> | HTN: <code>{htn_val}</code> | Sex: <code>{sex_val}</code> | Stay: <code>{stay_val:.1f}d</code></div>
            </div>
        </div>
        """.format(
            ph=phosphate, ph_min=auto_ph_min, ph_max=auto_ph_max, ph_std=auto_ph_std,
            k=potassium, k_max=auto_k_max, k_std=auto_k_std,
            k_col="#dc2626" if auto_hyperk else "#16a34a", k_flag="Positive" if auto_hyperk else "Normal",
            bc=bicarbonate, bc_min=auto_bc_min, bc_std=auto_bc_std, ca=calcium,
            alb=albumin, alb_min=auto_alb_min,
            alb_col="#dc2626" if auto_low_alb else "#16a34a", alb_flag="Low (<3.5)" if auto_low_alb else "Normal",
            glu=fasting_glucose, htn_val=1 if has_htn else 0, sex_val=1 if gender == "Male" else 0, stay_val=avg_los
        ), unsafe_allow_html=True)

        # Optional Manual Override Accordion
        with st.expander("⚙️ Customize / Override Statistical Aggregates", expanded=False):
            st.caption("Override the automatically derived statistical spreads with exact historical lab values:")
            ov1, ov2 = st.columns(2)
            with ov1:
                ov_ph_min = st.number_input("Override Phosphate Min", value=auto_ph_min, step=0.1)
                ov_ph_max = st.number_input("Override Phosphate Max", value=auto_ph_max, step=0.1)
                ov_ph_std = st.number_input("Override Phosphate Std", value=auto_ph_std, step=0.05)
                ov_k_max = st.number_input("Override Potassium Max", value=auto_k_max, step=0.1)
            with ov2:
                ov_k_std = st.number_input("Override Potassium Std", value=auto_k_std, step=0.05)
                ov_bc_min = st.number_input("Override Bicarbonate Min", value=auto_bc_min, step=0.5)
                ov_bc_std = st.number_input("Override Bicarbonate Std", value=auto_bc_std, step=0.1)
                ov_alb_min = st.number_input("Override Albumin Min", value=auto_alb_min, step=0.1)

        notes = st.text_area("Clinical Consultation Notes", value=p['notes'] if p else "", height=60)

        run_btn = st.button("💾 Run Virtual Biopsy & Save Record", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ===================== RIGHT: RESULTS =====================
    with right_col:
        # Build exact 20-feature input dictionary from derived/overridden inputs
        raw = {
            'gender_encoded': 1.0 if gender == "Male" else 0.0,
            'has_hypertension': 1.0 if has_htn else 0.0,
            'albumin_mean': float(albumin),
            'albumin_min': float(ov_alb_min),
            'potassium_mean': float(potassium),
            'potassium_max': float(ov_k_max),
            'potassium_std': float(ov_k_std),
            'sodium_std': float(auto_na_std),
            'calcium_min': float(calcium),
            'phosphate_mean': float(phosphate),
            'phosphate_min': float(ov_ph_min),
            'phosphate_max': float(ov_ph_max),
            'phosphate_std': float(ov_ph_std),
            'glucose_min': float(fasting_glucose),
            'bicarbonate_mean': float(bicarbonate),
            'bicarbonate_min': float(ov_bc_min),
            'bicarbonate_std': float(ov_bc_std),
            'low_albumin': 1.0 if auto_low_alb else 0.0,
            'hyperkalemia': 1.0 if auto_hyperk else 0.0,
            'avg_los': float(avg_los)
        }

        scaled = {}
        for feat in FEATURE_ORDER:
            mu = scaler_params[feat]['mean']
            sigma = scaler_params[feat]['scale']
            scaled[feat] = (raw[feat] - mu) / sigma

        input_df = pd.DataFrame([scaled])[FEATURE_ORDER]
        risk = float(model.predict_proba(input_df)[0][1])

        # Damage Level Classification
        if risk <= 0.25:
            dlvl, dtitle, dbadge, dcolor = 0, "Preserved Architecture", "damage-0", "#22c55e"
            ifta = "< 10% IFTA"
            micro = "Intact glomerular capillary loops with uniform basement membranes. Minimal mesangial expansion. Proximal tubules show preserved brush borders. No significant interstitial fibrosis."
        elif risk <= 0.50:
            dlvl, dtitle, dbadge, dcolor = 1, "Mild Chronic Injury", "damage-1", "#eab308"
            ifta = "10 – 25% IFTA"
            micro = "Focal glomerular basement membrane thickening with early mesangial matrix expansion. Scattered tubular vacuolization and mild interstitial mononuclear infiltrates."
        elif risk <= 0.75:
            dlvl, dtitle, dbadge, dcolor = 2, "Severe Tissue Damage", "damage-2", "#f97316"
            ifta = "26 – 50% IFTA"
            micro = "Segmental glomerulosclerosis involving >25% of glomeruli. Widespread tubular atrophy with thyroidization. Moderate interstitial fibrosis and arteriolar hyalinosis."
        else:
            dlvl, dtitle, dbadge, dcolor = 3, "Critical Failure", "damage-3", "#ef4444"
            ifta = "> 50% IFTA"
            micro = "Global glomerulosclerosis in >50% of glomeruli. Severe diffuse tubular drop-out with extensive interstitial fibrosis. Fibrointimal arterial narrowing and vascular obliteration."

        # --- PATIENT INFO BAR ---
        st.markdown(f"""
        <div class="patient-bar">
            <span><strong>Patient:</strong> {patient_name}</span>
            <span><strong>MRN:</strong> {patient_id}</span>
            <span><strong>Age/Sex:</strong> {age} {gender}</span>
            <span><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</span>
        </div>
        """, unsafe_allow_html=True)

        # --- DAMAGE LEVEL + GAUGE ---
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">Virtual Histopathology Result</div>', unsafe_allow_html=True)

        res_left, res_right = st.columns([3, 2])
        with res_left:
            st.markdown(f'<span class="damage-badge {dbadge}">DAMAGE LEVEL {dlvl}</span>', unsafe_allow_html=True)
            st.markdown(f"<h3 style='margin:8px 0 4px 0;color:#0f172a;'>{dtitle}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#64748b;font-size:0.9rem;margin:0;'>Estimated Chronic Scarring: <strong style=\"color:{dcolor}\">{ifta}</strong></p>", unsafe_allow_html=True)
        with res_right:
            st.markdown(make_gauge_svg(risk * 100, dcolor), unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;margin-top:12px;">
            <p style="font-weight:600;color:#374151;font-size:0.85rem;margin:0 0 6px 0;">Tissue Pathology Summary</p>
            <p style="color:#475569;font-size:0.88rem;line-height:1.6;margin:0;">{micro}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- TREATMENT ADVISORY ---
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">Treatment Advisory</div>', unsafe_allow_html=True)

        tx1, tx2 = st.columns(2)
        with tx1:
            disease_items = []
            if has_htn:
                disease_items.append("Initiate RAAS blockade (ACE inhibitor or ARB) — target BP < 130/80 mmHg")
            if fasting_glucose > 130.0:
                disease_items.append(f"Optimize glycemic control (SGLT2i / Metformin) — current glucose: {fasting_glucose:.0f} mg/dL")
            if not disease_items:
                disease_items.append("No active modifiable drivers identified. Maintain routine surveillance.")

            items_html = "".join(f"<li>{it}</li>" for it in disease_items)
            st.markdown(f"""
            <div class="tx-card tx-disease">
                <div class="tx-title">Disease-Modifying Treatment</div>
                <div class="tx-body">
                    Interventions that directly slow tissue scarring:
                    <ul>{items_html}</ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            primary_plan = "; ".join(disease_items)

        with tx2:
            supp_items = []
            if phosphate > 4.5:
                supp_items.append(f"Phosphate elevated ({phosphate:.1f} mg/dL) — consider phosphate binders")
            if potassium > 5.0:
                supp_items.append(f"Potassium elevated ({potassium:.1f} mEq/L) — dietary restriction or binders")
            if bicarbonate < 22.0:
                supp_items.append(f"Metabolic acidosis (HCO3: {bicarbonate:.1f}) — sodium bicarbonate supplements")
            if not supp_items:
                supp_items.append("Electrolytes within normal range. No acute intervention needed.")

            supp_html = "".join(f"<li>{it}</li>" for it in supp_items)
            st.markdown(f"""
            <div class="tx-card tx-supportive">
                <div class="tx-title">Supportive Care Only</div>
                <div class="tx-body">
                    Symptom management (will NOT reverse tissue damage):
                    <ul>{supp_html}</ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Safety Warning
        warning_text = (
            "Elevated phosphate and potassium are consequences of reduced kidney clearance, "
            "not the cause of tissue damage. Normalizing these lab values improves acute patient "
            "safety but does not halt or reverse glomerular sclerosis or interstitial fibrosis. "
            "Treatment priority must remain on blood pressure and glycemic control."
        )
        st.markdown(f"""
        <div class="tx-card tx-warning">
            <div class="tx-title">Prescribing Safety Notice</div>
            <div class="tx-body">{warning_text}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- TREATMENT RESPONSE SIMULATOR ---
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">Treatment Response Simulator</div>', unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns([2, 2, 1.5])
        with sc1:
            sim_bp = st.checkbox("Achieve target BP (< 130/80)", value=has_htn)
        with sc2:
            sim_glu = st.checkbox("Normalize glucose (< 110 mg/dL)", value=(fasting_glucose > 130.0))

        reduction = 0.0
        if has_htn and sim_bp:
            reduction += 0.0828
        if fasting_glucose > 130.0 and sim_glu:
            reduction += 0.0483

        projected = max(0.02, risk - reduction)

        if projected <= 0.25:
            p_label, p_color = "Level 0 (Preserved)", "#22c55e"
        elif projected <= 0.50:
            p_label, p_color = "Level 1 (Mild)", "#eab308"
        elif projected <= 0.75:
            p_label, p_color = "Level 2 (Severe)", "#f97316"
        else:
            p_label, p_color = "Level 3 (Critical)", "#ef4444"

        with sc3:
            st.metric("Risk Reduction", f"-{reduction*100:.1f}%")

        # Before/After bar
        st.markdown(f"""
        <div class="sim-card">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#475569;margin-bottom:6px;">
                <span>Before Treatment</span><span>After Treatment</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
                <div style="flex:1;background:#e2e8f0;border-radius:6px;height:24px;overflow:hidden;">
                    <div style="width:{risk*100:.0f}%;height:100%;background:{dcolor};border-radius:6px;transition:width 0.5s;"></div>
                </div>
                <span style="font-weight:600;font-size:0.85rem;color:#0f172a;min-width:50px;">{risk*100:.1f}%</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
                <div style="flex:1;background:#e2e8f0;border-radius:6px;height:24px;overflow:hidden;">
                    <div style="width:{projected*100:.0f}%;height:100%;background:{p_color};border-radius:6px;transition:width 0.5s;"></div>
                </div>
                <span style="font-weight:600;font-size:0.85rem;color:#0f172a;min-width:50px;">{projected*100:.1f}%</span>
            </div>
            <p style="margin:10px 0 0 0;font-size:0.85rem;color:#1e40af;">
                Projected outcome with treatment adherence: <strong style="color:{p_color}">{p_label}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- SAVE BUTTON ---
        if run_btn:
            p_id = patient_id.strip() if patient_id.strip() else "PAT-0042"
            p_nm = patient_name.strip() if patient_name.strip() else "Anonymous Patient"
            record = {
                'patient_id': p_id, 'patient_name': p_nm,
                'age': age, 'gender': gender,
                'damage_level': dlvl, 'damage_title': f"Damage Level {dlvl}: {dtitle}",
                'risk_probability': risk,
                'has_hypertension': 1 if has_htn else 0,
                'glucose_min': fasting_glucose,
                'phosphate_mean': phosphate, 'potassium_mean': potassium,
                'bicarbonate_min': bicarbonate, 'albumin_min': albumin,
                'primary_intervention': primary_plan,
                'causal_warning': warning_text,
                'counterfactual_reduction': reduction * 100,
                'notes': notes
            }
            rid = save_patient_record(record)
            st.success(f"✅ Biopsy successfully saved to database as Record #{rid}! You can review it in the 'Patient Registry' tab.")

# ====================================================
# TAB 2: PATIENT REGISTRY
# ====================================================
elif nav == "Patient Registry":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="med-card-header">Patient Longitudinal Registry</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1.2, 1.8, 3.5])
    with b1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with b2:
        if st.button("➕ Add New Patient"):
            st.session_state['selected_preset'] = "➕ New Patient (Enter Custom Data)"
            st.session_state['nav_tab'] = "Virtual Biopsy"
            st.rerun()
    with b3:
        if st.button("👥 Load Sample Cohort"):
            seed_sample_patients()
            st.rerun()

    df = get_all_records()
    if len(df) == 0:
        st.info("No records yet. Run a biopsy or load sample cohort.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("Total Biopsies", len(df))
        with m2: st.metric("Level 0", len(df[df['damage_level']==0]))
        with m3: st.metric("Level 1", len(df[df['damage_level']==1]))
        with m4: st.metric("Level 2", len(df[df['damage_level']==2]))
        with m5: st.metric("Level 3", len(df[df['damage_level']==3]))

        f1, f2 = st.columns(2)
        with f1:
            lvl_filter = st.selectbox("Filter by Damage Level", ["All", "Level 0", "Level 1", "Level 2", "Level 3"])
        with f2:
            search = st.text_input("Search patient name or ID")

        filtered = df.copy()
        if lvl_filter != "All":
            lvl_num = int(lvl_filter.split(" ")[1])
            filtered = filtered[filtered['damage_level'] == lvl_num]
        if search:
            filtered = filtered[
                filtered['patient_id'].str.contains(search, case=False, na=False) |
                filtered['patient_name'].str.contains(search, case=False, na=False)
            ]

        show_cols = ['id','timestamp','patient_id','patient_name','age','gender',
                     'damage_level','risk_probability','has_hypertension',
                     'glucose_min','phosphate_mean','potassium_mean','primary_intervention']
        tbl = filtered[show_cols].copy()
        tbl['risk_probability'] = (tbl['risk_probability']*100).round(1).astype(str)+'%'
        tbl['has_hypertension'] = tbl['has_hypertension'].map({1:'Yes',0:'No'})
        tbl.columns = ['#','Date','MRN','Name','Age','Sex','Level','Risk','HTN',
                       'Glucose','Phosphate','Potassium','Treatment']

        st.dataframe(tbl, use_container_width=True, hide_index=True)

        col_actions, col_del = st.columns([2, 5])
        with col_actions:
            csv = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv,
                              f"biopsy_registry_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
                              use_container_width=True)

        st.markdown("<hr style='margin:16px 0 12px 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)
        with st.expander("🗑️ Delete or Manage Records", expanded=False):
            del_c1, del_c2, del_c3 = st.columns([3.5, 1.5, 2.0])
            with del_c1:
                rec_choices = {f"Record #{r['id']} — {r['patient_name']} (MRN: {r['patient_id']})": r['id'] for _, r in df.iterrows()}
                chosen_label = st.selectbox("Select Record to Remove", list(rec_choices.keys()), label_visibility="collapsed")
            with del_c2:
                if st.button("🗑️ Delete Record", use_container_width=True):
                    chosen_id = rec_choices[chosen_label]
                    delete_record(chosen_id)
                    st.success(f"Deleted Record #{chosen_id}")
                    st.rerun()
            with del_c3:
                if st.button("⚠️ Clear All Records", use_container_width=True):
                    clear_all_records()
                    st.warning("All records deleted.")
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# TAB 3: RESEARCH EVIDENCE (FOR EXAMINERS)
# ====================================================
elif nav == "Research Evidence":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="med-card-header">Clinical Evidence & Scientific Methodology</div>', unsafe_allow_html=True)
    st.markdown("""
    This section documents the causal inference methodology, structural causal model, and comparative
    empirical evaluations supporting the Virtual Biopsy decision framework.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="med-card-header">Structural Causal Model (DAG)</div>', unsafe_allow_html=True)
    st.markdown("""
    Using Judea Pearl's Structural Causal Models via Microsoft DoWhy, we estimated the Average Treatment Effect (ATE)
    of all 20 non-leaky features to separate true causal drivers from misleading downstream symptoms.
    """)
    if os.path.exists(CAUSAL_GRAPH_IMG):
        st.image(CAUSAL_GRAPH_IMG, caption="Directed Acyclic Graph: Root Causes → Kidney Damage → Downstream Symptoms")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="med-card-header">Feature Evidence Matrix: Correlation vs Causal Effect</div>', unsafe_allow_html=True)
    if not causal_df.empty:
        cdf = causal_df.copy()
        def role(r):
            if abs(r['Causal_Effect']) > 0.01:
                return "Root Cause" if r['Treatable'] == "✅ Yes" else "Demographic"
            return "Downstream Symptom"
        cdf['Role'] = cdf.apply(role, axis=1)
        st.dataframe(cdf[['Feature','Category','Correlation','Causal_Effect','Role','Intervention']],
                     use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="med-card-header">RQ3: Non-Causal vs Causal-Only Model Comparison</div>', unsafe_allow_html=True)
    rq3_csv = os.path.join(BASE_PATH, 'outputs/results/rq3_causal_vs_noncausal.csv')
    if os.path.exists(rq3_csv):
        st.dataframe(pd.read_csv(rq3_csv), use_container_width=True, hide_index=True)
        st.caption("McNemar's Test: χ² = 339.43, p = 8.47e-76 (statistically significant)")
    rq3_fig = os.path.join(BASE_PATH, 'outputs/figures/rq3_causal_vs_noncausal_comparison.png')
    if os.path.exists(rq3_fig):
        st.image(rq3_fig, caption="Confusion Matrices: Observational vs Causal-Only Classifiers")
    st.markdown('</div>', unsafe_allow_html=True)
