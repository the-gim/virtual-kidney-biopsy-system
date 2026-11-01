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
    layout="wide",
    initial_sidebar_state="auto"
)

# --- PROFESSIONAL INSTITUTIONAL NAVY BLUE CSS ---
st.markdown("""
<style>
    /* Professional Institutional Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Palette */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0a192f !important;
    }

    /* Unobtrusive Clean Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.4rem !important;
        z-index: 99 !important;
    }
    #MainMenu, footer { visibility: hidden; }
    div[data-testid="stDecoration"], div[data-testid="stStatusWidget"], .stDeployButton { display: none; }

    /* Attached Sidebar Toggle Buttons (Clean Corporate Styling) */
    [data-testid="stSidebarCollapseButton"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        color: #0f2744 !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        background: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f2744 !important;
    }

    /* Dedicated Left-Attached Open Button when Sidebar is Closed */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: #0f2744 !important;
        border: 1px solid #0f2744 !important;
        border-radius: 0 6px 6px 0 !important;
        color: #ffffff !important;
        box-shadow: 2px 2px 6px rgba(15, 39, 68, 0.2) !important;
        transition: all 0.15s ease-in-out !important;
        position: fixed !important;
        top: 14px !important;
        left: 0 !important;
        z-index: 999999 !important;
        padding: 5px 8px !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: #1e3a5f !important;
        border-color: #1e3a5f !important;
        color: #ffffff !important;
        padding-left: 10px !important;
    }

    /* Layout: Stretches Full Width Automatically When Sidebar Collapses */
    .block-container {
        padding-top: 0.4rem !important;
        padding-bottom: 0.8rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Minimal Institutional Navy Sidebar */
    section[data-testid="stSidebar"] {
        width: 290px !important;
        min-width: 260px !important;
        max-width: 310px !important;
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 1px 0 4px rgba(0,0,0,0.02) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.8rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    /* Top Application Header */
    .top-app-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 16px;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-tag {
        background: #0f2744;
        color: #ffffff;
        font-weight: 800;
        font-size: 0.8rem;
        padding: 5px 9px;
        border-radius: 5px;
        letter-spacing: 0.05em;
    }
    .brand-text-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f2744;
        line-height: 1.2;
    }
    .brand-text-sub {
        font-size: 0.72rem;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.02em;
    }
    .system-status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #0f2744;
        padding: 4px 9px;
        border-radius: 5px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background: #166534;
        border-radius: 50%;
    }

    /* Professional Rounded Tab Selection (No Cutoff, No Scroll Chevron) */
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
        background-color: #f1f5f9 !important;
        padding: 3px 4px !important;
        border-radius: 7px !important;
        border: 1px solid #e2e8f0 !important;
        gap: 4px !important;
        margin-bottom: 10px !important;
        overflow-x: visible !important;
        flex-wrap: nowrap !important;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] [data-baseweb="tab-border"],
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] button[aria-label="Previous slide"],
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] button[aria-label="Next slide"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
        border-radius: 5px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 6px 12px !important;
        color: #475569 !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        margin: 0 !important;
        height: auto !important;
        white-space: nowrap !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
        color: #0f2744 !important;
        background-color: rgba(255, 255, 255, 0.7) !important;
    }
    div[data-testid="stTabs"] > div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f2744 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 1px 2px rgba(15, 39, 68, 0.06) !important;
    }

    /* Nested Sub-tabs Styling: Equal 3-way distribution across container */
    div[data-testid="stTabs"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        padding: 3px !important;
        border-radius: 6px !important;
        margin-bottom: 8px !important;
        display: flex !important;
        width: 100% !important;
    }
    div[data-testid="stTabs"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
        flex: 1 1 0 !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 6px 4px !important;
        font-size: 0.76rem !important;
    }

    /* Column Spacing & Zero-Cutoff Layout */
    div[data-testid="stHorizontalBlock"] {
        gap: 12px !important;
        align-items: stretch !important;
    }
    div[data-testid="column"] {
        min-width: 0 !important;
    }

    /* Custom Bordered Containers (Cards) */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        margin-bottom: 6px !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    /* Section Card Header */
    .card-title-sm {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 4px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 5px;
    }

    /* Patient Identity Strip */
    .patient-strip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 6px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.76rem;
    }
    .patient-strip-item {
        color: #64748b;
    }
    .patient-strip-item strong {
        color: #0f2744;
        margin-left: 4px;
    }

    /* Damage Badges */
    .damage-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.74rem;
        letter-spacing: 0.04em;
    }
    .pill-0 { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .pill-1 { background: #fefce8; color: #854d0e; border: 1px solid #fef08a; }
    .pill-2 { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
    .pill-3 { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

    /* Clinical Guidance Action Boxes */
    .action-box {
        border-radius: 6px;
        padding: 9px 12px;
        margin-bottom: 8px;
        font-size: 0.79rem;
    }
    .action-disease {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0f2744;
    }
    .action-disease .action-header { color: #0f2744; font-weight: 700; margin-bottom: 3px; font-size: 0.81rem; }
    
    .action-support {
        background: #fefce8;
        border: 1px solid #fef08a;
        border-left: 4px solid #854d0e;
    }
    .action-support .action-header { color: #854d0e; font-weight: 700; margin-bottom: 3px; font-size: 0.81rem; }
    
    .action-warning {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #991b1b;
    }
    .action-warning .action-header { color: #991b1b; font-weight: 700; margin-bottom: 3px; font-size: 0.81rem; }

    /* Counterfactual Simulator Card */
    .sim-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 11px 13px;
    }

    /* Primary Navy Button Override */
    div.stButton > button[kind="primary"] {
        background-color: #0f2744 !important;
        color: #ffffff !important;
        border: 1px solid #0f2744 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.45rem 0.9rem !important;
        transition: background-color 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1e3a5f !important;
        border-color: #1e3a5f !important;
        color: #ffffff !important;
    }

    /* Secondary Buttons */
    div.stButton > button {
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
    }
    div.stButton > button:hover {
        border-color: #94a3b8 !important;
        background-color: #f8fafc !important;
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
    "New Patient (Custom Entry)": {
        'id': 'PAT-NEW', 'name': '', 'age': 55, 'gender': 'Male',
        'htn': False, 'glucose_min': 95.0, 'avg_los': 3.0,
        'phosphate_mean': 3.5, 'phosphate_min': 3.0, 'phosphate_max': 4.0, 'phosphate_std': 0.3,
        'potassium_mean': 4.2, 'potassium_max': 4.5, 'potassium_std': 0.2, 'hyperkalemia': False, 'sodium_std': 1.5,
        'bicarbonate_mean': 24.0, 'bicarbonate_min': 22.0, 'bicarbonate_std': 1.2, 'calcium_min': 9.2,
        'albumin_mean': 4.0, 'albumin_min': 3.8, 'low_albumin': False,
        'notes': ''
    },
    "Case 1: Eleanor Vance (Diabetic Nephropathy)": {
        'id': 'PAT-9021', 'name': 'Eleanor Vance', 'age': 67, 'gender': 'Female',
        'htn': True, 'glucose_min': 185.0, 'avg_los': 14.0,
        'phosphate_mean': 6.2, 'phosphate_min': 5.4, 'phosphate_max': 7.4, 'phosphate_std': 1.2,
        'potassium_mean': 5.6, 'potassium_max': 6.2, 'potassium_std': 0.8, 'hyperkalemia': True, 'sodium_std': 3.2,
        'bicarbonate_mean': 16.0, 'bicarbonate_min': 13.0, 'bicarbonate_std': 3.5, 'calcium_min': 7.1,
        'albumin_mean': 2.4, 'albumin_min': 2.0, 'low_albumin': True,
        'notes': 'Type 2 DM 15 yrs, severe HTN, peripheral edema.'
    },
    "Case 2: Robert Sterling (Hypertensive Nephrosclerosis)": {
        'id': 'PAT-5821', 'name': 'Robert Sterling', 'age': 59, 'gender': 'Male',
        'htn': True, 'glucose_min': 110.0, 'avg_los': 7.0,
        'phosphate_mean': 4.8, 'phosphate_min': 4.2, 'phosphate_max': 5.6, 'phosphate_std': 0.7,
        'potassium_mean': 5.1, 'potassium_max': 5.6, 'potassium_std': 0.5, 'hyperkalemia': True, 'sodium_std': 2.5,
        'bicarbonate_mean': 21.0, 'bicarbonate_min': 18.5, 'bicarbonate_std': 2.2, 'calcium_min': 7.8,
        'albumin_mean': 3.1, 'albumin_min': 2.7, 'low_albumin': True,
        'notes': 'HTN 12 yrs, intermittent medication compliance.'
    },
    "Case 3: Sarah Jenkins (Early Tubular Stress)": {
        'id': 'PAT-8910', 'name': 'Sarah Jenkins', 'age': 44, 'gender': 'Female',
        'htn': False, 'glucose_min': 140.0, 'avg_los': 5.0,
        'phosphate_mean': 3.9, 'phosphate_min': 3.4, 'phosphate_max': 4.3, 'phosphate_std': 0.4,
        'potassium_mean': 4.4, 'potassium_max': 4.8, 'potassium_std': 0.3, 'hyperkalemia': False, 'sodium_std': 1.8,
        'bicarbonate_mean': 23.0, 'bicarbonate_min': 21.0, 'bicarbonate_std': 1.6, 'calcium_min': 8.5,
        'albumin_mean': 3.5, 'albumin_min': 3.2, 'low_albumin': False,
        'notes': 'Pre-diabetes, mild microalbuminuria.'
    },
    "Case 4: Arthur Pendelton (Healthy Baseline)": {
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
    r = 46
    cx, cy = 56, 56
    circumference = 2 * math.pi * r
    offset = circumference * (1 - percent / 100)
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;margin:0;width:100%;">
    <svg width="112" height="112" viewBox="0 0 112 112" style="max-width:100%;height:auto;">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e2e8f0" stroke-width="8"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="8"
            stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
            stroke-linecap="round" transform="rotate(-90 {cx} {cy})"
            style="transition: stroke-dashoffset 0.8s ease;"/>
        <text x="{cx}" y="{cy-2}" text-anchor="middle" font-size="20" font-weight="700" fill="#0f2744">{percent:.1f}%</text>
        <text x="{cx}" y="{cy+12}" text-anchor="middle" font-size="8.5" fill="#64748b" font-weight="600" letter-spacing="0.08em">RISK SCORE</text>
    </svg>
    </div>
    """

# ====================================================
# MINIMAL INSTITUTIONAL SIDEBAR
# ====================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #f1f5f9;">
        <div style="font-weight:700;font-size:0.86rem;color:#0f2744;letter-spacing:0.02em;">Patient Cohort Selection</div>
        <span style="font-size:0.65rem;background:#f8fafc;color:#0f2744;padding:2px 6px;border-radius:4px;font-weight:600;border:1px solid #e2e8f0;">Active</span>
    </div>
    """, unsafe_allow_html=True)

    preset_keys = list(PRESETS.keys())
    default_preset = st.session_state.get('selected_preset', preset_keys[1])
    default_idx = preset_keys.index(default_preset) if default_preset in preset_keys else 0

    def _sync_preset():
        st.session_state['selected_preset'] = st.session_state['preset_selectbox']

    preset_choice = st.selectbox(
        "Clinical Profile",
        preset_keys,
        index=default_idx,
        key="preset_selectbox",
        label_visibility="collapsed",
        on_change=_sync_preset
    )
    p = PRESETS[preset_choice]

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("New Patient", use_container_width=True):
            st.session_state['selected_preset'] = "New Patient (Custom Entry)"
            st.rerun()
    with c_btn2:
        if st.button("Load Cohort", use_container_width=True):
            seed_sample_patients()
            st.toast("Sample cohort loaded into registry.")

    st.markdown("<hr style='margin:8px 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Demographics</div>', unsafe_allow_html=True)

    sb_id = st.text_input("Patient ID / MRN", value=p['id'] if p and p['id'] != 'PAT-NEW' else "PAT-0042")
    sb_name = st.text_input("Full Name", value=p['name'] if p else "", placeholder="e.g. Eleanor Vance")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        sb_age = st.number_input("Age", min_value=18, max_value=99, value=int(p['age']) if p else 55)
    with col_a2:
        sb_gender = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if (p and p['gender'] == 'Male') else 1)

    with st.expander("Hospital Stay & Clinical Notes", expanded=False):
        sb_los = st.number_input("Stay Duration (Days)", min_value=1.0, max_value=60.0, value=float(p['avg_los']) if p else 4.0, step=1.0)
        sb_notes = st.text_area("Encounter Notes", value=p['notes'] if p else "", height=60)

    if 'sb_los' not in locals():
        sb_los = float(p['avg_los']) if p else 4.0
    if 'sb_notes' not in locals():
        sb_notes = p['notes'] if p else ""

    st.markdown("""
    <div style="margin-top:14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:6px 10px;font-size:0.68rem;color:#475569;line-height:1.5;">
        <div style="font-weight:700;color:#0f2744;margin-bottom:2px;">System Infrastructure</div>
        <div>Model: XGBoost (F1: 0.7732)</div>
        <div>Inference: Pearl SCM / DoWhy</div>
        <div>Storage: SQLite Encrypted</div>
    </div>
    """, unsafe_allow_html=True)

# ====================================================
# TOP BRAND BAR
# ====================================================
st.markdown("""
<div class="top-app-header">
    <div class="brand-group">
        <div class="brand-tag">VKB</div>
        <div>
            <div class="brand-text-title">Virtual Kidney Biopsy Portal</div>
            <div class="brand-text-sub">Clinical Decision Support System &bull; Histopathology & Causal Risk Estimation</div>
        </div>
    </div>
    <div class="system-status-indicator">
        <span class="status-dot"></span>
        Operational &bull; KDIGO Guidelines
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================================
# TOP WORKSPACE TABS
# ====================================================
tab_biopsy, tab_registry, tab_research = st.tabs([
    "Virtual Biopsy Assessment",
    "Patient Registry",
    "Scientific Evidence"
])

# ====================================================
# TAB 1: VIRTUAL BIOPSY ASSESSMENT
# ====================================================
with tab_biopsy:
    # Tight gap to prevent canyon and avoid cutoffs
    col_input, col_results = st.columns([1.0, 1.08], gap="small")

    # ------------------ LEFT COLUMN: INPUT WORKSPACE ------------------
    with col_input:
        with st.container(border=True):
            st.markdown('<div class="card-title-sm"><span>Clinical Biomarker Intake</span><span>MIMIC-IV Normalized</span></div>', unsafe_allow_html=True)

            # Short concise titles to prevent overflow and scroll chevrons
            input_subtab1, input_subtab2, input_subtab3 = st.tabs([
                "Routine Labs",
                "Root Causes",
                "Feature Details"
            ])

            with input_subtab1:
                st.caption("Standard basic metabolic and electrolyte panel:")
                l_col1, l_col2 = st.columns(2)
                with l_col1:
                    phosphate = st.number_input("Phosphate (mg/dL)", min_value=1.0, max_value=15.0, value=float(p['phosphate_mean']) if p else 3.5, step=0.1)
                    potassium = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=9.0, value=float(p['potassium_mean']) if p else 4.3, step=0.1)
                    albumin = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=float(p['albumin_mean']) if p else 3.8, step=0.1)
                with l_col2:
                    bicarbonate = st.number_input("Bicarbonate (mEq/L)", min_value=5.0, max_value=45.0, value=float(p['bicarbonate_mean']) if p else 24.0, step=0.5)
                    calcium = st.number_input("Calcium (mg/dL)", min_value=3.0, max_value=15.0, value=float(p['calcium_min']) if p else 8.8, step=0.1)
                    sodium = st.number_input("Sodium (mEq/L)", min_value=110.0, max_value=160.0, value=140.0, step=1.0)

            with input_subtab2:
                st.caption("Causal root causes verified through Structural Causal Modeling:")
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    has_htn = st.toggle("Documented Hypertension", value=p['htn'] if p else False, help="Root Cause: ATE = -0.0828 (p < 0.001)")
                with e_col2:
                    fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=40.0, max_value=500.0,
                                                      value=float(p['glucose_min']) if p else 100.0, step=5.0,
                                                      help="Root Cause: ATE = -0.0483 (p < 0.001)")

                st.markdown("""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;margin-top:8px;font-size:0.73rem;color:#475569;">
                    <strong>Physiological Rationale:</strong> Systemic arterial hypertension and glycemic dysregulation exert mechanical shear and biochemical injury on glomerular basement membranes, initiating permanent glomerulosclerosis.
                </div>
                """, unsafe_allow_html=True)

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

            with input_subtab3:
                st.markdown(f"""
                <div style="font-size:0.74rem;color:#334155;line-height:1.65;">
                    <div>Phosphate: Min <code>{auto_ph_min:.2f}</code> | Max <code>{auto_ph_max:.2f}</code> | Std <code>{auto_ph_std:.2f}</code></div>
                    <div>Potassium: Max <code>{auto_k_max:.2f}</code> | Std <code>{auto_k_std:.2f}</code> | Hyperkalemia: <code>{'Yes' if auto_hyperk else 'No'}</code></div>
                    <div>Acid-Base: HCO3 Min <code>{auto_bc_min:.1f}</code> | Std <code>{auto_bc_std:.2f}</code> | Ca Min <code>{calcium:.1f}</code></div>
                    <div>Albumin: Min <code>{auto_alb_min:.2f}</code> | Low Albumin: <code>{'Yes' if auto_low_alb else 'No'}</code></div>
                    <div>Sodium: Spread Std <code>{auto_na_std:.2f}</code></div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Manual Feature Overrides (Optional)", expanded=False):
                    ov_c1, ov_c2 = st.columns(2)
                    with ov_c1:
                        ov_ph_min = st.number_input("Override PO4 Min", value=auto_ph_min, step=0.1)
                        ov_ph_max = st.number_input("Override PO4 Max", value=auto_ph_max, step=0.1)
                        ov_ph_std = st.number_input("Override PO4 Std", value=auto_ph_std, step=0.05)
                        ov_k_max = st.number_input("Override K Max", value=auto_k_max, step=0.1)
                    with ov_c2:
                        ov_k_std = st.number_input("Override K Std", value=auto_k_std, step=0.05)
                        ov_bc_min = st.number_input("Override HCO3 Min", value=auto_bc_min, step=0.5)
                        ov_bc_std = st.number_input("Override HCO3 Std", value=auto_bc_std, step=0.1)
                        ov_alb_min = st.number_input("Override Alb Min", value=auto_alb_min, step=0.1)
            
            eff_ph_min = ov_ph_min if 'ov_ph_min' in locals() else auto_ph_min
            eff_ph_max = ov_ph_max if 'ov_ph_max' in locals() else auto_ph_max
            eff_ph_std = ov_ph_std if 'ov_ph_std' in locals() else auto_ph_std
            eff_k_max = ov_k_max if 'ov_k_max' in locals() else auto_k_max
            eff_k_std = ov_k_std if 'ov_k_std' in locals() else auto_k_std
            eff_bc_min = ov_bc_min if 'ov_bc_min' in locals() else auto_bc_min
            eff_bc_std = ov_bc_std if 'ov_bc_std' in locals() else auto_bc_std
            eff_alb_min = ov_alb_min if 'ov_alb_min' in locals() else auto_alb_min

            st.markdown("<hr style='margin:10px 0 8px 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)
            run_btn = st.button("Run Virtual Biopsy & Save Record", type="primary", use_container_width=True)

    # ------------------ RIGHT COLUMN: DIAGNOSTIC & SIMULATION WORKSPACE ------------------
    with col_results:
        with st.container(border=True):
            raw = {
                'gender_encoded': 1.0 if sb_gender == "Male" else 0.0,
                'has_hypertension': 1.0 if has_htn else 0.0,
                'albumin_mean': float(albumin),
                'albumin_min': float(eff_alb_min),
                'potassium_mean': float(potassium),
                'potassium_max': float(eff_k_max),
                'potassium_std': float(eff_k_std),
                'sodium_std': float(auto_na_std),
                'calcium_min': float(calcium),
                'phosphate_mean': float(phosphate),
                'phosphate_min': float(eff_ph_min),
                'phosphate_max': float(eff_ph_max),
                'phosphate_std': float(eff_ph_std),
                'glucose_min': float(fasting_glucose),
                'bicarbonate_mean': float(bicarbonate),
                'bicarbonate_min': float(eff_bc_min),
                'bicarbonate_std': float(eff_bc_std),
                'low_albumin': 1.0 if auto_low_alb else 0.0,
                'hyperkalemia': 1.0 if auto_hyperk else 0.0,
                'avg_los': float(sb_los)
            }

            scaled = {}
            for feat in FEATURE_ORDER:
                mu = scaler_params[feat]['mean']
                sigma = scaler_params[feat]['scale']
                scaled[feat] = (raw[feat] - mu) / sigma

            input_df = pd.DataFrame([scaled])[FEATURE_ORDER]
            risk = float(model.predict_proba(input_df)[0][1])

            # Damage Level Classification (Standard Restrained Clinical Colors)
            if risk <= 0.25:
                dlvl, dtitle, dbadge, dcolor = 0, "Preserved Architecture", "pill-0", "#166534"
                ifta = "< 10% IFTA"
                kdigo_grade = "KDIGO Stage G1 (eGFR >= 90)"
                micro = "Intact glomerular capillary loops with uniform basement membranes. Minimal mesangial expansion. Proximal tubules show preserved brush borders without interstitial fibrosis."
            elif risk <= 0.50:
                dlvl, dtitle, dbadge, dcolor = 1, "Mild Chronic Injury", "pill-1", "#854d0e"
                ifta = "10 – 25% IFTA"
                kdigo_grade = "KDIGO Stage G2 (eGFR 60 - 89)"
                micro = "Focal glomerular basement membrane thickening with early mesangial expansion. Scattered tubular vacuolization with early focal interstitial mononuclear infiltrates."
            elif risk <= 0.75:
                dlvl, dtitle, dbadge, dcolor = 2, "Severe Tissue Damage", "pill-2", "#9a3412"
                ifta = "26 – 50% IFTA"
                kdigo_grade = "KDIGO Stage G3a/b (eGFR 30 - 59)"
                micro = "Segmental glomerulosclerosis involving >25% of glomeruli. Widespread tubular atrophy with thyroidization. Moderate interstitial fibrosis and arteriolar hyalinosis."
            else:
                dlvl, dtitle, dbadge, dcolor = 3, "Critical Failure", "pill-3", "#991b1b"
                ifta = "> 50% IFTA"
                kdigo_grade = "KDIGO Stage G4/G5 (eGFR < 30)"
                micro = "Global glomerulosclerosis in >50% of glomeruli. Severe diffuse tubular drop-out with extensive interstitial fibrosis and fibrointimal vascular obliteration."

            p_display_name = sb_name.strip() if sb_name.strip() else "Anonymous Patient"
            p_display_id = sb_id.strip() if sb_id.strip() else "PAT-0042"
            st.markdown(f"""
            <div class="patient-strip">
                <span class="patient-strip-item">MRN:<strong>{p_display_id}</strong></span>
                <span class="patient-strip-item">Patient:<strong>{p_display_name}</strong></span>
                <span class="patient-strip-item">Demographics:<strong>{sb_age}y &bull; {sb_gender}</strong></span>
                <span class="patient-strip-item">Encounter:<strong>{datetime.now().strftime('%b %d, %Y')}</strong></span>
            </div>
            """, unsafe_allow_html=True)

            res_tab1, res_tab2, res_tab3 = st.tabs([
                "Histopathology",
                "Treatment Plan",
                "What-If Simulator"
            ])

            with res_tab1:
                r_top1, r_top2 = st.columns([1.7, 1.0], gap="small")
                with r_top1:
                    st.markdown(f'<div style="margin-top:2px;"><span class="damage-pill {dbadge}">DAMAGE LEVEL {dlvl}</span></div>', unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:1.15rem;font-weight:700;color:#0f2744;margin:4px 0 2px 0;'>{dtitle}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color:#64748b;font-size:0.8rem;margin-bottom:2px;'>Chronic Scarring: <strong style=\"color:{dcolor}\">{ifta}</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color:#475569;font-size:0.75rem;'>Staging Equivalent: <strong>{kdigo_grade}</strong></div>", unsafe_allow_html=True)
                with r_top2:
                    st.markdown(make_gauge_svg(risk * 100, dcolor), unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:8px 12px;margin:6px 0 8px 0;">
                    <div style="font-weight:700;color:#0f2744;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;">
                        Microscopic Tissue Pathology Summary
                    </div>
                    <div style="color:#334155;font-size:0.78rem;line-height:1.45;">{micro}</div>
                </div>
                """, unsafe_allow_html=True)

                # Diagnostic Biomarker Findings Grid (Responsive Flexbox, Zero Cutoff)
                st.markdown('<div class="card-title-sm" style="margin-top:6px;"><span>Key Biomarker Status & Staging Correlates</span><span>Normal Range Comparison</span></div>', unsafe_allow_html=True)
                
                po4_status = "Elevated" if phosphate > 4.5 else ("Low" if phosphate < 2.5 else "Normal")
                po4_color = "#991b1b" if phosphate > 4.5 else "#166534"
                k_status = "Hyperkalemic" if potassium > 5.0 else ("Hypokalemic" if potassium < 3.5 else "Normal")
                k_color = "#991b1b" if potassium > 5.0 else "#166534"
                hco3_status = "Acidosis" if bicarbonate < 22.0 else "Normal"
                hco3_color = "#991b1b" if bicarbonate < 22.0 else "#166534"
                alb_status = "Depleted" if albumin < 3.5 else "Normal"
                alb_color = "#9a3412" if albumin < 3.5 else "#166534"

                st.markdown(f"""
                <div style="display:flex;gap:6px;margin-top:2px;width:100%;box-sizing:border-box;">
                    <div style="flex:1 1 0;min-width:0;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:6px 2px;text-align:center;">
                        <div style="font-size:0.60rem;color:#64748b;text-transform:uppercase;font-weight:600;letter-spacing:0.02em;">Phosphate</div>
                        <div style="font-size:0.86rem;font-weight:700;color:#0f2744;line-height:1.2;">{phosphate:.1f}</div>
                        <div style="font-size:0.62rem;font-weight:600;color:{po4_color};line-height:1.2;">{po4_status}</div>
                    </div>
                    <div style="flex:1 1 0;min-width:0;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:6px 2px;text-align:center;">
                        <div style="font-size:0.60rem;color:#64748b;text-transform:uppercase;font-weight:600;letter-spacing:0.02em;">Potassium</div>
                        <div style="font-size:0.86rem;font-weight:700;color:#0f2744;line-height:1.2;">{potassium:.1f}</div>
                        <div style="font-size:0.62rem;font-weight:600;color:{k_color};line-height:1.2;">{k_status}</div>
                    </div>
                    <div style="flex:1 1 0;min-width:0;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:6px 2px;text-align:center;">
                        <div style="font-size:0.60rem;color:#64748b;text-transform:uppercase;font-weight:600;letter-spacing:0.02em;">Bicarbonate</div>
                        <div style="font-size:0.86rem;font-weight:700;color:#0f2744;line-height:1.2;">{bicarbonate:.1f}</div>
                        <div style="font-size:0.62rem;font-weight:600;color:{hco3_color};line-height:1.2;">{hco3_status}</div>
                    </div>
                    <div style="flex:1 1 0;min-width:0;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:6px 2px;text-align:center;">
                        <div style="font-size:0.60rem;color:#64748b;text-transform:uppercase;font-weight:600;letter-spacing:0.02em;">Albumin</div>
                        <div style="font-size:0.86rem;font-weight:700;color:#0f2744;line-height:1.2;">{albumin:.1f}</div>
                        <div style="font-size:0.62rem;font-weight:600;color:{alb_color};line-height:1.2;">{alb_status}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_tab2:
                disease_items = []
                if has_htn:
                    disease_items.append("Initiate RAAS blockade (ACE inhibitor or ARB) — target BP < 130/80 mmHg")
                if fasting_glucose > 130.0:
                    disease_items.append(f"Optimize glycemic control (SGLT2i / Metformin) — current glucose: {fasting_glucose:.0f} mg/dL")
                if not disease_items:
                    disease_items.append("No active modifiable root causes identified. Maintain routine longitudinal surveillance.")
                primary_plan = "; ".join(disease_items)

                supp_items = []
                if phosphate > 4.5:
                    supp_items.append(f"Phosphate elevated ({phosphate:.1f} mg/dL) — prescribe dietary phosphate binders")
                if potassium > 5.0:
                    supp_items.append(f"Potassium elevated ({potassium:.1f} mEq/L) — initiate dietary restriction or binding resins")
                if bicarbonate < 22.0:
                    supp_items.append(f"Metabolic acidosis (HCO3: {bicarbonate:.1f}) — prescribe sodium bicarbonate tablets")
                if not supp_items:
                    supp_items.append("Electrolyte & mineral markers within normal limits.")

                d_html = "".join(f"<li>{it}</li>" for it in disease_items)
                s_html = "".join(f"<li>{it}</li>" for it in supp_items)

                st.markdown(f"""
                <div class="action-box action-disease">
                    <div class="action-header">Disease-Modifying Targets (Etiologic Root Causes)</div>
                    <ul style="margin:2px 0 0 0;padding-left:18px;line-height:1.45;">{d_html}</ul>
                </div>
                <div class="action-box action-support">
                    <div class="action-header">Supportive Care Only (Non-Causal Symptoms)</div>
                    <ul style="margin:2px 0 0 0;padding-left:18px;line-height:1.45;">{s_html}</ul>
                </div>
                <div class="action-box action-warning">
                    <div class="action-header">Clinical Prescribing Advisory</div>
                    Elevated phosphate and potassium are passive consequences of decreased nephron clearance. Lowering them resolves acute lab numbers but does not slow renal scarring. Interventional priority must remain on BP and glucose.
                </div>
                """, unsafe_allow_html=True)

            with res_tab3:
                st.caption("Simulate counterfactual therapeutic outcomes ($Y_{X \\leftarrow x'}$):")
                sim_c1, sim_c2 = st.columns(2)
                with sim_c1:
                    sim_bp = st.checkbox("Achieve Target BP (< 130/80)", value=has_htn)
                with sim_c2:
                    sim_glu = st.checkbox("Normalize Fasting Glucose (< 110)", value=(fasting_glucose > 130.0))

                reduction = 0.0
                if has_htn and sim_bp:
                    reduction += 0.0828
                if fasting_glucose > 130.0 and sim_glu:
                    reduction += 0.0483

                projected = max(0.02, risk - reduction)
                if projected <= 0.25:
                    p_label, p_color = "Level 0 (Preserved)", "#166534"
                elif projected <= 0.50:
                    p_label, p_color = "Level 1 (Mild)", "#854d0e"
                elif projected <= 0.75:
                    p_label, p_color = "Level 2 (Severe)", "#9a3412"
                else:
                    p_label, p_color = "Level 3 (Critical)", "#991b1b"

                st.markdown(f"""
                <div class="sim-box">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <span style="font-size:0.75rem;color:#475569;">Absolute Risk Reduction:</span>
                        <span style="font-size:1.05rem;font-weight:700;color:#166534;">-{reduction*100:.1f}%</span>
                    </div>
                    <div style="font-size:0.7rem;color:#64748b;margin-bottom:2px;">Baseline Risk: {risk*100:.1f}%</div>
                    <div style="background:#e2e8f0;border-radius:4px;height:10px;overflow:hidden;margin-bottom:6px;">
                        <div style="width:{risk*100:.0f}%;height:100%;background:{dcolor};"></div>
                    </div>
                    <div style="font-size:0.7rem;color:#64748b;margin-bottom:2px;">Projected Post-Intervention: {projected*100:.1f}%</div>
                    <div style="background:#e2e8f0;border-radius:4px;height:10px;overflow:hidden;margin-bottom:8px;">
                        <div style="width:{projected*100:.0f}%;height:100%;background:{p_color};"></div>
                    </div>
                    <div style="font-size:0.78rem;color:#0f2744;font-weight:600;">
                        Projected Histological Outcome: <span style="color:{p_color};">{p_label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if run_btn:
            record = {
                'patient_id': p_display_id,
                'patient_name': p_display_name,
                'age': sb_age,
                'gender': sb_gender,
                'damage_level': dlvl,
                'damage_title': f"Damage Level {dlvl}: {dtitle}",
                'risk_probability': risk,
                'has_hypertension': 1 if has_htn else 0,
                'glucose_min': fasting_glucose,
                'phosphate_mean': phosphate,
                'potassium_mean': potassium,
                'bicarbonate_min': bicarbonate,
                'albumin_min': albumin,
                'primary_intervention': primary_plan,
                'causal_warning': "Treatment priority must remain on BP and glucose control.",
                'counterfactual_reduction': reduction * 100,
                'notes': sb_notes
            }
            rid = save_patient_record(record)
            st.success(f"Biopsy record #{rid} for {p_display_name} ({p_display_id}) successfully recorded in the registry.")

# ====================================================
# TAB 2: PATIENT REGISTRY
# ====================================================
with tab_registry:
    with st.container(border=True):
        st.markdown('<div class="card-title-sm"><span>Longitudinal Patient Encounters Database</span><span>SQLite Persistent</span></div>', unsafe_allow_html=True)

        df_records = get_all_records()
        if len(df_records) == 0:
            st.info("No records in the database. Use 'Load Cohort' in the sidebar or run a biopsy.")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: m1.metric("Total Biopsies", len(df_records))
            with m2: m2.metric("Level 0 (Normal)", len(df_records[df_records['damage_level']==0]))
            with m3: m3.metric("Level 1 (Mild)", len(df_records[df_records['damage_level']==1]))
            with m4: m4.metric("Level 2 (Severe)", len(df_records[df_records['damage_level']==2]))
            with m5: m5.metric("Level 3 (Failure)", len(df_records[df_records['damage_level']==3]))

            st.markdown("<hr style='margin:8px 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)
            f_col1, f_col2, f_col3 = st.columns([1.5, 2.5, 1])
            with f_col1:
                lvl_filter = st.selectbox("Filter Damage Grade", ["All", "Level 0", "Level 1", "Level 2", "Level 3"])
            with f_col2:
                search_query = st.text_input("Search MRN or Patient Name", placeholder="Type name or ID...")
            with f_col3:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                if st.button("Refresh Data", use_container_width=True):
                    st.rerun()

            filt = df_records.copy()
            if lvl_filter != "All":
                lvl_val = int(lvl_filter.split(" ")[1])
                filt = filt[filt['damage_level'] == lvl_val]
            if search_query:
                filt = filt[
                    filt['patient_id'].str.contains(search_query, case=False, na=False) |
                    filt['patient_name'].str.contains(search_query, case=False, na=False)
                ]

            display_cols = ['id','timestamp','patient_id','patient_name','age','gender',
                            'damage_level','risk_probability','has_hypertension',
                            'glucose_min','phosphate_mean','potassium_mean','primary_intervention']
            grid_data = filt[display_cols].copy()
            grid_data['risk_probability'] = (grid_data['risk_probability']*100).round(1).astype(str)+'%'
            grid_data['has_hypertension'] = grid_data['has_hypertension'].map({1:'Yes',0:'No'})
            grid_data.columns = ['#','Encounter Date','MRN','Name','Age','Sex','Damage','Risk','HTN',
                                 'Glucose','Phosphate','Potassium','Intervention Plan']

            st.dataframe(grid_data, use_container_width=True, hide_index=True)

            down_c, del_c = st.columns([2, 5])
            with down_c:
                csv_bytes = filt.to_csv(index=False).encode('utf-8')
                st.download_button("Export Registry CSV", csv_bytes,
                                   f"patient_biopsy_registry_{datetime.now().strftime('%Y%m%d')}.csv",
                                   "text/csv", use_container_width=True)
            with del_c:
                with st.expander("Manage Individual Records", expanded=False):
                    d1, d2, d3 = st.columns([3, 1.5, 1.5])
                    with d1:
                        opts = {f"Record #{r['id']} - {r['patient_name']} ({r['patient_id']})": r['id'] for _, r in df_records.iterrows()}
                        selected_rec = st.selectbox("Select Record", list(opts.keys()), label_visibility="collapsed")
                    with d2:
                        if st.button("Delete Record", use_container_width=True):
                            delete_record(opts[selected_rec])
                            st.success(f"Deleted record #{opts[selected_rec]}")
                            st.rerun()
                    with d3:
                        if st.button("Clear All Records", use_container_width=True):
                            clear_all_records()
                            st.warning("All records wiped.")
                            st.rerun()

# ====================================================
# TAB 3: SCIENTIFIC EVIDENCE
# ====================================================
with tab_research:
    with st.container(border=True):
        st.markdown('<div class="card-title-sm"><span>Scientific Validation & Causal Inference Dossier</span><span>MSc Research Evidence</span></div>', unsafe_allow_html=True)

        ev_tab1, ev_tab2, ev_tab3 = st.tabs([
            "Structural Causal Model (DAG)",
            "Causal Effects vs Observational Correlation",
            "RQ3: Associational vs Causal AI"
        ])

        with ev_tab1:
            st.markdown("""
            **Judea Pearl's Structural Causal Models (Microsoft DoWhy):**
            By applying the Backdoor Adjustment criterion over observational clinical graphs, we mathematically decoupled true etiologic root causes from downstream symptom biomarkers.
            """)
            if os.path.exists(CAUSAL_GRAPH_IMG):
                st.image(CAUSAL_GRAPH_IMG, caption="Directed Acyclic Graph (DAG): Root Causes → Kidney Damage → Downstream Symptoms", use_container_width=True)

        with ev_tab2:
            st.markdown("Comparison between raw Pearson correlation and true Average Treatment Effect (ATE):")
            if not causal_df.empty:
                cdf_view = causal_df.copy()
                if 'Treatable' in cdf_view.columns:
                    cdf_view['Treatable'] = cdf_view['Treatable'].astype(str).str.replace('\u2705', '', regex=False).str.replace('\u274c', '', regex=False).str.strip()
                def categorize_role(r):
                    if abs(r['Causal_Effect']) > 0.01:
                        return "Root Cause (Actionable)" if "Yes" in str(r.get('Treatable', '')) else "Root Cause (Demographic)"
                    return "Downstream Symptom (Passive)"
                cdf_view['Classification'] = cdf_view.apply(categorize_role, axis=1)
                st.dataframe(cdf_view[['Feature','Category','Correlation','Causal_Effect','Classification','Intervention']],
                             use_container_width=True, hide_index=True)
                st.caption("Key Finding: 17 out of 20 predictive features exhibit ATE = 0.0000 (symptoms that do not halt tissue destruction).")

        with ev_tab3:
            st.markdown("""
            **Research Question 3 Empirical Benchmark:**
            Quantifying the diagnostic vs. therapeutic trade-offs between Model A (20 features, symptoms included) and Model B (3 proven causal root causes).
            """)
            rq3_path = os.path.join(BASE_PATH, 'outputs/results/rq3_causal_vs_noncausal.csv')
            if os.path.exists(rq3_path):
                st.dataframe(pd.read_csv(rq3_path), use_container_width=True, hide_index=True)
                st.caption("McNemar's Paired Significance Test: χ² = 339.43, p = 8.47e-76 (statistically significant difference in recall).")
            rq3_plot = os.path.join(BASE_PATH, 'outputs/figures/rq3_causal_vs_noncausal_comparison.png')
            if os.path.exists(rq3_plot):
                st.image(rq3_plot, caption="Confusion Matrix Benchmark: Model A (Observational) vs Model B (Causal)", use_container_width=True)
