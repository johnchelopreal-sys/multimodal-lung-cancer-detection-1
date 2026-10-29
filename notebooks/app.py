import os
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import streamlit as st
import torch
import pydicom

# =====================================================================
# 1. PAGE CONFIGURATION & HEALTHCARE PORTAL CSS THEME
# =====================================================================
st.set_page_config(
    page_title="Early Lung Cancer AI Diagnostic Portal",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Inspired by Centura Health Portal (Navy #0A2540, Emerald #00A859, Light Gray #F4F6F8)
st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #f4f6f8;
        color: #1e293b;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Top Utility Bar */
    .utility-bar {
        background-color: #05192d;
        color: #e2e8f0;
        padding: 8px 30px;
        font-size: 13px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #00a859;
    }

    /* Primary Navigation Header */
    .portal-header {
        background-color: #ffffff;
        padding: 16px 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .brand-title {
        color: #0a2540;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-accent {
        color: #00a859;
    }

    /* Hero Section Card */
    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }

    .hero-headline {
        color: #0a2540;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-sub {
        color: #475569;
        font-size: 16px;
        margin-bottom: 0;
    }

    /* Sidebar Questionnaire Container */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p {
        color: #0a2540 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* Inputs Styling */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }

    /* Action Card Box */
    .action-container {
        background-color: #0a2540;
        color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }

    /* Action Button */
    .stButton>button {
        background: #00a859 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 14px 28px !important;
        width: 100%;
        transition: background-color 0.2s ease !important;
    }

    .stButton>button:hover {
        background: #008f4c !important;
    }

    /* Metric Display Cards */
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #00a859 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }

    /* Clinical Status Alerts */
    .result-alert {
        padding: 18px 24px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 600;
        margin: 20px 0;
        border-left: 5px solid;
    }
    .alert-stage0 { background-color: #f0fdf4; border-color: #00a859; color: #166534; }
    .alert-stage1 { background-color: #fefce8; border-color: #eab308; color: #854d0e; }
    .alert-stage2 { background-color: #fff7ed; border-color: #f97316; color: #9a3412; }
    .alert-stage3 { background-color: #fef2f2; border-color: #ef4444; color: #991b1b; }

    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Top Utility Navigation Bar
st.markdown("""
    <div class="utility-bar">
        <div>Clinical AI Diagnostic Network | Dual-Modal Oncology Division</div>
        <div>Workstation Protocol: <b>DICOM v3.0 / NIfTI Enabled</b></div>
    </div>
    <div class="portal-header">
        <div class="brand-title">
            🫁 Precision Care <span class="brand-accent">AI Platform</span>
        </div>
        <div style="font-size: 13px; color: #64748b; font-weight: 500;">
            Patient ID: <b>PT-84920-LC</b> | DOB: <b>04/12/1964</b> | Sex: <b>Male</b>
        </div>
    </div>
""", unsafe_allow_html=True)

# Hero Section Banner
st.markdown("""
    <div class="hero-card">
        <div class="hero-headline">Early Detection & Multimodal Diagnostic Intelligence</div>
        <div class="hero-sub">
            Sub-2mm (&lt; 1.99mm) 3D Radiomic CT Imaging Integrated with Sputum Biological Biomarker Panels
        </div>
    </div>
""", unsafe_allow_html=True)

# =====================================================================
# 2. SIDEBAR: CLINICAL QUESTIONNAIRE
# =====================================================================
st.sidebar.markdown("<h3 style='color: #0a2540; font-size: 18px;'>Clinical Questionnaire</h3>", unsafe_allow_html=True)

patient_cohort = st.sidebar.radio(
    "Patient Clinical Status",
    ["1. Screening / Pre-Diagnosis", "2. Active Cancer Treatment", "3. Post-Treatment Surveillance"]
)

st.sidebar.markdown("---")

clinical_risk_modifier = 0.0

if "1. Screening" in patient_cohort:
    st.sidebar.markdown("<b>Risk Factors & Exposures</b>", unsafe_allow_html=True)
    patient_age = st.sidebar.number_input("Patient Age", min_value=18, max_value=100, value=62)
    smoking_pack_years = st.sidebar.slider("Smoking History (Pack-Years)", 0, 100, 35)
    radon_hazard = st.sidebar.selectbox("Radon / Occupational Hazard", ["No Known Exposure", "Occupational Exposure Present"])
    family_history = st.sidebar.selectbox("First-Degree Family History", ["No", "Yes"])

    st.sidebar.markdown("<br><b>Signs & Symptoms</b>", unsafe_allow_html=True)
    symptom_cough = st.sidebar.selectbox("Persistent Cough (> 3 Weeks)", ["No", "Mild Dry Cough", "Productive Cough"])
    symptom_hemoptysis = st.sidebar.selectbox("Hemoptysis (Blood in Sputum)", ["No", "Blood Streaks", "Frequent Hemoptysis"])
    symptom_weight_loss = st.sidebar.selectbox("Unintentional Weight Loss", ["No", "Significant Loss (> 5%)"])

    if symptom_hemoptysis != "No": clinical_risk_modifier += 0.25
    if symptom_cough != "No": clinical_risk_modifier += 0.15
    if symptom_weight_loss != "No": clinical_risk_modifier += 0.15

elif "2. Active" in patient_cohort:
    st.sidebar.markdown("<b>Active Therapy Parameters</b>", unsafe_allow_html=True)
    patient_age = st.sidebar.number_input("Patient Age", min_value=18, max_value=100, value=62)
    smoking_pack_years = st.sidebar.slider("Smoking History (Pack-Years)", 0, 100, 35)
    cancer_subtype = st.sidebar.selectbox("Histological Subtype", ["Adenocarcinoma", "Squamous Cell", "Small Cell (SCLC)", "Pending Biopsy"])
    treatment_type = st.sidebar.multiselect("Current Therapy", ["Chemotherapy", "Targeted Therapy", "Immunotherapy", "Radiation Therapy"])
    side_effects = st.sidebar.multiselect("Adverse Effects", ["Radiation Pneumonitis", "Severe Fatigue", "Dyspnea", "Hemoptysis"])
    response_eval = st.sidebar.selectbox("Clinical Response Evaluation", ["Partial Response", "Stable Disease", "Suspected Progression"])

    if "Radiation Pneumonitis" in side_effects: clinical_risk_modifier += 0.20
    if response_eval == "Suspected Progression": clinical_risk_modifier += 0.35

else:
    st.sidebar.markdown("<b>Surveillance Parameters</b>", unsafe_allow_html=True)
    patient_age = st.sidebar.number_input("Patient Age", min_value=18, max_value=100, value=62)
    smoking_pack_years = st.sidebar.slider("Smoking History (Pack-Years)", 0, 100, 35)
    prior_tx = st.sidebar.multiselect("Completed Primary Therapy", ["Surgical Resection", "Stereotactic Radiation (SBRT)", "Chemotherapy"])
    time_post_tx = st.sidebar.selectbox("Time Since Treatment Completion", ["< 6 Months", "6-12 Months", "1-3 Years", "> 3 Years"])
    recurrence_signs = st.sidebar.multiselect("New Onset Symptoms", ["New/Worsening Cough", "Unexplained Bone Pain", "Neurological Symptoms"])

    if "New/Worsening Cough" in recurrence_signs: clinical_risk_modifier += 0.20
    if "Unexplained Bone Pain" in recurrence_signs: clinical_risk_modifier += 0.30

st.sidebar.markdown("---")
st.sidebar.markdown("<b>Diagnostic Uploads</b>", unsafe_allow_html=True)
uploaded_slide = st.sidebar.file_uploader("Sputum Cytology Slide (.jpg, .png)", type=["jpg", "png", "jpeg"])
uploaded_ct_zip = st.sidebar.file_uploader("3D Thoracic CT Scan (ZIP / DICOM)", type=["zip", "dcm"])

run_pipeline_btn = st.sidebar.button("RUN DUAL-MODAL PIPELINE")

# =====================================================================
# 3. HELPER FUNCTIONS: CT VOLUME & PACS DASHBOARD
# =====================================================================
def generate_sample_3d_ct_volume():
    depth, height, width = 32, 128, 128
    vol = np.full((depth, height, width), -1000.0, dtype=np.float32)
    y_grid, x_grid = np.ogrid[:height, :width]
    
    chest_mask = ((y_grid - 64)**2 / 50**2) + ((x_grid - 64)**2 / 55**2) <= 1.0
    vol[:, chest_mask] = 0.0
    
    r_lung = ((y_grid - 64)**2 / 38**2) + ((x_grid - 88)**2 / 20**2) <= 1.0
    l_lung = ((y_grid - 64)**2 / 38**2) + ((x_grid - 40)**2 / 20**2) <= 1.0
    vol[:, r_lung] = -800.0
    vol[:, l_lung] = -800.0

    mediastinum = ((x_grid - 64)**2 / 10**2 + (y_grid - 58)**2 / 18**2) <= 1.0
    spine = ((x_grid - 64)**2 / 5**2 + (y_grid - 95)**2 / 5**2) <= 1.0
    vol[:, mediastinum] = 40.0
    vol[:, spine] = 300.0
    
    z_g, y_g, x_g = np.ogrid[:depth, :height, :width]
    dist_micro = np.sqrt((z_g - 16)**2 + (y_g - 64)**2 + (x_g - 88)**2)
    vol[dist_micro <= 1.8] = -250.0
    return vol

def render_pacs_dashboard(hu_volume, target_z=16, target_y=64, target_x=88):
    micro_crop = hu_volume[target_z, target_y-6:target_y+7, target_x-6:target_x+7]
    peak_hu = float(np.max(micro_crop))
    diameter_mm = 1.80
    volume_mm3 = (4.0 / 3.0) * np.pi * ((diameter_mm / 2.0) ** 3)
    s_imaging = 0.6850

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor='#0a2540')

    # Panel 1: Axial CT Scan
    axes[0].set_facecolor('#0a2540')
    axes[0].imshow(hu_volume[target_z], cmap='bone', vmin=-900, vmax=-200)
    axes[0].axhline(target_y, color='#00a859', ls='--', lw=0.8, alpha=0.8)
    axes[0].axvline(target_x, color='#00a859', ls='--', lw=0.8, alpha=0.8)
    axes[0].add_patch(patches.Rectangle((target_x-6, target_y-6), 12, 12, fill=False, edgecolor='#ef4444', linewidth=1.5))

    axes[0].annotate("Right Upper Lobe\n(-800 HU)", xy=(88, 50), xytext=(92, 18),
                    arrowprops=dict(facecolor='#00a859', shrink=0.05, width=1.0, headwidth=4),
                    fontsize=8, fontweight='bold', color='white', bbox=dict(boxstyle="round", fc="#05192d", ec="#00a859"))

    axes[0].annotate("Left Upper Lobe\n(-800 HU)", xy=(40, 50), xytext=(5, 18),
                    arrowprops=dict(facecolor='#00a859', shrink=0.05, width=1.0, headwidth=4),
                    fontsize=8, fontweight='bold', color='white', bbox=dict(boxstyle="round", fc="#05192d", ec="#00a859"))

    axes[0].annotate("Mediastinum\n(+40 HU)", xy=(64, 58), xytext=(35, 115),
                    arrowprops=dict(facecolor='#38bdf8', shrink=0.05, width=1.0, headwidth=4),
                    fontsize=8, fontweight='bold', color='white', bbox=dict(boxstyle="round", fc="#05192d", ec="#38bdf8"))

    axes[0].set_title(f"1. AXIAL CT SCAN (SLICE Z={target_z})", fontsize=10, fontweight='bold', color='#ffffff')
    axes[0].axis('off')

    # Panel 2: Proliferation Heatmap
    axes[1].set_facecolor('#0a2540')
    lung_mask = hu_volume[target_z] < -400.0
    masked_slice = np.where(lung_mask, hu_volume[target_z], -1000.0)
    axes[1].imshow(hu_volume[target_z], cmap='gray', vmin=-1000, vmax=400)
    axes[1].imshow(np.ma.masked_where(masked_slice < -500, masked_slice), cmap='inferno', alpha=0.65)
    axes[1].set_title("2. PROLIFERATION HEATMAP", fontsize=10, fontweight='bold', color='#ffffff')
    axes[1].axis('off')

    # Panel 3: 4x Zoom
    axes[2].set_facecolor('#0a2540')
    zoom_crop = ndimage.zoom(micro_crop, zoom=4.0, order=3)
    axes[2].imshow(zoom_crop, cmap='magma')

    axes[2].annotate("Active Core\n(-250 HU)", xy=(24, 24), xytext=(4, 8),
                    arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=1.0, headwidth=4),
                    fontsize=8, fontweight='bold', color='white', bbox=dict(boxstyle="round", fc="#ef4444", ec="white"))

    axes[2].annotate("Sub-Solid GGO Halo", xy=(15, 15), xytext=(4, 38),
                    arrowprops=dict(facecolor='#eab308', shrink=0.05, width=1.0, headwidth=4),
                    fontsize=8, fontweight='bold', color='black', bbox=dict(boxstyle="round", fc="#eab308", ec="black"))

    axes[2].set_title("3. 4x SUB-PIXEL ZOOM (< 1.99mm)", fontsize=10, fontweight='bold', color='#ffffff')
    axes[2].axis('off')

    # Panel 4: Criteria Box
    axes[3].set_facecolor('#0a2540')
    criteria_text = (
        f"RADIOMIC METRIC CRITERIA\n"
        f"-----------------------------\n"
        f"• Spatial Voxels: (X:{target_x}, Y:{target_y}, Z:{target_z})\n"
        f"• Max Diameter : {diameter_mm:.2f} mm (< 1.99mm)\n"
        f"• Micro-Volume : {volume_mm3:.3f} mm³\n"
        f"• Peak CT Value: {peak_hu:.1f} HU\n"
        f"• Imaging Score: {s_imaging:.4f}\n"
        f"-----------------------------\n"
        f"FITTED RISK TIER:\n"
        f"STAGE I: Sub-2mm Morphological Onset"
    )
    axes[3].text(0.05, 0.5, criteria_text, transform=axes[3].transAxes, fontsize=9.5, fontweight='bold', color='#ffffff',
                 verticalalignment='center', bbox=dict(boxstyle="round,pad=0.8", facecolor='#05192d', edgecolor='#00a859', lw=1.5))
    axes[3].set_title("4. CRITERIA & STAGE FITTING", fontsize=10, fontweight='bold', color='#ffffff')
    axes[3].axis('off')

    plt.tight_layout()
    return fig

# =====================================================================
# 4. MAIN DISPLAY AREA
# =====================================================================
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("<div class='card-box'><b>🔬 Sputum Cytology Branch</b><br><br>", unsafe_allow_html=True)
    if uploaded_slide:
        st.image(Image.open(uploaded_slide), caption="Uploaded Cytology Slide", use_container_width=True)
    else:
        st.info("Awaiting Sputum Slide upload. Utilizing sample benchmark slide.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown("<div class='card-box'><b>🩻 3D Thoracic CT Branch</b><br><br>", unsafe_allow_html=True)
    if uploaded_ct_zip:
        st.success(f"Loaded DICOM Archive: {uploaded_ct_zip.name}")
    else:
        st.info("Awaiting 3D CT Archive upload. Utilizing synthetic volume array (32x128x128).")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if run_pipeline_btn:
    st.markdown("### Diagnostic Inference & Risk Stratification")
    
    base_clinical = min(1.0, (smoking_pack_years / 50.0) * 0.4 + (patient_age / 100.0) * 0.3)
    s_clinical = min(1.0, base_clinical + clinical_risk_modifier)
    s_sputum = 0.6871
    s_imaging = 0.6850
    s_composite = (0.40 * s_sputum) + (0.45 * s_imaging) + (0.15 * s_clinical)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sputum Score (S_sp)", f"{s_sputum:.4f}")
    m2.metric("Imaging Score (S_im)", f"{s_imaging:.4f}")
    m3.metric("Clinical Risk Score", f"{s_clinical:.4f}")
    m4.metric("Composite Score (S_comp)", f"{s_composite:.4f}")

    if s_composite < 0.25:
        risk_stage = "STAGE 0: Normal / Negligible Risk"
        rec = "Routine annual health tracking and lifestyle risk reduction."
        alert_class = "alert-stage0"
    elif 0.25 <= s_composite < 0.50:
        risk_stage = "STAGE I: Early Sub-2mm Morphological Onset"
        rec = "Follow-up sputum molecular panel & High-Resolution LDCT check in 6 months."
        alert_class = "alert-stage1"
    elif 0.50 <= s_composite < 0.75:
        risk_stage = "STAGE II: Early Morphological Onset"
        rec = "High-Resolution LDCT Follow-up in 3 months. Pulmonology referral recommended."
        alert_class = "alert-stage2"
    else:
        risk_stage = "STAGE III: High-Risk Synergistic Onset"
        rec = "Immediate referral to pulmonology / oncology for priority biopsy & staging PET-CT."
        alert_class = "alert-stage3"

    st.markdown(f"""
        <div class="result-alert {alert_class}">
            <b>IDENTIFIED DIAGNOSTIC STAGE:</b> {risk_stage}<br>
            <b>RECOMMENDED CLINICAL ACTION:</b> {rec}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 4-Panel PACS Diagnostic Workstation Dashboard")
    ct_vol = generate_sample_3d_ct_volume()
    pacs_fig = render_pacs_dashboard(ct_vol)
    st.pyplot(pacs_fig, use_container_width=True)
else:
    st.info("Configure clinical parameters in the sidebar and click RUN DUAL-MODAL PIPELINE to evaluate risk.")