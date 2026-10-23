import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path so frontend can import backend / config if needed
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import api_client

st.set_page_config(
    page_title="CareFlow Hospital System | Clinical Triage & Intake",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom Clean Clinical Dashboard CSS
st.markdown("""
<style>
    /* Clean typography & spacing */
    .main {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    /* Hide Streamlit default multipage navigation so users use intentional portals */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    .header-box {
        background: linear-gradient(135deg, #0F766E 0%, #1E3A8A 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px -2px rgba(15, 118, 110, 0.2);
    }
    .header-title {
        font-size: 30px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 15px;
        opacity: 0.92;
        margin-top: 8px;
        line-height: 1.5;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 14px 18px;
        border-radius: 8px;
        color: #92400E;
        font-size: 13.5px;
        margin-bottom: 24px;
        line-height: 1.5;
    }
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-desc {
        font-size: 14px;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    .hospital-badge {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        font-size: 11.5px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Hospital Header
st.markdown("""
<div class="header-box">
    <div class="header-title">🏥 CareFlow Health System</div>
    <div class="header-subtitle">
        Clinical Symptom Intake, Multilingual Emergency Triage & Decision Support Command Center
    </div>
</div>
""", unsafe_allow_html=True)

# Clinical Safety Notice
st.markdown("""
<div class="disclaimer-banner">
    <strong>⚠️ Clinical Protocol Notice:</strong>
    CareFlow provides automated conversational symptom intake and algorithmic triage classification.
    All assessments require evaluation by a licensed healthcare professional.
    <strong>In a life-threatening medical emergency, immediately call 112 / 911 or visit the nearest hospital emergency room.</strong>
</div>
""", unsafe_allow_html=True)

# Sidebar System Health Status
health = api_client.get_health()
st.sidebar.title("⚙️ System Status")

if health.get("status") == "healthy":
    st.sidebar.success("🟢 Hospital Backend API: Online")
    st.sidebar.info("🧠 Clinical Intelligence: **Google MedGemma**")
    st.sidebar.info("🌳 Urgency Engine: **LightGBM ML Classifier**")
    st.sidebar.info("🎙️ Voice Pipeline: **Deepgram Nova-2 & Aura**")
    st.sidebar.caption(f"Triage Protocol: {health.get('triage_rule_version', 'v1.0.0-clinical')}")
    st.sidebar.caption("Data Store: Verified SQLite Database")
else:
    st.sidebar.error("🔴 Hospital Backend API: Offline or Unreachable")
    st.sidebar.caption("Ensure FastAPI is running: `uvicorn backend.main:app --port 8000`")

st.sidebar.divider()
st.sidebar.subheader("Navigation")
st.sidebar.page_link("pages/patient.py", label="Patient Intake Portal", icon="🩺")
st.sidebar.page_link("pages/model_showcase.py", label="Model Inference & Architecture", icon="🧠")

# Role-Based Portals
col_patient, col_reviewer = st.columns(2)

with col_patient:
    st.markdown("""
    <div class="card">
        <div>
            <span class="hospital-badge">PATIENT ACCESS PORTAL</span>
            <div class="card-title">🩺 Patient Symptom Intake</div>
            <div class="card-desc">
                A secure, patient-friendly conversational interface designed for confidential symptom reporting.
                <ul>
                    <li><strong>No Login or PII Required</strong>: A private, anonymous Case ID is generated for clinical privacy.</li>
                    <li><strong>Trilingual Communication</strong>: Fluid support for English, Hindi, and Hinglish with instant translation.</li>
                    <li><strong>Integrated Multimodal Input</strong>: Type symptoms, speak via high-accuracy medical voice recognition, or upload medical prescriptions.</li>
                    <li><strong>Real-Time Red Flag Detection</strong>: Immediate safety advisories if critical symptoms (e.g., severe chest pain) are detected.</li>
                    <li><strong>Review & Edit</strong>: Patients can verify and edit their reported statements before final clinical submission.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Patient Intake Portal →", type="primary", use_container_width=True, key="btn_launch_patient"):
        st.switch_page("pages/patient.py")

with col_reviewer:
    st.markdown("""
    <div class="card">
        <div>
            <span class="hospital-badge" style="background:#FEE2E2; color:#991B1B;">STAFF & CLINICIAN ACCESS ONLY</span>
            <div class="card-title">📋 Clinical Reviewer Dashboard</div>
            <div class="card-desc">
                Authorized command center for triage nurses, medical officers, and emergency physicians.
                <ul>
                    <li><strong>Dual-Signal Triage</strong>: Deterministic emergency protocols (P0-P3) coupled with LightGBM ML risk predictions.</li>
                    <li><strong>Evidence-Grounded Synthesis</strong>: Structured clinical summary with verbatim patient quote citations.</li>
                    <li><strong>Uncertainty & Conflict Detection</strong>: Highlighting conflicting reports and missing clinical dimensions.</li>
                    <li><strong>Clinician Override & Audit Trail</strong>: Healthcare professionals can adjust priority with required medical rationale.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Open Reviewer Dashboard →", type="primary", use_container_width=True, key="btn_auth_reviewer"):
        st.switch_page("pages/reviewer.py")

st.divider()

# Hospital System Architecture & Standards (Purely factual, no fine-tuning claims)
col_arch1, col_arch2 = st.columns(2)

with col_arch1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🛡️ Clinical Protocol Standards</div>
        <div class="card-desc">
            CareFlow implements deterministic clinical safety rules adapted from the <strong>Manchester Triage System (MTS)</strong> and <strong>Emergency Severity Index (ESI)</strong>:
            <ul>
                <li><strong>Priority 0 (Resuscitation / Immediate)</strong>: Life-threatening emergencies requiring immediate resuscitation.</li>
                <li><strong>Priority 1 (Emergency)</strong>: High-acuity critical conditions requiring urgent care within 10 minutes.</li>
                <li><strong>Priority 2 (Urgent)</strong>: Severe symptoms requiring clinical attention within 60 minutes.</li>
                <li><strong>Priority 3 (Semi-Urgent / Routine)</strong>: Stable symptoms appropriate for standard outpatient consultation.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_arch2:
    st.markdown("""
    <div class="card">
        <div class="card-title">⚙️ Integrated Clinical Intelligence</div>
        <div class="card-desc">
            <strong>Clinical Reasoning:</strong> Google MedGemma Clinical Language Model for contextual medical inquiry.<br/>
            <strong>Tabular Risk Engine:</strong> LightGBM 15-Feature Gradient Boosted Classifier.<br/>
            <strong>Voice Intelligence:</strong> Deepgram Nova-2 Medical STT & Aura Natural Speech Synthesis.<br/>
            <strong>Document Processing:</strong> High-precision OCR for prescription and laboratory diagnostic reports.<br/>
            <strong>Human-in-the-Loop:</strong> Mandatory clinician review with comprehensive audit logs and override tracking.
        </div>
    </div>
    """, unsafe_allow_html=True)
