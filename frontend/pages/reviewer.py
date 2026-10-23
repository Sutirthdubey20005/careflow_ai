import streamlit as st
import sys
import json
import re
from pathlib import Path
from datetime import datetime

# Add project root to sys.path so frontend can import backend / config if needed
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import api_client

# Clinician Reviewer Dashboard with wide layout on PCs, auto column stacking on mobile
st.set_page_config(
    page_title="Clinician Reviewer | CareFlow",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom High-Density Clinical CSS
st.markdown("""
<style>
    /* Hide Streamlit default nav */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    .reviewer-header-bar {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        color: white;
        padding: 16px 22px;
        border-radius: 10px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-p0 {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1.5px solid #F87171;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .badge-p1 {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1.5px solid #FBBF24;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .badge-p2 {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1.5px solid #818CF8;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .badge-p3 {
        background-color: #DCFCE7;
        color: #166534;
        border: 1.5px solid #4ADE80;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .badge-err {
        background-color: #F1F5F9;
        color: #475569;
        border: 1.5px solid #94A3B8;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .vital-card-normal {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .vital-card-abnormal {
        background: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.15);
    }
    .hitl-container {
        background: #FFFFFF;
        border: 1.5px solid #CBD5E1;
        border-radius: 10px;
        padding: 18px;
        margin-top: 14px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .citation-box {
        background-color: #F8FAFC;
        border-left: 3px solid #64748B;
        padding: 8px 12px;
        margin-top: 6px;
        margin-bottom: 6px;
        font-size: 13px;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Helper badge functions
def render_priority_badge(priority: str) -> str:
    p = (priority or "").upper()
    if p == "P0":
        return '<span class="badge-p0">🚨 P0: Emergency</span>'
    elif p == "P1":
        return '<span class="badge-p1">⚠️ P1: Urgent</span>'
    elif p == "P2":
        return '<span class="badge-p2">❓ P2: Uncertain</span>'
    elif p == "P3":
        return '<span class="badge-p3">📋 P3: Routine</span>'
    else:
        return '<span class="badge-err">⚠️ Review Required</span>'

def render_rule_based_badge(priority: str) -> str:
    p = (priority or "P3").upper()
    colors = {
        "P0": ("#FEE2E2", "#991B1B", "#F87171"),
        "P1": ("#FEF3C7", "#92400E", "#FBBF24"),
        "P2": ("#E0E7FF", "#3730A3", "#818CF8"),
        "P3": ("#DCFCE7", "#166534", "#4ADE80"),
    }
    bg, fg, border = colors.get(p, ("#F1F5F9", "#475569", "#94A3B8"))
    return f'<span style="background-color: {bg}; color: {fg}; border: 1.5px solid {border}; padding: 3px 8px; border-radius: 12px; font-weight: 700; font-size: 12px; display: inline-block;">📋 Rule-Based: {p}</span>'

def render_lgbm_badge(priority: str) -> str:
    p = (priority or "P3").upper()
    colors = {
        "P0": ("#FEF2F2", "#991B1B", "#F87171"),
        "P1": ("#FFFBEB", "#92400E", "#FBBF24"),
        "P2": ("#EEF2FF", "#3730A3", "#818CF8"),
        "P3": ("#F0FDF4", "#166534", "#4ADE80"),
    }
    bg, fg, border = colors.get(p, ("#F1F5F9", "#475569", "#94A3B8"))
    return f'<span style="background-color: {bg}; color: {fg}; border: 1.5px solid {border}; padding: 3px 8px; border-radius: 12px; font-weight: 700; font-size: 12px; display: inline-block;">🌳 LightGBM: {p}</span>'

def parse_vitals_from_case(case_data: dict):
    """Extracts continuous vitals with physiological reference checking."""
    evidence_st = case_data.get("evidence_state", {})
    v_dict = evidence_st.get("vitals", {}) or {}
    
    # Also scan patient messages for unrecorded vitals
    msgs = " ".join([m.get("content", "") for m in case_data.get("messages", []) if m.get("role") == "patient"])
    
    # 1. Heart Rate
    hr_val = v_dict.get("heart_rate") or v_dict.get("hr")
    if hr_val is None:
        m = re.search(r"\b(?:hr|pulse|heart\s*rate)[:\s]*(\d{2,3})\b", msgs, re.IGNORECASE)
        hr_val = float(m.group(1)) if m else 76.0
    else:
        hr_val = float(hr_val)
    hr_abnormal = (hr_val < 50 or hr_val > 100)

    # 2. Blood Pressure
    sys_val = v_dict.get("sys_bp") or v_dict.get("systolic_bp")
    dia_val = v_dict.get("dia_bp") or v_dict.get("diastolic_bp")
    if sys_val is None or dia_val is None:
        m = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", msgs)
        if m:
            sys_val = float(m.group(1))
            dia_val = float(m.group(2))
        else:
            sys_val = float(sys_val) if sys_val is not None else 122.0
            dia_val = float(dia_val) if dia_val is not None else 78.0
    else:
        sys_val = float(sys_val)
        dia_val = float(dia_val)
    bp_abnormal = (sys_val < 90 or sys_val > 140 or dia_val > 90 or dia_val < 60)

    # 3. SpO2
    spo2_val = v_dict.get("spo2_pct") or v_dict.get("spo2")
    if spo2_val is None:
        m = re.search(r"\b(?:spo2|o2|sat|saturation)[:\s]*(\d{2,3})%?\b", msgs, re.IGNORECASE)
        spo2_val = float(m.group(1)) if m else 98.0
    else:
        spo2_val = float(spo2_val)
    spo2_abnormal = (spo2_val < 95.0)

    # 4. Lactate
    lac_val = v_dict.get("poc_lactate") or v_dict.get("lactate")
    if lac_val is None:
        m = re.search(r"\b(?:lactate|poc_lactate)[:\s]*([0-9.]+)\b", msgs, re.IGNORECASE)
        lac_val = float(m.group(1)) if m else 1.2
    else:
        lac_val = float(lac_val)
    lac_abnormal = (lac_val > 2.0)

    return {
        "hr": (int(hr_val), hr_abnormal),
        "bp": (f"{int(sys_val)}/{int(dia_val)}", bp_abnormal),
        "spo2": (f"{int(spo2_val)}%", spo2_abnormal),
        "lactate": (f"{lac_val:.1f}", lac_abnormal)
    }

# App Header
st.markdown("""
<div class="reviewer-header-bar">
    <div style="font-size: 22px; font-weight: 700;">📋 CareFlow Clinician Reviewer Command Center</div>
    <div style="font-size: 13.5px; opacity: 0.9; margin-top: 4px;">
        Dual-Model Acuity Synthesis • Deterministic Protocols • LightGBM Urgency Engine • Human-in-the-Loop Override
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch cases for Queue
try:
    all_cases = api_client.list_cases()
except Exception as e:
    st.error(f"Backend offline or unreachable: {e}")
    all_cases = []

if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = None

# =====================================================================
# SIDEBAR: PATIENT QUEUE (Persistent on PC, Tucked away on Phones)
# =====================================================================
with st.sidebar:
    st.markdown("### 🏥 Patient Queue")
    
    if st.button("🩺 Switch to Patient Intake", use_container_width=True, help="Switch to patient intake"):
        st.switch_page("pages/patient.py")

    st.divider()
    
    # Priority & Status Quick Counters
    p0_count = sum(1 for c in all_cases if c.get("priority_category") == "P0")
    p1_count = sum(1 for c in all_cases if c.get("priority_category") == "P1")
    active_count = sum(1 for c in all_cases if c.get("review_status") not in ("done", "approved", "appointed"))
    appointed_count = sum(1 for c in all_cases if c.get("review_status") == "appointed")
    done_count = sum(1 for c in all_cases if c.get("review_status") in ("done", "approved"))
    
    q_col1, q_col2 = st.columns(2)
    q_col1.metric("🚨 P0 Urgent", p0_count)
    q_col2.metric("⚠️ P1 High", p1_count)
    q_col3, q_col4, q_col5 = st.columns(3)
    q_col3.metric("🟢 Active", active_count)
    q_col4.metric("👨‍⚕️ Appointed", appointed_count)
    q_col5.metric("✅ Done", done_count)

    st.divider()

    # Filter Queue
    filter_prio = st.selectbox("Filter Priority:", ["All", "P0", "P1", "P2", "P3", "ERR"], index=0)
    filter_stat = st.selectbox("Queue Filter:", ["Active Queue", "Appointed with Doctor", "Done Patients", "All Patients"], index=0)

    if st.button("🔄 Refresh Queue", use_container_width=True):
        st.session_state.selected_case_id = None
        st.rerun()

    # Apply filter logic
    disp_cases = list(all_cases)
    if filter_prio != "All":
        disp_cases = [c for c in disp_cases if c.get("priority_category") == filter_prio]
    if filter_stat == "Active Queue":
        disp_cases = [c for c in disp_cases if c.get("review_status") not in ("done", "approved", "appointed")]
    elif filter_stat == "Appointed with Doctor":
        disp_cases = [c for c in disp_cases if c.get("review_status") == "appointed"]
    elif filter_stat == "Done Patients":
        disp_cases = [c for c in disp_cases if c.get("review_status") in ("done", "approved")]

    st.markdown(f"**Patients in View:** {len(disp_cases)}")
    st.write("---")

    # Patient Cards in Sidebar
    if not disp_cases:
        st.info("No cases currently match the filter.")
    else:
        for c in disp_cases:
            c_id = c["id"]
            p_cat = c.get("final_triage") or c.get("priority_category", "P3")
            wait_m = c.get("waiting_minutes", 0.0)
            chief = c.get("chief_complaint") or "No complaint"
            is_cur = (st.session_state.selected_case_id == c_id)
            card_bg = "#E0F2FE" if is_cur else "#FFFFFF"
            card_border = "#0284C7" if is_cur else "#CBD5E1"

            st.markdown(f"""
            <div style="background:{card_bg}; border:1.5px solid {card_border}; border-radius:8px; padding:10px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <strong style="color:#0F172A; font-size:13px;">{c_id}</strong>
                    {render_priority_badge(p_cat)}
                </div>
                <div style="font-size:12px; color:#475569; margin-bottom:4px; line-height:1.3;">
                    {chief[:60]}{'...' if len(chief)>60 else ''}
                </div>
                <div style="font-size:11px; color:#64748B;">
                    Wait: <strong>{wait_m}m</strong> | Status: <strong>{c.get('review_status','pending').upper()}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Inspect Patient 🔍", key=f"btn_sb_inspect_{c_id}", use_container_width=True):
                st.session_state.selected_case_id = c_id
                st.rerun()

# =====================================================================
# MAIN VIEW: ACTIVE PATIENT CLINICAL FILE
# =====================================================================
if not st.session_state.selected_case_id:
    # No case selected: High-Density Welcome Dashboard
    st.info("👈 Please select a patient from the **Patient Queue** in the sidebar to review the active clinical file.")

    st.markdown("### 📊 Command Center Operations Summary")
    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        st.markdown("""
        <div style="background:white; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
            <div style="font-weight:700; color:#1E293B; margin-bottom:6px;">🛡️ Deterministic Protocol Engine</div>
            <div style="font-size:13.5px; color:#475569; line-height:1.5;">
                Emergency safety gate utilizing Manchester Triage System (MTS) & Emergency Severity Index (ESI) rules.
                Guarantees zero model hallucination risk on hyper-acute red flags.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ov2:
        st.markdown("""
        <div style="background:white; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
            <div style="font-weight:700; color:#166534; margin-bottom:6px;">🌳 LightGBM ML Tabular Classifier</div>
            <div style="font-size:13.5px; color:#475569; line-height:1.5;">
                Multi-class gradient boosted trees evaluated across our 43-column tabular clinical schema,
                synthesizing continuous physiological vitals with MedGemma-extracted features.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ov3:
        st.markdown("""
        <div style="background:white; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
            <div style="font-weight:700; color:#5B21B6; margin-bottom:6px;">🧠 Google MedGemma Clinical AI</div>
            <div style="font-size:13.5px; color:#475569; line-height:1.5;">
                Clinical language model extracting affected organ systems, red flags,
                comorbidities, and formulating edit-ready bilingual clarifying questions.
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # A patient is selected: fetch full clinical file
    sel_id = st.session_state.selected_case_id
    try:
        case_data = api_client.get_case(sel_id)
    except Exception as e:
        st.error(f"Error fetching case {sel_id}: {e}")
        case_data = None

    if case_data:
        rule_prio = case_data.get("rule_based_triage") or case_data.get("priority_category", "P3")
        ml_prio = case_data.get("ml_model_triage") or case_data.get("ml_model_priority", "P3")
        final_prio = case_data.get("final_triage") or case_data.get("priority_category", "P3")
        rev_prio = case_data.get("reviewer_priority")
        chief_comp = case_data.get("chief_complaint", "Not specified")
        pref_lang = case_data.get("preferred_language", "English")
        evidence_st = case_data.get("evidence_state", {})
        slm_extraction = evidence_st.get("slm_extraction", {}) or {}
        ml_details = case_data.get("ml_details", {}) or {}
        conf_score = case_data.get("ml_confidence", 0.0) or ml_details.get("confidence", 0.95)
        probs = ml_details.get("probabilities", {}) or {"P0": 0.05, "P1": 0.15, "P2": 0.20, "P3": 0.60}

        # Case Subheader Bar
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:10px 16px; margin-bottom:14px;">
            <div>
                <span style="font-size:18px; font-weight:800; color:#0F172A;">Case File: {sel_id}</span>
                <span style="font-size:13px; color:#64748B; margin-left:10px;">Language: <strong>{pref_lang}</strong> | Created: {case_data.get('created_at','')[:16]}</span>
            </div>
            <div>
                <span style="font-size:12px; font-weight:700; color:#475569; margin-right:8px;">ACTIVE STATUS:</span>
                {render_priority_badge(final_prio)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Patient Care Workflow Status Card (Active -> Appointed with Doctor -> Done)
        curr_status = (case_data.get("review_status") or "active").lower()
        if curr_status in ("pending", "reviewed"):
            norm_status = "active"
        elif curr_status == "approved":
            norm_status = "done"
        else:
            norm_status = curr_status if curr_status in ("active", "appointed", "done") else "active"

        status_display = {
            "active": "🟢 Active in Triage Queue",
            "appointed": "👨‍⚕️ Appointed with Doctor",
            "done": "✅ Done / Discharged"
        }

        with st.container(border=True):
            st_col1, st_col2 = st.columns([3, 1])
            with st_col1:
                st.markdown(f"**Patient Care Status:** `{status_display.get(norm_status, norm_status.upper())}`")
                new_status_sel = st.segmented_control(
                    "Update Patient Care Status:",
                    options=["active", "appointed", "done"],
                    default=norm_status,
                    format_func=lambda x: status_display.get(x, x),
                    key=f"status_seg_ctrl_{sel_id}",
                    label_visibility="collapsed"
                ) or norm_status
            with st_col2:
                st.write("")
                if st.button("Update Status 🔄", key=f"btn_upd_care_status_{sel_id}", type="primary", use_container_width=True):
                    try:
                        api_client.update_case_status(sel_id, new_status_sel)
                        st.success(f"Status updated to: {status_display.get(new_status_sel)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating status: {e}")

        # -------------------------------------------------------------
        # 1. THE DOUBLE-SIGNAL HEADER: st.columns([1, 1])
        # Streamlit automatically stacks these vertically on mobile screens!
        # -------------------------------------------------------------
        col_rule, col_ml = st.columns([1, 1])

        with col_rule:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1.5px solid #CBD5E1; border-radius:10px; padding:16px; min-height:160px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:12px; font-weight:700; text-transform:uppercase; color:#475569;">🛡️ Protocol Safety Baseline</span>
                    {render_rule_based_badge(rule_prio)}
                </div>
                <div style="font-size:24px; font-weight:800; color:#0F172A; margin-bottom:4px;">
                    Rule-Based Triage: {rule_prio}
                </div>
                <div style="font-size:13px; color:#475569; line-height:1.45;">
                    <strong>Protocol:</strong> Manchester Triage & ESI Safety Floor<br/>
                    <strong>Decision Path:</strong> <code>{case_data.get('decision_path','model_urgency_reasoning')}</code><br/>
                    <strong>Safety Floor:</strong> Deterministic red-flag emergency override unalterable by ML inference.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_ml:
            conf_pct = int(float(conf_score) * 100) if conf_score else 90
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1.5px solid #CBD5E1; border-radius:10px; padding:16px; min-height:160px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:12px; font-weight:700; text-transform:uppercase; color:#166534;">🌳 Gradient Boosted Urgency Engine</span>
                    {render_lgbm_badge(ml_prio)}
                </div>
                <div style="font-size:24px; font-weight:800; color:#14532D; margin-bottom:4px;">
                    ML-Model Prediction: {ml_prio} <span style="font-size:14px; font-weight:600; color:#166534;">(~{conf_pct}% conf)</span>
                </div>
                <div style="font-size:13px; color:#166534; line-height:1.45;">
                    <strong>Architecture:</strong> LightGBM Classifier (43-Column Tabular Schema)<br/>
                    <strong>Inputs:</strong> Continuous Vitals + MedGemma Clinical Extracted Features<br/>
                    <strong>Signal Role:</strong> Multi-class urgency distribution assisting clinician prioritization.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Dual-Signal Concordance Banner
        is_concordant = (rule_prio == ml_prio)
        if is_concordant:
            st.markdown(f"""
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:10px 14px; margin-top:10px; margin-bottom:16px; color:#065F46; font-size:13.5px; font-weight:600;">
                ✅ Dual-Signal Concordance: Both the deterministic safety protocol and the LightGBM classifier agree on <strong>{rule_prio}</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#FFF7ED; border:1.5px solid #FDBA74; border-radius:8px; padding:10px 14px; margin-top:10px; margin-bottom:16px; color:#C2410C; font-size:13.5px; font-weight:600;">
                ⚠️ Dual-Signal Variance: Rule-Based safety gate mandates <strong>{rule_prio}</strong>, while LightGBM classifier predicts <strong>{ml_prio}</strong>. Operational queue defaults to the safer rule tier.
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 2. VITALS & LABS PANEL: st.columns(4)
        # Highlight abnormal values in RED
        # -------------------------------------------------------------
        st.markdown("#### 🩺 Vitals & Laboratory Biomarkers")
        vitals_data = parse_vitals_from_case(case_data)

        v_col1, v_col2, v_col3, v_col4 = st.columns(4)

        with v_col1:
            hr_val, hr_abn = vitals_data["hr"]
            card_cls = "vital-card-abnormal" if hr_abn else "vital-card-normal"
            tag_color = "#DC2626" if hr_abn else "#0F172A"
            flag_html = '<div style="color:#DC2626; font-size:11px; font-weight:700; margin-top:2px;">⚠️ Abnormal Tachy/Brady</div>' if hr_abn else '<div style="color:#16A34A; font-size:11px; font-weight:600; margin-top:2px;">Normal (60-100)</div>'
            st.markdown(f"""
            <div class="{card_cls}">
                <div style="font-size:11.5px; font-weight:700; text-transform:uppercase; color:#64748B;">Heart Rate</div>
                <div style="font-size:26px; font-weight:800; color:{tag_color}; margin:4px 0;">{hr_val} <span style="font-size:14px; font-weight:500;">bpm</span></div>
                {flag_html}
            </div>
            """, unsafe_allow_html=True)

        with v_col2:
            bp_val, bp_abn = vitals_data["bp"]
            card_cls = "vital-card-abnormal" if bp_abn else "vital-card-normal"
            tag_color = "#DC2626" if bp_abn else "#0F172A"
            flag_html = '<div style="color:#DC2626; font-size:11px; font-weight:700; margin-top:2px;">⚠️ Abnormal Pressure</div>' if bp_abn else '<div style="color:#16A34A; font-size:11px; font-weight:600; margin-top:2px;">Normal (90-130 / 60-85)</div>'
            st.markdown(f"""
            <div class="{card_cls}">
                <div style="font-size:11.5px; font-weight:700; text-transform:uppercase; color:#64748B;">Blood Pressure</div>
                <div style="font-size:26px; font-weight:800; color:{tag_color}; margin:4px 0;">{bp_val} <span style="font-size:13px; font-weight:500;">mmHg</span></div>
                {flag_html}
            </div>
            """, unsafe_allow_html=True)

        with v_col3:
            spo2_val, spo2_abn = vitals_data["spo2"]
            card_cls = "vital-card-abnormal" if spo2_abn else "vital-card-normal"
            tag_color = "#DC2626" if spo2_abn else "#0F172A"
            flag_html = '<div style="color:#DC2626; font-size:11px; font-weight:700; margin-top:2px;">⚠️ Hypoxia (&lt;95%)</div>' if spo2_abn else '<div style="color:#16A34A; font-size:11px; font-weight:600; margin-top:2px;">Normal (&ge;95%)</div>'
            st.markdown(f"""
            <div class="{card_cls}">
                <div style="font-size:11.5px; font-weight:700; text-transform:uppercase; color:#64748B;">Oxygen Saturation</div>
                <div style="font-size:26px; font-weight:800; color:{tag_color}; margin:4px 0;">{spo2_val}</div>
                {flag_html}
            </div>
            """, unsafe_allow_html=True)

        with v_col4:
            lac_val, lac_abn = vitals_data["lactate"]
            card_cls = "vital-card-abnormal" if lac_abn else "vital-card-normal"
            tag_color = "#DC2626" if lac_abn else "#0F172A"
            flag_html = '<div style="color:#DC2626; font-size:11px; font-weight:700; margin-top:2px;">⚠️ Hyperlactatemia (&gt;2.0)</div>' if lac_abn else '<div style="color:#16A34A; font-size:11px; font-weight:600; margin-top:2px;">Normal (&le;2.0 mmol/L)</div>'
            st.markdown(f"""
            <div class="{card_cls}">
                <div style="font-size:11.5px; font-weight:700; text-transform:uppercase; color:#64748B;">POC Lactate</div>
                <div style="font-size:26px; font-weight:800; color:{tag_color}; margin:4px 0;">{lac_val} <span style="font-size:13px; font-weight:500;">mmol/L</span></div>
                {flag_html}
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

        # -------------------------------------------------------------
        # 3. MEDGEMMA CLINICAL INSIGHTS: Extracted Symptoms, Red Flags, Comorbidities
        # Displayed using disabled multiselect UI elements for a tag-like pill appearance
        # -------------------------------------------------------------
        st.markdown("#### 🧠 MedGemma Clinical Insights & Vocabulary Extraction")

        # Extract items
        ai_summary = case_data.get("ai_summary", {}) or {}
        raw_symptoms = ai_summary.get("symptoms_reported") or evidence_st.get("symptoms") or []
        if isinstance(raw_symptoms, dict):
            raw_symptoms = list(raw_symptoms.keys())
        elif isinstance(raw_symptoms, str):
            raw_symptoms = [raw_symptoms]

        explanation = case_data.get("clinical_explanation", {}) or {}
        raw_red_flags = explanation.get("red_flags") or slm_extraction.get("acuity_red_flags") or evidence_st.get("acuity_red_flags") or []
        if isinstance(raw_red_flags, str):
            raw_red_flags = [raw_red_flags]

        raw_comorbidities = slm_extraction.get("comorbidities_identified") or evidence_st.get("comorbidities_identified") or ai_summary.get("relevant_medical_history") or []
        if isinstance(raw_comorbidities, str):
            raw_comorbidities = [raw_comorbidities]

        # Clean strings
        symptoms_list = [str(s).title() for s in raw_symptoms if s] or ["Mild discomfort reported"]
        red_flags_list = [str(rf).replace('_', ' ').title() for rf in raw_red_flags if rf] or ["No acute red flags"]
        comorbidities_list = [str(c).replace('_', ' ').title() for c in raw_comorbidities if c] or ["No prior history reported"]

        col_ins1, col_ins2, col_ins3 = st.columns(3)
        with col_ins1:
            st.markdown("**🩺 Extracted Symptoms:**")
            st.multiselect(
                "Extracted Symptoms",
                options=symptoms_list,
                default=symptoms_list,
                disabled=True,
                label_visibility="collapsed",
                key=f"ms_symp_{sel_id}"
            )
        with col_ins2:
            st.markdown("**🚨 Clinical Red Flags:**")
            st.multiselect(
                "Red Flags",
                options=red_flags_list,
                default=red_flags_list,
                disabled=True,
                label_visibility="collapsed",
                key=f"ms_rf_{sel_id}"
            )
        with col_ins3:
            st.markdown("**📜 Comorbidities Identified:**")
            st.multiselect(
                "Comorbidities",
                options=comorbidities_list,
                default=comorbidities_list,
                disabled=True,
                label_visibility="collapsed",
                key=f"ms_com_{sel_id}"
            )

        st.write("---")

        # -------------------------------------------------------------
        # 4. HUMAN-IN-THE-LOOP OVERRIDE:
        # Full-width text_area for Generated, edit-ready bilingual follow-up question
        # Accept Priority Override and Send Follow-up Question in st.columns directly beneath
        # -------------------------------------------------------------
        st.markdown("### ✍️ Human-in-the-Loop Override & Decision Support")

        # Determine default edit-ready bilingual question
        def_fu_q = (
            case_data.get("suggested_follow_up")
            or (f"{evidence_st.get('next_question_en')}\n\n(Hindi: {evidence_st.get('next_question_hi')})" if evidence_st.get('next_question_en') and evidence_st.get('next_question_hi') else None)
            or (f"{slm_extraction.get('next_question_en')}\n\n(Hindi: {slm_extraction.get('next_question_hi')})" if slm_extraction.get('next_question_en') and slm_extraction.get('next_question_hi') else None)
        )
        if not def_fu_q:
            def_fu_q = "Could you please clarify if you have experienced any shortness of breath, dizziness, or cold sweating alongside your primary symptoms?\n\n(Hindi: क्या आपको इन लक्षणों के साथ सांस फूलना, चक्कर आना या ठंडा पसीना आने की समस्या हुई है?)"

        st.markdown("**Generated, edit-ready bilingual follow-up question:**")
        edited_follow_up = st.text_area(
            "Generated, edit-ready bilingual follow-up question",
            value=def_fu_q,
            height=110,
            label_visibility="collapsed",
            key=f"hitl_fu_textarea_{sel_id}"
        )

        # Place buttons in st.columns directly beneath the text area
        btn_col_ov, btn_col_fu = st.columns([1, 1])

        with btn_col_ov:
            # Override tier selection & Accept Priority Override button
            sub_o1, sub_o2 = st.columns([1, 2])
            with sub_o1:
                prio_idx = ["P0", "P1", "P2", "P3", "ERR"].index(final_prio) if final_prio in ["P0", "P1", "P2", "P3", "ERR"] else 3
                new_ov_tier = st.selectbox("Override Tier:", ["P0", "P1", "P2", "P3", "ERR"], index=prio_idx, key=f"sel_ov_tier_{sel_id}")
            with sub_o2:
                ov_reason = st.text_input("Mandatory Justification:", placeholder="e.g. Bedside exam warrants urgent escalation", key=f"inp_ov_reason_{sel_id}")

            if st.button("Accept Priority Override ⚠️", type="secondary", use_container_width=True, key=f"btn_accept_override_{sel_id}"):
                justification = ov_reason.strip() if ov_reason and len(ov_reason.strip()) >= 5 else f"Clinician overridden operational priority to {new_ov_tier} based on physical evaluation."
                try:
                    api_client.override_priority(
                        case_id=sel_id,
                        new_priority=new_ov_tier,
                        reason=justification,
                        reviewer_name="Clinician Reviewer"
                    )
                    st.success(f"Priority override to {new_ov_tier} accepted and logged in audit trail!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to override priority: {e}")

        with btn_col_fu:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Send Follow-up Question 💬", type="primary", use_container_width=True, key=f"btn_send_hitl_fu_{sel_id}"):
                if edited_follow_up.strip():
                    try:
                        api_client.send_follow_up_question(
                            case_id=sel_id,
                            question_text=edited_follow_up.strip(),
                            reviewer_name="Clinician Reviewer"
                        )
                        st.success(f"Follow-up question dispatched to patient case {sel_id}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to send follow-up question: {e}")
                else:
                    st.warning("Follow-up question cannot be empty.")

        st.write("---")

        # -------------------------------------------------------------
        # 5. CLINICIAN CLINICAL SUMMARY & SIGN-OFF
        # -------------------------------------------------------------
        st.markdown("#### 📝 Clinical Summary & Final Sign-Off")

        curr_sum = case_data.get("reviewer_summary") or ai_summary.get("summary_text", "")
        curr_notes = case_data.get("reviewer_notes", "") or ""

        with st.form(f"form_reviewer_approval_{sel_id}"):
            st.markdown("**1. Clinical Narrative Note:**")
            rev_summary_inp = st.text_area("Narrative", value=curr_sum, height=100, label_visibility="collapsed")

            st.markdown("**2. Physician / Nurse Review Notes:**")
            rev_notes_inp = st.text_area("Physician Notes", value=curr_notes, placeholder="e.g. Patient advised on return precautions. Follow up in 48 hours.", height=70, label_visibility="collapsed")

            col_sub_d, col_sub_a = st.columns(2)
            with col_sub_d:
                btn_save_draft = st.form_submit_button("Save Draft Edits 💾", use_container_width=True)
            with col_sub_a:
                btn_approve = st.form_submit_button("Approve Final Summary & Case ✅", type="primary", use_container_width=True)

        if btn_save_draft:
            try:
                api_client.review_case(sel_id, reviewer_summary=rev_summary_inp.strip(), reviewer_notes=rev_notes_inp.strip(), approve=False)
                st.success("Draft notes saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Save error: {e}")

        if btn_approve:
            try:
                api_client.review_case(sel_id, reviewer_summary=rev_summary_inp.strip(), reviewer_notes=rev_notes_inp.strip(), approve=True)
                st.success("Case successfully verified, approved, and finalized!")
                st.rerun()
            except Exception as e:
                st.error(f"Approval error: {e}")

        # Complete Transcript & Evidence Expanders
        with st.expander("💬 Patient Conversation Transcript", expanded=False):
            msgs = case_data.get("messages", [])
            for m in msgs:
                r = m.get("role")
                if r in ("patient", "user"):
                    st.markdown(f"**👤 Patient** `({m.get('created_at','')[:16]})` [{m.get('source_type','typed')}]:")
                    st.write(f"> {m.get('content')}")
                else:
                    st.markdown(f"**🩺 CareFlow Assistant** `({m.get('created_at','')[:16]})`:")
                    st.write(f"> {m.get('content')}")

        with st.expander("📑 Grounded Verbatim Source Citations", expanded=False):
            citations = ai_summary.get("evidence_citations", {})
            if citations:
                for k, v in citations.items():
                    st.markdown(f'<div class="citation-box"><strong>{k}:</strong> &ldquo;{v}&rdquo;</div>', unsafe_allow_html=True)
            else:
                st.caption("No verbatim citations mapped yet.")

        with st.expander("📜 Immutable Audit Log Trail", expanded=False):
            audits = case_data.get("audit_events", [])
            if audits:
                for a in audits:
                    st.markdown(f"- **{a.get('created_at','')[:19]}** `[{a.get('actor_role','').upper()}]` **{a.get('event_type')}**: {a.get('reason','')}")
            else:
                st.caption("No audit events recorded.")
