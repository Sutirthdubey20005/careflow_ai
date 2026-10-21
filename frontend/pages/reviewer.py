import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import api_client

st.set_page_config(
    page_title="Reviewer Dashboard | CareFlow",
    page_icon="📋",
    layout="wide"
)

# Clinical Dashboard CSS
st.markdown("""
<style>
    .reviewer-header {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .badge-p0 {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-p1 {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FBBF24;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-p2 {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1px solid #818CF8;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-p3 {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #4ADE80;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-err {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #94A3B8;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .card-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .summary-badge-awaiting {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        color: #B45309;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13.5px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .summary-badge-approved {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        color: #15803D;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13.5px;
        font-weight: 600;
        margin-bottom: 12px;
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

# Helper function for priority badge HTML
def render_priority_badge(priority: str) -> str:
    p = (priority or "").upper()
    if p == "P0":
        return '<span class="badge-p0">🚨 P0: Emergency Escalation</span>'
    elif p == "P1":
        return '<span class="badge-p1">⚠️ P1: Urgent Review</span>'
    elif p == "P2":
        return '<span class="badge-p2">❓ P2: Incomplete / Uncertain</span>'
    elif p == "P3":
        return '<span class="badge-p3">📋 P3: Routine Review</span>'
    else:
        return '<span class="badge-err">⚠️ Review Required — AI Unavailable</span>'

# Header
st.markdown("""
<div class="reviewer-header">
    <h2 style="margin:0; font-size: 24px;">📋 CareFlow Healthcare Reviewer Dashboard</h2>
    <div style="font-size: 14px; opacity: 0.9; margin-top: 4px;">
        Evidence-Grounded Clinical Decision Support, Queue Management & Override Controls
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Disclaimer
st.markdown("""
<div style="background-color: #FEF3C7; border-left: 4px solid #D97706; padding: 10px 14px; border-radius: 6px; color: #92400E; font-size: 13px; margin-bottom: 16px;">
    <strong>Clinical Decision Support Disclaimer:</strong> CareFlow outputs are algorithmic aids and do NOT constitute a medical diagnosis.
    All triage priorities and summary records must be reviewed, verified, and approved by a qualified clinician.
</div>
""", unsafe_allow_html=True)

# Select active case in session state
if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = None

# =====================================================================
# SECTION 1: QUEUE METRICS & FILTER CONTROLS
# =====================================================================
try:
    all_cases = api_client.list_cases()
except Exception as e:
    st.error(f"Failed to fetch cases from backend: {e}")
    all_cases = []

# Queue Metrics Cards
total_count = len(all_cases)
p0_count = sum(1 for c in all_cases if c["priority_category"] == "P0")
p1_count = sum(1 for c in all_cases if c["priority_category"] == "P1")
p2_count = sum(1 for c in all_cases if c["priority_category"] == "P2")
p3_count = sum(1 for c in all_cases if c["priority_category"] == "P3")
pending_count = sum(1 for c in all_cases if c["review_status"] == "pending")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total In Queue", total_count)
m2.metric("🚨 P0 Emergency", p0_count)
m3.metric("⚠️ P1 Urgent", p1_count)
m4.metric("❓ P2 Uncertain", p2_count)
m5.metric("📋 P3 Routine", p3_count)
m6.metric("⏳ Awaiting Review", pending_count)

st.write("---")

# Filter Controls
f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1])
with f_col1:
    filter_priority = st.selectbox("Filter Priority", ["All", "P0", "P1", "P2", "P3", "ERR"], index=0)
with f_col2:
    filter_status = st.selectbox("Filter Status", ["All", "pending", "reviewed", "approved", "reopened"], index=0)
with f_col3:
    filter_uncertainty = st.selectbox("Filter Uncertainty / Conflict", ["All", "Has Uncertainty or Conflict", "No Conflicts"], index=0)
with f_col4:
    if st.button("🔄 Refresh Queue", use_container_width=True):
        st.session_state.selected_case_id = None
        st.rerun()

# Apply filters
filtered_cases = list(all_cases)
if filter_priority != "All":
    filtered_cases = [c for c in filtered_cases if c["priority_category"] == filter_priority]
if filter_status != "All":
    filtered_cases = [c for c in filtered_cases if c["review_status"] == filter_status]
if filter_uncertainty == "Has Uncertainty or Conflict":
    filtered_cases = [c for c in filtered_cases if c["has_uncertainty_or_conflict"]]
elif filter_uncertainty == "No Conflicts":
    filtered_cases = [c for c in filtered_cases if not c["has_uncertainty_or_conflict"]]

# Layout: Split Queue Table and Case Details
col_queue, col_detail = st.columns([1, 1.4])

# =====================================================================
# QUEUE TABLE (Left Pane)
# =====================================================================
with col_queue:
    st.subheader(f"Triage Queue ({len(filtered_cases)})")

    if not filtered_cases:
        st.info("No cases currently match the selected queue filters.")
    else:
        for c in filtered_cases:
            case_id = c["id"]
            priority = c["priority_category"]
            p_badge = render_priority_badge(priority)
            chief = c["chief_complaint"] or "Chief complaint not provided"
            wait_min = c.get("waiting_minutes", 0.0)
            status = c["review_status"].upper()
            
            missing_flag = "⚠️ Missing Info" if c["has_missing_info"] else ""
            conflict_flag = "⚡ Conflict/Uncertainty" if c["has_uncertainty_or_conflict"] else ""
            flags = " • ".join([f for f in [missing_flag, conflict_flag] if f])

            is_selected = (st.session_state.selected_case_id == case_id)
            box_style = "border: 2px solid #0284C7; background: #F0F9FF;" if is_selected else "background: #FFFFFF;"

            st.markdown(f"""
            <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 10px; {box_style}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <strong>{case_id}</strong>
                    <span>{p_badge}</span>
                </div>
                <div style="font-size: 14px; color: #1E293B; margin-bottom: 4px;">{chief[:80]}</div>
                <div style="font-size: 12px; color: #64748B; display: flex; justify-content: space-between;">
                    <span>Wait: <strong>{wait_min}m</strong> | Status: <strong>{status}</strong></span>
                    <span style="color: #B45309; font-weight: 600;">{flags}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Inspect Case {case_id}", key=f"sel_{case_id}", use_container_width=True):
                st.session_state.selected_case_id = case_id
                st.rerun()

# =====================================================================
# CASE DETAIL & REVIEWER CONTROLS (Right Pane)
# =====================================================================
with col_detail:
    if not st.session_state.selected_case_id:
        st.info("👈 Select a case from the queue on the left to review summary, evidence citations, and triage reasons.")
    else:
        sel_id = st.session_state.selected_case_id
        try:
            case_data = api_client.get_case(sel_id)
        except Exception as e:
            st.error(f"Error fetching case {sel_id}: {e}")
            case_data = None

        if case_data:
            st.subheader(f"Case Review: {sel_id}")

            # Top Status Bar
            c_prio = case_data.get("priority_category", "P3")
            rev_prio = case_data.get("reviewer_priority")
            review_st = case_data.get("review_status", "pending")
            ai_st = case_data.get("ai_status", "pending")
            rule_ver = case_data.get("triage_rule_version", "v1.0.0-prototype")

            st.markdown(f"""
            <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 16px;">
                {render_priority_badge(c_prio)}
                <span style="font-size: 13px; color: #475569;">
                    Protocol: <code>{rule_ver}</code> | AI Engine: <code>MedGemma ({ai_st})</code>
                </span>
            </div>
            """, unsafe_allow_html=True)

            if rev_prio and rev_prio != c_prio:
                st.info(f"ℹ️ Operational Priority Overridden by Clinician: **{rev_prio}** (Original Algorithmic Priority: **{c_prio}**)")

            # Triage Trigger Reasons
            reasons = case_data.get("priority_reasons", [])
            if reasons:
                with st.expander("🔍 Operational Triage Trigger Reasons & Evidence", expanded=True):
                    for r in reasons:
                        st.markdown(f"- **{r}**")

            # MedGemma Dynamic Reasoned Urgency Signals
            evidence_st = case_data.get("evidence_state", {})
            urgency_sigs = evidence_st.get("urgency_signals", []) or case_data.get("ai_summary", {}).get("urgency_signals", [])
            if urgency_sigs:
                with st.expander("🤖 MedGemma Evaluated Clinical Urgency Signals", expanded=True):
                    st.caption("Generalized clinical urgency reasoning dynamically generated by MedGemma from patient narrative & documents:")
                    for sig in urgency_sigs:
                        s_name = sig.get("signal", "Signal").replace("_", " ").title()
                        s_sev = str(sig.get("severity_level", "urgent")).upper()
                        s_rat = sig.get("rationale", "")
                        s_q = sig.get("source_quote", "")
                        
                        sev_color = "#DC2626" if s_sev == "EMERGENCY" else ("#D97706" if s_sev == "URGENT" else "#2563EB")
                        quote_html = f'<div style="font-size: 12px; color: #64748B; margin-top: 4px;"><em>Grounded Quote:</em> &ldquo;{s_q}&rdquo;</div>' if s_q else ''
                        st.markdown(f"""
                        <div style="border-left: 4px solid {sev_color}; background: #F8FAFC; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px; border: 1px solid #E2E8F0;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <strong style="color: #0F172A; font-size: 14px;">{s_name}</strong>
                                <span style="background: {sev_color}15; color: {sev_color}; font-weight: 700; font-size: 11px; padding: 2px 8px; border-radius: 10px; border: 1px solid {sev_color}40;">
                                    {s_sev}
                                </span>
                            </div>
                            <div style="font-size: 13px; color: #334155; margin-bottom: 2px;"><strong>Clinical Rationale:</strong> {s_rat}</div>
                            {quote_html}
                        </div>
                        """, unsafe_allow_html=True)

            # Uploaded Documents & OCR Inspection
            doc_events = [ev for ev in case_data.get("audit_events", []) if ev.get("event_type") == "document_uploaded"]
            ocr_msgs = [m for m in case_data.get("messages", []) if m.get("source_type") in ("ocr", "ocr_extraction")]

            if doc_events or ocr_msgs:
                doc_count = max(len(doc_events), len(ocr_msgs))
                with st.expander(f"📄 Uploaded Medical Documents & OCR Extractions ({doc_count})", expanded=False):
                    if doc_events:
                        for dev in doc_events:
                            details = dev.get("new_value") or dev.get("details") or {}
                            fname = details.get("filename", "document")
                            chars = details.get("char_count", 0)
                            engine = details.get("ocr_engine", "native OCR")
                            conf = details.get("confidence")
                            conf_str = f" | Confidence: ~{int(conf * 100)}%" if conf is not None else ""
                            preview = details.get("preview", "")
                            st.markdown(f"**Document:** `{fname}` ({chars} characters via <code>{engine}</code>{conf_str})", unsafe_allow_html=True)
                            if preview:
                                st.text_area(f"OCR Extracted Content ({fname}):", value=preview, height=110, disabled=True, key=f"ocr_view_{dev.get('id', fname)}")
                    else:
                        for m in ocr_msgs:
                            st.text_area(f"Document Message OCR:", value=m.get("content", ""), height=110, disabled=True, key=f"msg_ocr_{m.get('id')}")

            # Warnings / Missing Info / Uncertainties
            missing = case_data.get("missing_information", [])
            uncertainties = case_data.get("uncertainties", [])
            contradictions = evidence_st.get("contradictions", [])

            if missing or uncertainties or contradictions:
                st.markdown("""
                <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 12px; margin-bottom: 16px;">
                    <div style="font-weight: 700; color: #991B1B; margin-bottom: 4px;">⚠️ Clinical Attention Items</div>
                """, unsafe_allow_html=True)
                if contradictions:
                    st.markdown(f"**⚡ Contradictions:** {'; '.join(contradictions)}")
                if missing:
                    st.markdown(f"**❓ Missing Information:** {'; '.join(missing)}")
                if uncertainties:
                    st.markdown(f"**💭 Unresolved Uncertainties:** {'; '.join(uncertainties)}")
                st.markdown("</div>", unsafe_allow_html=True)

            # AI Clinical Summary Card
            ai_sum = case_data.get("ai_summary", {})
            citations = ai_sum.get("evidence_citations", {})

            if review_st == "approved":
                st.markdown('<div class="summary-badge-approved">✅ Clinician Approved Summary</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="summary-badge-awaiting">🤖 AI-Generated Summary — Awaiting Professional Review</div>', unsafe_allow_html=True)

            # Display Structured Fields
            with st.container():
                st.markdown(f"**Chief Complaint:** {ai_sum.get('chief_complaint', 'Not entered')}")
                st.markdown(f"**Symptoms Reported:** {', '.join(ai_sum.get('symptoms_reported', [])) or 'None'}")
                st.markdown(f"**Onset & Duration:** {ai_sum.get('onset_and_duration', 'Not reported')}")
                st.markdown(f"**Progression:** {ai_sum.get('progression', 'Not reported')}")
                st.markdown(f"**Severity:** {ai_sum.get('severity', 'Not reported')}")
                
                negatives = ai_sum.get("explicit_negative_findings", [])
                if negatives:
                    st.markdown(f"**Explicit Negative Findings:** {', '.join(negatives)}")

                st.markdown(f"**Clinical Narrative:** {ai_sum.get('summary_text', 'No summary generated.')}")

            # Grounded Citations Section
            if citations:
                with st.expander("📑 Grounded Source Evidence Citations", expanded=False):
                    st.caption("Verbatim patient quotes grounding the model extraction:")
                    for fact, quote in citations.items():
                        st.markdown(f'<div class="citation-box"><strong>{fact}:</strong> &ldquo;{quote}&rdquo;</div>', unsafe_allow_html=True)

            # Atomic Evidence Items & Source Provenance
            evidence_items = case_data.get("evidence_items", [])
            if evidence_items:
                with st.expander(f"🔬 Atomic Evidence Items & Modality Provenance ({len(evidence_items)})", expanded=False):
                    st.caption("Extracted clinical facts with explicit source modality, confidence, and verification status:")
                    for item in evidence_items:
                        f_name = item.get("field", "").replace("_", " ").title()
                        f_val = str(item.get("value", ""))
                        f_status = item.get("status", "reported").upper()
                        f_source = item.get("source_type", "patient_message")
                        f_conf = item.get("confidence")
                        f_quote = item.get("source_quote", "")

                        # Modality Badge
                        if "voice" in f_source.lower():
                            mod_badge = f'<span style="background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE; font-size:11px; padding:2px 6px; border-radius:6px; font-weight:600;">🎤 Voice (Conf: {int(f_conf*100) if f_conf is not None else "N/A"}%)</span>'
                        elif "ocr" in f_source.lower() or "document" in f_source.lower():
                            mod_badge = '<span style="background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; font-size:11px; padding:2px 6px; border-radius:6px; font-weight:600;">📄 Medical Report OCR</span>'
                        else:
                            mod_badge = '<span style="background:#F1F5F9; color:#475569; border:1px solid #CBD5E1; font-size:11px; padding:2px 6px; border-radius:6px; font-weight:600;">💬 Typed Text</span>'

                        # Status style
                        stat_color = "#15803D" if f_status == "REPORTED" else ("#DC2626" if f_status in ["CONFLICTING", "UNCERTAIN"] else "#475569")
                        stat_badge = f'<span style="background:{stat_color}15; color:{stat_color}; font-size:11px; font-weight:700; padding:2px 6px; border-radius:6px; border:1px solid {stat_color}40;">{f_status}</span>'

                        quote_str = f'<div style="font-size:12px; color:#64748B; margin-top:3px;"><em>Grounded Quote:</em> &ldquo;{f_quote}&rdquo;</div>' if f_quote else ""

                        st.markdown(f"""
                        <div style="border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; background: white;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong>{f_name}:</strong> <span style="color:#0F172A; font-weight:500;">{f_val}</span>
                                <div>{mod_badge} {stat_badge}</div>
                            </div>
                            {quote_str}
                        </div>
                        """, unsafe_allow_html=True)

            st.write("---")

            # Complete Conversation History
            with st.expander("💬 Complete Patient Conversation Transcript", expanded=False):
                msgs = case_data.get("messages", [])
                for m in msgs:
                    if m["role"] == "patient":
                        m_source = m.get("source_type", "patient_typed")
                        if m_source in ("voice", "voice_transcript"):
                            conf_val = m.get("transcription_confidence")
                            conf_str = f" | Conf: {int(conf_val*100)}%" if conf_val is not None else ""
                            lang_str = f" | {m.get('source_language')}" if m.get("source_language") else ""
                            mod_label = f'<span style="background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE; font-size:11px; font-weight:700; padding:2px 6px; border-radius:6px; margin-left:6px;">🎤 Voice{lang_str}{conf_str}</span>'
                        elif m_source in ("ocr", "ocr_extraction"):
                            mod_label = '<span style="background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; font-size:11px; font-weight:700; padding:2px 6px; border-radius:6px; margin-left:6px;">📄 Report OCR</span>'
                        else:
                            mod_label = '<span style="background:#F1F5F9; color:#475569; border:1px solid #CBD5E1; font-size:11px; font-weight:700; padding:2px 6px; border-radius:6px; margin-left:6px;">💬 Typed</span>'
                        role_label = f"👤 Patient {mod_label}"
                    else:
                        role_label = "🤖 CareFlow Assistant"

                    st.markdown(f"**{role_label}** `({m.get('created_at', '')[:19]})`:", unsafe_allow_html=True)
                    st.write(f"> {m['content']}")
                    if m.get("raw_transcript") and m.get("raw_transcript").strip() != m["content"].strip():
                        st.caption(f"🎤 *Original ASR transcript (before patient edits):* &ldquo;{m['raw_transcript']}&rdquo;")

            # Reviewer Controls & Approval Form
            st.subheader("Clinician Actions & Approval")

            current_summary = case_data.get("reviewer_summary") or ai_sum.get("summary_text", "")
            current_notes = case_data.get("reviewer_notes", "") or ""

            with st.form("reviewer_action_form"):
                st.markdown("**1. Edit Referral / Intake Clinical Summary:**")
                edited_summary = st.text_area(
                    "Clinical Summary Note",
                    value=current_summary,
                    height=130,
                    label_visibility="collapsed"
                )

                st.markdown("**2. Reviewer Clinical Notes:**")
                notes = st.text_area(
                    "Clinician Notes / Instructions",
                    value=current_notes,
                    placeholder="e.g. Advised routine follow-up with PCP within 5 days; patient educated on return precautions.",
                    height=80,
                    label_visibility="collapsed"
                )

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    save_draft = st.form_submit_button("Save Edits & Notes 💾", use_container_width=True)
                with btn_col2:
                    approve_case = st.form_submit_button("Approve Final Summary & Case ✅", type="primary", use_container_width=True)

            if save_draft:
                try:
                    api_client.review_case(
                        case_id=sel_id,
                        reviewer_summary=edited_summary.strip(),
                        reviewer_notes=notes.strip(),
                        approve=False
                    )
                    st.success("Reviewer edits and notes saved successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving review: {e}")

            if approve_case:
                try:
                    api_client.review_case(
                        case_id=sel_id,
                        reviewer_summary=edited_summary.strip(),
                        reviewer_notes=notes.strip(),
                        approve=True
                    )
                    st.success("Case successfully approved and finalized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error approving case: {e}")

            # Reopen Case Option
            if review_st in ["approved", "reviewed"]:
                if st.button("↩️ Reopen Case (Request Further Clarification)", use_container_width=True):
                    try:
                        api_client.reopen_case(sel_id)
                        st.info("Case status updated to 'reopened'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error reopening case: {e}")

            # Priority Override Section
            with st.expander("⚙️ Override Operational Priority", expanded=False):
                st.caption("Change the triage queue category. A mandatory justification is required for the audit trail.")
                with st.form("override_form"):
                    new_priority = st.selectbox(
                        "New Priority Category",
                        ["P0", "P1", "P2", "P3", "ERR"],
                        index=["P0", "P1", "P2", "P3", "ERR"].index(c_prio) if c_prio in ["P0", "P1", "P2", "P3", "ERR"] else 3
                    )
                    override_reason = st.text_input(
                        "Mandatory Justification Reason (min 5 chars):",
                        placeholder="e.g. Clinical history of coronary disease warrants P1 urgent assessment."
                    )
                    reviewer_name = st.text_input("Reviewer Name:", value="Attending Clinician")
                    submit_override = st.form_submit_button("Execute Priority Override ⚠️", type="primary")

                if submit_override:
                    if len(override_reason.strip()) < 5:
                        st.error("A mandatory reason of at least 5 characters is required to override triage priority.")
                    else:
                        try:
                            api_client.override_priority(
                                case_id=sel_id,
                                new_priority=new_priority,
                                reason=override_reason.strip(),
                                reviewer_name=reviewer_name.strip()
                            )
                            st.success(f"Priority successfully overridden to {new_priority}.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Override error: {e}")

            # Immutable Audit Event Log
            with st.expander("📜 Immutable Audit Log Trail", expanded=False):
                audit_logs = case_data.get("audit_events", [])
                if audit_logs:
                    for ev in audit_logs:
                        ev_type = ev.get("event_type", "event")
                        actor = ev.get("actor_role", "system")
                        ts = ev.get("created_at", "")[:19]
                        reason = ev.get("reason", "No reason recorded")
                        st.markdown(f"- **{ts}** `[{actor.upper()}]` **{ev_type}**: {reason}")
                else:
                    st.caption("No audit events recorded.")
