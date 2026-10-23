import streamlit as st
import sys
import hashlib
import json
from pathlib import Path

# Add project root to sys.path so frontend can import backend / config if needed
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import api_client

# Responsive centered layout for mobile app feel on phones and clean web widget on PCs
st.set_page_config(
    page_title="Patient Intake | CareFlow Health System",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# Professional Hospital-Grade CSS
st.markdown("""
<style>
    /* Generous top clearance so headers and language selectors are NEVER cut off */
    .block-container {
        max-width: 780px !important;
        padding-top: 4.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }
    
    /* Transparent streamlit top header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 10 !important;
    }

    /* Hide Streamlit default multipage navigation; custom staff gate in sidebar */
    [data-testid="stSidebarNav"], 
    nav[data-testid="stSidebarNav"] {
        display: none !important;
    }

    .stApp {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Hospital Branding Header */
    .hospital-header-card {
        background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px -2px rgba(15, 118, 110, 0.25);
    }
    .hospital-brand-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .hospital-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hospital-subtitle {
        font-size: 13px;
        opacity: 0.92;
        margin-top: 4px;
    }
    .status-pill {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Emergency Warning Banner */
    .emergency-banner {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 12px 16px;
        border-radius: 8px;
        color: #991B1B;
        font-size: 13.5px;
        margin-bottom: 16px;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(220, 38, 38, 0.08);
    }

    /* Language & Session Bar */
    .lang-session-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Integrated Multimodal Input Dock */
    .input-dock {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 16px;
        margin-top: 14px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .dock-mode-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .dock-title {
        font-size: 14px;
        font-weight: 700;
        color: #0F766E;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .dock-desc {
        font-size: 12.5px;
        color: #64748B;
        line-height: 1.45;
        margin-bottom: 10px;
    }

    /* Badges */
    .voice-badge {
        background-color: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        font-size: 11.5px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .ocr-badge {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        font-size: 11.5px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 4px;
    }

    /* Confirmation Card */
    .card-confirm {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# Translation catalogs for dynamic multilingual responses
GREETINGS = {
    "English": "Hello! I am CareFlow's clinical intake assistant. Please describe the symptoms or health concerns you are experiencing today (type below or use voice/document upload).",
    "Hindi": "नमस्ते! मैं केयरफ़्लो का क्लिनिकल इनटेक सहायक हूँ। कृपया आज आप जिन लक्षणों या स्वास्थ्य समस्याओं का सामना कर रहे हैं, उनके बारे में विस्तार से बताएं।",
    "Hinglish": "Namaste! Main CareFlow ka clinical intake assistant hoon. Kripya aaj aap jin symptoms ya health issues ko face kar rahe hain, unke baare me batayein (type kar sakte hain ya voice/document use kar sakte hain)."
}

CLINICAL_QUESTIONS_TRANSLATION = {
    "Does the chest pain radiate or spread to your left arm, shoulder, neck, jaw, or back?": {
        "Hindi": "क्या सीने का दर्द आपके बाएं हाथ, कंधे, गर्दन, जबड़े या पीठ की तरफ फैल रहा है?",
        "Hinglish": "Kya chest pain aapke left arm, shoulder, neck, jaw ya back ki taraf radiate kar raha hai?"
    },
    "Are you experiencing any shortness of breath, cold sweats, nausea, or lightheadedness along with the chest pain?": {
        "Hindi": "क्या सीने में दर्द के साथ सांस फूलना, ठंडा पसीना, उल्टी या चक्कर आ रहे हैं?",
        "Hinglish": "Kya chest pain ke sath saans phoolna, cold sweats, ulti ya chakkar aa rahe hain?"
    },
    "Did the headache peak at maximum intensity suddenly within seconds (like a thunderclap), or did it build gradually?": {
        "Hindi": "क्या यह सिरदर्द अचानक कुछ ही सेकंड में असहनीय तेज हो गया, या धीरे-धीरे बढ़ा?",
        "Hinglish": "Kya ye sir dard achanak seconds me bahut tez ho gaya (thunderclap ki tarah), ya gradually badha?"
    },
    "Do you have any neck stiffness, pain when bending your head forward, fever, or sensitivity to bright light?": {
        "Hindi": "क्या आपको गर्दन में अकड़न, सिर आगे झुकाने में दर्द, बुखार या तेज रोशनी से परेशानी हो रही है?",
        "Hinglish": "Kya gardan me stiffness, aage jhukane me dard, bukhar ya bright light se pareshani ho rahi hai?"
    },
    "Are you able to speak in full sentences without gasping, and does lying flat make your breathing harder?": {
        "Hindi": "क्या आप बिना सांस फूले पूरे वाक्य बोल पा रहे हैं, और क्या सीधे लेटने पर सांस ज्यादा फूलती है?",
        "Hinglish": "Kya aap bina ruke full sentences bol pa rahe hain, aur letne par saans zyada phoolti hai?"
    },
    "Have you noticed any audible wheezing, cough with phlegm, fever, or swelling of your face or ankles?": {
        "Hindi": "क्या सांस लेते समय सीटी जैसी आवाज (घरघराहट), खांसी में बलगम, बुखार या चेहरे/पैरों में सूजन है?",
        "Hinglish": "Kya saans lete waqt wheezing sound, khansi me balgam, bukhar ya face/pairon me sujan hai?"
    },
    "Where exactly in your abdomen is the pain located, and are you able to keep water or fluids down without vomiting?": {
        "Hindi": "पेट में दर्द ठीक किस जगह पर है, और क्या आप पानी या तरल चीजें बिना उल्टी किए पचा पा रहे हैं?",
        "Hinglish": "Pet me dard exactly kahan par hai, aur kya aap water ya liquids bina ulti ke digest kar pa rahe hain?"
    },
    "Does the pain radiate through to your back or shoulder, and have you had fever, chills, or yellowing of the eyes?": {
        "Hindi": "क्या यह दर्द पीठ या कंधे की तरफ जाता है, और क्या आपको बुखार, कंपकंपी या आंखों में पीलापन है?",
        "Hinglish": "Kya dard back ya shoulder me radiate kar raha hai, aur kya bukhar ya eyes me yellowness hai?"
    },
    "How many days have you had the fever, and what was the highest temperature measured?": {
        "Hindi": "आपको बुखार कितने दिनों से है, और थर्मामीटर पर सबसे अधिक तापमान कितना मापा गया?",
        "Hinglish": "Aapko bukhar kitne dino se hai, aur thermometer par highest temperature kitna measure hua?"
    },
    "Do you have any neck stiffness, confusion, extreme shivering, or any new rash on your skin?": {
        "Hindi": "क्या आपको गर्दन में अकड़न, भ्रम या अत्यधिक कंपकंपी है, या त्वचा पर कोई नया दाना/चकत्ता है?",
        "Hinglish": "Kya gardan me stiffness, confusion, extreme shivering, ya skin par koi naya rash hai?"
    },
    "When did your symptoms first begin, and have they been constant, worsening, or coming and going?": {
        "Hindi": "यह लक्षण सबसे पहले कब शुरू हुए और क्या यह समय के साथ बढ़ रहे हैं, घट रहे हैं या लगातार बने हुए हैं?",
        "Hinglish": "Ye symptoms sabse pehle kab shuru hue, aur kya ye continuously badh rahe hain ya kam-zyada ho rahe hain?"
    },
    "On a scale from 0 to 10 (where 0 is no discomfort and 10 is unbearable), how severe is your symptom right now?": {
        "Hindi": "0 से 10 के पैमाने पर (जहाँ 0 कोई दर्द नहीं और 10 असहनीय दर्द है), अभी आपका दर्द कितना तीव्र है?",
        "Hinglish": "0 se 10 ke scale par (jahan 0 koi dard nahi aur 10 unbearable pain hai), abhi dard kitna severe hai?"
    },
    "Are you currently taking any prescription medications, or do you have any relevant medical conditions or allergies?": {
        "Hindi": "क्या आप कोई नियमित दवाइयाँ ले रहे हैं, या आपको कोई पुरानी बीमारी या एलर्जी है?",
        "Hinglish": "Kya aap koi regular medications le rahe hain, ya aapko koi existing medical conditions ya allergies hain?"
    }
}

def get_localized_question(msg_obj: dict, target_lang: str) -> str:
    """Translates or fetches localized question text based on active trilingual choice."""
    if not msg_obj:
        return ""
    if target_lang == "Hindi" and msg_obj.get("content_hi"):
        return msg_obj["content_hi"]
    if target_lang == "Hinglish" and msg_obj.get("content_hinglish"):
        return msg_obj["content_hinglish"]
    if target_lang == "English" and msg_obj.get("content_en"):
        return msg_obj["content_en"]

    base_text = msg_obj.get("content_en") or msg_obj.get("content", "")
    base_clean = base_text.strip()
    if base_clean in CLINICAL_QUESTIONS_TRANSLATION:
        trans_dict = CLINICAL_QUESTIONS_TRANSLATION[base_clean]
        if target_lang in trans_dict:
            return trans_dict[target_lang]

    # Dynamic fallback via backend Google translation
    try:
        from backend.services.translation import translation_service
        if target_lang == "Hindi":
            return translation_service.translate_to_hindi(base_clean, default_hindi=base_clean)
        elif target_lang == "Hinglish":
            return translation_service.translate_to_hinglish(base_clean, default_hinglish=base_clean)
        elif target_lang == "English":
            return translation_service.translate_to_english(base_clean)
    except Exception:
        pass

    return base_text

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "case_id" not in st.session_state:
    st.session_state.case_id = None
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "max_turns" not in st.session_state:
    st.session_state.max_turns = 8
if "is_emergency" not in st.session_state:
    st.session_state.is_emergency = False
if "intake_mode" not in st.session_state:
    st.session_state.intake_mode = "chat"  # 'chat', 'review', 'submitted'
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "processed_audio_hashes" not in st.session_state:
    st.session_state.processed_audio_hashes = set()
if "dock_input_mode" not in st.session_state:
    st.session_state.dock_input_mode = "type"

def reset_intake_session():
    """Cleans up current patient intake session."""
    st.session_state.messages = []
    st.session_state.case_id = None
    st.session_state.turn_count = 0
    st.session_state.is_emergency = False
    st.session_state.intake_mode = "chat"
    st.session_state.uploaded_docs = []
    st.session_state.processed_audio_hashes = set()
    st.session_state.dock_input_mode = "type"

# Sidebar: Hospital Info, Case Management, and Staff Access Code Gate
with st.sidebar:
    st.markdown("### 🏥 CareFlow Health System")
    st.caption("Patient Clinical Intake & Triage Portal")

    if st.session_state.case_id:
        st.divider()
        st.markdown(f"**Active Case:** `{st.session_state.case_id}`")
        if st.button("New Intake Session 🔄", use_container_width=True, key="sidebar_new_intake"):
            reset_intake_session()
            st.rerun()

    st.divider()
    st.markdown("### 📋 Clinician Reviewer Portal")
    if st.button("Open Reviewer Dashboard 📋", type="primary", use_container_width=True, key="btn_patient_to_reviewer"):
        st.switch_page("pages/reviewer.py")

    st.divider()
    st.page_link("pages/model_showcase.py", label="🧠 Model Architecture", icon="🧠")

# Wrap entire intake portal inside clean centered container
with st.container():
    # 1. Hospital Header Bar
    case_badge_text = f"Case: {st.session_state.case_id}" if st.session_state.case_id else "Confidential Session"
    st.markdown(f"""
    <div class="hospital-header-card">
        <div class="hospital-brand-row">
            <div>
                <div class="hospital-title">🏥 CareFlow Health System</div>
                <div class="hospital-subtitle">Clinical Triage & Patient Symptom Intake Portal</div>
            </div>
            <div class="status-pill">
                🔒 {case_badge_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Language Selection & Session Reset Bar (Fully visible with comfortable margin)
    with st.container():
        col_lang, col_reset = st.columns([3, 1])
        with col_lang:
            active_lang = st.segmented_control(
                "🌐 Preferred Language / भाषा चुनें",
                ["English", "Hindi", "Hinglish"],
                default="Hinglish",
                key="segmented_lang_control",
                label_visibility="visible"
            ) or "Hinglish"
        with col_reset:
            st.write("") # spacing alignment
            st.write("")
            if st.button("New Intake 🔄", key="btn_top_reset", use_container_width=True, help="Start a new confidential intake session"):
                reset_intake_session()
                st.rerun()

    # Map to ASR code for transcription
    asr_lang = "hi" if active_lang == "Hindi" else ("en" if active_lang == "English" else "auto")

    # 3. Emergency Disclaimer Notice
    st.markdown("""
    <div class="emergency-banner">
        <strong>🚨 Emergency Notice:</strong> If you are experiencing severe chest pain, extreme shortness of breath, sudden weakness, or uncontrolled bleeding, 
        <strong>immediately call 112 / 911</strong> or proceed to the nearest hospital emergency department.
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # MODE 1: CHAT CONVERSATION INTERFACE
    # -----------------------------------------------------------------
    if st.session_state.intake_mode == "chat":
        # Render Chat History
        # If no messages yet, show assistant greeting
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🩺"):
                st.write(GREETINGS.get(active_lang, GREETINGS["Hinglish"]))

        # Render all conversational messages in history
        for msg in st.session_state.messages:
            if msg.get("role") in ("patient", "user"):
                with st.chat_message("user", avatar="👤"):
                    src = msg.get("source_type", "patient_typed")
                    if src == "voice":
                        st.markdown('<span class="voice-badge">🎤 Voice Recording</span>', unsafe_allow_html=True)
                    elif src == "ocr":
                        st.markdown('<span class="ocr-badge">📄 Medical Document</span>', unsafe_allow_html=True)
                    st.write(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                with st.chat_message("assistant", avatar="🩺"):
                    q_text = get_localized_question(msg, active_lang)
                    st.write(q_text)

        # Helper function to send message & advance questioning
        def handle_patient_input(text_content: str, source_type: str = "patient_typed", raw_transcript: str = None, conf: float = None):
            if not text_content.strip():
                return
            with st.spinner("CareFlow is analyzing symptoms and preparing next clinical question..."):
                try:
                    # 1. Initialize case if not exists
                    if not st.session_state.case_id:
                        new_case = api_client.create_case(
                            preferred_language=active_lang,
                            chief_complaint=text_content.strip()
                        )
                        st.session_state.case_id = new_case["id"]

                    # 2. Append message to backend
                    msg_res = api_client.send_message(
                        case_id=st.session_state.case_id,
                        content=text_content.strip(),
                        raw_transcript=raw_transcript,
                        source_type=source_type,
                        source_language=active_lang,
                        transcription_confidence=conf,
                        is_confirmed=True
                    )
                    st.session_state.messages.append({
                        "role": "patient",
                        "content": text_content.strip(),
                        "raw_transcript": raw_transcript,
                        "source_type": source_type,
                        "source_language": active_lang,
                        "transcription_confidence": conf
                    })
                    if msg_res.get("is_emergency"):
                        st.session_state.is_emergency = True

                    # 3. Request next MedGemma follow-up question
                    q_res = api_client.get_next_question(st.session_state.case_id)
                    st.session_state.turn_count += 1

                    if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count >= st.session_state.max_turns:
                        st.session_state.intake_mode = "review"
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": q_res["next_question"],
                            "content_en": q_res.get("next_question_en") or q_res["next_question"],
                            "content_hi": q_res.get("next_question_hi")
                        })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error communicating with clinical backend: {e}")

        # -------------------------------------------------------------
        # UNIFIED CLINICAL INPUT DOCK (Integrated Toolbar & Input Area)
        # -------------------------------------------------------------
        with st.container(border=True):
            # Row 1: Quick Action Response Chips (Skip, Unknown, Review)
            col_act1, col_act2, col_act3 = st.columns([1, 1, 1.2])
            with col_act1:
                skip_btn_label = "⏩ Skip Question" if active_lang == "English" else ("⏩ प्रश्न छोड़ें" if active_lang == "Hindi" else "⏩ Skip Question")
                if st.button(skip_btn_label, key="btn_skip_q", use_container_width=True):
                    skip_text = "I prefer to skip this question." if active_lang == "English" else ("मैं इस प्रश्न को छोड़ना चाहता हूँ।" if active_lang == "Hindi" else "Main is question ko skip karna chahta hoon.")
                    handle_patient_input(skip_text, source_type="patient_typed")
            with col_act2:
                unk_btn_label = "❓ Not Sure" if active_lang == "English" else ("❓ पता नहीं" if active_lang == "Hindi" else "❓ Not Sure / Don't Know")
                if st.button(unk_btn_label, key="btn_unk_q", use_container_width=True):
                    unk_text = "I do not know or am not sure of the answer." if active_lang == "English" else ("मुझे इसका उत्तर निश्चित रूप से नहीं पता है।" if active_lang == "Hindi" else "Mujhe sure nahi pata is baare me.")
                    handle_patient_input(unk_text, source_type="patient_typed")
            with col_act3:
                rev_btn_label = "📋 Review & Submit" if active_lang == "English" else ("📋 समीक्षा और जमा करें" if active_lang == "Hindi" else "📋 Review & Submit")
                if st.button(rev_btn_label, key="btn_early_sub", use_container_width=True):
                    if st.session_state.messages:
                        st.session_state.intake_mode = "review"
                        st.rerun()
                    else:
                        st.warning("Please enter your symptoms before reviewing.")

            # Row 2: Unified Input Mode Selector (Type vs Mic vs Report Upload)
            selected_mode = st.segmented_control(
                "Input Method",
                ["💬 Type Symptoms", "🎙️ Voice Input", "📎 Attach Report"],
                default="💬 Type Symptoms",
                key="input_dock_mode_control",
                label_visibility="collapsed"
            ) or "💬 Type Symptoms"

            # Mode Details rendering seamlessly inside the dock
            if selected_mode == "🎙️ Voice Input":
                st.markdown("""
                <div class="dock-mode-card">
                    <div class="dock-title">🎙️ Voice Symptom Dictation</div>
                    <div class="dock-desc">
                        Press the microphone button below to record your symptoms. Speak naturally in <strong>English, Hindi, or Hinglish</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                audio_prompt = st.audio_input(label="Voice recording input", label_visibility="collapsed", key="dock_audio_input")
                audio_file_upl = st.file_uploader("Or upload an audio clip (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a", "ogg"], key="dock_audio_file_upl")
                
                incoming_audio = None
                incoming_filename = "patient_speech.wav"
                if audio_prompt is not None:
                    incoming_audio = audio_prompt.getvalue()
                elif audio_file_upl is not None:
                    incoming_audio = audio_file_upl.getvalue()
                    incoming_filename = audio_file_upl.name

                if incoming_audio is not None:
                    audio_bytes = incoming_audio
                    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
                    if audio_hash not in st.session_state.processed_audio_hashes:
                        st.session_state.processed_audio_hashes.add(audio_hash)
                        with st.spinner("Transcribing your voice recording..."):
                            try:
                                trans_res = api_client.transcribe_audio(
                                    file_bytes=audio_bytes,
                                    filename=incoming_filename,
                                    language=asr_lang
                                )
                                transcript = trans_res.get("transcript", "").strip()
                                conf = trans_res.get("confidence")
                                if transcript:
                                    st.info(f"🎤 Transcribed: &ldquo;{transcript}&rdquo;")
                                    handle_patient_input(
                                        text_content=transcript,
                                        source_type="voice",
                                        raw_transcript=transcript,
                                        conf=conf
                                    )
                                else:
                                    st.warning("No speech detected. Please speak clearly or type below.")
                            except Exception as e:
                                st.error(f"Voice transcription error: {e}")

            elif selected_mode == "📎 Attach Report":
                st.markdown("""
                <div class="dock-mode-card">
                    <div class="dock-title">📎 Attach Medical Prescription or Lab Report</div>
                    <div class="dock-desc">
                        Upload an image or PDF of prescriptions or diagnostic test results. Clinical OCR will extract text into your intake file.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                upl_doc = st.file_uploader("Upload Medical Document", type=["png", "jpg", "jpeg", "pdf"], key="dock_file_upl", label_visibility="collapsed")
                if upl_doc is not None:
                    if st.button("Scan & Attach Document 📄", key="btn_dock_attach_doc", type="primary", use_container_width=True):
                        with st.spinner("Extracting medical information via clinical OCR..."):
                            try:
                                if not st.session_state.case_id:
                                    new_case = api_client.create_case(
                                        preferred_language=active_lang,
                                        chief_complaint=f"Attached Medical Document: {upl_doc.name}"
                                    )
                                    st.session_state.case_id = new_case["id"]

                                doc_res = api_client.upload_document(st.session_state.case_id, upl_doc.getvalue(), upl_doc.name)
                                st.session_state.uploaded_docs.append(upl_doc.name)
                                st.session_state.messages.append({
                                    "role": "patient",
                                    "content": f"[Attached Medical Document: {upl_doc.name} ({doc_res.get('char_count', 0)} characters extracted)]",
                                    "source_type": "ocr"
                                })
                                # Advance question
                                q_res = api_client.get_next_question(st.session_state.case_id)
                                st.session_state.turn_count += 1
                                if q_res.get("next_question"):
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": q_res["next_question"],
                                        "content_en": q_res.get("next_question_en") or q_res["next_question"],
                                        "content_hi": q_res.get("next_question_hi")
                                    })
                                st.success(f"Document '{upl_doc.name}' attached to clinical file.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error scanning document: {e}")

        # Primary Chat Input (always accessible and pinned at the bottom)
        input_placeholder = "Describe your symptoms, location, pain intensity, duration... (Type here)" if active_lang == "English" else ("अपने लक्षणों, दर्द और परेशानी के बारे में यहाँ लिखें..." if active_lang == "Hindi" else "Apne symptoms, pain intensity aur duration yahan type karein...")
        chat_text = st.chat_input(input_placeholder)
        if chat_text:
            handle_patient_input(chat_text, source_type="patient_typed")

    # -----------------------------------------------------------------
    # MODE 2: REVIEW & EDIT BEFORE CLINICIAN SUBMISSION
    # -----------------------------------------------------------------
    elif st.session_state.intake_mode == "review":
        st.subheader("📋 Review Your Clinical Statements")
        st.write("Please review your reported symptoms. You may edit your words before submitting to the hospital clinical queue.")

        patient_msgs = [m for m in st.session_state.messages if m.get("role") in ("patient", "user")]
        edited_inputs = []

        for idx, m in enumerate(patient_msgs):
            st.markdown(f"**Item {idx + 1} ({m.get('source_type', 'typed')}):**")
            val = st.text_area(
                f"Item {idx + 1}",
                value=m.get("content", ""),
                height=80,
                key=f"edit_patient_item_{idx}",
                label_visibility="collapsed"
            )
            edited_inputs.append((m.get("content", ""), val))

        col_sub1, col_sub2 = st.columns([2, 1])
        with col_sub1:
            if st.button("Submit Intake for Healthcare Professional Review ✅", type="primary", use_container_width=True):
                with st.spinner("Submitting clinical intake to hospital reviewer queue..."):
                    try:
                        # Send any corrected items
                        for i, (orig, edited) in enumerate(edited_inputs):
                            if orig != edited and edited.strip():
                                api_client.send_message(
                                    case_id=st.session_state.case_id,
                                    content=f"Correction to Item #{i+1}: {edited.strip()}",
                                    source_type="patient_correction"
                                )
                        # Submit case
                        api_client.submit_case(st.session_state.case_id)
                        st.session_state.intake_mode = "submitted"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error submitting intake: {e}")

        with col_sub2:
            if st.button("← Back to Chat", use_container_width=True):
                st.session_state.intake_mode = "chat"
                st.rerun()

    # -----------------------------------------------------------------
    # MODE 3: SUBMISSION CONFIRMATION
    # -----------------------------------------------------------------
    elif st.session_state.intake_mode == "submitted":
        st.balloons()
        st.success("🎉 Your clinical intake has been successfully submitted to the hospital triage queue!")

        st.markdown(f"""
        <div class="card-confirm">
            <h3 style="margin-top:0; color:#0F766E;">Clinical Intake Confirmation</h3>
            <p><strong>Case ID:</strong> <code>{st.session_state.case_id}</code></p>
            <p><strong>Status:</strong> Queued for Licensed Healthcare Professional Review</p>
            <p>Your responses have been securely transferred to the hospital command center. A clinician will inspect your file, LightGBM clinical risk predictions, and MedGemma evidence summary.</p>
            <div style="background:#F1F5F9; border-radius:8px; padding:14px; font-size:13px; color:#475569; margin-top:14px; line-height:1.5;">
                <strong>Emergency Reminder:</strong> CareFlow is an automated clinical assistant prototype and does not provide a definitive diagnosis. If your symptoms worsen or you develop severe chest pain or shortness of breath, call 112 / 911 immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Another Patient Intake 🔄", type="primary", use_container_width=True):
            reset_intake_session()
            st.rerun()
