import streamlit as st
import sys
import hashlib
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import api_client

st.set_page_config(
    page_title="Patient Intake | CareFlow",
    page_icon="🩺",
    layout="centered"
)

# Clean Clinical CSS
st.markdown("""
<style>
    .patient-header {
        background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .privacy-notice {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 6px;
        color: #1E40AF;
        font-size: 13.5px;
        margin-bottom: 20px;
    }
    .emergency-banner {
        background-color: #FEE2E2;
        border: 2px solid #DC2626;
        padding: 16px;
        border-radius: 8px;
        color: #991B1B;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 24px;
    }
    .chat-bubble-patient {
        background-color: #E2E8F0;
        color: #1E293B;
        padding: 12px 16px;
        border-radius: 12px 12px 0 12px;
        margin-left: 20%;
        margin-bottom: 12px;
        font-size: 14.5px;
    }
    .chat-bubble-assistant {
        background-color: #CCFBF1;
        color: #115E59;
        border: 1px solid #99F6E4;
        padding: 14px 18px;
        border-radius: 12px 12px 12px 0;
        margin-right: 20%;
        margin-bottom: 12px;
        font-size: 15px;
        font-weight: 500;
    }
    .card-review {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# BILINGUAL CLINICAL QUESTION & CONVERSATION TRANSLATION DICTIONARY
# Guarantees full-screen seamless language switching (English <-> Hindi)
# =====================================================================
CLINICAL_QUESTIONS_TRANSLATION = {
    "Does the chest pain radiate or spread to your left arm, shoulder, neck, jaw, or back?": "क्या सीने का दर्द आपके बाएं हाथ, कंधे, गर्दन, जबड़े या पीठ की तरफ फैल रहा है?",
    "Are you experiencing any shortness of breath, cold sweats, nausea, or lightheadedness along with the chest pain?": "क्या सीने में दर्द के साथ सांस फूलना, ठंडा पसीना, उल्टी या चक्कर आ रहे हैं?",
    "Did the headache peak at maximum intensity suddenly within seconds (like a thunderclap), or did it build gradually?": "क्या यह सिरदर्द अचानक कुछ ही सेकंड में असहनीय तेज हो गया, या धीरे-धीरे बढ़ा?",
    "Do you have any neck stiffness, pain when bending your head forward, fever, or sensitivity to bright light?": "क्या आपको गर्दन में अकड़न, सिर आगे झुकाने में दर्द, बुखार या तेज रोशनी से परेशानी हो रही है?",
    "Are you able to speak in full sentences without gasping, and does lying flat make your breathing harder?": "क्या आप बिना सांस फूले पूरे वाक्य बोल पा रहे हैं, और क्या सीधे लेटने पर सांस ज्यादा फूलती है?",
    "Have you noticed any audible wheezing, cough with phlegm, fever, or swelling of your face or ankles?": "क्या सांस लेते समय सीटी जैसी आवाज (घरघराहट), खांसी में बलगम, बुखार या चेहरे/पैरों में सूजन है?",
    "Where exactly in your abdomen is the pain located, and are you able to keep water or fluids down without vomiting?": "पेट में दर्द ठीक किस जगह पर है, और क्या आप पानी या तरल चीजें बिना उल्टी किए पचा पा रहे हैं?",
    "Does the pain radiate through to your back or shoulder, and have you had fever, chills, or yellowing of the eyes?": "क्या यह दर्द पीठ या कंधे की तरफ जाता है, और क्या आपको बुखार, कंपकंपी या आंखों में पीलापन है?",
    "How many days have you had the fever, and what was the highest temperature measured?": "आपको बुखार कितने दिनों से है, और थर्मामीटर पर सबसे अधिक तापमान कितना मापा गया?",
    "Do you have any neck stiffness, confusion, extreme shivering, or any new rash on your skin?": "क्या आपको गर्दन में अकड़न, भ्रम या अत्यधिक कंपकंपी है, या त्वचा पर कोई नया दाना/चकत्ता है?",
    "When did your symptoms first begin, and have they been constant, worsening, or coming and going?": "यह लक्षण सबसे पहले कब शुरू हुए और क्या यह समय के साथ बढ़ रहे हैं, घट रहे हैं या लगातार बने हुए हैं?",
    "On a scale from 0 to 10 (where 0 is no discomfort and 10 is unbearable), how severe is your symptom right now?": "0 से 10 के पैमाने पर (जहाँ 0 कोई दर्द नहीं और 10 असहनीय दर्द है), अभी आपका दर्द कितना तीव्र है?",
    "Are you currently taking any prescription medications, or do you have any relevant medical conditions or allergies?": "क्या आप कोई नियमित दवाइयाँ ले रहे हैं, या आपको कोई पुरानी बीमारी या एलर्जी है?",
    "Could you clarify any other symptoms you are experiencing?": "क्या आप अपने लक्षणों के बारे में कुछ और विस्तार से बता सकते हैं?",
    "What other symptoms or discomfort are you experiencing?": "आप इसके अलावा और क्या लक्षण या परेशानी महसूस कर रहे हैं?",
    "What symptoms or health concerns are you experiencing today?": "आज आप किन लक्षणों या स्वास्थ्य संबंधी परेशानियों का सामना कर रहे हैं?"
}

REVERSE_QUESTIONS_TRANSLATION = {v: k for k, v in CLINICAL_QUESTIONS_TRANSLATION.items()}

COMMON_PATIENT_PHRASES = {
    "chest pain": "सीने में दर्द",
    "fever": "बुखार",
    "headache": "सिरदर्द",
    "cough": "खांसी",
    "vomiting": "उल्टी",
    "shortness of breath": "सांस लेने में तकलीफ",
    "stomach pain": "पेट में दर्द",
    "weakness": "कमजोरी",
    "chills": "कंपकंपी / ठंड"
}
REVERSE_PATIENT_PHRASES = {v: k for k, v in COMMON_PATIENT_PHRASES.items()}

def translate_question_text(text: str, target_lang: str) -> str:
    """Translates clinical question text between English and Hindi dynamically."""
    if not text:
        return ""
    t_clean = text.strip()
    if target_lang == "Hindi":
        if t_clean in CLINICAL_QUESTIONS_TRANSLATION:
            return CLINICAL_QUESTIONS_TRANSLATION[t_clean]
        for en, hi in CLINICAL_QUESTIONS_TRANSLATION.items():
            if en in t_clean:
                return t_clean.replace(en, hi)
    elif target_lang == "English":
        if t_clean in REVERSE_QUESTIONS_TRANSLATION:
            return REVERSE_QUESTIONS_TRANSLATION[t_clean]
        for hi, en in REVERSE_QUESTIONS_TRANSLATION.items():
            if hi in t_clean:
                return t_clean.replace(hi, en)
    return text

def get_patient_message_translation(text: str, target_lang: str) -> str:
    """Provides subtitle translation for patient messages when switching language."""
    if not text or text.startswith("[Attached"):
        return ""
    has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
    if target_lang == "English" and has_devanagari:
        # Hindi -> English translation hint
        matched = [en for hi, en in REVERSE_PATIENT_PHRASES.items() if hi in text]
        if matched:
            return f"Translated: Mentions {', '.join(matched)}"
    elif target_lang == "Hindi" and not has_devanagari:
        # English -> Hindi translation hint
        matched = [hi for en, hi in COMMON_PATIENT_PHRASES.items() if en in text.lower()]
        if matched:
            return f"अनुवाद: {', '.join(matched)} का उल्लेख"
    return ""

# Bilingual Dictionary
UI_TEXT = {
    "English": {
        "title": "🩺 CareFlow Clinical Intake",
        "subtitle": "Private & Confidential Clinical Symptom Intake",
        "privacy_title": "🛡️ Privacy Protection Notice:",
        "privacy_body": "Please do <strong>NOT</strong> enter your name, telephone number, home address, social security / Aadhaar number, or any other identifying personal information. You will be assigned a random, non-identifying Case ID.",
        "emergency_title": "🚨 POTENTIAL MEDICAL EMERGENCY DETECTED:",
        "emergency_body": "If you are experiencing severe chest pain, trouble breathing, sudden weakness, or another life-threatening emergency, <strong>immediately call 911 / 112 (or local emergency services) or proceed to the nearest Emergency Room.</strong> Do not wait for online clinical review.",
        "start_title": "Begin New Clinical Intake",
        "lang_label": "Preferred Language / भाषा चुनें",
        "complaint_label": "What symptoms or health concerns are you experiencing today?",
        "complaint_placeholder": "e.g., I have had a severe sore throat and high fever for 2 days, and it hurts when I swallow. / मुझे दो दिन से तेज़ बुखार और गले में दर्द है...",
        "doc_upload_header": "📄 Attach Medical Document or Prescription (Optional)",
        "doc_upload_help": "Upload an image (PNG, JPG) or PDF of a prescription, lab test, or clinical report. Text will be automatically extracted and reviewed.",
        "doc_extracted_alert": "✅ Medical document parsed and attached to your clinical record.",
        "start_btn": "Start Intake Assessment →",
        "case_id_label": "Case ID",
        "question_count": "Question {curr} of {max}",
        "you_label": "You",
        "assistant_label": "CareFlow Assistant",
        "q_label": "CareFlow Follow-Up Question:",
        "input_label": "Your Response:",
        "input_placeholder": "Type your answer here in English or Hindi (हिन्दी)...",
        "send_btn": "Send Response 💬",
        "skip_btn": "Skip ⏩",
        "unknown_btn": "I Don't Know ❓",
        "early_submit_btn": "Finish & Review Now (Submit Early) 📋",
        "upload_during_chat": "📎 Upload Additional Medical Document / Lab Report (PDF or Image)",
        "upload_btn": "Extract & Ingest Document 📤",
        "voice_header": "🎤 Voice Input (Record or Upload Audio)",
        "voice_help": "Record symptoms in English, Hindi (हिन्दी), or Hinglish. You will review and edit the transcript before sending.",
        "voice_transcript_title": "🎤 Voice Transcript (Review & Edit before sending):",
        "voice_confirm_btn": "Confirm & Send Voice Response 💬",
        "voice_discard_btn": "Discard Recording ❌",
        "doc_preview_title": "📄 Extracted Document Text (Review & Edit before sending):",
        "doc_confirm_btn": "Confirm & Ingest Document Text 📋",
        "doc_discard_btn": "Discard Document ❌",
        "review_title": "Review Your Intake Information",
        "review_subtitle": "Please review the information you provided. You may correct or add details before submitting for clinician review.",
        "submit_btn": "Submit Intake for Healthcare Professional Review ✅",
        "return_btn": "← Return to Questions",
        "success_title": "🎉 Your clinical intake has been successfully submitted!",
        "confirm_card_title": "Intake Confirmation",
        "confirm_status": "Queued for Registered Healthcare Professional Review",
        "confirm_body": "Your responses and uploaded medical documentation have been securely transferred to the clinical reviewer dashboard. A registered medical reviewer will inspect your case notes and clinical summary.",
        "safety_reminder": "<strong>Important Safety Reminder:</strong> CareFlow is an automated clinical assistant prototype. It does not diagnose medical conditions. If your symptoms worsen or you develop severe chest pain, difficulty breathing, or sudden numbness, seek emergency medical care immediately.",
        "restart_btn": "Start Another Patient Intake"
    },
    "हिन्दी (Hindi)": {
        "title": "🩺 CareFlow क्लिनिकल इनटेक",
        "subtitle": "सुरक्षित एवं गोपनीय क्लिनिकल लक्षण मूल्यांकन",
        "privacy_title": "🛡️ गोपनीयता संरक्षण सूचना:",
        "privacy_body": "कृपया अपना <strong>नाम, फ़ोन नंबर, घर का पता, आधार कार्ड नंबर</strong> या कोई अन्य पहचान योग्य व्यक्तिगत जानकारी दर्ज <strong>न करें</strong>। आपको एक सुरक्षित, गैर-पहचान योग्य केस आईडी प्रदान की जाएगी।",
        "emergency_title": "🚨 संभावित आपातकालीन स्थिति पहचानी गई:",
        "emergency_body": "यदि आप सीने में तेज़ दर्द, सांस लेने में अत्यधिक तकलीफ़, अचानक कमजोरी या कोई अन्य गंभीर आपातकालीन स्थिति महसूस कर रहे हैं, तो <strong>तुरंत आपातकालीन सेवा (112 / 108) पर कॉल करें या नज़दीकी अस्पताल के इमरजेंसी वार्ड में जाएं।</strong> ऑनलाइन समीक्षा की प्रतीक्षा न करें।",
        "start_title": "नया क्लिनिकल इनटेक शुरू करें",
        "lang_label": "पसंदीदा भाषा (Preferred Language)",
        "complaint_label": "आज आप किन लक्षणों या स्वास्थ्य संबंधी परेशानियों का सामना कर रहे हैं?",
        "complaint_placeholder": "उदा: मुझे दो दिन से तेज़ बुखार, गले में ख़राश और दर्द है, और निगलने में बहुत तकलीफ़ हो रही है...",
        "doc_upload_header": "📄 मेडिकल पर्ची या दस्तावेज़ अपलोड करें (वैकल्पिक)",
        "doc_upload_help": "डॉक्टर की पर्ची, लैब टेस्ट या रिपोर्ट की फ़ोटो (PNG, JPG) या PDF अपलोड करें। दस्तावेज़ से टेक्स्ट सुरक्षित रूप से निकाला जाएगा।",
        "doc_extracted_alert": "✅ दस्तावेज़ से टेक्स्ट सफलतापूर्वक प्राप्त कर लिया गया है।",
        "start_btn": "इनटेक मूल्यांकन शुरू करें →",
        "case_id_label": "केस आईडी",
        "question_count": "प्रश्न {curr} / {max}",
        "you_label": "आप",
        "assistant_label": "CareFlow सहायक",
        "q_label": "CareFlow अनुवर्ती प्रश्न:",
        "input_label": "आपका उत्तर:",
        "input_placeholder": "यहाँ अपना उत्तर हिंदी या अंग्रेजी में लिखें...",
        "send_btn": "उत्तर भेजें 💬",
        "skip_btn": "छोड़ें ⏩",
        "unknown_btn": "मुझे नहीं पता ❓",
        "early_submit_btn": "समीक्षा करें एवं अभी जमा करें 📋",
        "upload_during_chat": "📎 अतिरिक्त मेडिकल दस्तावेज़ या रिपोर्ट अपलोड करें (PDF/Image OCR)",
        "upload_btn": "दस्तावेज़ से टेक्स्ट निकालें एवं जोड़ें 📤",
        "voice_header": "🎤 बोलकर बताएं (आवाज़ रिकॉर्ड करें या ऑडियो अपलोड करें)",
        "voice_help": "हिंदी, अंग्रेजी या हिंग्लिश में बोलें। भेजने से पहले आप टेक्स्ट की समीक्षा और सुधार कर सकेंगे।",
        "voice_transcript_title": "🎤 वॉइस ट्रांसक्रिप्ट (भेजने से पहले समीक्षा एवं सुधार करें):",
        "voice_confirm_btn": "पुष्टि करें और उत्तर भेजें 💬",
        "voice_discard_btn": "रद्द करें ❌",
        "doc_preview_title": "📄 दस्तावेज़ से प्राप्त टेक्स्ट (समीक्षा एवं सुधार करें):",
        "doc_confirm_btn": "पुष्टि करें और दस्तावेज़ जोड़ें 📋",
        "doc_discard_btn": "दस्तावेज़ हटाएं ❌",
        "review_title": "अपनी इनटेक जानकारी की समीक्षा करें",
        "review_subtitle": "कृपया अपने दिए गए उत्तरों की समीक्षा करें। डॉक्टर/समीक्षक के पास जमा करने से पहले आप सुधार कर सकते हैं।",
        "submit_btn": "स्वास्थ्य विशेषज्ञ समीक्षा हेतु इनटेक जमा करें ✅",
        "return_btn": "← प्रश्नों पर वापस जाएँ",
        "success_title": "🎉 आपका क्लिनिकल इनटेक सफलतापूर्वक जमा हो गया है!",
        "confirm_card_title": "इनटेक पावती (Confirmation)",
        "confirm_status": "पंजीकृत स्वास्थ्य समीक्षक समीक्षा हेतु कतारबद्ध",
        "confirm_body": "आपके उत्तर और अपलोड किए गए मेडिकल दस्तावेज़ क्लिनिकल समीक्षक डैशबोर्ड पर सुरक्षित रूप से स्थानांतरित कर दिए गए हैं। एक पंजीकृत चिकित्सक आपके केस और सारांश की समीक्षा करेंगे।",
        "safety_reminder": "<strong>महत्वपूर्ण सुरक्षा चेतावनी:</strong> CareFlow एक स्वचालित क्लिनिकल सहायक प्रोटोटाइप है। यह कोई चिकित्सा निदान (डायग्नोसिस) नहीं करता है। यदि आपके लक्षण बिगड़ते हैं या आपको सीने में तेज़ दर्द अथवा सांस लेने में तकलीफ़ होती है, तो तुरंत आपातकालीन चिकित्सा सहायता लें।",
        "restart_btn": "नया इनटेक शुरू करें"
    },
    "Auto-detect / Hinglish": {
        "title": "🩺 CareFlow Clinical Intake",
        "subtitle": "Private & Confidential Multilingual Clinical Intake (English / Hindi / Hinglish)",
        "privacy_title": "🛡️ Privacy Protection Notice:",
        "privacy_body": "Please do <strong>NOT</strong> enter your name, phone number, address, or Aadhaar number. A confidential Case ID will be assigned to your session.",
        "emergency_title": "🚨 POTENTIAL MEDICAL EMERGENCY DETECTED:",
        "emergency_body": "If you have severe chest pain, breathing difficulty, sudden weakness, or emergency symptoms, <strong>call 112 / 911 or go to the nearest emergency room immediately.</strong> Do not wait for online clinical review.",
        "start_title": "Begin New Clinical Intake (नया इनटेक शुरू करें)",
        "lang_label": "Preferred Language / भाषा चुनें",
        "complaint_label": "What symptoms or health concerns are you experiencing today? / आज क्या तकलीफ है?",
        "complaint_placeholder": "Type or speak in English, Hindi (हिन्दी), or Hinglish (e.g. 2 din se severe chest pain aur bukhar hai...)",
        "doc_upload_header": "📄 Attach Medical Document or Prescription (Optional)",
        "doc_upload_help": "Upload an image (PNG, JPG) or PDF of a prescription, lab test, or clinical report. Key clinical facts will be scanned automatically.",
        "doc_extracted_alert": "✅ Medical document parsed and attached to your clinical record.",
        "start_btn": "Start Intake Assessment →",
        "case_id_label": "Case ID",
        "question_count": "Question {curr} of {max}",
        "you_label": "You",
        "assistant_label": "CareFlow Assistant",
        "q_label": "CareFlow Follow-Up Question:",
        "input_label": "Your Response / आपका उत्तर:",
        "input_placeholder": "Type in English, Hindi, or Hinglish...",
        "send_btn": "Send Response 💬",
        "skip_btn": "Skip ⏩",
        "unknown_btn": "I Don't Know ❓",
        "early_submit_btn": "Finish & Review Now (Submit Early) 📋",
        "upload_during_chat": "📎 Attach Medical Document / Lab Report (PDF or Image)",
        "upload_btn": "Attach Document 📤",
        "voice_header": "🎤 Voice Input (Record or Upload Audio)",
        "voice_help": "Record symptoms in English, Hindi (हिन्दी), or Hinglish. You will review and edit the transcript before sending.",
        "voice_transcript_title": "🎤 Voice Transcript (Review & Edit before sending):",
        "voice_confirm_btn": "Confirm & Send Voice Response 💬",
        "voice_discard_btn": "Discard Recording ❌",
        "doc_preview_title": "📄 Attached Medical Document:",
        "doc_confirm_btn": "Confirm & Ingest Document 📋",
        "doc_discard_btn": "Remove Document ❌",
        "review_title": "Review Your Intake Information",
        "review_subtitle": "Please review the information you provided. You may correct or add details before submitting for clinician review.",
        "submit_btn": "Submit Intake for Healthcare Professional Review ✅",
        "return_btn": "← Return to Questions",
        "success_title": "🎉 Your clinical intake has been successfully submitted!",
        "confirm_card_title": "Intake Confirmation",
        "confirm_status": "Queued for Registered Healthcare Professional Review",
        "confirm_body": "Your responses and uploaded medical documentation have been securely transferred to the clinical reviewer dashboard. A registered medical reviewer will inspect your case notes and clinical summary.",
        "safety_reminder": "<strong>Important Safety Reminder:</strong> CareFlow is an automated clinical assistant prototype. It does not diagnose medical conditions. If your symptoms worsen, seek emergency medical care immediately.",
        "restart_btn": "Start Another Patient Intake"
    }
}

# Initialize Session State
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "🌐 Auto-Detect Language (स्वचालित भाषा पहचान)"
if "case_id" not in st.session_state:
    st.session_state.case_id = None
if "intake_step" not in st.session_state:
    st.session_state.intake_step = "start"  # 'start', 'chatting', 'review_edit', 'submitted'
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "is_emergency" not in st.session_state:
    st.session_state.is_emergency = False
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "max_turns" not in st.session_state:
    st.session_state.max_turns = 8
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "pending_voice" not in st.session_state:
    st.session_state.pending_voice = None
if "pending_voice_start" not in st.session_state:
    st.session_state.pending_voice_start = None
if "pending_doc_preview" not in st.session_state:
    st.session_state.pending_doc_preview = None

# Language selector bar in sidebar / top
lang_options = ["🌐 Auto-Detect Language (स्वचालित भाषा पहचान)", "🇬🇧 English", "🇮🇳 हिन्दी (Hindi)"]
with st.sidebar:
    st.markdown("### 🌐 Language / भाषा")
    curr_idx = 0
    if st.session_state.selected_lang in lang_options:
        curr_idx = lang_options.index(st.session_state.selected_lang)
    chosen_lang = st.radio(
        "Interface & Speech Mode:",
        options=lang_options,
        index=curr_idx,
        key="lang_radio"
    )
    if chosen_lang != st.session_state.selected_lang:
        st.session_state.selected_lang = chosen_lang
        # Dynamic full-screen translation: translate current question & conversation history
        new_target = "Hindi" if "हिन्दी" in chosen_lang else "English"
        if st.session_state.current_question:
            st.session_state.current_question = translate_question_text(st.session_state.current_question, new_target)
        for m in st.session_state.messages:
            if m.get("role") == "assistant":
                m["content"] = translate_question_text(m.get("content", ""), new_target)
        st.rerun()

# Determine active UI dictionary and target language
if "हिन्दी" in st.session_state.selected_lang:
    current_target_lang = "Hindi"
    t = UI_TEXT["हिन्दी (Hindi)"]
    backend_lang = "Hindi"
    asr_lang = "hi"
elif "English" in st.session_state.selected_lang:
    current_target_lang = "English"
    t = UI_TEXT["English"]
    backend_lang = "English"
    asr_lang = "en"
else:
    current_target_lang = "English"
    t = UI_TEXT["Auto-detect / Hinglish"]
    backend_lang = "English"
    asr_lang = "auto"  # Triggers Deepgram detect_language=True

# Header
st.markdown(f"""
<div class="patient-header">
    <h2 style="margin:0; font-size: 24px;">{t['title']}</h2>
    <div style="font-size: 14px; opacity: 0.9; margin-top: 4px;">{t['subtitle']}</div>
</div>
""", unsafe_allow_html=True)

# Privacy Warning
st.markdown(f"""
<div class="privacy-notice">
    <strong>{t['privacy_title']}</strong><br/>
    {t['privacy_body']}
</div>
""", unsafe_allow_html=True)

# Emergency Warning Banner (Conditional)
if st.session_state.is_emergency:
    st.markdown(f"""
    <div class="emergency-banner">
        <strong>{t['emergency_title']}</strong><br/>
        {t['emergency_body']}
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# STEP 1: START INTAKE (Typed, Browser Mic, Audio Upload, & Document OCR)
# =====================================================================
if st.session_state.intake_step == "start":
    st.subheader(t["start_title"])

    # CASE A: PENDING INITIAL VOICE TRANSCRIPT REVIEW
    if st.session_state.pending_voice_start is not None:
        pv = st.session_state.pending_voice_start
        p_trans = pv.get("transcript", "")
        p_lang = pv.get("language", backend_lang)
        p_conf = pv.get("confidence")
        conf_pct = int(p_conf * 100) if p_conf is not None else 0
        conf_status = "Good acoustic signal" if (p_conf or 0) >= 0.70 else "⚠️ Lower acoustic confidence — please verify accuracy"

        st.markdown(f"""
        <div style="background-color: #F5F3FF; border: 2px solid #C4B5FD; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <h4 style="margin: 0 0 6px 0; color: #5B21B6; font-size: 15px;">{t['voice_transcript_title']}</h4>
            <div style="font-size: 13px; color: #6D28D9;">
                Language: <strong>{p_lang.title() if p_lang else 'Auto-detect'}</strong> | Estimated Confidence: <strong>~{conf_pct}% ({conf_status})</strong>
            </div>
            <div style="font-size: 11.5px; color: #7C3AED; margin-top: 4px;">
                ℹ️ <em>Confidence is an automated acoustic estimate, not a guarantee of medical accuracy. Please review and edit your words below.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("last_start_voice_hash") != pv.get("audio_hash"):
            st.session_state["voice_transcript_start_edit"] = p_trans
            st.session_state["last_start_voice_hash"] = pv.get("audio_hash")

        edited_start_voice = st.text_area(
            "Review or edit your transcribed complaint before starting intake:",
            value=st.session_state.get("voice_transcript_start_edit", p_trans),
            key="voice_transcript_start_edit",
            height=110
        )
        col_sv1, col_sv2 = st.columns([2, 1])
        with col_sv1:
            if st.button("Confirm & Start Intake Assessment →", type="primary", key="btn_confirm_start_voice", use_container_width=True):
                if edited_start_voice.strip():
                    with st.spinner("Initializing private intake session & analyzing voice input..."):
                        try:
                            # 1. Create case in backend
                            new_case = api_client.create_case(
                                preferred_language=backend_lang,
                                chief_complaint=edited_start_voice.strip()
                            )
                            case_id = new_case["id"]
                            st.session_state.case_id = case_id

                            # 2. Add first message as verified voice input
                            msg_res = api_client.send_message(
                                case_id=case_id,
                                content=edited_start_voice.strip(),
                                raw_transcript=p_trans,
                                source_type="voice",
                                source_language=p_lang,
                                transcription_confidence=p_conf,
                                is_confirmed=True
                            )
                            st.session_state.messages.append({
                                "role": "patient",
                                "content": edited_start_voice.strip(),
                                "raw_transcript": p_trans,
                                "source_type": "voice",
                                "source_language": p_lang,
                                "transcription_confidence": p_conf
                            })
                            if msg_res.get("is_emergency"):
                                st.session_state.is_emergency = True

                            # 3. Clean up start voice state
                            st.session_state.pending_voice_start = None
                            st.session_state.pop("voice_transcript_start_edit", None)
                            st.session_state.pop("last_start_voice_hash", None)

                            # 4. Request first follow-up question
                            q_res = api_client.get_next_question(case_id)
                            st.session_state.turn_count = 1

                            if q_res.get("next_question"):
                                st.session_state.current_question = q_res["next_question"]
                                st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})
                                st.session_state.intake_step = "chatting"
                            else:
                                st.session_state.intake_step = "review_edit"

                            st.rerun()
                        except Exception as e:
                            st.error(f"Error starting intake session: {e}")
                else:
                    st.warning("Please enter or edit your symptoms before continuing.")
        with col_sv2:
            if st.button(t["voice_discard_btn"], key="btn_discard_start_voice", use_container_width=True):
                st.session_state.pending_voice_start = None
                st.session_state.pop("voice_transcript_start_edit", None)
                st.session_state.pop("last_start_voice_hash", None)
                st.rerun()

    # STANDARD INITIAL INTAKE INPUTS (Typed / Voice Recording / Document OCR)
    else:
        st.markdown("#### ✍️ Option 1: Type Your Symptoms")
        chief_complaint = st.text_area(
            t["complaint_label"],
            placeholder=t["complaint_placeholder"],
            height=100,
            key="chief_complaint_typed"
        )

        st.markdown("#### 🎙️ Option 2: Record or Upload Audio (English / हिन्दी / Hinglish)")
        st.caption("Speak naturally in English, Hindi, or Hinglish. You will review and edit your transcript before starting intake.")
        col_audio1, col_audio2 = st.columns(2)
        rec_audio_start = None
        with col_audio1:
            if hasattr(st, "audio_input"):
                rec_audio_start = st.audio_input("Record via Microphone", key="mic_start")
        with col_audio2:
            upl_audio_start = st.file_uploader(
                "Upload Voice Audio (WAV, MP3, M4A, OGG, WEBM)",
                type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
                key="audio_start"
            )

        chosen_start_audio = rec_audio_start or upl_audio_start
        if chosen_start_audio is not None:
            audio_bytes = chosen_start_audio.getvalue()
            curr_hash = hashlib.sha256(audio_bytes).hexdigest()
            if st.session_state.pending_voice_start and st.session_state.get("last_start_voice_hash") != curr_hash:
                st.session_state.pending_voice_start = None
                st.session_state.pop("voice_transcript_start_edit", None)

            if st.button("Transcribe Initial Voice 🎙️", type="secondary", key="btn_transcribe_start", use_container_width=True):
                with st.spinner("Transcribing voice audio..."):
                    try:
                        fname = getattr(chosen_start_audio, "name", "recording.wav") or "recording.wav"
                        t_res = api_client.transcribe_audio(
                            file_bytes=audio_bytes,
                            filename=fname,
                            language=asr_lang
                        )
                        t_res["audio_hash"] = curr_hash
                        st.session_state.pending_voice_start = t_res
                        st.session_state["voice_transcript_start_edit"] = t_res.get("transcript", "")
                        st.session_state["last_start_voice_hash"] = curr_hash
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ {e}")

        # Document upload accordion before/with chief complaint
        with st.expander(t["doc_upload_header"], expanded=False):
            st.caption(t["doc_upload_help"])
            doc_file_start = st.file_uploader(
                "Upload Prescription or Medical Document (PDF, PNG, JPG, JPEG)",
                type=["pdf", "png", "jpg", "jpeg"],
                key="doc_start"
            )
            if doc_file_start is not None:
                st.info(f"📁 Selected: **{doc_file_start.name}** ({doc_file_start.size} bytes). This will be scanned and attached to your intake.")

        st.write("---")
        if st.button(t["start_btn"], type="primary", use_container_width=True):
            if not chief_complaint.strip() and not doc_file_start:
                st.warning("Please type your symptoms, record voice audio, or attach a medical document.")
            else:
                with st.spinner("Initializing private intake session & analyzing medical input..."):
                    try:
                        # 1. Create case in backend
                        initial_complaint = chief_complaint.strip() if chief_complaint.strip() else "Medical document attached for intake assessment"
                        new_case = api_client.create_case(
                            preferred_language=backend_lang,
                            chief_complaint=initial_complaint
                        )
                        case_id = new_case["id"]
                        st.session_state.case_id = case_id

                        # 2. If a document was uploaded, process OCR via upload endpoint
                        if doc_file_start is not None:
                            file_bytes = doc_file_start.getvalue()
                            up_res = api_client.upload_document(case_id, file_bytes, doc_file_start.name)
                            st.session_state.uploaded_docs.append({
                                "name": doc_file_start.name,
                                "char_count": up_res.get("char_count", 0),
                                "ocr_preview": up_res.get("extracted_text_snippet", "")
                            })
                            if up_res.get("is_emergency"):
                                st.session_state.is_emergency = True

                        # 3. Add first message to backend if chief complaint was typed
                        if chief_complaint.strip():
                            msg_res = api_client.send_message(
                                case_id=case_id,
                                content=chief_complaint.strip(),
                                source_type="patient_typed",
                                source_language=backend_lang,
                                is_confirmed=True
                            )
                            st.session_state.messages.append({
                                "role": "patient",
                                "content": chief_complaint.strip(),
                                "source_type": "patient_typed",
                                "source_language": backend_lang
                            })
                            if msg_res.get("is_emergency"):
                                st.session_state.is_emergency = True
                        elif doc_file_start is not None:
                            st.session_state.messages.append({
                                "role": "patient",
                                "content": f"[Attached Medical Document: {doc_file_start.name}]",
                                "source_type": "ocr"
                            })

                        # 4. Request first follow-up question
                        q_res = api_client.get_next_question(case_id)
                        st.session_state.turn_count = 1

                        if q_res.get("next_question"):
                            st.session_state.current_question = q_res["next_question"]
                            st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})
                            st.session_state.intake_step = "chatting"
                        else:
                            st.session_state.intake_step = "review_edit"

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error connecting to CareFlow backend: {e}")

# =====================================================================
# STEP 2: CONVERSATIONAL ADAPTIVE QUESTIONING & IN-CHAT OCR UPLOAD
# =====================================================================
elif st.session_state.intake_step == "chatting":
    q_str = t["question_count"].format(curr=st.session_state.turn_count, max=st.session_state.max_turns)
    st.caption(f"{t['case_id_label']}: **{st.session_state.case_id}** | {q_str}")
    
    # Progress bar
    progress_val = min(st.session_state.turn_count / st.session_state.max_turns, 1.0)
    st.progress(progress_val)

    # Conversation history
    st.write("---")
    for msg in st.session_state.messages[:-1]:  # Show history except latest assistant prompt
        if msg["role"] == "patient":
            src = msg.get("source_type", "patient_typed")
            badge = ""
            if src == "voice":
                badge = '<span style="background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE; font-size:11px; font-weight:700; padding:2px 6px; border-radius:10px; margin-right:6px;">🎤 Voice Transcript</span>'
            elif src == "ocr":
                badge = '<span style="background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; font-size:11px; font-weight:700; padding:2px 6px; border-radius:10px; margin-right:6px;">📄 Document OCR</span>'
            p_text = msg["content"]
            sub_tr = get_patient_message_translation(p_text, current_target_lang)
            sub_html = f'<div style="font-size:12px; opacity:0.8; margin-top:4px; font-style:italic;">{sub_tr}</div>' if sub_tr else ""
            st.markdown(f'<div class="chat-bubble-patient">{badge}<strong>{t["you_label"]}:</strong> {p_text}{sub_html}</div>', unsafe_allow_html=True)
        else:
            disp_asst = translate_question_text(msg["content"], current_target_lang)
            st.markdown(f'<div class="chat-bubble-assistant"><strong>{t["assistant_label"]}:</strong> {disp_asst}</div>', unsafe_allow_html=True)

    # Current Question Card (Dynamically translated to selected language)
    current_q_raw = st.session_state.current_question or ("Could you clarify any other symptoms you are experiencing?" if current_target_lang == "English" else "क्या आप अपने लक्षणों के बारे में कुछ और विस्तार से बता सकते हैं?")
    current_q = translate_question_text(current_q_raw, current_target_lang)
    st.markdown(f'<div class="chat-bubble-assistant" style="margin-right: 0; background: #E0F2FE; border-color: #BAE6FD; color: #0369A1;">'
                f'<strong>{t["q_label"]}</strong><br/><span style="font-size: 16px;">{current_q}</span></div>', unsafe_allow_html=True)

    backend_lang = "Hindi" if "Hindi" in st.session_state.selected_lang else "English"

    # -------------------------------------------------------------
    # CASE A: PENDING VOICE TRANSCRIPT REVIEW (Human-in-the-loop)
    # -------------------------------------------------------------
    if st.session_state.pending_voice is not None:
        pv = st.session_state.pending_voice
        p_trans = pv.get("transcript", "")
        p_lang = pv.get("language", backend_lang)
        p_conf = pv.get("confidence")
        conf_pct = int(p_conf * 100) if p_conf is not None else 0
        conf_status = "Good acoustic signal" if (p_conf or 0) >= 0.70 else "⚠️ Lower acoustic confidence — please verify accuracy"

        st.markdown(f"""
        <div style="background-color: #F5F3FF; border: 2px solid #C4B5FD; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <h4 style="margin: 0 0 6px 0; color: #5B21B6; font-size: 15px;">{t['voice_transcript_title']}</h4>
            <div style="font-size: 13px; color: #6D28D9;">
                Language: <strong>{p_lang.title() if p_lang else 'Auto-detect'}</strong> | Estimated Confidence: <strong>~{conf_pct}% ({conf_status})</strong>
            </div>
            <div style="font-size: 11.5px; color: #7C3AED; margin-top: 4px;">
                ℹ️ <em>Confidence is an automated acoustic estimate, not a guarantee of medical accuracy. Please review and edit your words below.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("last_voice_hash") != pv.get("audio_hash"):
            st.session_state["voice_transcript_edit"] = p_trans
            st.session_state["last_voice_hash"] = pv.get("audio_hash")

        edited_voice_text = st.text_area(
            "Review or edit transcribed text before submitting:",
            value=st.session_state.get("voice_transcript_edit", p_trans),
            key="voice_transcript_edit",
            height=100
        )
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            if st.button(t["voice_confirm_btn"], type="primary", key="btn_confirm_voice", use_container_width=True):
                if edited_voice_text.strip():
                    with st.spinner("Submitting verified voice response..."):
                        try:
                            send_res = api_client.send_message(
                                case_id=st.session_state.case_id,
                                content=edited_voice_text.strip(),
                                raw_transcript=p_trans,
                                source_type="voice",
                                source_language=p_lang,
                                transcription_confidence=p_conf,
                                is_confirmed=True
                            )
                            st.session_state.messages.append({
                                "role": "patient",
                                "content": edited_voice_text.strip(),
                                "raw_transcript": p_trans,
                                "source_type": "voice",
                                "source_language": p_lang,
                                "transcription_confidence": p_conf
                            })
                            if send_res.get("is_emergency"):
                                st.session_state.is_emergency = True
                            st.session_state.pending_voice = None
                            st.session_state.pop("voice_transcript_edit", None)
                            st.session_state.pop("last_voice_hash", None)

                            q_res = api_client.get_next_question(st.session_state.case_id)
                            st.session_state.turn_count += 1
                            if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count > st.session_state.max_turns:
                                st.session_state.intake_step = "review_edit"
                            else:
                                st.session_state.current_question = q_res["next_question"]
                                st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error submitting voice response: {e}")
                else:
                    st.warning("Please enter text or discard the recording.")
        with col_v2:
            if st.button(t["voice_discard_btn"], key="btn_discard_voice", use_container_width=True):
                st.session_state.pending_voice = None
                st.session_state.pop("voice_transcript_edit", None)
                st.session_state.pop("last_voice_hash", None)
                st.rerun()

    # -------------------------------------------------------------
    # CASE B: STANDARD PATIENT INPUT (Typed / Voice / Report OCR)
    # -------------------------------------------------------------
    else:
        # Patient Typed Answer Form
        with st.form("patient_answer_form", clear_on_submit=True):
            patient_answer = st.text_input(t["input_label"], placeholder=t["input_placeholder"], key="current_input")
            col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
            with col_btn1:
                submit_answer = st.form_submit_button(t["send_btn"], type="primary", use_container_width=True)
            with col_btn2:
                skip_question = st.form_submit_button(t["skip_btn"], use_container_width=True)
            with col_btn3:
                unknown_answer = st.form_submit_button(t["unknown_btn"], use_container_width=True)

        content_to_send = None
        if submit_answer:
            if patient_answer.strip():
                content_to_send = patient_answer.strip()
            else:
                st.warning("Please enter your answer or click 'Skip' / 'I Don't Know'.")
        elif skip_question:
            content_to_send = "I choose to skip this question." if "English" in st.session_state.selected_lang else "मैं इस प्रश्न को छोड़ना चाहता हूँ।"
        elif unknown_answer:
            content_to_send = "I do not know or am not sure of the answer to this question." if "English" in st.session_state.selected_lang else "मुझे इस प्रश्न का उत्तर निश्चित रूप से नहीं पता है।"

        if content_to_send:
            with st.spinner("Processing your response..."):
                try:
                    send_res = api_client.send_message(
                        st.session_state.case_id,
                        content_to_send,
                        source_type="patient_typed",
                        source_language=backend_lang,
                        is_confirmed=True
                    )
                    st.session_state.messages.append({
                        "role": "patient",
                        "content": content_to_send,
                        "source_type": "patient_typed",
                        "source_language": backend_lang
                    })

                    if send_res.get("is_emergency"):
                        st.session_state.is_emergency = True

                    q_res = api_client.get_next_question(st.session_state.case_id)
                    st.session_state.turn_count += 1

                    if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count > st.session_state.max_turns:
                        st.session_state.intake_step = "review_edit"
                    else:
                        st.session_state.current_question = q_res["next_question"]
                        st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})

                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing message: {e}")

        # Multimodal Input Expanders
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            with st.expander(t["voice_header"], expanded=False):
                st.caption(t["voice_help"])
                rec_audio = None
                if hasattr(st, "audio_input"):
                    rec_audio = st.audio_input("Record audio (Mic)", key="mic_audio_chat")
                upl_audio = st.file_uploader(
                    "Or upload voice file (WAV, MP3, M4A, OGG, WEBM, FLAC)",
                    type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
                    key="upl_audio_chat"
                )
                chosen_audio = rec_audio or upl_audio
                if chosen_audio is not None:
                    audio_bytes = chosen_audio.getvalue()
                    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
                    if st.session_state.pending_voice and st.session_state.get("last_voice_hash") != audio_hash:
                        st.session_state.pending_voice = None
                        st.session_state.pop("voice_transcript_edit", None)

                    if st.button("Transcribe Audio 🎙️", type="primary", key="btn_run_transcribe", use_container_width=True):
                        with st.spinner("Transcribing voice recording..."):
                            try:
                                fname = getattr(chosen_audio, "name", "recording.wav") or "recording.wav"
                                t_res = api_client.transcribe_audio(
                                    file_bytes=audio_bytes,
                                    filename=fname,
                                    language=asr_lang
                                )
                                t_res["audio_hash"] = audio_hash
                                st.session_state.pending_voice = t_res
                                st.session_state["voice_transcript_edit"] = t_res.get("transcript", "")
                                st.session_state["last_voice_hash"] = audio_hash
                                st.rerun()
                            except Exception as e:
                                st.error(f"⚠️ {e}")

        with col_exp2:
            with st.expander(t["upload_during_chat"], expanded=False):
                st.caption("Upload prescription or report image/PDF to attach to your record.")
                doc_in_chat = st.file_uploader(
                    "Upload image or PDF (PNG, JPG, PDF)",
                    type=["pdf", "png", "jpg", "jpeg", "webp"],
                    key="doc_chat"
                )
                if doc_in_chat is not None:
                    if st.button("Attach & Scan Document 📄", type="secondary", key="btn_run_attach_doc", use_container_width=True):
                        with st.spinner("Scanning and securely attaching medical document..."):
                            try:
                                file_bytes = doc_in_chat.getvalue()
                                up_res = api_client.upload_document(
                                    case_id=st.session_state.case_id,
                                    file_bytes=file_bytes,
                                    filename=doc_in_chat.name
                                )
                                c_count = up_res.get("char_count", 0)
                                engine = up_res.get("engine_used", "winocr")
                                conf = up_res.get("confidence", 0.9)

                                st.session_state.uploaded_docs.append({
                                    "name": doc_in_chat.name,
                                    "char_count": c_count,
                                    "engine_used": engine,
                                    "confidence": conf
                                })
                                st.session_state.messages.append({
                                    "role": "patient",
                                    "content": f"[Attached Medical Document: {doc_in_chat.name} ({c_count} chars extracted via {engine})]",
                                    "source_type": "ocr"
                                })
                                if up_res.get("is_emergency"):
                                    st.session_state.is_emergency = True

                                # Advance question
                                q_res = api_client.get_next_question(st.session_state.case_id)
                                st.session_state.turn_count += 1
                                if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count > st.session_state.max_turns:
                                    st.session_state.intake_step = "review_edit"
                                else:
                                    st.session_state.current_question = q_res["next_question"]
                                    st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})

                                st.rerun()
                            except Exception as e:
                                st.error(f"Error attaching document: {e}")

    # Early submission option
    st.write("---")
    col_early, _ = st.columns([2, 1])
    with col_early:
        if st.button(t["early_submit_btn"], use_container_width=True):
            st.session_state.intake_step = "review_edit"
            st.rerun()

# =====================================================================
# STEP 3: PRE-SUBMISSION REVIEW & EDIT SCREEN
# =====================================================================
elif st.session_state.intake_step == "review_edit":
    st.subheader(t["review_title"])
    st.write(t["review_subtitle"])

    if st.session_state.uploaded_docs:
        with st.expander(f"📄 Uploaded Documents ({len(st.session_state.uploaded_docs)})", expanded=True):
            for d in st.session_state.uploaded_docs:
                st.markdown(f"- **{d['name']}** ({d['char_count']} chars extracted via OCR)")
                if d.get("ocr_preview"):
                    st.caption(f"Preview: {d['ocr_preview'][:200]}...")

    patient_responses = [m["content"] for m in st.session_state.messages if m["role"] == "patient"]

    edited_texts = []
    for idx, resp in enumerate(patient_responses):
        with st.container():
            st.markdown(f"**Item {idx+1}:**")
            new_text = st.text_area(
                f"Response #{idx+1}",
                value=resp,
                key=f"edit_resp_{idx}",
                label_visibility="collapsed"
            )
            edited_texts.append(new_text)

    col_sub, col_add = st.columns([2, 1])
    with col_sub:
        if st.button(t["submit_btn"], type="primary", use_container_width=True):
            with st.spinner("Submitting intake to clinical reviewer queue..."):
                try:
                    # Check if patient modified any responses, if so send updated statement
                    for i, (orig, edited) in enumerate(zip(patient_responses, edited_texts)):
                        if orig != edited and edited.strip():
                            api_client.send_message(
                                st.session_state.case_id,
                                f"Correction to Item #{i+1}: {edited.strip()}"
                            )

                    # Finalize submission
                    submit_res = api_client.submit_case(st.session_state.case_id)
                    st.session_state.intake_step = "submitted"
                    st.rerun()
                except Exception as e:
                    st.error(f"Submission error: {e}")

    with col_add:
        if st.button(t["return_btn"], use_container_width=True):
            st.session_state.intake_step = "chatting"
            st.rerun()

# =====================================================================
# STEP 4: SUBMISSION CONFIRMATION
# =====================================================================
elif st.session_state.intake_step == "submitted":
    st.balloons()
    st.success(t["success_title"])

    st.markdown(f"""
    <div class="card-review">
        <h3>{t['confirm_card_title']}</h3>
        <p><strong>{t['case_id_label']}:</strong> <code>{st.session_state.case_id}</code></p>
        <p><strong>Status:</strong> {t['confirm_status']}</p>
        <p>{t['confirm_body']}</p>
        <div style="background: #F1F5F9; padding: 12px; border-radius: 6px; font-size: 13.5px; color: #475569;">
            {t['safety_reminder']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(t["restart_btn"], use_container_width=True):
        st.session_state.case_id = None
        st.session_state.intake_step = "start"
        st.session_state.messages = []
        st.session_state.current_question = None
        st.session_state.is_emergency = False
        st.session_state.turn_count = 0
        st.session_state.uploaded_docs = []
        st.rerun()
