import logging
from typing import List, Dict, Any, Optional
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.schemas.clinical import FollowUpQuestionOutput
from backend.config import settings

logger = logging.getLogger("careflow.questioning")

QUESTION_SYSTEM_PROMPT = """You are an empathetic, clinical triage intake assistant.
TASK: FOLLOW_UP_QUESTION

Your goal is to formulate ONE single clarifying follow-up question to collect clinically critical missing information.
RULES:
1. Ask ONLY ONE focused question at a time.
2. Prioritize screening for critical red-flag symptoms (e.g. pain radiation, shortness of breath, neck stiffness) and essential dimensions (onset, duration, severity scale, medications/allergies).
3. Do NOT tell the patient they have any disease, do NOT give diagnostic labels, and do NOT tell them they are safe.
4. Do NOT assume skipped or unknown answers are negative or absent.
5. ANTI-REPETITION MANDATE: NEVER repeat any previously asked question or re-ask a topic that has already been asked. If the patient already answered, skipped, or said 'I don't know', that topic is considered addressed. Always move to an unasked dimension.
6. If enough information has been gathered, or if further questioning is unlikely to add value, set "next_question": null and "is_terminal": true.
7. Return your output strictly as a JSON object matching this schema:
{
  "next_question": "the single question to ask or null",
  "reason_for_question": "clinical justification",
  "target_field": "clinical attribute name",
  "is_terminal": false
}
"""

class AdaptiveQuestioningService:
    """Service generating the single next clinically relevant question without repetition."""

    SYMPTOM_QUESTION_CATALOG = {
        "hematemesis": {
            "keywords": ["vomiting blood", "threw up blood", "throwing up blood", "blood in vomit", "hematemesis", "khoon ki ulti", "ulti me khoon", "खून की उल्टी", "उल्टी में खून"],
            "questions": [
                {
                    "target_field": "hematemesis_appearance_and_volume",
                    "reason": "Assessing active upper GI hemorrhage volume and appearance (bright red vs coffee grounds).",
                    "English": "Is the blood you vomited bright red or dark like coffee grounds, and roughly how much have you brought up?",
                    "Hindi": "उल्टी में निकला खून ताजा लाल है या कॉफी के रंग जैसा गहरा काला, और लगभग कितनी मात्रा में निकला है?",
                    "Hinglish": "Ulti me jo khoon nikla wo bright red hai ya coffee grounds jaisa dark, aur lagbhag kitna amount nikla hai?"
                },
                {
                    "target_field": "hematemesis_hemodynamic_stability",
                    "reason": "Screening for hemodynamic instability, syncope, or orthostasis secondary to acute blood loss.",
                    "English": "Are you feeling faint, dizzy, severely weak, or noticing dark black tarry stools?",
                    "Hindi": "क्या आपको चक्कर, बेहोशी, अत्यधिक कमजोरी महसूस हो रही है, या काला मल आ रहा है?",
                    "Hinglish": "Kya aapko chakkar, behoshi, extreme weakness feel ho rahi hai, ya dark black stool aa raha hai?"
                }
            ]
        },
        "chest_pain": {
            "keywords": ["chest pain", "pressure in chest", "chest tightness", "tightness in chest", "seene me dard", "chhati me dard", "dil me dard", "सीने में दर्द", "छाती में दर्द", "दिल में दर्द"],
            "questions": [
                {
                    "target_field": "pain_radiation",
                    "reason": "Screening for classic acute coronary radiation warning signs to left arm, neck, jaw, or back.",
                    "English": "Does the chest pain radiate or spread to your left arm, shoulder, neck, jaw, or back?",
                    "Hindi": "क्या सीने का दर्द आपके बाएं हाथ, कंधे, गर्दन, जबड़े या पीठ की तरफ फैल रहा है?",
                    "Hinglish": "Kya chest pain aapke left arm, shoulder, neck, jaw ya back ki taraf radiate kar raha hai?"
                },
                {
                    "target_field": "associated_cardiac_symptoms",
                    "reason": "Assessing associated symptoms of cardiopulmonary compromise (dyspnea, diaphoresis).",
                    "English": "Are you experiencing any shortness of breath, cold sweats, nausea, or lightheadedness along with the chest pain?",
                    "Hindi": "क्या सीने में दर्द के साथ सांस फूलना, ठंडा पसीना, उल्टी या चक्कर आ रहे हैं?",
                    "Hinglish": "Kya chest pain ke sath saans phoolna, cold sweats, ulti ya chakkar aa rahe hain?"
                }
            ]
        },
        "headache": {
            "keywords": ["headache", "head pain", "sir dard", "sar dard", "shar dard", "सिरदर्द", "सिर में दर्द", "headahed"],
            "questions": [
                {
                    "target_field": "headache_onset_severity",
                    "reason": "Screening for sudden-onset thunderclap headache vs subacute presentation.",
                    "English": "Did the headache peak at maximum intensity suddenly within seconds (like a thunderclap), or did it build gradually?",
                    "Hindi": "क्या यह सिरदर्द अचानक कुछ ही सेकंड में असहनीय तेज हो गया, या धीरे-धीरे बढ़ा?",
                    "Hinglish": "Kya ye sir dard achanak seconds me bahut tez ho gaya (thunderclap ki tarah), ya gradually badha?"
                },
                {
                    "target_field": "neck_stiffness_photophobia",
                    "reason": "Screening for signs of meningeal irritation or focal neurological deficits.",
                    "English": "Do you have any neck stiffness, pain when bending your head forward, fever, or sensitivity to bright light?",
                    "Hindi": "क्या आपको गर्दन में अकड़न, सिर आगे झुकाने में दर्द, बुखार या तेज रोशनी से परेशानी हो रही है?",
                    "Hinglish": "Kya gardan me stiffness, aage jhukane me dard, bukhar ya bright light se pareshani ho rahi hai?"
                }
            ]
        },
        "shortness_of_breath": {
            "keywords": ["shortness of breath", "trouble breathing", "breathless", "saans lene me takleef", "dam phoolna", "सांस लेने में तकलीफ", "सांस फूलना"],
            "questions": [
                {
                    "target_field": "respiratory_severity_orthopnea",
                    "reason": "Assessing functional respiratory reserve and orthopnea.",
                    "English": "Are you able to speak in full sentences without gasping, and does lying flat make your breathing harder?",
                    "Hindi": "क्या आप बिना सांस फूले पूरे वाक्य बोल पा रहे हैं, और क्या सीधे लेटने पर सांस ज्यादा फूलती है?",
                    "Hinglish": "Kya aap bina ruke full sentences bol pa rahe hain, aur letne par saans zyada phoolti hai?"
                },
                {
                    "target_field": "respiratory_associated_signs",
                    "reason": "Screening for airway reactivity, wheezing, or facial/lip edema.",
                    "English": "Have you noticed any audible wheezing, cough with phlegm, fever, or swelling of your face or ankles?",
                    "Hindi": "क्या सांस लेते समय सीटी जैसी आवाज (घरघराहट), खांसी में बलगम, बुखार या चेहरे/पैरों में सूजन है?",
                    "Hinglish": "Kya saans lete waqt wheezing sound, khansi me balgam, bukhar ya face/pairon me sujan hai?"
                }
            ]
        },
        "abdominal_pain": {
            "keywords": ["abdominal pain", "stomach pain", "belly pain", "pet dard", "pet me dard", "पेट में दर्द", "पेट दर्द"],
            "questions": [
                {
                    "target_field": "abdominal_location_fluid_tolerance",
                    "reason": "Localizing abdominal pain and assessing oral fluid tolerance.",
                    "English": "Where exactly in your abdomen is the pain located, and are you able to keep water or fluids down without vomiting?",
                    "Hindi": "पेट में दर्द ठीक किस जगह पर है, और क्या आप पानी या तरल चीजें बिना उल्टी किए पचा पा रहे हैं?",
                    "Hinglish": "Pet me dard exactly kahan par hai, aur kya aap water ya liquids bina ulti ke digest kar pa rahe hain?"
                },
                {
                    "target_field": "abdominal_associated_features",
                    "reason": "Screening for acute biliary, pancreatic, or peritonitic features.",
                    "English": "Does the pain radiate through to your back or shoulder, and have you had fever, chills, or yellowing of the eyes?",
                    "Hindi": "क्या यह दर्द पीठ या कंधे की तरफ जाता है, और क्या आपको बुखार, कंपकंपी या आंखों में पीलापन है?",
                    "Hinglish": "Kya dard back ya shoulder me radiate kar raha hai, aur kya bukhar ya eyes me yellowness hai?"
                }
            ]
        },
        "fever": {
            "keywords": ["fever", "high temperature", "chills", "bukhar", "taap", "बुखार", "तेज बुखार"],
            "questions": [
                {
                    "target_field": "fever_duration_temperature",
                    "reason": "Quantifying fever duration and measured peak temperature.",
                    "English": "How many days have you had the fever, and what was the highest temperature measured?",
                    "Hindi": "आपको बुखार कितने दिनों से है, और थर्मामीटर पर सबसे अधिक तापमान कितना मापा गया?",
                    "Hinglish": "Aapko bukhar kitne dino se hai, aur thermometer par highest temperature kitna measure hua?"
                },
                {
                    "target_field": "fever_red_flag_signs",
                    "reason": "Screening for systemic infection red flags: altered sensorium, nuchal rigidity, rash.",
                    "English": "Do you have any neck stiffness, confusion, extreme shivering, or any new rash on your skin?",
                    "Hindi": "क्या आपको गर्दन में अकड़न, भ्रम या अत्यधिक कंपकंपी है, या त्वचा पर कोई नया दाना/चकत्ता है?",
                    "Hinglish": "Kya gardan me stiffness, confusion, extreme shivering, ya skin par koi naya rash hai?"
                }
            ]
        }
    }

    # Fallback rotation of general clinical dimensions
    CLINICAL_DIMENSION_QUESTIONS = [
        ("onset_and_duration", {
            "reason": "Onset and duration are essential for triage acuity.",
            "English": "When did your symptoms first begin, and have they been constant, worsening, or coming and going?",
            "Hindi": "यह लक्षण सबसे पहले कब शुरू हुए और क्या यह समय के साथ बढ़ रहे हैं, घट रहे हैं या लगातार बने हुए हैं?",
            "Hinglish": "Ye symptoms sabse pehle kab shuru hue, aur kya ye continuously badh rahe hain ya kam-zyada ho rahe hain?"
        }),
        ("severity", {
            "reason": "Quantifying symptom severity to establish clinical acuity.",
            "English": "On a scale from 0 to 10 (where 0 is no discomfort and 10 is unbearable), how severe is your symptom right now?",
            "Hindi": "0 से 10 के पैमाने पर (जहाँ 0 कोई दर्द नहीं और 10 असहनीय दर्द है), अभी आपका दर्द कितना तीव्र है?",
            "Hinglish": "0 se 10 ke scale par (jahan 0 koi dard nahi aur 10 unbearable pain hai), abhi dard kitna severe hai?"
        }),
        ("medications_and_history", {
            "reason": "Identifying relevant baseline health status, regular medications, and allergies.",
            "English": "Are you currently taking any prescription medications, or do you have any relevant medical conditions or allergies?",
            "Hindi": "क्या आप कोई नियमित दवाइयाँ ले रहे हैं, या आपको कोई पुरानी बीमारी या एलर्जी है?",
            "Hinglish": "Kya aap koi regular medications le rahe hain, ya aapko koi existing medical conditions ya allergies hain?"
        })
    ]

    @classmethod
    def _detect_language(cls, preferred_language: str, text: str) -> str:
        """Determines best language category: English, Hindi (Devanagari), or Hinglish."""
        if preferred_language:
            p_low = preferred_language.lower()
            if "hindi" in p_low and "hinglish" not in p_low:
                return "Hindi"
            if "hinglish" in p_low:
                return "Hinglish"
            if "english" in p_low:
                return "English"

        # Check for Devanagari Unicode block
        import re
        if re.search(r'[\u0900-\u097F]', text):
            return "Hindi"

        # Check for common Hinglish tokens
        hinglish_words = ["dard", "bukhar", "ulti", "chakkar", "seene", "saans", "gardan", "pet", "nahi", "bahut", "bhi", "tha", "raha", "hai"]
        words = text.lower().split()
        if any(w in words for w in hinglish_words):
            return "Hinglish"

        return "English"

    @classmethod
    def generate_next_question(
        cls,
        patient_messages: List[str],
        current_question_count: int,
        assistant_messages: Optional[List[str]] = None,
        evidence_state: Optional[Dict[str, Any]] = None,
        preferred_language: str = "English"
    ) -> FollowUpQuestionOutput:
        """
        Determines the next follow-up question, guaranteeing no repetitive questions.
        Prioritizes symptom-specific queries before falling back to general dimensions.
        """
        assistant_messages = assistant_messages or []

        # Enforce maximum question limit or early completion
        if current_question_count >= settings.MAX_FOLLOWUP_QUESTIONS:
            logger.info(f"Reached maximum question limit ({settings.MAX_FOLLOWUP_QUESTIONS}). Ending questioning.")
            return FollowUpQuestionOutput(
                next_question=None,
                reason_for_question=f"Reached maximum limit of {settings.MAX_FOLLOWUP_QUESTIONS} follow-up questions. Intake ready for human clinician review.",
                target_field="limit_reached",
                is_terminal=True
            )

        full_patient_text = " ".join(patient_messages)
        lang = cls._detect_language(preferred_language, full_patient_text)

        # Build prompt with safe utterance enclosure
        user_content_blocks = [
            f"<patient_utterance>{msg.strip()}</patient_utterance>"
            for msg in patient_messages if msg.strip()
        ]
        
        # Include previously asked questions
        prev_q_blocks = [
            f"<previously_asked_question>{q.strip()}</previously_asked_question>"
            for q in assistant_messages if q.strip()
        ]

        user_prompt = (
            f"The patient has provided the following messages (turn count: {current_question_count}/{settings.MAX_FOLLOWUP_QUESTIONS}, preferred language: {lang}):\n\n"
            + "\n".join(user_content_blocks)
            + "\n\n"
            + (("PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT ANY OF THESE):\n" + "\n".join(prev_q_blocks) + "\n\n") if prev_q_blocks else "")
            + "Formulate the single most clinically relevant follow-up question to clarify an unaddressed critical dimension. "
            + "If all key areas have been covered, set 'next_question': null and 'is_terminal': true."
        )

        try:
            raw_json = medgemma_service.generate_json(
                system_prompt=QUESTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_retries=1
            )
            result = FollowUpQuestionOutput.model_validate(raw_json)

            # Deduplication Verification: Check if generated question repeats any previous question
            if result.next_question and cls._is_duplicate_question(result.next_question, assistant_messages):
                logger.warning(f"Detected duplicate question '{result.next_question}'. Finding unaddressed alternative.")
                result = cls._get_unasked_fallback_question(assistant_messages, patient_messages, lang)

            return result

        except Exception as e:
            logger.error(f"Failed to generate follow-up question via model: {e}")
            # Fallback gracefully to structured symptom/dimension catalog
            return cls._get_unasked_fallback_question(assistant_messages, patient_messages, lang)

    @classmethod
    def _is_duplicate_question(cls, question: str, previous_questions: List[str]) -> bool:
        """Checks if a question has essentially already been asked."""
        q_norm = question.lower().strip()
        for prev in previous_questions:
            p_norm = prev.lower().strip()
            # Exact or substring match
            if q_norm == p_norm or q_norm in p_norm or p_norm in q_norm:
                return True
            # Check key semantic word overlap
            q_keywords = set(w for w in q_norm.split() if len(w) > 4)
            p_keywords = set(w for w in p_norm.split() if len(w) > 4)
            if q_keywords and p_keywords:
                overlap = len(q_keywords.intersection(p_keywords)) / max(len(q_keywords), len(p_keywords))
                if overlap > 0.6:
                    return True
        return False

    @classmethod
    def _get_unasked_fallback_question(
        cls,
        previous_questions: List[str],
        patient_messages: List[str],
        language: str = "English"
    ) -> FollowUpQuestionOutput:
        """Selects the next unaddressed symptom-specific or general clinical dimension."""
        full_patient_text = " ".join(patient_messages).lower()
        prev_text = " ".join(previous_questions).lower()

        # Step 1: Check Symptom-Specific Questions
        for symptom_key, symp_data in cls.SYMPTOM_QUESTION_CATALOG.items():
            if any(k in full_patient_text for k in symp_data["keywords"]):
                for q_spec in symp_data["questions"]:
                    target_field = q_spec["target_field"]
                    q_text = q_spec.get(language, q_spec["English"])

                    # Check if already asked
                    if cls._is_duplicate_question(q_text, previous_questions):
                        continue
                    if any(target_field in prev for prev in previous_questions):
                        continue

                    # Check if already explicitly answered
                    if target_field == "pain_radiation" and any(w in full_patient_text for w in ["radiat", "spread", "left arm", "jaw", "neck", "baaye hath", "गर्दन", "फैल"]):
                        continue
                    if target_field == "hematemesis_appearance_and_volume" and any(w in full_patient_text for w in ["bright red", "dark", "coffee ground", "laal", "cup"]):
                        continue

                    return FollowUpQuestionOutput(
                        next_question=q_text,
                        reason_for_question=q_spec["reason"],
                        target_field=target_field,
                        is_terminal=False
                    )

        # Step 2: Fall back to General Dimensions
        for target_field, dim_spec in cls.CLINICAL_DIMENSION_QUESTIONS:
            q_text = dim_spec.get(language, dim_spec["English"])

            # Check if this topic was already asked
            if cls._is_duplicate_question(q_text, previous_questions):
                continue
            if any(target_field in prev for prev in previous_questions):
                continue

            # Check if already answered in patient messages
            if target_field == "onset_and_duration" and any(w in full_patient_text for w in ["yesterday", "ago", "started", "since", "days", "hours", "din se", "kal se", "दिन से"]):
                continue
            if target_field == "severity" and any(w in full_patient_text for w in ["/10", "out of 10", "mild", "severe", "moderate", "tez dard", "असहनीय"]):
                continue

            return FollowUpQuestionOutput(
                next_question=q_text,
                reason_for_question=dim_spec["reason"],
                target_field=target_field,
                is_terminal=False
            )

        # If all dimensions have been asked or addressed, conclude questioning
        terminal_msg = {
            "Hindi": "आपकी सभी आवश्यक प्राथमिक जानकारी दर्ज कर ली गई है। डॉक्टर जल्द ही आपकी समीक्षा करेंगे।",
            "Hinglish": "Aapki primary clinical information record ho gayi hai. Doctor jald hi case review karenge.",
            "English": "All primary clinical intake dimensions have been gathered. Ready for clinician review."
        }.get(language, "All primary clinical intake dimensions have been gathered.")

        return FollowUpQuestionOutput(
            next_question=None,
            reason_for_question=terminal_msg,
            target_field="intake_complete",
            is_terminal=True
        )

adaptive_questioning_service = AdaptiveQuestioningService()
