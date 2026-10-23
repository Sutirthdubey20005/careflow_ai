import os
import logging
import re
from typing import List, Dict, Any, Optional
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.schemas.clinical import FollowUpQuestionOutput
from backend.config import settings

logger = logging.getLogger("careflow.questioning")

QUESTION_SYSTEM_PROMPT = """You are CareFlow's empathetic, clinical triage intake assistant powered by MedGemma.
TASK: FOLLOW_UP_QUESTION

Your goal is to formulate ONE genuine, empathetic conversational reaction and single high-yield clarifying clinical follow-up question.

RULES:
1. GENUINE REACTION & EMPATHY: Always start by warmly and empathetically acknowledging the patient's specific symptom or situation (e.g. "I'm sorry you're experiencing...", "That sounds very uncomfortable...", "Thank you for letting me know about..."). Never sound robotic, dismissive, or cold.
2. SYMPTOM-SPECIFIC CLINICAL RELEVANCE: Tailor the inquiry directly to the pathophysiology and red flags of their specific complaint (e.g., for sore throat: swallowing saliva, airway patency, trismus; for rash: mucosal involvement, breathing difficulty, blistering; for ankle trauma: inability to bear weight, neurovascular signs; for dysuria: hematuria, flank pain, fever).
3. AVOID REDUNDANT DIMENSIONS: If the patient has ALREADY provided an onset timeline (e.g., "since yesterday", "for 3 days"), DO NOT ask when it started! If they have ALREADY provided a severity or pain level (e.g., "8/10", "mild", "severe"), DO NOT ask for a 0-10 scale!
4. Ask ONLY ONE focused question at a time.
5. Do NOT tell the patient they have any disease, do NOT give diagnostic labels, and do NOT tell them they are safe.
6. Do NOT assume skipped or unknown answers are negative or absent.
7. ANTI-REPETITION MANDATE: NEVER repeat any previously asked question or re-ask a topic that has already been asked. If the patient already answered, skipped, or said 'I don't know', that topic is considered addressed. Always move to an unasked dimension.
8. If enough information has been gathered, or if further questioning is unlikely to add value, set "next_question": null and "is_terminal": true.
9. Return your output strictly as a JSON object matching this schema:
{
  "next_question": "the empathetic opening reaction + single question to ask or null",
  "reason_for_question": "clinical justification",
  "target_field": "clinical attribute name",
  "is_terminal": false
}
"""

class AdaptiveQuestioningService:
    """Service generating the single next clinically relevant question with genuine empathy and without repetition."""

    SYMPTOM_QUESTION_CATALOG = {
        "hematemesis": {
            "keywords": ["vomiting blood", "threw up blood", "throwing up blood", "blood in vomit", "hematemesis", "khoon ki ulti", "ulti me khoon", "खून की उल्टी", "उल्टी में खून"],
            "questions": [
                {
                    "target_field": "hematemesis_appearance_and_volume",
                    "reason": "Assessing active upper GI hemorrhage volume and appearance (bright red vs coffee grounds).",
                    "English": "I am very sorry to hear that, and seeing blood can certainly be frightening. Is the blood you vomited bright red or dark like coffee grounds, and roughly how much have you brought up?",
                    "Hindi": "उल्टी में खून देखना वास्तव में बहुत चिंताजनक हो सकता है। उल्टी में निकला खून ताजा लाल है या कॉफी के रंग जैसा गहरा काला, और लगभग कितनी मात्रा में निकला है?",
                    "Hinglish": "Ulti me khoon dekhna sach me scary hota hai. Ulti me jo khoon nikla wo bright red hai ya coffee grounds jaisa dark, aur lagbhag kitna amount nikla hai?"
                },
                {
                    "target_field": "hematemesis_hemodynamic_stability",
                    "reason": "Screening for hemodynamic instability, syncope, or orthostasis secondary to acute blood loss.",
                    "English": "Because blood loss can affect blood pressure, are you feeling faint, dizzy, severely weak, or noticing dark black tarry stools?",
                    "Hindi": "खून की कमी से कमजोरी हो सकती है, इसलिए क्या आपको चक्कर, बेहोशी, अत्यधिक कमजोरी महसूस हो रही है, या काला मल आ रहा है?",
                    "Hinglish": "Blood loss se blood pressure kam ho sakta hai, kya aapko chakkar, behoshi, extreme weakness feel ho rahi hai, ya dark black stool aa raha hai?"
                }
            ]
        },
        "chest_pain": {
            "keywords": ["chest pain", "pressure in chest", "chest tightness", "tightness in chest", "seene me dard", "chhati me dard", "dil me dard", "सीने में दर्द", "छाती में दर्द", "दिल में दर्द"],
            "questions": [
                {
                    "target_field": "pain_radiation",
                    "reason": "Screening for classic acute coronary radiation warning signs to left arm, neck, jaw, or back.",
                    "English": "I understand that chest discomfort can be concerning, and we want to take it very seriously. Does the chest pain radiate or spread to your left arm, shoulder, neck, jaw, or back?",
                    "Hindi": "सीने में दर्द होना काफी चिंताजनक हो सकता है और हम इसे बहुत गंभीरता से ले रहे हैं। क्या सीने का दर्द आपके बाएं हाथ, कंधे, गर्दन, जबड़े या पीठ की तरफ फैल रहा है?",
                    "Hinglish": "Chest pain ko hum bohot seriously lete hain. Kya chest pain aapke left arm, shoulder, neck, jaw ya back ki taraf radiate kar raha hai?"
                },
                {
                    "target_field": "associated_cardiac_symptoms",
                    "reason": "Assessing associated symptoms of cardiopulmonary compromise (dyspnea, diaphoresis).",
                    "English": "To help us evaluate your heart and lungs, are you experiencing any shortness of breath, cold sweats, nausea, or lightheadedness along with the chest pain?",
                    "Hindi": "हृदय और फेफड़ों की स्थिति समझने के लिए, क्या सीने में दर्द के साथ सांस फूलना, ठंडा पसीना, उल्टी या चक्कर आ रहे हैं?",
                    "Hinglish": "Cardiopulmonary symptoms check karne ke liye, kya chest pain ke sath saans phoolna, cold sweats, ulti ya chakkar aa rahe hain?"
                }
            ]
        },
        "shortness_of_breath": {
            "keywords": ["shortness of breath", "trouble breathing", "breathless", "saans lene me takleef", "dam phoolna", "सांस लेने में तकलीफ", "सांस फूलना"],
            "questions": [
                {
                    "target_field": "respiratory_severity_orthopnea",
                    "reason": "Assessing functional respiratory reserve and orthopnea.",
                    "English": "Struggling to catch your breath can feel very scary. Are you able to speak in full sentences without gasping, and does lying flat make your breathing harder?",
                    "Hindi": "सांस लेने में तकलीफ होना बहुत बेचैन करने वाला हो सकता है। क्या आप बिना सांस फूले पूरे वाक्य बोल पा रहे हैं, और क्या सीधे लेटने पर सांस ज्यादा फूलती है?",
                    "Hinglish": "Saans lene me dikkat bohot uncomfortable hoti hai. Kya aap bina ruke full sentences bol pa rahe hain, aur letne par saans zyada phoolti hai?"
                },
                {
                    "target_field": "respiratory_associated_signs",
                    "reason": "Screening for airway reactivity, wheezing, or facial/lip edema.",
                    "English": "To check your airway, have you noticed any audible wheezing, cough with phlegm, fever, or swelling of your face, lips, or ankles?",
                    "Hindi": "सांस की नली की जांच के लिए, क्या सांस लेते समय सीटी जैसी आवाज (घरघराहट), खांसी में बलगम, बुखार या चेहरे/पैरों में सूजन है?",
                    "Hinglish": "Airway check karne ke liye, kya saans lete waqt wheezing sound, khansi me balgam, bukhar ya face/pairon me sujan hai?"
                }
            ]
        },
        "headache": {
            "keywords": ["headache", "head pain", "sir dard", "sar dard", "shar dard", "सिरदर्द", "सिर में दर्द", "headahed"],
            "questions": [
                {
                    "target_field": "headache_onset_severity",
                    "reason": "Screening for sudden-onset thunderclap headache vs subacute presentation.",
                    "English": "Dealing with a severe headache can be exhausting. Did the headache peak at maximum intensity suddenly within seconds (like a thunderclap), or did it build gradually?",
                    "Hindi": "तेज सिरदर्द से आपको काफी परेशानी हो रही होगी। क्या यह सिरदर्द अचानक कुछ ही सेकंड में असहनीय तेज हो गया (बिजली कड़कने जैसा), या धीरे-धीरे बढ़ा?",
                    "Hinglish": "Tez sir dard bohot exhausting hota hai. Kya ye sir dard achanak seconds me bahut tez ho gaya (thunderclap ki tarah), ya gradually badha?"
                },
                {
                    "target_field": "neck_stiffness_photophobia",
                    "reason": "Screening for signs of meningeal irritation or focal neurological deficits.",
                    "English": "To rule out neurological irritation, do you have any neck stiffness, pain when bending your head forward, fever, or sensitivity to bright light?",
                    "Hindi": "नसों में तनाव जांचने के लिए, क्या आपको गर्दन में अकड़न, सिर आगे झुकाने में दर्द, बुखार या तेज रोशनी से परेशानी हो रही है?",
                    "Hinglish": "Neurological check ke liye, kya gardan me stiffness, aage jhukane me dard, bukhar ya bright light se pareshani ho rahi hai?"
                }
            ]
        },
        "abdominal_pain": {
            "keywords": ["abdominal pain", "stomach pain", "belly pain", "pet dard", "pet me dard", "पेट में दर्द", "पेट दर्द"],
            "questions": [
                {
                    "target_field": "abdominal_location_fluid_tolerance",
                    "reason": "Localizing abdominal pain and assessing oral fluid tolerance.",
                    "English": "I'm sorry to hear that your stomach is hurting so much. Where exactly in your abdomen is the pain located, and are you able to keep water or fluids down without vomiting?",
                    "Hindi": "पेट में दर्द से आपको काफी तकलीफ हो रही होगी। पेट में दर्द ठीक किस जगह पर है, और क्या आप पानी या तरल चीजें बिना उल्टी किए पचा पा रहे हैं?",
                    "Hinglish": "Pet me dard ki wajah se kaafi pareshani ho rahi hogi. Pet me dard exactly kahan par hai, aur kya aap water ya liquids bina ulti ke digest kar pa rahe hain?"
                },
                {
                    "target_field": "abdominal_associated_features",
                    "reason": "Screening for acute biliary, pancreatic, or peritonitic features.",
                    "English": "Does the pain radiate through to your back or shoulder, and have you had fever, chills, or yellowing of your eyes or skin?",
                    "Hindi": "क्या यह दर्द पीठ या कंधे की तरफ जाता है, और क्या आपको बुखार, कंपकंपी या आंखों में पीलापन महसूस हुआ है?",
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
                    "English": "Running a fever can leave you feeling completely drained. How many days have you had the fever, and what was the highest temperature measured?",
                    "Hindi": "बुखार की वजह से शरीर में बहुत कमजोरी और बेचैनी हो रही होगी। आपको बुखार कितने दिनों से है, और थर्मामीटर पर सबसे अधिक तापमान कितना मापा गया?",
                    "Hinglish": "Bukhar se body me bohot weakness ho sakti hai. Aapko bukhar kitne dino se hai, aur thermometer par highest temperature kitna measure hua?"
                },
                {
                    "target_field": "fever_red_flag_signs",
                    "reason": "Screening for systemic infection red flags: altered sensorium, nuchal rigidity, rash.",
                    "English": "To check for systemic infection signs, do you have any neck stiffness, confusion, extreme shivering, or any new rash on your skin?",
                    "Hindi": "संक्रमण की गंभीरता जांचने के लिए, क्या आपको गर्दन में अकड़न, भ्रम या अत्यधिक कंपकंपी है, या त्वचा पर कोई नया दाना/चकत्ता है?",
                    "Hinglish": "Systemic infection screen karne ke liye, kya gardan me stiffness, confusion, extreme shivering, ya skin par koi naya rash hai?"
                }
            ]
        },
        "sore_throat": {
            "keywords": ["sore throat", "throat pain", "throat hurt", "gale me dard", "gala kharab", "गले में दर्द", "गले में खराश"],
            "questions": [
                {
                    "target_field": "throat_airway_swallowing",
                    "reason": "Screening for peritonsillar abscess, epiglottitis, or airway compromise.",
                    "English": "I'm sorry your throat is hurting so much. Are you having difficulty swallowing your own saliva (drooling), trouble breathing, or difficulty opening your mouth wide?",
                    "Hindi": "गले के तेज दर्द से आपको काफी तकलीफ हो रही होगी। क्या आपको अपनी लार निगलने में परेशानी (लार टपकना), सांस लेने में दिक्कत या मुंह पूरा खोलने में दर्द हो रहा है?",
                    "Hinglish": "Gale me dard se kaafi takleef ho rahi hogi. Kya saliva swallow karne me dikkat, saans lene me problem, ya muh pura open karne me dard ho raha hai?"
                },
                {
                    "target_field": "throat_associated_systemic",
                    "reason": "Assessing for tonsillar exudate, high fever, and lymphadenopathy.",
                    "English": "Have you noticed any white patches on your tonsils, high fever, or painful swollen lumps along your neck?",
                    "Hindi": "क्या गले के अंदर टॉन्सिल पर सफेद चकत्ते, तेज बुखार या गर्दन में दर्दनाक गांठें/सूजन महसूस हो रही हैं?",
                    "Hinglish": "Kya tonsils par white spots, high fever, ya neck me swollen painful glands feel ho rahe hain?"
                }
            ]
        },
        "rash_dermatology": {
            "keywords": ["rash", "skin redness", "hives", "itching", "चकत्ते", "खुजली", "khujli", "red spots", "skin peeling", "blisters"],
            "questions": [
                {
                    "target_field": "rash_mucosal_airway_involvement",
                    "reason": "Screening for serious dermatological emergencies (SJS, anaphylaxis, drug reaction).",
                    "English": "I understand how uncomfortable and frustrating a spreading rash can be. Has the rash started blistering or peeling, and have you noticed any swelling around your lips, tongue, or eyes?",
                    "Hindi": "त्वचा पर दाने और तेज खुजली वास्तव में बहुत परेशान करने वाली होती है। क्या दानों में छाले पड़ रहे हैं, त्वचा छिल रही है, या आपके होंठों, जीभ या आंखों के आसपास सूजन आ रही है?",
                    "Hinglish": "Skin par rash aur khujli bohot uncomfortable hoti hai. Kya rash me chhale (blisters) pad rahe hain, ya lips, tongue ya eyes ke aas-paas koi swelling aa rahi hai?"
                },
                {
                    "target_field": "rash_distribution_triggers",
                    "reason": "Assessing anatomical spread and potential medication or environmental triggers.",
                    "English": "Where on your body did the rash first appear, has it spread, and did you recently start any new medications or foods?",
                    "Hindi": "शरीर के किस हिस्से पर चकत्ते सबसे पहले दिखे, क्या यह फैल रहे हैं, और क्या आपने हाल ही में कोई नई दवा या भोजन लिया था?",
                    "Hinglish": "Body ke kis part par rash pehle aaya, kya ye spread ho raha hai, aur kya recently koi nayi medicine ya food liya tha?"
                }
            ]
        },
        "musculoskeletal_trauma": {
            "keywords": ["ankle", "knee", "fall", "twisted", "sprain", "broken", "fracture", "swollen joint", "leg hurt", "arm hurt", "pair me moch", "chot lagi", "चोट", "मोच", "हड्डी टूट"],
            "questions": [
                {
                    "target_field": "trauma_weight_bearing_deformity",
                    "reason": "Assessing fracture probability using Ottawa criteria and gross structural deformity.",
                    "English": "I'm really sorry to hear about your injury—joint sprains and falls can be very painful. Are you able to bear weight and take even a few steps on that limb, or is there any visible deformity?",
                    "Hindi": "चोट लगने और दर्द के बारे में जानकर मुझे खेद है। क्या आप उस पैर या हाथ पर थोड़ा भी वजन रख पा रहे हैं, या कोई हड्डी टेढ़ी/सूजी हुई दिखाई दे रही है?",
                    "Hinglish": "Aapko chot lagne aur dard ke baare me sunkar bura laga. Kya aap us leg/arm par thoda bhi weight daal pa rahe hain, ya koi visible deformity/haddi tedhi dikh rahi hai?"
                },
                {
                    "target_field": "trauma_neurovascular_integrity",
                    "reason": "Screening for compartment syndrome or acute neurovascular compromise.",
                    "English": "Do you feel any numbness, tingling, or unusual coldness in your toes or fingers beyond the injured area?",
                    "Hindi": "क्या चोट लगे हिस्से के आगे उंगलियों या पंजों में सुन्नपन, झनझनाहट या असामान्य ठंडक महसूस हो रही है?",
                    "Hinglish": "Kya injured area ke aage fingers ya toes me numbness, tingling ya unusual thandak feel ho rahi hai?"
                }
            ]
        },
        "urinary_dysuria": {
            "keywords": ["urination", "burning urine", "uti", "dysuria", "blood in urine", "peshab me jalan", "peshab me dard", "पेशाब में जलन", "पेशाब में खून"],
            "questions": [
                {
                    "target_field": "urinary_fever_flank_hematuria",
                    "reason": "Screening for upper tract infection (pyelonephritis) or renal calculi.",
                    "English": "I'm sorry you are dealing with painful urination, which can be very distressing. Have you noticed any visible blood in your urine, lower back or flank pain, or sudden chills with fever?",
                    "Hindi": "पेशाब में जलन और दर्द वास्तव में बहुत कष्टकारी होता है। क्या आपने पेशाब में खून, कमर या पीठ के निचले हिस्से में दर्द, या कंपकंपी के साथ बुखार महसूस किया है?",
                    "Hinglish": "Peshab me jalan aur dard kaafi takleefdeh hota hai. Kya urine me blood, kamar ya flank me dard, ya shivering ke sath bukhar aa raha hai?"
                },
                {
                    "target_field": "urinary_retention_frequency",
                    "reason": "Screening for acute urinary retention or pelvic outflow obstruction.",
                    "English": "Are you having difficulty passing urine or feeling like your bladder will not fully empty despite frequent urges?",
                    "Hindi": "क्या आपको पेशाब करने में रुकावट महसूस हो रही है, या बार-बार इच्छा होने के बावजूद मूत्राशय पूरी तरह खाली नहीं हो रहा?",
                    "Hinglish": "Kya peshab pass karne me rukawat feel ho rahi hai, ya baar baar urge hone ke baad bhi bladder empty nahi ho raha?"
                }
            ]
        },
        "eye_ophthalmic": {
            "keywords": ["eye pain", "red eye", "vision", "blurred vision", "eye discharge", "pus in eye", "aankh me dard", "aankh laal", "आंख में दर्द", "आंख लाल"],
            "questions": [
                {
                    "target_field": "eye_vision_photophobia",
                    "reason": "Screening for acute glaucoma, corneal ulcer, or uveitis threatening vision.",
                    "English": "Eye discomfort can be particularly distressing, and we want to protect your vision. Have you noticed any sudden loss or blurring in your vision, severe pain, or extreme sensitivity to light?",
                    "Hindi": "आंखों में तकलीफ बहुत चिंताजनक हो सकती है। क्या आपको दृष्टि में धुंधलापन, नजर कमजोर होना, तेज चुभन या रोशनी से अत्यधिक परेशानी महसूस हो रही है?",
                    "Hinglish": "Aankhon me pareshani bohot delicate hoti hai. Kya vision me blurriness, extreme eye pain, ya bright light se bohot chubhan ho rahi hai?"
                },
                {
                    "target_field": "eye_chemical_discharge",
                    "reason": "Screening for chemical ocular injury or bacterial keratitis.",
                    "English": "Did any chemical, foreign object, or dust splash into your eye, or is there any thick yellow or green discharge?",
                    "Hindi": "क्या आंख में कोई केमिकल, धूल या बाहरी कण चला गया था, या आंख से गाढ़ा पीला/हरा मवाद बह रहा है?",
                    "Hinglish": "Kya eye me koi chemical ya dust chala gaya tha, ya eye se thick yellow/green discharge aa raha hai?"
                }
            ]
        },
        "dizziness_vertigo": {
            "keywords": ["dizziness", "dizzy", "lightheaded", "vertigo", "spinning", "chakkar", "चक्कर"],
            "questions": [
                {
                    "target_field": "vertigo_nature_and_syncope",
                    "reason": "Differentiating peripheral vertigo from central neurologic or hemodynamic syncope.",
                    "English": "Feeling dizzy can be very unsettling. Does the room feel like it is spinning around you when you move your head, and have you had any sudden fainting or blackouts?",
                    "Hindi": "चक्कर आना बहुत असहज करने वाला अनुभव होता है। क्या सिर हिलाने पर कमरा गोल घूमता महसूस होता है, या आपको कभी बेहोशी या आंखों के आगे अंधेरा महसूस हुआ?",
                    "Hinglish": "Chakkar aana bohot uneasy feel karata hai. Kya sir ghumane par room spin karta hua lagta hai, ya koi fainting/blackout hua hai?"
                },
                {
                    "target_field": "vertigo_neurological_screen",
                    "reason": "Screening for posterior circulation stroke or focal deficits.",
                    "English": "Along with the dizziness, have you experienced any weakness in your arms or legs, numbness, double vision, or slurred speech?",
                    "Hindi": "चक्कर के साथ-साथ क्या आपको हाथ-पैरों में कमजोरी, सुन्नपन, दोहरी दृष्टि या बोलने में लड़खड़ाहट महसूस हुई?",
                    "Hinglish": "Chakkar ke sath kya hath-pair me weakness, double vision, numbness ya speech slurring notice hui hai?"
                }
            ]
        },
        "back_pain": {
            "keywords": ["back pain", "lower back", "spine", "sciatica", "kamar", "peeth", "कमर", "पीठ"],
            "questions": [
                {
                    "target_field": "back_cauda_equina_red_flags",
                    "reason": "Screening for emergency cauda equina syndrome requiring immediate spinal decompression.",
                    "English": "Back pain can severely disrupt your daily life, and I'm sorry you are dealing with this. Have you noticed any numbness around your groin or buttocks (saddle area), or any loss of bowel or bladder control?",
                    "Hindi": "कमर का दर्द रोजमर्रा के कामों में बहुत परेशानी पैदा करता है। क्या आपको जांघों के बीच या नितंबों के आसपास सुन्नपन महसूस हो रहा है, या पेशाब/शौच पर नियंत्रण खोने की कोई समस्या हुई है?",
                    "Hinglish": "Kamar dard sach me bohot pareshan karta hai. Kya aapko groin area me numbness feel ho rahi hai, ya urine/bowel control me koi achanak problem aayi hai?"
                },
                {
                    "target_field": "back_radiation_weakness",
                    "reason": "Assessing radiculopathy and progressive motor deficit.",
                    "English": "Does the pain shoot down past your knee into your foot, and have you had difficulty lifting your foot or toes when walking?",
                    "Hindi": "क्या यह दर्द घुटने से नीचे पंजे तक जाता है, और क्या चलते समय पंजा या उंगलियां ऊपर उठाने में कमजोरी लग रही है?",
                    "Hinglish": "Kya ye dard ghutne se niche pair tak shoot kar raha hai, aur chalne me foot drop ya weakness hai?"
                }
            ]
        },
        "cough_respiratory": {
            "keywords": ["cough", "coughing", "khansi", "खांसी", "balgam", "phlegm"],
            "questions": [
                {
                    "target_field": "cough_hemoptysis_pleurisy",
                    "reason": "Screening for hemoptysis, pulmonary embolism, or pneumonia.",
                    "English": "A persistent cough can be really exhausting. Have you been coughing up any blood or rust-colored phlegm, and is there any sharp chest pain when taking a deep breath?",
                    "Hindi": "लगातार खांसी आना बहुत थका देने वाला होता है। क्या आपको खांसी में खून या गहरा बलगम आ रहा है, और क्या गहरी सांस लेने पर सीने में कोई तेज चुभन होती है?",
                    "Hinglish": "Continuous khansi bohot exhausting hoti hai. Kya khansi me blood ya thick phlegm aa raha hai, aur deep breath lene par chest me koi sharp pain hota hai?"
                },
                {
                    "target_field": "cough_duration_systemic",
                    "reason": "Assessing chronic infectious etiology (e.g. tuberculosis) vs acute infection.",
                    "English": "How long have you had this cough, and have you had night sweats, unexplained weight loss, or high fever?",
                    "Hindi": "आपको यह खांसी कितने समय से है, और क्या रात में पसीना आना, वजन घटना या तेज बुखार की समस्या हुई है?",
                    "Hinglish": "Ye khansi kitne time se hai, aur kya raat me pasina, weight loss ya high fever notice hua hai?"
                }
            ]
        },
        "diarrhea_gi": {
            "keywords": ["diarrhea", "loose motion", "watery stool", "stomach cramps", "dast", "pet kharab", "दस्त", "पेट खराब"],
            "questions": [
                {
                    "target_field": "diarrhea_blood_hydration",
                    "reason": "Screening for invasive enteritis, dysentery, and severe dehydration.",
                    "English": "Frequent diarrhea can be exhausting and dehydrating. Have you noticed any red blood or dark black color in your stools, or are you unable to keep fluids down without vomiting?",
                    "Hindi": "दस्त और पेट में मरोड़ से बहुत कमजोरी और डिहाइड्रेशन हो सकता है। क्या मल में खून आ रहा है, या उल्टी के कारण आप पानी भी नहीं पचा पा रहे हैं?",
                    "Hinglish": "Loose motions aur cramps se bohot weakness ho sakti hai. Kya stool me blood aa raha hai, ya vomiting ki wajah se fluids digest nahi ho pa rahe?"
                },
                {
                    "target_field": "diarrhea_frequency_dizziness",
                    "reason": "Quantifying volume loss and orthostatic intolerance.",
                    "English": "Roughly how many loose bowel movements have you had today, and are you feeling dizzy or lightheaded when standing up?",
                    "Hindi": "आज लगभग कितनी बार पतले दस्त हुए हैं, और क्या खड़े होने पर चक्कर या कमजोरी महसूस हो रही है?",
                    "Hinglish": "Aaj lagbhag kitni baar loose motions hue hain, aur kya khade hone par chakkar ya fainting lag rahi hai?"
                }
            ]
        },
        "ear_pain_ent": {
            "keywords": ["ear pain", "earache", "ear discharge", "kaan me dard", "कान में दर्द", "कान बहना"],
            "questions": [
                {
                    "target_field": "ear_discharge_mastoid",
                    "reason": "Screening for tympanic membrane perforation and acute mastoiditis.",
                    "English": "Earaches can be intensely painful and frustrating. Are you experiencing any fluid or pus draining from the ear, sudden hearing loss, or swelling and tenderness right behind your ear?",
                    "Hindi": "कान का दर्द बहुत असहनीय हो सकता है। क्या कान से कोई मवाद या पानी बह रहा है, सुनने में अचानक कमी आई है, या कान के ठीक पीछे कोई सूजन या दर्द है?",
                    "Hinglish": "Kaan ka dard bohot sharp hota hai. Kya kaan se koi discharge/fluid beh raha hai, hearing kam hui hai, ya kaan ke peeche sujan hai?"
                }
            ]
        },
        "dental_oral": {
            "keywords": ["toothache", "tooth pain", "dental", "gum swelling", "daant me dard", "दांत में दर्द", "मसूड़ों में सूजन"],
            "questions": [
                {
                    "target_field": "dental_facial_swelling_trismus",
                    "reason": "Screening for spreading odontogenic infection, Ludwig's angina, or space abscess.",
                    "English": "Toothaches can be throbbing and miserable. Have you noticed any swelling in your face, cheek, or under your jaw, or any difficulty opening your mouth or swallowing?",
                    "Hindi": "दांत का दर्द वास्तव में बहुत असहनीय होता है। क्या आपके चेहरे, गाल या जबड़े के नीचे कोई सूजन आई है, या मुंह खोलने और निगलने में कठिनाई हो रही है?",
                    "Hinglish": "Daant ka dard sach me bohot bura hota hai. Kya face, cheek ya jaw ke neeche sujan aa gayi hai, ya muh kholne me dikkat ho rahi hai?"
                }
            ]
        },
        "burn_wound": {
            "keywords": ["burn", "scald", "cut", "wound", "bleeding wound", "jal gaya", "chot", "घाव", "जल गया"],
            "questions": [
                {
                    "target_field": "burn_wound_depth_extent",
                    "reason": "Assessing tissue depth, blistering, and anatomical surface involvement.",
                    "English": "Burns and wounds require careful attention to prevent infection. Is the skin blistered, white, or charred, and roughly how large is the affected area compared to your hand?",
                    "Hindi": "जले और खुले घावों की तुरंत देखभाल जरूरी होती है। क्या त्वचा पर फफोले पड़े हैं या त्वचा सफेद/काली पड़ गई है, और प्रभावित हिस्सा हथेली की तुलना में कितना बड़ा है?",
                    "Hinglish": "Jalna aur wound infection se bachane ke liye care mangta hai. Kya skin par blisters hain ya skin charred ho gayi hai, aur area hand ke compare me kitna bada hai?"
                }
            ]
        },
        "palpitations": {
            "keywords": ["palpitations", "racing heart", "heart pounding", "heart fluttering", "dil ki dhadkan", "घबराहट", "दिल की धड़कन"],
            "questions": [
                {
                    "target_field": "palpitations_syncope_duration",
                    "reason": "Screening for hemodynamically unstable cardiac arrhythmias.",
                    "English": "A fluttering or racing heartbeat can understandably cause anxiety. Does your heart race while resting, and have you felt faint, lightheaded, or had any chest tightness?",
                    "Hindi": "दिल की तेज धड़कन या घबराहट बेचैन कर सकती है। क्या बैठे-बैठे भी दिल तेजी से धड़क रहा है, और क्या इसके साथ चक्कर, बेहोशी या सीने में भारीपन महसूस हुआ?",
                    "Hinglish": "Dil ki beating tez hona bohot uneasy feel karata hai. Kya rest karte waqt bhi heart race kar raha hai, aur kya chakkar ya chest tightness hui hai?"
                }
            ]
        },
        "allergic_reaction": {
            "keywords": ["allergic reaction", "swollen lip", "swollen tongue", "anaphylaxis", "allergy", "एलर्जी", "होंठ सूज", "जीभ सूज"],
            "questions": [
                {
                    "target_field": "allergy_airway_swelling",
                    "reason": "Emergency screening for progressive upper airway angioedema and anaphylaxis.",
                    "English": "Allergic reactions need rapid assessment. Are your lips, tongue, or throat swelling, or are you having difficulty swallowing or catching your breath?",
                    "Hindi": "एलर्जी की तुरंत जांच जरूरी होती है। क्या आपके होंठ, जीभ या गले में सूजन आ रही है, या आपको निगलने या सांस लेने में परेशानी हो रही है?",
                    "Hinglish": "Allergic reactions ko instantly assess karna zaroori hai. Kya lips, tongue ya throat me sujan hai, ya saans lene aur nigalne me dikkat ho rahi hai?"
                }
            ]
        }
    }

    # Empathetic Fallback Rotation for Essential Clinical Dimensions
    CLINICAL_DIMENSION_QUESTIONS = [
        ("onset_and_duration", {
            "reason": "Onset and duration are essential for clinical acuity determination.",
            "English": "Thank you for explaining your symptoms. To help us establish how acutely this began, when did your symptoms first start, and have they been constant, worsening, or coming and going?",
            "Hindi": "अपने लक्षण बताने के लिए धन्यवाद। स्थिति की शुरुआत समझने के लिए, यह लक्षण सबसे पहले कब शुरू हुए और क्या यह समय के साथ बढ़ रहे हैं, घट रहे हैं या लगातार बने हुए हैं?",
            "Hinglish": "Symptom share karne ke liye shukriya. Timing samajhne ke liye, ye symptoms sabse pehle kab shuru hue, aur kya ye continuously badh rahe hain ya kam-zyada ho rahe hain?"
        }),
        ("severity", {
            "reason": "Quantifying symptom severity to establish clinical acuity.",
            "English": "I understand this is causing discomfort. To help our medical team gauge the severity right now, on a scale from 0 to 10 (where 0 is no discomfort and 10 is unbearable), how severe is your symptom?",
            "Hindi": "हम आपकी परेशानी को समझते हैं। डॉक्टरों को गंभीरता का सही अंदाजा देने के लिए, 0 से 10 के पैमाने पर (जहाँ 0 कोई दर्द नहीं और 10 असहनीय दर्द है), अभी आपका दर्द कितना तीव्र है?",
            "Hinglish": "Hum samajh sakte hain ki ye uncomfortable hai. Acuity gauge karne ke liye, 0 se 10 ke scale par (jahan 0 koi dard nahi aur 10 unbearable pain hai), abhi dard kitna severe hai?"
        }),
        ("medications_and_history", {
            "reason": "Identifying relevant baseline health status, regular medications, and allergies for clinical safety.",
            "English": "To ensure any advice or treatment given is completely safe for you, are you currently taking any prescription medications, or do you have any relevant medical conditions or allergies?",
            "Hindi": "आपकी सुरक्षा और सही उपचार सुनिश्चित करने के लिए, क्या आप कोई नियमित दवाइयाँ ले रहे हैं, या आपको कोई पुरानी बीमारी या एलर्जी है?",
            "Hinglish": "Aapki safety aur sahi treatment ensure karne ke liye, kya aap koi regular medications le rahe hain, ya aapko koi existing medical conditions ya allergies hain?"
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
        if re.search(r'[\u0900-\u097F]', text):
            return "Hindi"

        # Check for common Hinglish tokens
        hinglish_words = ["dard", "bukhar", "ulti", "chakkar", "seene", "saans", "gardan", "pet", "nahi", "bahut", "bhi", "tha", "raha", "hai", "kaafi", "zyada", "chot"]
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
        preferred_language: str = "English",
        provider: Optional[str] = None
    ) -> FollowUpQuestionOutput:
        """
        Determines the next follow-up question using the selected MedGemma provider.
        Maintains bilingual question state (EN + HI), guarantees no repetitive questions,
        and targets unaddressed clinical dimensions.
        """
        assistant_messages = assistant_messages or []

        # Enforce maximum question limit or early completion
        if current_question_count >= settings.MAX_FOLLOWUP_QUESTIONS:
            logger.info(f"Reached maximum question limit ({settings.MAX_FOLLOWUP_QUESTIONS}). Ending questioning.")
            return FollowUpQuestionOutput(
                next_question=None,
                next_question_en=None,
                next_question_hi=None,
                reason_for_question=f"Reached maximum limit of {settings.MAX_FOLLOWUP_QUESTIONS} follow-up questions. Intake ready for human clinician review.",
                target_field="limit_reached",
                is_terminal=True,
                selected_provider=provider or settings.DEFAULT_MEDGEMMA_PROVIDER
            )

        full_patient_text = " ".join(patient_messages)
        lang = cls._detect_language(preferred_language, full_patient_text)

        # Extract vitals if present in evidence_state
        vitals = {}
        if evidence_state:
            vitals = evidence_state.get("vitals", {})

# Run selected MedGemma SLM provider to extract features and generate bilingual question
        slm_turn = {}
        slm_error = None
        extracted_rep = evidence_state.get("extracted_report") if evidence_state else None
        known_facts = evidence_state.get("known_clinical_facts") if evidence_state else None


        try:
            slm_turn = medgemma_service.generate_slm_triage_turn(
                conversation_history=patient_messages,
                vitals=vitals,
                extracted_report=extracted_rep,
                previous_questions=assistant_messages,
                known_clinical_facts=known_facts,
                preferred_language=lang,
                provider=provider
            )
        except Exception as slm_err:
            logger.warning(f"SLM questioning extraction encountered error: {slm_err}")
            slm_error = slm_err
            # If mock fallbacks are disabled for testing, raise actual error immediately
            if os.getenv("ENABLE_MOCK_FALLBACK", "false").lower() != "true":
                raise

        q_en = slm_turn.get("next_question_en")
        q_hi = slm_turn.get("next_question_hi")

        # Step 1: Return the neural SLM's generated question if valid and not already asked
        if slm_turn and (q_en or q_hi):
            cand_en = (q_en or "").strip()
            cand_hi = (q_hi or "").strip()
            is_dup = False
            if cand_en and cls._is_duplicate_question(cand_en, assistant_messages):
                is_dup = True
            if not is_dup and cand_hi and cls._is_duplicate_question(cand_hi, assistant_messages):
                is_dup = True

            if not is_dup and (cand_en or cand_hi):
                q_text = cand_hi if (lang.lower() == "hindi" and cand_hi) else (cand_en or cand_hi)
                return FollowUpQuestionOutput(
                    next_question=q_text,
                    next_question_en=cand_en or q_text,
                    next_question_hi=cand_hi or q_text,
                    reason_for_question=f"Targeting {slm_turn.get('primary_system_flag', 'system')} indicators (boundary: {slm_turn.get('target_triage_boundary', 'P2')})",
                    target_field=slm_turn.get("primary_system_flag", "clinical_inquiry"),
                    is_terminal=False,
                    slm_extraction=slm_turn,
                    selected_provider=provider or slm_turn.get("selected_provider", "finetuned_medgemma"),
                    validation_status=slm_turn.get("validation_status", "valid")
                )

        # Step 2: Fallback to targeted symptom inquiry or unasked clinical dimension catalog
        cat_q = cls._get_unasked_fallback_question(assistant_messages, patient_messages, lang)
        if cat_q:
            cat_q.slm_extraction = slm_turn
            if not cat_q.next_question_en:
                cat_q.next_question_en = cat_q.next_question if lang.lower() != "hindi" else (q_en or cat_q.next_question)
            if not cat_q.next_question_hi:
                cat_q.next_question_hi = cat_q.next_question if lang.lower() == "hindi" else (q_hi or cat_q.next_question)
            cat_q.selected_provider = provider or slm_turn.get("selected_provider", "finetuned_medgemma")
            cat_q.validation_status = slm_turn.get("validation_status", "valid")
            return cat_q

        # Step 3: Guarded Mock Catalog Fallback (strictly disabled during live neural testing)
        enable_mock = os.getenv("ENABLE_MOCK_FALLBACK", "false").lower() == "true"
        if not enable_mock:
            raise RuntimeError(
                f"Neural SLM inference did not produce a valid question. "
                f"Mock fallback is disabled (ENABLE_MOCK_FALLBACK=false). Cause: {slm_error or 'Empty SLM output'}"
            )

        # Fallback question
        return FollowUpQuestionOutput(
            next_question="Could you describe any other symptoms or discomfort you are experiencing?",
            next_question_en="Could you describe any other symptoms or discomfort you are experiencing?",
            next_question_hi="क्या आप अपने अन्य लक्षणों या किसी अन्य परेशानी के बारे में बता सकते हैं?",
            reason_for_question="General exploratory inquiry to capture unaddressed patient symptoms.",
            target_field="general_symptoms",
            is_terminal=False,
            slm_extraction=slm_turn,
            selected_provider=provider or "finetuned_medgemma"
        )

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
        """Selects the next unaddressed symptom-specific or general clinical dimension with genuine empathy."""
        full_patient_text = " ".join(patient_messages).lower()

        # Step 1: Check Symptom-Specific Questions across catalog
        for symptom_key, symp_data in cls.SYMPTOM_QUESTION_CATALOG.items():
            if any(k in full_patient_text for k in symp_data["keywords"]):
                for q_spec in symp_data["questions"]:
                    target_field = q_spec["target_field"]
                    q_text = q_spec.get(language, q_spec["English"])

                    # Check if already asked in any language form or target field
                    if any(cls._is_duplicate_question(q_spec.get(l, ""), previous_questions) for l in ("English", "Hindi", "Hinglish") if l in q_spec):
                        continue
                    if any(target_field in prev.lower() for prev in previous_questions):
                        continue
                    if target_field == "pain_radiation" and any(any(w in p.lower() for w in ["radiat", "left arm", "jaw", "neck", "बाएं हाथ", "कंधे", "पीठ की तरफ"]) for p in previous_questions):
                        continue
                    if target_field == "hematemesis_appearance_and_volume" and any(any(w in p.lower() for w in ["coffee", "bright red", "कितनी मात्रा", "amount"]) for p in previous_questions):
                        continue

                    # Check if already explicitly answered in patient narrative
                    if target_field == "pain_radiation" and any(w in full_patient_text for w in ["radiat", "spread", "left arm", "jaw", "neck", "baaye hath", "गर्दन", "फैल"]):
                        continue
                    if target_field == "hematemesis_appearance_and_volume" and any(w in full_patient_text for w in ["bright red", "coffee ground", "laal", "dark red", "fresh blood"]):
                        continue
                    if target_field == "trauma_weight_bearing_deformity" and any(w in full_patient_text for w in ["cannot walk", "can't walk", "unable to walk", "cant bear weight", "chal nahi", "चल नहीं"]):
                        continue

                    return FollowUpQuestionOutput(
                        next_question=q_text,
                        next_question_en=q_spec.get("English", q_text),
                        next_question_hi=q_spec.get("Hindi", q_text),
                        reason_for_question=q_spec["reason"],
                        target_field=target_field,
                        is_terminal=False
                    )

        # Step 2: Fall back to General Dimensions, strictly avoiding already-answered topics
        for target_field, dim_spec in cls.CLINICAL_DIMENSION_QUESTIONS:
            q_text = dim_spec.get(language, dim_spec["English"])

            # Check if this topic was already asked
            if cls._is_duplicate_question(q_text, previous_questions):
                continue
            if any(target_field in prev for prev in previous_questions):
                continue

            # Check if already answered in patient messages
            if target_field == "onset_and_duration" and any(w in full_patient_text for w in ["yesterday", "ago", "started", "since", "days", "hours", "weeks", "months", "today", "morning", "night", "din se", "kal se", "दिन से", "घंटे से", "kal", "aaj"]):
                continue
            if target_field == "severity" and (
                any(w in full_patient_text for w in ["/10", "out of 10", "mild", "severe", "moderate", "unbearable", "excruciating", "extreme", "tez dard", "असहनीय", "बहुत तेज", "bahut tez"]) or
                any(pt.strip() in [str(i) for i in range(1, 11)] for pt in patient_messages)
            ):
                continue
            if target_field == "medications_and_history" and any(w in full_patient_text for w in ["taking", "medication", "medicine", "pill", "prescript", "dawa", "allergy", "allergic", "no meds", "koi dawai"]):
                continue

            return FollowUpQuestionOutput(
                next_question=q_text,
                next_question_en=dim_spec.get("English", q_text),
                next_question_hi=dim_spec.get("Hindi", q_text),
                reason_for_question=dim_spec["reason"],
                target_field=target_field,
                is_terminal=False
            )

        # If all relevant dimensions have been asked or addressed, conclude questioning
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
