import re
import logging
from typing import Dict, Any, List, Optional
from backend.schemas.clinical import TriageResult, TriageTrigger
from backend.config import settings

logger = logging.getLogger("careflow.triage")

# Documented Protocol Source:
# Adapted for illustrative clinical research prototype from Emergency Severity Index (ESI)
# and Manchester Triage System (MTS) Red Flag Discriminators.
# PROTOCOL_VERSION = "v1.0.0-prototype"

class DeterministicTriageEngine:
    """
    Independent, rule-based clinical triage engine.
    
    Evaluates grounded evidence and facts extracted from patient intake.
    MedGemma DOES NOT determine the urgency category.
    
    Triage Hierarchy:
    1. System Health / Model Failure Check -> 'ERR' (Review required — AI unavailable)
    2. Emergency Red Flags (P0) -> Immediate Escalation
    3. Urgent Indicators (P1) -> Priority Clinician Assessment
    4. Unresolved Contradictions or Missing Critical Flags (P2) -> Incomplete / Uncertain
    5. Routine Review (P3) -> Standard Non-Urgent Queue
    """

    PROTOCOL_VERSION = settings.TRIAGE_RULE_VERSION

    # P0: Emergency Red Flag Rules (Bilingual English & Hindi)
    P0_RULES = [
        {
            "id": "RULE-P0-CARD-01",
            "name": "Acute Coronary Syndrome Red Flag",
            "criterion": "Chest pain radiating to left arm, neck, or jaw",
            "pattern": r"(chest\s*pain|pressure\s*in\s*chest|chest\s*tightness|सीने\s*में\s*(?:.{0,25}?)(?:दर्द|दबाव|भारी)|छाती\s*में\s*(?:.{0,25}?)(?:दर्द|दबाव)|सीना\s*भारी|chhati\s*me\s*dard|seene\s*me\s*(?:.{0,25}?)(?:dard|dabav)).*(radiat|arm|jaw|neck|बाएं\s*हाथ|बाँह|बांह|गर्दन|जबड़े|haath\s*me|gardan)|(radiat|arm|jaw|neck|बाएं\s*हाथ|बाँह|बांह|गर्दन|जबड़े).*(chest\s*pain|chest\s*pressure|सीने\s*में\s*(?:.{0,25}?)(?:दर्द|दबाव)|छाती\s*में\s*(?:.{0,25}?)(?:दर्द|दबाव))",
            "action": "Immediate emergency department transfer; activate acute coronary protocol."
        },
        {
            "id": "RULE-P0-RESP-02",
            "name": "Acute Severe Respiratory Compromise",
            "criterion": "Severe shortness of breath, inability to speak full sentences, or stridor",
            "pattern": r"(can't\s*breathe|cannot\s*breathe|gasping|severe\s*shortness\s*of\s*breath|stridor|choking|सांस\s*फूल|दम\s*फूल|सांस\s*लेने\s*में\s*(?:.{0,20}?)(?:तकलीफ|दिक्कत|परेशानी)|दम\s*घुट|सांस\s*नहीं\s*आ\s*रही|saans\s*lene\s*me\s*takleef|dam\s*phool\s*raha)",
            "action": "Immediate respiratory assessment and oxygenation support."
        },
        {
            "id": "RULE-P0-NEURO-03",
            "name": "Acute Stroke Symptom Complex (FAST)",
            "criterion": "Sudden facial drooping, arm weakness, or slurred speech",
            "pattern": r"(facial\s*droop|face\s*droop|slurred\s*speech|weakness\s*on\s*one\s*side|sudden\s*numbness|can't\s*move\s*arm|चेहरा\s*टेढ़ा|बोलने\s*में\s*दिक्कत|लकवा|ek\s*taraf\s*kamzori)",
            "action": "Activate emergency stroke pathway; record time of last known normal."
        },
        {
            "id": "RULE-P0-HEM-04",
            "name": "Severe Hemorrhage or Hematemesis",
            "criterion": "Active hematemesis (vomiting blood), coughing up blood, uncontrolled bleeding, or sudden syncope",
            "pattern": r"(?:vomit(?:ing|ed)?\s*blood|throwing\s*up\s*blood|threw\s*up\s*blood|blood\s*in\s*(?:my\s*)?vomit|hematemesis|coffee[\s\-]ground\s*(?:emesis|vomit)|coughing\s*up\s*blood|hemoptysis|uncontrolled\s*bleeding|passed\s*out|syncope|खून\s*की\s*उल्टी|उल्टी\s*में\s*खून|उल्टी\s*के\s*साथ\s*खून|रक्त\s*वमन|बेहोश|khoon\s*ki\s*ulti|ulti\s*me(?:n)?\s*(?:khoon|blood)|behosh)",
            "action": "Immediate resuscitation, airway protection, and emergency surgical/endoscopic evaluation."
        },
        {
            "id": "RULE-P0-ANAPH-05",
            "name": "Anaphylaxis Warning Signs",
            "criterion": "Swelling of lips/tongue/throat with respiratory distress or sudden generalized hives",
            "pattern": r"(swollen\s*tongue|swollen\s*throat|throat\s*closing|lips\s*swelling|anaphylaxis|गला\s*बंद|जीभ\s*सूज|होंठ\s*सूज)",
            "action": "Immediate intramuscular epinephrine and airway management."
        }
    ]

    # P1: Urgent Review Rules (Bilingual English & Hindi)
    P1_RULES = [
        {
            "id": "RULE-P1-INF-01",
            "name": "Meningeal Irritation in Febrile Illness",
            "criterion": "High fever accompanied by neck stiffness or photophobia",
            "pattern": r"(fever|high\s*temp|बुखार|bukhar).*(stiff.*neck|neck.*stiff|cannot\s*bend\s*neck|hurts\s*to\s*bend.*head|गर्दन\s*में\s*अकड़न|gardan\s*me\s*akad)|(stiff.*neck|neck.*stiff|गर्दन\s*में\s*अकड़न).*(fever|temp|बुखार|bukhar)",
            "action": "Urgent in-person physician assessment within 1 hour."
        },
        {
            "id": "RULE-P1-PAIN-02",
            "name": "Severe Uncontrolled Pain",
            "criterion": "Reported pain severity >= 8/10 or severe/unbearable pain",
            "pattern": r"(?:\b([89]|10)\s*(?:out\s*of\s*10|/10|\s*/\s*10|out\s*of\s*ten)\b|\b(?:pain|severity|scale|level|dard|rating)\s*(?:is|=|:)?\s*([89]|10)\b|\b([89]|10)\s*(?:pain|severity|level|rating|dard)\b|(?:severe|unbearable|very\s*high|high|extreme|excruciating)\s+(?:pain|headache|headahed|chest\s*pain|abdominal\s*pain|stomach\s*pain|back\s*pain|cramps|ache)|severe\s*unbearable\s*pain|unbearable\s*pain|excruciating\s*pain|असहनीय\s*दर्द|बहुत\s*(?:ज़्यादा|ज्यादा|तेज़|तेज)\s*दर्द|बहुत\s*(?:shar\s*|sar\s*|sir\s*)?dard|bahut\s*(?:zyada\s*|jada\s*|tez\s*)?(?:shar\s*|sar\s*|sir\s*)?dard)",
            "action": "Urgent clinical review for acute pain etiology and analgesia."
        },
        {
            "id": "RULE-P1-CARD-05",
            "name": "Acute Central Chest Pain or Pressure",
            "criterion": "New or acute chest pain, tightness, or cardiac discomfort without confirmed radiation",
            "pattern": r"(chest\s*pain|pressure\s*in\s*chest|chest\s*tightness|tightness\s*in\s*(?:my\s*)?chest|lower\s*chest\s*is\s*paining|सीने\s*में\s*(?:.{0,25}?)(?:दर्द|दबाव|भारी)|छाती\s*में\s*(?:.{0,25}?)(?:दर्द|दबाव)|सीना\s*भारी|chhati\s*me\s*(?:.{0,25}?)(?:dard|dabav)|seene\s*me\s*(?:.{0,25}?)(?:dard|dabav)|dil\s*me\s*(?:.{0,25}?)(?:dard|pain)|दिल\s*में\s*(?:.{0,25}?)(?:दर्द))",
            "action": "Urgent clinical and ECG evaluation within 30-60 minutes."
        },
        {
            "id": "RULE-P1-ABD-03",
            "name": "Acute Abdomen with Intolerance",
            "criterion": "Severe abdominal pain with intractable vomiting or fever",
            "pattern": r"(abdominal\s*pain|stomach\s*pain|belly\s*pain|पेट\s*में\s*दर्द|pet\s*me\s*dard).*(vomit|throwing\s*up|उल्टी|ulti)|(vomit|throwing\s*up|उल्टी).*(abdominal\s*pain|stomach\s*pain|पेट\s*में\s*दर्द)",
            "action": "Urgent surgical / abdominal workup within 2 hours."
        },
        {
            "id": "RULE-P1-RESP-04",
            "name": "Subacute Respiratory Distress",
            "criterion": "Shortness of breath with fever or cough, without immediate collapse",
            "pattern": r"(shortness\s*of\s*breath|trouble\s*breathing|सांस\s*लेने\s*में\s*परेशानी).*(fever|cough|बुखार|खांसी)|(fever|cough|बुखार|खांसी).*(shortness\s*of\s*breath|trouble\s*breathing)",
            "action": "Urgent chest evaluation and pulse oximetry."
        },
        {
            "id": "RULE-P1-GI-06",
            "name": "Suspected Gastrointestinal Bleeding",
            "criterion": "Dark black tarry stools (melena) or suspected gastrointestinal bleeding",
            "pattern": r"(?:black\s*tarry\s*stool|melena|dark\s*black\s*stool|काला\s*मल|kali\s*latrine|latrine\s*me\s*khoon)",
            "action": "Urgent physician assessment, IV access, and GI endoscopy consultation within 1 to 2 hours."
        }
    ]

    @staticmethod
    def _is_negated(match: re.Match, text: str) -> bool:
        """Determines if a matched symptom pattern is explicitly negated in context."""
        start = match.start()
        end = match.end()
        pre_text = text[max(0, start - 45):start].lower()
        post_text = text[end:min(len(text), end + 35)].lower()

        # If there is a clause boundary before the match, only inspect the current clause
        clause_break = max(
            pre_text.rfind("."), pre_text.rfind(";"), pre_text.rfind("!"),
            pre_text.rfind(" but "), pre_text.rfind(" however "), pre_text.rfind(" lekin "), pre_text.rfind(" par ")
        )
        if clause_break != -1:
            pre_text = pre_text[clause_break:]

        # Prefix negation in English/Hindi/Hinglish
        if re.search(r"\b(no|not|without|denies|denied|never|none|zero)\b(?:\s+\w+){0,3}\s*$", pre_text.strip()):
            return True
        if re.search(r"(नहीं|न|कोई नहीं|बिना)\s*$", pre_text.strip()):
            return True
        if re.search(r"\b(nahi|nahin|na)\b(?:\s+\w+){0,2}\s*$", pre_text.strip()):
            return True

        # Postfix negation in Hindi/Hinglish/English
        post_clause_break = min(
            [pos for pos in [post_text.find("."), post_text.find(";"), post_text.find("!"), post_text.find(" but "), post_text.find(" lekin ")] if pos != -1] or [len(post_text)]
        )
        post_clause = post_text[:post_clause_break].strip()

        if re.search(r"^\s*(?:nahi\b|nahin\b|nai\b|absent\b|resolved\b|is not present\b|nahi hai|nahin hai|nai hai|नहीं|नहीं है|नहीं हुई|न है)", post_clause):
            return True

        return False

    def evaluate(
        self,
        evidence_state: Dict[str, Any],
        raw_messages: List[str],
        ai_status: str = "ready"
    ) -> TriageResult:
        """
        Evaluates clinical evidence and generates a transparent, auditable TriageResult.
        Combines deterministic protocol safety checks with MedGemma model-reasoned urgency signals.
        """
        # 1. Check for AI / System Failure
        if ai_status == "failed":
            logger.warning("AI status is 'failed'. Triage defaulting to ERR.")
            return TriageResult(
                priority_category="ERR",
                priority_label="⚠️ Review Required — AI Unavailable",
                reasons=["AI model service or schema validation failed. Manual clinician review mandatory."],
                warning_signs=["System failure: AI-extracted evidence unavailable."],
                triggered_rules=[
                    TriageTrigger(
                        rule_id="RULE-SYS-01",
                        rule_name="System Failure Safeguard",
                        priority="ERR",
                        criterion="AI extraction failure or invalid model output",
                        matched_evidence=f"ai_status: {ai_status}",
                        action_protocol="Manual triage by registered nurse or medical reviewer."
                    )
                ],
                missing_information=["Full automated clinical extraction"],
                uncertainties=["All fields require manual clinician assessment"],
                triage_rule_version=self.PROTOCOL_VERSION
            )

        all_patient_text = " ".join(raw_messages).lower()
        model_urgency_signals = evidence_state.get("urgency_signals", [])

        triggered_rules: List[TriageTrigger] = []
        warning_signs: List[str] = []
        reasons: List[str] = []

        # 2. Check Model-Reasoned Emergency Signals (MedGemma Semantic Red Flags)
        for sig in model_urgency_signals:
            if isinstance(sig, dict) and sig.get("severity_level") == "emergency":
                sig_name = sig.get("signal", "Critical Red Flag")
                sig_rationale = sig.get("rationale", "Emergency clinical marker identified by MedGemma")
                sig_quote = sig.get("source_quote", "Reported in intake")
                trigger = TriageTrigger(
                    rule_id="RULE-P0-MODEL-01",
                    rule_name=f"AI Clinical Red Flag: {sig_name}",
                    priority="P0",
                    criterion=sig_rationale,
                    matched_evidence=f"Patient stated: '{sig_quote[:120]}'",
                    action_protocol="Immediate clinical emergency escalation."
                )
                triggered_rules.append(trigger)
                warning_signs.append(f"AI Red Flag: {sig_name}")
                reasons.append(f"AI identified emergency signal ({sig_name}): {sig_rationale}")

        # 3. Check Protocol-Based P0 Emergency Rules
        for rule in self.P0_RULES:
            for match in re.finditer(rule["pattern"], all_patient_text):
                if self._is_negated(match, all_patient_text):
                    continue
                matched_snippet = match.group(0)
                source_msg = next((msg for msg in raw_messages if matched_snippet in msg.lower()), matched_snippet)
                trigger = TriageTrigger(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    priority="P0",
                    criterion=rule["criterion"],
                    matched_evidence=f"Patient stated: '{source_msg[:120]}'",
                    action_protocol=rule["action"]
                )
                triggered_rules.append(trigger)
                warning_signs.append(f"{rule['name']}: {rule['criterion']}")
                reasons.append(f"Triggered {rule['id']} ({rule['name']}): '{source_msg[:90]}'")
                break

        if triggered_rules:
            return TriageResult(
                priority_category="P0",
                priority_label="🚨 P0: Emergency Escalation",
                reasons=reasons,
                warning_signs=warning_signs,
                triggered_rules=triggered_rules,
                missing_information=evidence_state.get("missing_information", []),
                uncertainties=evidence_state.get("uncertainties", []),
                triage_rule_version=self.PROTOCOL_VERSION
            )

        # 4. Evaluate P1 (Urgent) Rules: Check Model-Reasoned Urgent Signals First
        for sig in model_urgency_signals:
            if isinstance(sig, dict) and sig.get("severity_level") == "urgent":
                sig_name = sig.get("signal", "Urgent Condition")
                sig_rationale = sig.get("rationale", "Urgent clinical acuity marker identified by MedGemma")
                sig_quote = sig.get("source_quote", "Reported in intake")
                trigger = TriageTrigger(
                    rule_id="RULE-P1-MODEL-01",
                    rule_name=f"AI Urgent Indicator: {sig_name}",
                    priority="P1",
                    criterion=sig_rationale,
                    matched_evidence=f"Patient stated: '{sig_quote[:120]}'",
                    action_protocol="Prompt clinical physician assessment within 1 to 2 hours."
                )
                triggered_rules.append(trigger)
                warning_signs.append(f"AI Urgent Flag: {sig_name}")
                reasons.append(f"AI identified urgent signal ({sig_name}): {sig_rationale}")

        # Evaluate Protocol-Based P1 Rules
        for rule in self.P1_RULES:
            for match in re.finditer(rule["pattern"], all_patient_text):
                if self._is_negated(match, all_patient_text):
                    continue
                matched_snippet = match.group(0)
                source_msg = next((msg for msg in raw_messages if matched_snippet in msg.lower()), matched_snippet)
                
                trigger = TriageTrigger(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    priority="P1",
                    criterion=rule["criterion"],
                    matched_evidence=f"Patient stated: '{source_msg[:120]}'",
                    action_protocol=rule["action"]
                )
                triggered_rules.append(trigger)
                warning_signs.append(f"{rule['name']}: {rule['criterion']}")
                reasons.append(f"Triggered {rule['id']} ({rule['name']}): '{source_msg[:90]}'")
                break

        # Check if patient directly provided standalone severe pain score (e.g. answering severity question with '8', '9', '10')
        standalone_high_pain = [msg.strip() for msg in raw_messages if msg.strip() in ("8", "9", "10", "8/10", "9/10", "10/10", "10 out of 10", "9 out of 10", "8 out of 10")]
        if standalone_high_pain and not any(t.rule_id == "RULE-P1-PAIN-02" for t in triggered_rules):
            trigger = TriageTrigger(
                rule_id="RULE-P1-PAIN-02",
                rule_name="Severe Uncontrolled Pain",
                priority="P1",
                criterion="Reported pain severity >= 8/10",
                matched_evidence=f"Patient reported pain score: '{standalone_high_pain[-1]}'",
                action_protocol="Urgent clinical review for acute pain etiology and analgesia."
            )
            triggered_rules.append(trigger)
            warning_signs.append("Severe Uncontrolled Pain: Reported pain severity >= 8/10")
            reasons.append(f"Triggered RULE-P1-PAIN-02 (Severe Uncontrolled Pain): '{standalone_high_pain[-1]}'")

        # Check structured severity fact in evidence_state
        facts = evidence_state.get("facts", {})
        sev_fact = facts.get("severity", {})
        sev_val = str(sev_fact.get("value", "") if isinstance(sev_fact, dict) else sev_fact).lower()
        if (any(term in sev_val for term in ["8/10", "9/10", "10/10", "severe", "unbearable", "very high"]) or sev_val in ["8", "9", "10"]) and not any(t.rule_id == "RULE-P1-PAIN-02" for t in triggered_rules):
            trigger = TriageTrigger(
                rule_id="RULE-P1-PAIN-02",
                rule_name="Severe Uncontrolled Pain",
                priority="P1",
                criterion="Extracted pain severity >= 8/10 or severe",
                matched_evidence=f"Evidence fact severity: '{sev_val}'",
                action_protocol="Urgent clinical review for acute pain etiology and analgesia."
            )
            triggered_rules.append(trigger)
            warning_signs.append("Severe Uncontrolled Pain: Extracted pain severity >= 8/10 or severe")
            reasons.append(f"Triggered RULE-P1-PAIN-02 (Severe Uncontrolled Pain): Severity fact '{sev_val}'")

        if triggered_rules:
            return TriageResult(
                priority_category="P1",
                priority_label="⚠️ P1: Urgent Review",
                reasons=reasons,
                warning_signs=warning_signs,
                triggered_rules=triggered_rules,
                missing_information=evidence_state.get("missing_information", []),
                uncertainties=evidence_state.get("uncertainties", []),
                triage_rule_version=self.PROTOCOL_VERSION
            )

        # 4. Evaluate P2 (Incomplete or Uncertain)
        # Check for contradictions, unresolved uncertainties, or multiple unknown dimensions
        contradictions = evidence_state.get("contradictions", [])
        uncertainties = evidence_state.get("uncertainties", [])
        missing_info = evidence_state.get("missing_information", [])
        unknowns = evidence_state.get("unknown_information", [])

        p2_triggers = []
        if contradictions:
            p2_triggers.append(
                TriageTrigger(
                    rule_id="RULE-P2-CONFLICT-01",
                    rule_name="Unresolved Contradictory Evidence",
                    priority="P2",
                    criterion="Conflicting statements detected in patient conversation",
                    matched_evidence="; ".join(contradictions),
                    action_protocol="Clinician callback or targeted re-interview to resolve conflicting reports."
                )
            )
            reasons.append(f"Contradictory information present: {'; '.join(contradictions)}")

        if uncertainties:
            p2_triggers.append(
                TriageTrigger(
                    rule_id="RULE-P2-UNCERTAIN-02",
                    rule_name="High Clinical Uncertainty",
                    priority="P2",
                    criterion="Ambiguous or uncertain patient reports requiring clinical clarification",
                    matched_evidence="; ".join(uncertainties),
                    action_protocol="Reviewer clarification required."
                )
            )
            reasons.append(f"High clinical uncertainty: {'; '.join(uncertainties)}")

        if len(unknowns) >= 2:
            p2_triggers.append(
                TriageTrigger(
                    rule_id="RULE-P2-INCOMPLETE-03",
                    rule_name="Multiple Unknown Clinical Dimensions",
                    priority="P2",
                    criterion="Multiple key clinical dimensions marked unknown or skipped",
                    matched_evidence="; ".join(unknowns),
                    action_protocol="Clinician callback to complete missing clinical intake history."
                )
            )
            reasons.append(f"Multiple unknown clinical dimensions: {len(unknowns)} skipped/unknown items")

        if p2_triggers:
            return TriageResult(
                priority_category="P2",
                priority_label="❓ P2: Incomplete or Uncertain",
                reasons=reasons,
                warning_signs=[],
                triggered_rules=p2_triggers,
                missing_information=missing_info,
                uncertainties=uncertainties,
                triage_rule_version=self.PROTOCOL_VERSION
            )

        # 5. Fallback to P3 (Routine Review)
        return TriageResult(
            priority_category="P3",
            priority_label="📋 P3: Routine Review",
            reasons=["No emergency (P0), urgent (P1), or contradictory/uncertain (P2) criteria detected. Patient meets criteria for routine scheduled assessment."],
            warning_signs=[],
            triggered_rules=[
                TriageTrigger(
                    rule_id="RULE-P3-STD-01",
                    rule_name="Standard Routine Criteria",
                    priority="P3",
                    criterion="Absence of acute red-flag or escalation markers; stable clinical picture",
                    matched_evidence="All reported symptoms are non-escalating and manageable in standard intake.",
                    action_protocol="Standard clinic queue; routine medical appointment."
                )
            ],
            missing_information=missing_info,
            uncertainties=uncertainties,
            triage_rule_version=self.PROTOCOL_VERSION
        )

# Global singleton
triage_engine = DeterministicTriageEngine()
