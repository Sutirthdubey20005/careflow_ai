"""
CareFlow Multi-Policy Questioning Benchmark.

Compares 4 question-selection policies across question budgets (4, 6, 8):
1. Fixed: scripted clinical baseline (onset -> severity -> progression -> associated -> red flags -> negatives)
2. MedGemma-only: conversational zero-shot questioning without explicit evidence state tracking
3. State-aware heuristic: deterministic slot-filling based on missing evidence items
4. CareFlow-BQ: budgeted questioning combining evidence-gap prioritization, clinical red-flag discovery, and MedGemma conversational formulation

Metrics computed:
- extraction_quality: F1/recall of true patient clinical facts extracted
- critical_question_recall: proportion of high-risk red-flag questions probed
- required_info_coverage: proportion of essential clinical slots populated
- average_questions: number of questions asked before completion
- redundancy: count of redundant questions asking already answered slots
- missed_escalation_triggers: count of emergency/urgent cases where critical red flag was unprobed
- inappropriate_stops: count of premature terminations while high-risk gaps remained
- latency_ms: average latency per decision turn

Clearly labeled as research benchmark fixtures, NOT real clinical patient records.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from backend.services.medgemma import medgemma_service
from backend.services.extraction import clinical_extraction_service
from backend.services.questioning import adaptive_questioning_service
from backend.services.triage_rules import triage_engine

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Standardized benchmark patient profiles (Synthetic research fixtures)
BENCHMARK_PROFILES = [
    {
        "id": "POLICY-SCENARIO-01-CARDIAC",
        "title": "Acute Retrosternal Chest Discomfort (High Risk P0)",
        "chief_complaint": "I have severe tight chest pain that started 45 minutes ago.",
        "true_priority": "P0",
        "true_symptoms": ["chest pain", "shortness of breath", "dizziness"],
        "true_onset": "45 minutes ago",
        "true_severity": "9/10 severe crushing pressure",
        "true_progression": "worsening rapidly",
        "true_radiation": "radiates down left arm and into jaw",
        "true_negatives": ["no prior heart attacks", "no fever"],
        "critical_slots": ["radiation", "severity", "onset"],
        "essential_slots": ["onset", "severity", "progression", "radiation", "negatives"],
        "responses": {
            "onset": "It started suddenly about 45 minutes ago while sitting at my desk.",
            "severity": "The pain is an 8 or 9 out of 10, feels like a heavy weight pressing on my chest.",
            "progression": "It has been getting steadily worse since it started.",
            "radiation": "Yes, the tightness radiates down my left arm and up toward my jaw.",
            "associated": "I feel quite dizzy, nauseous, and short of breath.",
            "negatives": "I do not have a fever and have never had a heart attack before.",
            "treatments": "I tried resting and drinking water, but it didn't help at all.",
            "general": "I feel very unwell, weak, and cold sweat."
        }
    },
    {
        "id": "POLICY-SCENARIO-02-MENINGEAL",
        "title": "Febrile Headache with Nuchal Rigidity (Urgent P1)",
        "chief_complaint": "Severe throbbing headache with high fever for the past 2 days.",
        "true_priority": "P1",
        "true_symptoms": ["headache", "fever", "neck stiffness", "photophobia"],
        "true_onset": "2 days ago",
        "true_severity": "8/10 intense throbbing",
        "true_progression": "getting worse today",
        "true_radiation": "radiates down neck into upper spine",
        "true_negatives": ["no head injury", "no cough"],
        "critical_slots": ["radiation", "severity"],
        "essential_slots": ["onset", "severity", "progression", "radiation", "negatives"],
        "responses": {
            "onset": "The headache and fever started 2 days ago.",
            "severity": "It is an 8 out of 10, very painful especially when exposed to bright light.",
            "progression": "My neck became stiff today, making it worse than yesterday.",
            "radiation": "My neck is very stiff and the pain shoots down my neck when I bend forward.",
            "associated": "Bright lights hurt my eyes and I feel mild nausea.",
            "negatives": "I have not hit my head, and I have no cough or runny nose.",
            "treatments": "I took acetaminophen 500mg, but the fever and headache persisted.",
            "general": "I just want to lie down in a dark quiet room."
        }
    },
    {
        "id": "POLICY-SCENARIO-03-ABDOMINAL",
        "title": "Right Lower Quadrant Acute Pain (Urgent P1)",
        "chief_complaint": "Sharp abdominal pain in my lower right side that started last night.",
        "true_priority": "P1",
        "true_symptoms": ["abdominal pain", "nausea", "low grade fever"],
        "true_onset": "12 hours ago",
        "true_severity": "7/10 sharp pain",
        "true_progression": "started around navel, now localized to right lower belly",
        "true_radiation": "localized right lower quadrant",
        "true_negatives": ["no diarrhea", "no urinary pain"],
        "critical_slots": ["severity", "progression"],
        "essential_slots": ["onset", "severity", "progression", "radiation", "negatives"],
        "responses": {
            "onset": "It started about 12 hours ago last night.",
            "severity": "It is about 7 out of 10, sharp and painful whenever I walk or cough.",
            "progression": "It started as a dull ache around my belly button and moved to my lower right side.",
            "radiation": "It doesn't radiate anywhere else, it stays in the lower right abdomen.",
            "associated": "I have nausea and had one episode of vomiting this morning.",
            "negatives": "No diarrhea, no burning when urinating, no blood in stool.",
            "treatments": "I have not taken any pain relievers yet because I didn't want to upset my stomach.",
            "general": "Walking makes the jarring pain worse."
        }
    },
    {
        "id": "POLICY-SCENARIO-04-RESPIRATORY",
        "title": "Mild Upper Respiratory Cold (Routine P3)",
        "chief_complaint": "Runny nose, sneezing, and slight tickle in my throat for 3 days.",
        "true_priority": "P3",
        "true_symptoms": ["runny nose", "sneezing", "sore throat"],
        "true_onset": "3 days ago",
        "true_severity": "2/10 mild discomfort",
        "true_progression": "stable, not getting worse",
        "true_radiation": "none",
        "true_negatives": ["no fever", "no shortness of breath", "no chest pain"],
        "critical_slots": ["negatives"],
        "essential_slots": ["onset", "severity", "progression", "radiation", "negatives"],
        "responses": {
            "onset": "It started 3 days ago after I was out in the cold rain.",
            "severity": "Very mild, maybe a 2 out of 10. Just annoying congestion.",
            "progression": "Pretty steady, not getting worse.",
            "radiation": "No pain spreading anywhere.",
            "associated": "Occasional dry cough and mild fatigue.",
            "negatives": "No fever, no difficulty breathing, no chest pain, no wheezing.",
            "treatments": "I have been drinking warm ginger tea and resting.",
            "general": "Mainly just tired and stuffy nose."
        }
    }
]


class BenchmarkPatientOracle:
    """Simulates realistic patient responses based on scenario ground-truth."""

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile
        self.answered_slots = set()

    def answer_question(self, question: str) -> Tuple[str, Optional[str], bool]:
        """
        Determines the clinical slot being asked, returns (answer_text, slot_key, is_redundant).
        """
        q_lower = question.lower()
        slot_key = None

        if any(w in q_lower for w in ["when", "how long", "start", "began", "onset", "duration", "hours", "days"]):
            slot_key = "onset"
        elif any(w in q_lower for w in ["scale", "1 to 10", "1-10", "severe", "severity", "how bad", "intensity"]):
            slot_key = "severity"
        elif any(w in q_lower for w in ["worse", "better", "improving", "progress", "progressing", "changed", "changing"]):
            slot_key = "progression"
        elif any(w in q_lower for w in ["spread", "radiat", "arm", "jaw", "back", "neck", "shoulder"]):
            slot_key = "radiation"
        elif any(w in q_lower for w in ["associated", "fever", "dizz", "nausea", "breath", "other symptom"]):
            slot_key = "associated"
        elif any(w in q_lower for w in ["not have", "deni", "negative", "prior", "history", "before", "definitely not"]):
            slot_key = "negatives"
        elif any(w in q_lower for w in ["medicat", "pill", "taken", "treatment", "relief"]):
            slot_key = "treatments"
        else:
            slot_key = "general"

        is_redundant = slot_key in self.answered_slots and slot_key != "general"
        self.answered_slots.add(slot_key)

        response_text = self.profile["responses"].get(slot_key, self.profile["responses"]["general"])
        if is_redundant:
            response_text = f"As I mentioned earlier, {response_text.lower()}"

        return response_text, slot_key, is_redundant


# ---------------------------------------------------------------------------
# Policy Implementations
# ---------------------------------------------------------------------------

class FixedPolicy:
    """
    Scripted clinical intake baseline.
    Asks fixed questions in predetermined clinical order regardless of state.
    """
    SCRIPTED_QUESTIONS = [
        ("onset", "When did these symptoms first begin, and how long have they been going on?"),
        ("severity", "On a scale from 1 to 10, how severe is your discomfort or pain?"),
        ("progression", "Are your symptoms getting progressively worse, staying about the same, or improving?"),
        ("radiation", "Does any pain or discomfort radiate or spread anywhere else, such as your arm, back, neck, or jaw?"),
        ("associated", "Are you experiencing any other symptoms like fever, shortness of breath, nausea, or dizziness?"),
        ("negatives", "Do you have any prior medical conditions, or are there specific symptoms you definitely do not have?"),
        ("treatments", "Have you taken any medications or remedies for this, and did they provide any relief?"),
        ("general", "Is there any other information about your health or symptoms you would like to share?")
    ]

    def select_question(self, turn_idx: int, budget: int, evidence_state: Dict[str, Any], history: List[str]) -> Optional[str]:
        if turn_idx >= budget or turn_idx >= len(self.SCRIPTED_QUESTIONS):
            return None
        return self.SCRIPTED_QUESTIONS[turn_idx][1]


class MedGemmaOnlyPolicy:
    """
    Zero-shot conversational policy using MedGemma directly without explicit evidence state tracking.
    """
    def select_question(self, turn_idx: int, budget: int, evidence_state: Dict[str, Any], history: List[str]) -> Optional[str]:
        if turn_idx >= budget:
            return None
        patient_msgs = [m.replace("Patient: ", "").strip() for m in history if m.startswith("Patient: ") or not m.startswith("CareFlow: ")]
        assistant_msgs = [m.replace("CareFlow: ", "").strip() for m in history if m.startswith("CareFlow: ")]
        result = adaptive_questioning_service.generate_next_question(
            patient_messages=patient_msgs,
            current_question_count=turn_idx,
            assistant_messages=assistant_msgs,
            evidence_state=None
        )
        if result.is_terminal or not result.next_question:
            return None
        return result.next_question


class StateAwareHeuristicPolicy:
    """
    Deterministic slot-filling policy that inspects missing slots in evidence state.
    """
    SLOT_PROMPTS = {
        "onset": "When did your symptoms first begin, and how long have you had them?",
        "severity": "On a scale of 1 to 10, how severe is your pain or discomfort?",
        "progression": "Have your symptoms been getting worse, staying steady, or improving?",
        "radiation": "Does your pain or discomfort radiate or spread to any other part of your body?",
        "associated": "Are you experiencing any associated symptoms such as fever, dizziness, or nausea?",
        "negatives": "Are there any symptoms you definitely do not have or prior health conditions?"
    }

    def select_question(self, turn_idx: int, budget: int, evidence_state: Dict[str, Any], history: List[str]) -> Optional[str]:
        if turn_idx >= budget:
            return None

        facts = evidence_state.get("facts", {})
        # Check essential slots
        for slot in ["onset", "severity", "progression", "radiation", "associated", "negatives"]:
            if slot not in facts:
                return self.SLOT_PROMPTS.get(slot)

        # If all filled, terminate early
        return None


class CareFlowBQPolicy:
    """
    CareFlow Budgeted Questioning (CareFlow-BQ).
    Balances evidence gaps, clinical triage priority, and remaining turn budget.
    Prioritizes high-acuity life-threat questions first when red flags are possible,
    and terminates safely when diagnostic triage certainty is reached.
    """
    def select_question(self, turn_idx: int, budget: int, evidence_state: Dict[str, Any], history: List[str]) -> Optional[str]:
        turns_left = budget - turn_idx
        if turns_left <= 0:
            return None

        facts = evidence_state.get("facts", {})
        chief = history[0] if history else ""
        chief_lower = chief.lower()

        # 1. High-risk red flag detection
        is_cardiac_suspect = any(w in chief_lower for w in ["chest", "heart", "pressure"])
        is_neuro_suspect = any(w in chief_lower for w in ["headache", "neck", "stiff", "confus"])

        # Prioritize red-flag exclusion if relevant
        if is_cardiac_suspect and "radiation" not in facts:
            return "Does the chest discomfort radiate to your left arm, shoulder, neck, back, or jaw, or cause shortness of breath?"
        if is_neuro_suspect and "radiation" not in facts:
            return "Do you have any neck stiffness, trouble bending your neck forward, or sensitivity to light?"

        # 2. Critical triage severity and onset
        if "severity" not in facts:
            return "On a scale from 1 to 10, how would you rate the severity of your symptoms right now?"
        if "onset" not in facts:
            return "When did this start, and did it come on suddenly or develop gradually?"

        # 3. Progression & Associated
        if "progression" not in facts and turns_left >= 2:
            return "Are your symptoms getting rapidly worse, staying the same, or getting better?"
        if "associated" not in facts and turns_left >= 2:
            return "Are you noticing any other symptoms like fever, sweating, dizziness, or nausea?"

        # 4. Negatives
        if "negatives" not in facts and turns_left >= 1:
            return "Do you have any known medical conditions, or are there key symptoms like fever or shortness of breath that you do NOT have?"

        # If high-priority triage info is satisfied and turns are low, safely stop
        if "severity" in facts and "onset" in facts and ("radiation" in facts or not (is_cardiac_suspect or is_neuro_suspect)):
            return None

        return None


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_policy_benchmark(
    budgets: List[int] = [4, 6, 8],
    output_prefix: str = "eval_policies"
) -> Dict[str, Any]:
    """
    Runs all 4 policies across the specified question budgets on all benchmark profiles.
    Computes standard metrics and saves JSONL and summary JSON.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{output_prefix}_{timestamp}"
    jsonl_path = RESULTS_DIR / f"{run_id}.jsonl"
    summary_path = RESULTS_DIR / f"{run_id}_summary.json"

    policies = {
        "fixed": FixedPolicy(),
        "medgemma_only": MedGemmaOnlyPolicy(),
        "state_aware": StateAwareHeuristicPolicy(),
        "careflow_bq": CareFlowBQPolicy()
    }

    all_turn_records = []
    case_summaries = []

    print("\n" + "="*65)
    print(f"  CareFlow Questioning Policy Benchmark (Budgets: {budgets})")
    print(f"  Run ID: {run_id}")
    print("="*65 + "\n")

    for budget in budgets:
        for p_name, policy in policies.items():
            for profile in BENCHMARK_PROFILES:
                oracle = BenchmarkPatientOracle(profile)
                history = [profile["chief_complaint"]]
                evidence_state = {"facts": {"chief_complaint": profile["chief_complaint"]}}
                
                turns_asked = 0
                redundant_turns = 0
                critical_slots_asked = set()
                latencies = []

                # Initial extraction from chief complaint
                extraction = clinical_extraction_service.extract_clinical_facts([profile["chief_complaint"]])
                for item in extraction.evidence_items:
                    evidence_state["facts"][item.field] = item.value

                # Multi-turn interaction loop
                for turn_idx in range(budget):
                    t0 = time.perf_counter()
                    q = policy.select_question(turn_idx, budget, evidence_state, history)
                    lat = (time.perf_counter() - t0) * 1000.0
                    latencies.append(lat)

                    if not q:
                        break  # Policy stopped early

                    turns_asked += 1
                    history.append(f"CareFlow: {q}")

                    # Oracle responds
                    ans, slot_key, is_redundant = oracle.answer_question(q)
                    history.append(f"Patient: {ans}")
                    if is_redundant:
                        redundant_turns += 1
                    if slot_key in profile["critical_slots"]:
                        critical_slots_asked.add(slot_key)

                    # Update evidence state
                    if slot_key:
                        evidence_state["facts"][slot_key] = ans

                    record = {
                        "run_id": run_id,
                        "budget": budget,
                        "policy": p_name,
                        "profile_id": profile["id"],
                        "turn": turn_idx + 1,
                        "question": q,
                        "response": ans,
                        "slot_key": slot_key,
                        "is_redundant": is_redundant,
                        "latency_ms": round(lat, 2)
                    }
                    all_turn_records.append(record)

                # Final evaluation metrics for this trial
                total_critical = len(profile["critical_slots"])
                crit_recall = len(critical_slots_asked) / total_critical if total_critical > 0 else 1.0

                filled_essential = sum(1 for s in profile["essential_slots"] if s in evidence_state["facts"])
                info_coverage = filled_essential / len(profile["essential_slots"])

                # Extraction quality: symptom recall
                extracted_facts_text = " ".join(str(v) for v in evidence_state["facts"].values()).lower()
                symptoms_found = sum(1 for sym in profile["true_symptoms"] if sym.lower() in extracted_facts_text)
                extraction_quality = symptoms_found / len(profile["true_symptoms"]) if profile["true_symptoms"] else 1.0

                # Missed escalation check: if true emergency P0/P1 and critical slots were missed
                missed_escalation = 1 if (profile["true_priority"] in ["P0", "P1"] and crit_recall < 0.6) else 0

                # Inappropriate stop: stopped early while critical slots remained unprobed
                inappropriate_stop = 1 if (turns_asked < budget and crit_recall < 1.0 and profile["true_priority"] in ["P0", "P1"]) else 0

                case_summary = {
                    "budget": budget,
                    "policy": p_name,
                    "profile_id": profile["id"],
                    "true_priority": profile["true_priority"],
                    "questions_asked": turns_asked,
                    "redundancy_count": redundant_turns,
                    "redundancy_rate": round(redundant_turns / max(turns_asked, 1), 3),
                    "critical_question_recall": round(crit_recall, 3),
                    "required_info_coverage": round(info_coverage, 3),
                    "extraction_quality": round(extraction_quality, 3),
                    "missed_escalation_triggers": missed_escalation,
                    "inappropriate_stops": inappropriate_stop,
                    "avg_turn_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2)
                }
                case_summaries.append(case_summary)

    # Save detailed JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_turn_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate summaries by (policy, budget)
    aggregated = {}
    for p_name in policies.keys():
        aggregated[p_name] = {}
        for b in budgets:
            matches = [c for c in case_summaries if c["policy"] == p_name and c["budget"] == b]
            n = len(matches)
            if n > 0:
                aggregated[p_name][f"budget_{b}"] = {
                    "avg_questions": round(sum(c["questions_asked"] for c in matches) / n, 2),
                    "avg_redundancy_rate": round(sum(c["redundancy_rate"] for c in matches) / n, 3),
                    "avg_critical_recall": round(sum(c["critical_question_recall"] for c in matches) / n, 3),
                    "avg_info_coverage": round(sum(c["required_info_coverage"] for c in matches) / n, 3),
                    "avg_extraction_quality": round(sum(c["extraction_quality"] for c in matches) / n, 3),
                    "total_missed_escalations": sum(c["missed_escalation_triggers"] for c in matches),
                    "total_inappropriate_stops": sum(c["inappropriate_stops"] for c in matches),
                    "avg_latency_ms": round(sum(c["avg_turn_latency_ms"] for c in matches) / n, 2),
                }

    final_report = {
        "run_id": run_id,
        "timestamp": timestamp,
        "budgets": budgets,
        "policies": list(policies.keys()),
        "total_trials": len(case_summaries),
        "total_turns": len(all_turn_records),
        "aggregated_results": aggregated,
        "case_summaries": case_summaries
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"Benchmark complete!")
    print(f"JSONL traces: {jsonl_path.name}")
    print(f"Summary JSON: {summary_path.name}\n")

    # Print clean comparison table
    print(f"{'Policy':<22} | {'Budget':<6} | {'Avg Qs':<7} | {'Redund%':<8} | {'CritRec':<8} | {'Coverage':<8} | {'MissEsc':<7}")
    print("-" * 78)
    for p_name in policies.keys():
        for b in budgets:
            stats = aggregated[p_name].get(f"budget_{b}", {})
            print(
                f"{p_name:<22} | {b:<6} | {stats.get('avg_questions', 0):<7.1f} | "
                f"{stats.get('avg_redundancy_rate', 0)*100:<7.1f}% | "
                f"{stats.get('avg_critical_recall', 0)*100:<7.1f}% | "
                f"{stats.get('avg_info_coverage', 0)*100:<7.1f}% | "
                f"{stats.get('total_missed_escalations', 0):<7}"
            )

    return final_report


if __name__ == "__main__":
    run_policy_benchmark()
