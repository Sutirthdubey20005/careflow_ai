# CareFlow: MedGemma-Powered Clinical Intake and Reviewer Dashboard

**CareFlow** is an end-to-end, text-only clinical intake and triage review assistant built for research and hackathons. It leverages **Google MedGemma** for conversational clinical extraction and grounded summarization, paired with a **deterministic, auditable rule-based triage engine** and a **clinician-in-the-loop review dashboard**.

> **⚠️ RESEARCH PROTOTYPE DISCLAIMER**
> CareFlow is an exploratory clinical decision support and intake assistant prototype. It is **NOT** a certified medical diagnostic device or clinically validated emergency triage system. It does not diagnose medical conditions or provide treatment instructions. All clinical summaries and priority classifications must be verified, edited, and approved by a qualified healthcare professional. In a life-threatening medical emergency, call 911 (or local emergency services) or visit the nearest emergency department immediately.

---

## 1. System Architecture & Component Separation

CareFlow strictly isolates generative AI capabilities from safety-critical triage decisions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Streamlit Frontend                              │
│   ├── Patient Interface (/patient): Conversational intake & review          │
│   ├── Reviewer Dashboard (/reviewer): Queue, citations & approval controls  │
│   └── Evaluation Suite (/evaluation): Live benchmark verification           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP Requests
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                              FastAPI Backend                                │
│   ├── /cases, /messages, /next-question, /submit, /review, /override        │
│   └── Database: SQLite (SQLAlchemy ORM with Foreign Key Enforcement)        │
└──────────────┬──────────────────────────────────────────────┬───────────────┘
               │                                              │
┌──────────────▼──────────────┐                ┌──────────────▼──────────────┐
│  Generative AI (MedGemma)   │                │ Deterministic Triage Engine │
│ ─────────────────────────── │                │ ─────────────────────────── │
│ • Untrusted prompt defense  │                │ • Separate from MedGemma    │
│ • Clinical fact extraction  │                │ • P0 (Emergency Escalation) │
│ • Adaptive single questions │                │ • P1 (Urgent Review)        │
│ • Grounded clinical summary │                │ • P2 (Uncertain/Conflict)   │
│ • Verbatim quote citations  │                │ • P3 (Routine Review)       │
│ • Single-retry JSON repair  │                │ • ERR (AI Failure Fallback) │
└─────────────────────────────┘                └─────────────────────────────┘
```

### Responsibility Breakdown

| Component | Technology | Responsibility |
| :--- | :--- | :--- |
| **Generative AI** | Google MedGemma 4B Instruction-Tuned (`google/medgemma-1.5-4b-it`) | • Structured clinical extraction<br/>• Adaptive single follow-up question generation<br/>• Factual, evidence-grounded clinical summaries<br/>• Verbatim patient quote citations |
| **Deterministic Triage** | Rule-based engine (`backend/services/triage_rules.py`) | • Transparent priority categorization (`P0`, `P1`, `P2`, `P3`, `ERR`)<br/>• Emergency criteria checked first; never downgraded<br/>• Fail-safe fallback: model errors trigger `ERR: Review required — AI unavailable`<br/>• Documented protocol version (`v1.0.0-prototype`) |
| **Human Reviewer** | Clinician Dashboard (`frontend/pages/reviewer.py`) | • Edit and approve final clinical summaries<br/>• Priority overrides (requires mandatory recorded justification)<br/>• Enter clinical referral notes & orders<br/>• Reopen cases if further patient clarification is required |

---

## 2. Model Execution Modes & Hardware Guidance

CareFlow provides a configurable model adapter (`backend/services/medgemma.py`) with three execution modes:

### A. `mock` Mode (Default for Development & CI)
- **Zero external dependencies, zero GPU required**.
- Deterministic, high-fidelity clinical parser extracting real symptoms, onset, severity, denials, and contradictions directly from patient text.
- Executes in `< 5 ms`, enabling instant testing on any standard laptop or automated CI pipeline without downloading multi-gigabyte weights.

### B. `hosted` Mode (Recommended for Hackathons / Cloud)
- Connects to an external endpoint serving `google/medgemma-1.5-4b-it` (e.g. Hugging Face Dedicated Inference Endpoint, vLLM, or Text Generation Inference).
- Reads `MEDGEMMA_ENDPOINT` and `MEDGEMMA_API_KEY` from environment variables.
- Zero local RAM overhead.

### C. `local` Mode (Local Hugging Face Transformers)
- Loads `google/medgemma-1.5-4b-it` locally via `transformers` and `torch`.
- **Memory Requirements:**
  - 16-bit Float: **~8 GB to 10 GB VRAM / RAM**.
  - 4-bit Quantization (`bitsandbytes`): **~3.5 GB to 4.5 GB RAM**.
  - **16 GB RAM Laptop with No Dedicated GPU:** If loading locally on CPU, 4-bit quantization or CPU offloading can be used (`MEDGEMMA_LOCAL_QUANTIZATION=4bit`), but inference latency on CPU may range from 15 to 45 seconds per response. For real-time testing, `mock` or `hosted` mode is strongly recommended.

---

## 3. Operational Triage Protocol

Triage categories are evaluated deterministically against explicit evidence:

| Priority | Label | Clinical Meaning & Triggers | Operational Protocol |
| :---: | :--- | :--- | :--- |
| **P0** | 🚨 Emergency Escalation | Chest pain radiating to arm/jaw, severe acute dyspnea, sudden stroke signs (FAST), severe hemorrhage, anaphylaxis. | Immediate emergency escalation / ED transfer. Checked first; cannot be downgraded. |
| **P1** | ⚠️ Urgent Review | High fever (>39°C) with stiff neck, severe uncontrolled pain (>= 8/10), acute abdomen with intractable vomiting. | Prompt clinician evaluation (within 1–2 hours). |
| **P2** | ❓ Incomplete / Uncertain | Contradictory statements detected across conversation turns, or high clinical ambiguity. | Clinician callback or targeted re-interview to clarify discrepancies. |
| **P3** | 📋 Routine Review | Stable, mild-to-moderate symptoms (e.g., uncomplicated upper respiratory symptoms) with no escalation triggers. | Standard intake queue for scheduled outpatient review. |
| **ERR** | ⚠️ Review Required — AI Unavailable | Model failure, schema parsing failure after retry, or missing model credentials. | Manual clinician triage; system never silently assigns a routine priority. |

---

## 4. Quick Start & Setup Guide

### Prerequisites
- Python 3.10+
- Virtual Environment tool (`venv`)

### 1. Clone & Set Up Virtual Environment

```bash
cd "c:\Users\DELL\Documents\Inter IIT\GR"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

CareFlow includes a `.env.example` template:

```bash
cp .env.example .env
```

Default configuration (`mock` mode):
```ini
MEDGEMMA_MODE=mock
MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it
MEDGEMMA_ENDPOINT=
MEDGEMMA_API_KEY=
DATABASE_URL=sqlite:///./careflow.db
MAX_FOLLOWUP_QUESTIONS=8
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_URL=http://127.0.0.1:8000
STREAMLIT_SERVER_PORT=8501
```

To switch to a hosted MedGemma endpoint:
```ini
MEDGEMMA_MODE=hosted
MEDGEMMA_ENDPOINT=https://your-endpoint.endpoints.huggingface.cloud
MEDGEMMA_API_KEY=your_hf_token_here
```

### 3. Launch the Backend API

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: `http://127.0.0.1:8000/health`
- Interactive OpenAPI / Swagger Docs: `http://127.0.0.1:8000/docs`

### 4. Launch the Streamlit Frontend

In a separate terminal (with `.venv` activated):

```bash
streamlit run frontend/app.py --server.port 8501
```
- Open in browser: `http://localhost:8501`

---

## 5. Running Automated Tests

CareFlow features a comprehensive automated `pytest` test suite covering **52 test cases across 10 test modules**:

```bash
pytest -v
```

### Verified Test Suites
1. **Safety Defect Regressions (`tests/test_safety_fixes.py`):**
   - Fabricated cardiac denials strictly prevented (chest pain complaints never fabricate prior history denials).
   - Interim P1 urgency persistence (interim high fever/severe pain cannot be erased by follow-up answers).
   - Document upload actor attribution (`ai_system`, never `reviewer`).
2. **Multilingual Processing (`tests/test_multilingual.py`):**
   - English, Devanagari Hindi, and Hinglish symptom extraction.
   - Cross-lingual duration, progression, and severity understanding without translation artifacts.
   - Cross-turn timeline contradiction detection across English and Hindi.
3. **Voice Audio Transcription (`tests/test_transcription.py`):**
   - Standalone `/transcribe-audio` endpoint verification.
   - Format validation, empty file rejection, size limit enforcement (25 MB max).
   - Secure temporary file cleanup with zero leaked disk files.
4. **Medical Report OCR (`tests/test_ocr_service.py`):**
   - Native `winocr` and multilingual `PaddleOCR` integration with metadata tracking.
   - Document extraction preview endpoint (`/cases/{id}/extract-document`) without premature case ingestion.
5. **Questioning Policy Benchmark (`tests/test_eval_policies.py`):**
   - Verification of Fixed, MedGemma-only, State-aware, and CareFlow-BQ policies under budgets 4, 6, 8.
   - Patient oracle simulation and redundancy detection.
6. **Core Operational Protocols (`tests/test_api.py`, `tests/test_database.py`, `tests/test_evidence_state.py`, `tests/test_extraction.py`, `tests/test_triage_rules.py`):
   - Deterministic P0 emergency precedence, P1 urgent triage, P2 conflict handling, P3 routine review.
   - Safe degradation to `ERR: Review required — AI unavailable` on model failure.

---

## 6. Running Research Benchmarks & Policy Evaluation

CareFlow provides two repeatable evaluation benchmarks:

### Suite A: Standardized Clinical Pipeline Fixtures (7 Tests)
```bash
python -m evaluation.run_eval
```
- Tests 7 standardized scenarios: Acute Coronary P0, Febrile Meningeal P1, Severe Flank Pain P1, Contradictory Duration P2, Unknown Dimension P3, Routine Cold P3, and AI Failure Fallback ERR.
- Logs traces to `evaluation_results/eval_<timestamp>.jsonl` and summary to `evaluation_results/eval_<timestamp>_summary.json`.

### Suite B: Multi-Policy Questioning Benchmark (Budgets 4, 6, 8)
```bash
python -m evaluation.eval_policies
```
- Empirically benchmarks 4 question-selection policies across question budgets (4, 6, 8):
  1. **Fixed:** Scripted clinical sequence.
  2. **MedGemma-only:** Zero-shot conversational generation from history.
  3. **State-aware:** Deterministic missing-slot fulfillment.
  4. **CareFlow-BQ:** Budgeted questioning prioritizing high-acuity life-threats.
- Evaluates: Extraction quality, critical question recall %, required info coverage %, redundancy rate %, missed escalation triggers, and turn latency.

Both benchmarks can also be executed and inspected visually via the Streamlit interface at **`http://localhost:8501/evaluation`**.

---

## 7. API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Reports backend status, database status, and MedGemma readiness |
| `POST` | `/cases` | Creates a new patient intake with a random non-identifying Case ID |
| `GET` | `/cases` | Lists cases sorted by priority (`P0` -> `P1` -> `P2` -> `P3` -> `ERR`) and wait time |
| `GET` | `/cases/{id}` | Fetches full case details, messages, evidence items, and audit logs |
| `POST` | `/cases/{id}/messages` | Appends patient message, updates evidence state, runs interim red-flag check (supports `source_type`, `transcription_confidence`, `is_confirmed`) |
| `POST` | `/cases/{id}/next-question` | Generates single next clarifying question via MedGemma (max 8 turns) |
| `POST` | `/cases/{id}/upload-document` | Uploads and extracts text from medical image/PDF directly into case |
| `POST` | `/transcribe-audio` | Transcribes audio without auto-submitting; returns transcript, language, confidence, and model size |
| `POST` | `/cases/{id}/extract-document` | Previews OCR text extraction without committing to case transcript |
| `POST` | `/cases/{id}/submit` | Finalizes intake, produces clinical summary, applies deterministic triage rules |
| `POST` | `/cases/{id}/generate-summary` | Regenerates clinical summary draft |
| `POST` | `/cases/{id}/review` | Saves reviewer edits, clinical notes, and approval status |
| `POST` | `/cases/{id}/override-priority` | Overrides priority with mandatory reason (audit logged) |
| `POST` | `/cases/{id}/reopen` | Reopens an approved case for further clarification |

---

## 8. Database Schema Overview

CareFlow uses SQLite with SQLAlchemy models and foreign key enforcement:

- **`cases`**: `id` (PK, e.g. `CF-2026-XXXX`), `created_at`, `updated_at`, `preferred_language`, `chief_complaint`, `evidence_state_json`, `ai_summary_json`, `ai_status`, `priority_category`, `priority_reasons_json`, `missing_information_json`, `uncertainties_json`, `triage_rule_version`, `review_status`, `reviewer_summary`, `reviewer_priority`, `reviewer_notes`, `reviewed_at`.
- **`messages`**: `id` (PK), `case_id` (FK), `role` (`patient` / `assistant`), `content`, `source_type` (`patient_typed`, `voice`, `ocr`), `source_language`, `transcription_confidence`, `is_confirmed`, `created_at`.
- **`evidence`**: `id` (PK), `case_id` (FK), `field`, `value_json`, `status` (`reported`, `unknown`, `conflicting`, `uncertain`), `source_message_id` (FK), `source_quote`, `source_type`, `confidence`, `created_at`.
- **`audit_events`**: `id` (PK), `case_id` (FK), `event_type`, `old_value_json`, `new_value_json`, `reason`, `actor_role` (`patient`, `ai_system`, `triage_engine`, `reviewer`), `created_at`.

---

## 9. Local Windows 11 Run Guide

For complete, copy-paste Windows 11 setup commands (including PowerShell execution policies, D: drive Hugging Face cache redirection, and local audio setup), see [LOCAL_WINDOWS_TEST.md](LOCAL_WINDOWS_TEST.md).
