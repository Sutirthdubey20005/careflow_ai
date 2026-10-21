# CareFlow — Local Windows 11 Testing Guide

This guide provides step-by-step, copy-paste PowerShell instructions for running and verifying CareFlow on a Windows 11 laptop. It covers offline mock mode (instant, 0 GPU), local voice transcription via `faster-whisper`, report OCR via native `winocr` and `PaddleOCR`, and configurable Google MedGemma execution.

---

## 1. System Requirements & Directory Setup

- **Operating System:** Windows 11 (64-bit)
- **Python:** Python 3.10, 3.11, 3.12, or 3.13
- **Shell:** Windows PowerShell 5.1 or PowerShell 7+
- **Disk Space:**
  - Mock mode: **< 500 MB**
  - With faster-whisper (`base` / `tiny`): **~1.5 GB**
  - With local MedGemma 4B weights: **~10 GB to 15 GB** (Recommended on secondary `D:` drive if `C:` is constrained)

---

## 2. Setting Up the Environment on Windows

Open **PowerShell** in the CareFlow project folder:

```powershell
# Verify Python version (should be 3.10+)
python --version

# If PowerShell script execution is restricted, enable local scripts:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

# Create dedicated virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install documented core dependencies
pip install -r requirements.txt
```

---

## 3. Storage & Cache Configuration (D: Drive Support)

If your primary `C:` drive has limited storage, you can redirect Hugging Face model downloads, PyTorch caches, and CareFlow temporary files to another drive (e.g., `D:\`):

```powershell
# Create cache directories on D: drive
New-Item -ItemType Directory -Force -Path "D:\huggingface"
New-Item -ItemType Directory -Force -Path "D:\careflow_temp"

# Set environment variables for the current PowerShell session:
$env:HF_HOME = "D:\huggingface"
$env:CAREFLOW_TEMP_DIR = "D:\careflow_temp"
```

To make these permanent across sessions:
```powershell
[System.Environment]::SetEnvironmentVariable('HF_HOME', 'D:\huggingface', [System.EnvironmentVariableTarget]::User)
[System.Environment]::SetEnvironmentVariable('CAREFLOW_TEMP_DIR', 'D:\careflow_temp', [System.EnvironmentVariableTarget]::User)
```

---

## 4. Configuring Environment Variables (`.env`)

Create your `.env` configuration file from the template:

```powershell
Copy-Item .env.example .env
```

### Mode Options in `.env`:

#### Option A: `mock` Mode (Default — Zero GPU, 100% Offline)
```ini
MEDGEMMA_MODE=mock
MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it
ASR_MODEL_SIZE=base
ASR_DEVICE=cpu
OCR_ENGINE=auto
DATABASE_URL=sqlite:///./careflow.db
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_URL=http://127.0.0.1:8000
STREAMLIT_SERVER_PORT=8501
```

#### Option B: `hosted` Mode (Hugging Face Endpoint or Inference API)
```ini
MEDGEMMA_MODE=hosted
MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it
MEDGEMMA_ENDPOINT=https://your-endpoint.endpoints.huggingface.cloud
HF_TOKEN=hf_your_token_here
```

#### Option C: `local` Mode (Local Weights via Transformers)
```ini
MEDGEMMA_MODE=local
MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it
HF_TOKEN=hf_your_token_here
HF_HOME=D:\huggingface
MEDGEMMA_LOCAL_QUANTIZATION=4bit
```
> **Note on Gated Access:** `google/medgemma-1.5-4b-it` requires accepting terms on Hugging Face at https://huggingface.co/google/medgemma-1.5-4b-it. If terms are pending, the backend returns a clear HTTP 403 error message directing you to request access, without crashing.

---

## 5. Running Automated Verification Tests

Verify all 62 unit, integration, safety regression, and multimodal intake tests:

```powershell
# Run the complete test suite
pytest -v
```

Expected result:
```
====================== 62 passed in ~4s =======================
```

### Test Suite Breakdown:
- `tests/test_intake_500_and_voice.py`: HTTP 500 regression verification (`messages.source_type` dynamic SQLite migration), empty/whitespace message 400 validation, English/Hindi/Hinglish message ingestion, model extraction failure clean rollback, faster-whisper (`base` and `tiny`) transcription, PyAV compatibility, and complete end-to-end multimodal patient & reviewer journey.
- `tests/test_safety_fixes.py`: Fabricated denial prevention, interim P1 persistence, upload actor attribution.
- `tests/test_multilingual.py`: English, Devanagari Hindi, and Hinglish denials, durations, severities, and cross-lingual contradictions.
- `tests/test_transcription.py`: `/transcribe-audio` endpoint, format validation, size limit enforcement, temp file cleanup.
- `tests/test_ocr_service.py`: Document extraction preview, engine metadata, format safety, text decoding.
- `tests/test_eval_policies.py`: Fixed, MedGemma-only, State-aware, and CareFlow-BQ policies across budget constraints.
- `tests/test_api.py`, `tests/test_database.py`, `tests/test_evidence_state.py`, `tests/test_extraction.py`, `tests/test_triage_rules.py`: Full operational triage protocol and database integrity.

---

## 6. Running Research Benchmarks & Multi-Policy Evaluation

CareFlow includes two repeatable benchmark suites:

### Suite 1: Standardized Clinical Pipeline Fixtures (7 Tests)
```powershell
python -m evaluation.run_eval
```
- Tests Acute Coronary P0, Febrile Meningeal P1, Severe Pain P1, Contradiction P2, Unknown P3, Routine Cold P3, and AI Failure ERR.
- Logs detailed traces to `evaluation_results/eval_<timestamp>.jsonl`.

### Suite 2: Multi-Policy Questioning Benchmark (Budgets 4, 6, 8)
```powershell
python -m evaluation.eval_policies
```
- Empirically compares:
  1. **Fixed:** Scripted clinical baseline.
  2. **MedGemma-only:** Zero-shot conversational questioning.
  3. **State-aware:** Deterministic slot-filling.
  4. **CareFlow-BQ:** Budgeted questioning with red-flag prioritization.
- Computes: Average questions, redundancy rate %, critical question recall %, required information coverage %, and missed escalation triggers.
- Logs interaction traces and summary JSON in `evaluation_results/`.

---

## 7. Starting the Full Application

Open two separate PowerShell windows (both with `.venv` activated):

### Terminal 1: Backend API (FastAPI)
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Health Check:** Open `http://127.0.0.1:8000/health` in your browser.
- **Interactive API Documentation:** Open `http://127.0.0.1:8000/docs`.

### Terminal 2: Clinical Frontend (Streamlit)
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py --server.port 8501
```
- **Application Portal:** Open `http://localhost:8501`.

---

## 8. Manual Testing Walkthroughs

### Walkthrough A: Multilingual Text Intake (Hindi & Hinglish)
1. Navigate to **Patient Intake** (`http://localhost:8501/patient`).
2. Select **हिन्दी (Hindi)** from the language selector in the sidebar.
3. Enter chief complaint in Devanagari or Hinglish:
   - Example 1 (Hinglish): `"Mujhe 2 din se tez bukhar hai aur gale me dard hai."`
   - Example 2 (Devanagari): `"मुझे दो दिन से तेज़ बुखार और सिरदर्द है।"`
4. Observe follow-up questions generated in your preferred language.
5. Answer follow-up questions or use **मुझे नहीं पता ❓** (I Don't Know) / **छोड़ें ⏩** (Skip).
6. Verify that "I don't know" is safely recorded as unknown without fabricating denials.

### Walkthrough B: Voice Recording & Human-in-the-Loop Confirmation
1. In the Patient Intake conversational screen, open **🎤 Voice Input (Record or Upload Audio)**.
2. Either click the microphone button to record audio directly via the browser, or upload an audio file (`.wav`, `.mp3`, `.m4a`).
3. Click **Transcribe Audio 🎙️**.
4. Observe the **Voice Transcript Review Card**:
   - The transcript is displayed in an editable text area.
   - Acoustic confidence and detected language are explicitly shown.
5. Edit any misheard words and click **Confirm & Send Voice Response 💬** (or discard with **❌**).
6. Verify the message is added to conversation history with the `🎤 Voice Transcript` badge.

### Walkthrough C: Medical Report OCR & Clinician Provenance
1. In Patient Intake, expand **📎 Upload Additional Medical Document / Lab Report**.
2. Select an image (`.png`, `.jpg`) or `.pdf` of a prescription or lab report.
3. Click **Extract & Preview Document Text 🔍**.
4. Observe the **Document OCR Review Card**:
   - OCR extracted text is shown for patient verification.
   - Character count and OCR engine used (`winocr` / `PaddleOCR` / `plain_text`) are displayed.
5. Confirm ingestion.
6. Open the **Reviewer Dashboard** (`http://localhost:8501/reviewer`):
   - Inspect the case in the triage queue.
   - Open **🔬 Atomic Evidence Items & Modality Provenance** to view each extracted clinical fact with its origin modality (`🎤 Voice`, `📄 Medical Report OCR`, `💬 Typed`).

---

## 9. Troubleshooting & FAQ for Windows 11

### Q: `Activate.ps1` gives an Execution_Policies error in PowerShell
**Fix:** Run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
```

### Q: How to install optional local ASR (`faster-whisper`)
If you want to use the local Whisper engine rather than mock mode:
```powershell
pip install faster-whisper
```
CareFlow will automatically detect `faster-whisper` and load `base` or `tiny` as specified in `ASR_MODEL_SIZE`.

### Q: How to install optional `PaddleOCR` for Hindi Devanagari OCR
```powershell
pip install paddlepaddle paddleocr
```
CareFlow automatically uses `PaddleOCR` as a multilingual fallback when available, falling back to `winocr` or text extraction otherwise.

### Q: How to ensure model downloads don't fill drive C:
Set `$env:HF_HOME = "D:\huggingface"` in PowerShell before starting the backend or running scripts.
CareFlow automatically passes `HF_HOME` to Hugging Face Hub.

### Q: PyAV `metadata_errors` or OpenMP conflict on Windows 11
CareFlow's `TranscriptionService` includes an automated compatibility patch that resolves the PyAV >= 14 `metadata_errors` deprecation when running `faster-whisper`. Additionally, CPU thread isolation prevents runtime collisions between `faster-whisper` (CTranslate2) and `PaddleOCR` (PaddlePaddle).
