import os
import sys
import time
import threading
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
CLIENT_MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT_SECONDS", "300"))

_backend_started = False
_backend_lock = threading.Lock()

def _ensure_backend_running(backend_url: str):
    """
    Ensures the FastAPI backend is running.
    On Streamlit Community Cloud (where only Streamlit executes), this automatically
    spawns uvicorn in the background inside the container so everything works
    out-of-the-box without requiring a separate server.
    """
    global _backend_started
    if _backend_started:
        return
    if "127.0.0.1" not in backend_url and "localhost" not in backend_url:
        return

    with _backend_lock:
        if _backend_started:
            return
        try:
            r = requests.get(f"{backend_url}/health", timeout=1)
            if r.status_code == 200:
                _backend_started = True
                return
        except Exception:
            pass

        # Primary method: Spawn uvicorn as an independent background subprocess
        spawned = False
        try:
            root_dir = str(Path(__file__).resolve().parent.parent)
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
                cwd=root_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            spawned = True
        except Exception as e:
            print(f"[CareFlow] Subprocess uvicorn launch error: {e}")

        # Secondary fallback: In-thread uvicorn Server with signal handlers disabled
        if not spawned:
            def _run_server():
                try:
                    import uvicorn
                    from backend.main import app as fastapi_app
                    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
                    server = uvicorn.Server(config)
                    server.install_signal_handlers = lambda: None
                    server.run()
                except Exception as ex:
                    print(f"[CareFlow] In-thread uvicorn notice: {ex}")

            server_thread = threading.Thread(target=_run_server, daemon=True)
            server_thread.start()

        # Spin-wait up to 8 seconds for backend to become healthy
        for _ in range(16):
            time.sleep(0.5)
            try:
                if requests.get(f"{backend_url}/health", timeout=1).status_code == 200:
                    _backend_started = True
                    break
            except Exception:
                pass

class CareFlowAPIClient:
    """Client for CareFlow FastAPI backend endpoints."""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url
        _ensure_backend_running(self.base_url)

    def get_health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def create_case(
        self,
        preferred_language: str = "English",
        chief_complaint: Optional[str] = None,
        selected_model_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = {"preferred_language": preferred_language, "chief_complaint": chief_complaint}
        if selected_model_provider:
            payload["selected_model_provider"] = selected_model_provider
        resp = requests.post(f"{self.base_url}/cases", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def list_cases(
        self,
        priority: Optional[str] = None,
        review_status: Optional[str] = None,
        has_uncertainty: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        params = {}
        if priority and priority != "All":
            params["priority"] = priority
        if review_status and review_status != "All":
            params["review_status"] = review_status
        if has_uncertainty is not None:
            params["has_uncertainty"] = has_uncertainty

        resp = requests.get(f"{self.base_url}/cases", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_case(self, case_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/cases/{case_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def send_message(
        self,
        case_id: str,
        content: str,
        raw_transcript: Optional[str] = None,
        source_type: str = "patient_typed",
        source_language: Optional[str] = None,
        confidence: Optional[float] = None,
        transcription_confidence: Optional[float] = None,
        is_confirmed: bool = True
    ) -> Dict[str, Any]:
        conf = transcription_confidence if transcription_confidence is not None else confidence
        payload = {
            "content": content,
            "raw_transcript": raw_transcript,
            "source_type": source_type,
            "source_language": source_language,
            "transcription_confidence": conf,
            "is_confirmed": is_confirmed
        }
        resp = requests.post(f"{self.base_url}/cases/{case_id}/messages", json=payload, timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def transcribe_audio(
        self,
        file_bytes: bytes,
        filename: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        translate_to_english: bool = False
    ) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        params = {}
        if language:
            params["language"] = language
        if model_size:
            params["model_size"] = model_size
        if translate_to_english:
            params["translate_to_english"] = "true"
        resp = requests.post(f"{self.base_url}/transcribe-audio", files=files, params=params, timeout=CLIENT_MODEL_TIMEOUT)
        if not resp.ok:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            raise RuntimeError(err_detail or f"HTTP {resp.status_code}")
        return resp.json()

    def text_to_speech(self, text: str, model: Optional[str] = "aura-asteria-en") -> bytes:
        payload = {"text": text, "model": model or "aura-asteria-en"}
        resp = requests.post(f"{self.base_url}/text-to-speech", json=payload, timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    def preview_document_extraction(self, case_id: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        resp = requests.post(f"{self.base_url}/cases/{case_id}/extract-document", files=files, timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def get_next_question(self, case_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/next-question", timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def submit_case(self, case_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/submit", timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def regenerate_summary(self, case_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/generate-summary", timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def review_case(
        self,
        case_id: str,
        reviewer_summary: Optional[str] = None,
        reviewer_priority: Optional[str] = None,
        reviewer_notes: Optional[str] = None,
        approve: bool = False
    ) -> Dict[str, Any]:
        payload = {
            "reviewer_summary": reviewer_summary,
            "reviewer_priority": reviewer_priority,
            "reviewer_notes": reviewer_notes,
            "approve": approve
        }
        resp = requests.post(f"{self.base_url}/cases/{case_id}/review", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def override_priority(
        self,
        case_id: str,
        new_priority: str,
        reason: Optional[str] = None,
        reviewer_name: str = "Reviewer"
    ) -> Dict[str, Any]:
        default_reason = f"Clinician adjusted operational priority to {new_priority} based on clinical review."
        payload = {
            "new_priority": new_priority,
            "reason": (reason.strip() if reason and len(reason.strip()) >= 5 else default_reason),
            "reviewer_name": reviewer_name or "Reviewer"
        }
        resp = requests.post(f"{self.base_url}/cases/{case_id}/override-priority", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def update_case_status(self, case_id: str, status: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/status", json={"status": status}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def mark_case_done(self, case_id: str, notes: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if notes:
            params["notes"] = notes
        resp = requests.post(f"{self.base_url}/cases/{case_id}/mark-done", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def delete_case(self, case_id: str) -> Dict[str, Any]:
        resp = requests.delete(f"{self.base_url}/cases/{case_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def reopen_case(self, case_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/reopen", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def upload_document(self, case_id: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        resp = requests.post(f"{self.base_url}/cases/{case_id}/upload-document", files=files, timeout=40)
        resp.raise_for_status()
        return resp.json()

    def get_model_providers(self) -> Dict[str, Any]:
        """Fetches availability status of Standard MedGemma vs Fine-Tuned MedGemma."""
        resp = requests.get(f"{self.base_url}/model-providers", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def set_case_model_provider(self, case_id: str, provider: str) -> Dict[str, Any]:
        """Switches active MedGemma provider for the given case."""
        payload = {"provider": provider}
        resp = requests.post(f"{self.base_url}/cases/{case_id}/model-provider", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def compare_models(
        self,
        conversation_history: List[str],
        vitals: Optional[Dict[str, Any]] = None,
        preferred_language: str = "en"
    ) -> Dict[str, Any]:
        """Runs identical patient input through both standard and fine-tuned MedGemma side-by-side."""
        payload = {
            "conversation_history": conversation_history,
            "vitals": vitals or {},
            "preferred_language": preferred_language
        }
        resp = requests.post(f"{self.base_url}/model-comparison", json=payload, timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def get_ai_models_status(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/ai-models/status", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def run_live_pipeline(
        self,
        text: str,
        vitals: Optional[Dict[str, Any]] = None,
        preferred_language: str = "English"
    ) -> Dict[str, Any]:
        payload = {
            "text": text,
            "vitals": vitals or {},
            "preferred_language": preferred_language
        }
        resp = requests.post(f"{self.base_url}/ai-pipeline/run", json=payload, timeout=CLIENT_MODEL_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def send_follow_up_question(
        self,
        case_id: str,
        question_text: str,
        reviewer_name: str = "Clinician Reviewer"
    ) -> Dict[str, Any]:
        """Dispatches an edit-ready clinician follow-up question to the patient case."""
        payload = {"question_text": question_text, "reviewer_name": reviewer_name}
        try:
            resp = requests.post(f"{self.base_url}/cases/{case_id}/send-follow-up", json=payload, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return self.send_message(case_id, content=question_text, source_type="clinician_follow_up")

api_client = CareFlowAPIClient()


