import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

class CareFlowAPIClient:
    """Client for CareFlow FastAPI backend endpoints."""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    def get_health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def create_case(self, preferred_language: str = "English", chief_complaint: Optional[str] = None) -> Dict[str, Any]:
        payload = {"preferred_language": preferred_language, "chief_complaint": chief_complaint}
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
        resp = requests.post(f"{self.base_url}/cases/{case_id}/messages", json=payload, timeout=20)
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
        resp = requests.post(f"{self.base_url}/transcribe-audio", files=files, params=params, timeout=45)
        if not resp.ok:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            raise RuntimeError(err_detail or f"HTTP {resp.status_code}")
        return resp.json()

    def text_to_speech(self, text: str, model: Optional[str] = "aura-asteria-en") -> bytes:
        payload = {"text": text, "model": model or "aura-asteria-en"}
        resp = requests.post(f"{self.base_url}/text-to-speech", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.content

    def preview_document_extraction(self, case_id: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        resp = requests.post(f"{self.base_url}/cases/{case_id}/extract-document", files=files, timeout=40)
        resp.raise_for_status()
        return resp.json()

    def get_next_question(self, case_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/next-question", timeout=20)
        resp.raise_for_status()
        return resp.json()

    def submit_case(self, case_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/submit", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def regenerate_summary(self, case_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/cases/{case_id}/generate-summary", timeout=30)
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
        reason: str,
        reviewer_name: str = "Reviewer"
    ) -> Dict[str, Any]:
        payload = {
            "new_priority": new_priority,
            "reason": reason,
            "reviewer_name": reviewer_name
        }
        resp = requests.post(f"{self.base_url}/cases/{case_id}/override-priority", json=payload, timeout=15)
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

api_client = CareFlowAPIClient()
