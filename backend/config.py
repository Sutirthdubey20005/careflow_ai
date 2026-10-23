import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for CareFlow
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

class Settings:
    BASE_DIR: Path = BASE_DIR
    # 3-Mode Model Router: 'base_ollama' (default base medgemma-4b-it), 'finetuned_ollama', or 'external_api'
    MEDGEMMA_BACKEND_MODE: str = os.getenv("MEDGEMMA_BACKEND_MODE", "base_ollama").lower()
    BASE_OLLAMA_MODEL: str = os.getenv("BASE_OLLAMA_MODEL", "medgemma-4b-it")

    # Universal OpenAI-Compatible External API (Groq, OpenRouter, Colab vLLM, Gemini)
    EXTERNAL_API_BASE_URL: str = os.getenv("EXTERNAL_API_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    EXTERNAL_API_KEY: str = os.getenv("EXTERNAL_API_KEY", "")
    EXTERNAL_API_MODEL: str = os.getenv("EXTERNAL_API_MODEL", "gemma2-9b-it")

    # Model execution mode: 'mock', 'hosted', or 'local' (or 'peft')
    MEDGEMMA_MODE: str = os.getenv("MEDGEMMA_MODE", "hosted").lower()
    ENABLE_MOCK_FALLBACK: bool = os.getenv("ENABLE_MOCK_FALLBACK", "true").lower() == "true"
    
    # Model Identifier & Fine-Tuned LoRA Adapter
    MEDGEMMA_MODEL_ID: str = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
    MEDGEMMA_BASE_MODEL_ID: str = os.getenv("MEDGEMMA_BASE_MODEL_ID", "unsloth/medgemma-4b-it-unsloth-bnb-4bit")
    MEDGEMMA_ADAPTER_PATH: str = os.getenv("MEDGEMMA_ADAPTER_PATH", str(BASE_DIR / "careflow-medgemma-final"))
    
    # Provider Abstraction Configuration:
    # 'standard_medgemma' (local base medgemma-4b-it / hosted API) vs 'finetuned_medgemma'
    DEFAULT_MEDGEMMA_PROVIDER: str = os.getenv("DEFAULT_MEDGEMMA_PROVIDER", "standard_medgemma")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "careflow-slm")
    STANDARD_MEDGEMMA_MODEL_ID: str = os.getenv("STANDARD_MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
    
    # Hosted Endpoint & API Key (supports MEDGEMMA_API_KEY or standard HF_TOKEN)
    MEDGEMMA_ENDPOINT: str = os.getenv("MEDGEMMA_ENDPOINT", "")
    MEDGEMMA_API_KEY: str = os.getenv("MEDGEMMA_API_KEY", os.getenv("HF_TOKEN", ""))
    
    # Local Quantization: '4bit', '8bit', or 'none'
    MEDGEMMA_LOCAL_QUANTIZATION: str = os.getenv("MEDGEMMA_LOCAL_QUANTIZATION", "4bit")
    MEDGEMMA_LOCAL_GGUF_PATH: str = os.getenv("MEDGEMMA_LOCAL_GGUF_PATH", str(BASE_DIR / "models" / "medgemma-4b-it-Q4_K_M.gguf"))
    
    # Model timeouts and retries
    MODEL_TIMEOUT_SECONDS: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "300"))
    MODEL_MAX_RETRIES: int = int(os.getenv("MODEL_MAX_RETRIES", "1"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))

    # Deepgram Configuration (Hinglish & Accent Optimized STT, Translation, TTS)
    USE_DEEPGRAM: bool = os.getenv("USE_DEEPGRAM", "true").lower() == "true"
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    DEEPGRAM_MODEL: str = os.getenv("DEEPGRAM_MODEL", "nova-2")
    DEEPGRAM_LANGUAGE: str = os.getenv("DEEPGRAM_LANGUAGE", "en-IN")

    # Legacy ASR / Whisper Voice Transcription Configuration (Preserved under feature flag)
    ASR_MODE: str = os.getenv("ASR_MODE", "auto").lower()              # 'auto', 'local', 'mock', 'hosted'
    ASR_MODEL_SIZE: str = os.getenv("ASR_MODEL_SIZE", "base").lower()  # 'base', 'tiny', 'small'
    ASR_DEVICE: str = os.getenv("ASR_DEVICE", "cpu").lower()            # 'cpu' or 'cuda'
    WHISPER_ENDPOINT: str = os.getenv("WHISPER_ENDPOINT", "https://api.groq.com/openai/v1/audio/transcriptions")
    WHISPER_API_KEY: str = os.getenv("WHISPER_API_KEY", os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-large-v3")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # OCR Engine Configuration: 'auto', 'winocr', 'paddleocr'
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "auto").lower()

    # Cache and Storage Directories (e.g. D:\huggingface on Windows)
    HF_HOME: str = os.getenv("HF_HOME", "")
    CAREFLOW_TEMP_DIR: str = os.getenv("CAREFLOW_TEMP_DIR", "")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'careflow.db'}")

    # Workflow Limits
    MAX_FOLLOWUP_QUESTIONS: int = int(os.getenv("MAX_FOLLOWUP_QUESTIONS", "8"))
    
    # Server network
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    
    # Triage protocol version
    TRIAGE_RULE_VERSION: str = "v1.0.0-prototype"

    @classmethod
    def get_masked_key(cls, key: str) -> str:
        """Returns safe masked representation of an API key for logs/diagnostics."""
        if not key:
            return "none"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

# Set HF cache environment variable if configured
if os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = os.getenv("HF_HOME")

settings = Settings()
