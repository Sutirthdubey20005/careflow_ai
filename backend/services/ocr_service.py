import io
import os
import logging
from typing import Optional, Dict, Any, Tuple
from PIL import Image

from backend.config import settings
from backend.schemas.multilingual import DocumentExtractionResult

logger = logging.getLogger("careflow.ocr")


class MedicalDocumentOCRService:
    """
    Extracts text from uploaded medical documents (prescription photos, lab images, PDFs).

    Architecture & Fallback Chain:
    1. Primary: Windows native OCR (winocr) for fast local CPU execution on Windows.
    2. Fallback: PaddleOCR for multilingual recognition (English + Hindi/Devanagari).
    3. Fallback: pytesseract / PDF embedded layer extraction.
    4. Safe degradation: returns structured unverified notice if image is unreadable.
    """

    def __init__(self):
        self._paddle_ocr = None

    def _get_paddle_ocr(self, lang: str = "en"):
        """Lazy loader for PaddleOCR to avoid startup delays or unnecessary memory usage."""
        if self._paddle_ocr is not None:
            return self._paddle_ocr
        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR engine...")
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            return self._paddle_ocr
        except Exception as e:
            logger.warning(f"PaddleOCR is unavailable or failed to initialize: {e}")
            return None

    def _extract_with_paddle(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Runs PaddleOCR on PIL image, returning concatenated text and average confidence."""
        engine = self._get_paddle_ocr()
        if engine is None:
            return None, 0.0

        try:
            import numpy as np
            img_arr = np.array(image)
            result = engine.ocr(img_arr, cls=True)

            if not result or not result[0]:
                return None, 0.0

            lines = []
            scores = []
            for line_info in result[0]:
                if len(line_info) >= 2 and len(line_info[1]) >= 2:
                    text_str, score = line_info[1][0], line_info[1][1]
                    if text_str and text_str.strip():
                        lines.append(text_str.strip())
                        scores.append(float(score))

            if not lines:
                return None, 0.0

            avg_score = round(sum(scores) / len(scores), 3) if scores else 0.8
            return "\n".join(lines), avg_score
        except Exception as e:
            logger.warning(f"PaddleOCR execution failed: {e}")
            return None, 0.0

    @staticmethod
    def extract_from_pdf(file_bytes: bytes) -> Tuple[str, str, float]:
        """Extracts text content from a PDF document using pypdf."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt and txt.strip():
                    pages_text.append(f"--- Page {idx+1} ---\n{txt.strip()}")

            if not pages_text:
                return "[Note: PDF contains scanned pages without embedded text layer.]", "pypdf", 0.5
            return "\n\n".join(pages_text), "pypdf", 0.95
        except Exception as e:
            logger.error(f"Error parsing PDF document: {e}")
            return f"[Error parsing PDF: {str(e)}]", "pypdf_error", 0.0

    def extract_from_image(self, file_bytes: bytes) -> Tuple[str, str, float]:
        """
        Extracts text from an image with multi-tier engine support:
        1. winocr (Windows Native)
        2. PaddleOCR (Multilingual fallback)
        3. pytesseract (Legacy fallback)
        """
        try:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            engine_choice = settings.OCR_ENGINE.lower()

            # Path 1: If PaddleOCR is explicitly requested, try it first
            if engine_choice == "paddleocr":
                paddle_text, paddle_conf = self._extract_with_paddle(image)
                if paddle_text:
                    return paddle_text, "paddleocr", paddle_conf

            # Path 2: Try native Windows OCR (winocr)
            try:
                import winocr
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    ocr_res = executor.submit(winocr.recognize_pil_sync, image).result(timeout=15)

                if isinstance(ocr_res, dict) and "text" in ocr_res and ocr_res["text"].strip():
                    return ocr_res["text"].strip(), "winocr", 0.92
                elif hasattr(ocr_res, "text") and ocr_res.text and ocr_res.text.strip():
                    return ocr_res.text.strip(), "winocr", 0.92
            except Exception as ocr_err:
                logger.warning(f"Native winocr recognition error: {ocr_err}")

            # Path 3: Fallback to PaddleOCR if winocr failed or was not selected
            if engine_choice != "winocr":
                paddle_text, paddle_conf = self._extract_with_paddle(image)
                if paddle_text:
                    return paddle_text, "paddleocr", paddle_conf

            # Path 4: Fallback to pytesseract if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(image).strip()
                if text:
                    return text, "pytesseract", 0.85
            except Exception:
                pass

            return (
                "[Image uploaded: text could not be extracted automatically. Please verify or manually type key findings.]",
                "none",
                0.0
            )

        except Exception as e:
            logger.error(f"Error processing image for OCR: {e}")
            return f"[Error reading image file: {str(e)}]", "error", 0.0

    def extract_with_metadata(self, file_bytes: bytes, filename: str) -> DocumentExtractionResult:
        """
        Extracts document text with complete metadata, engine provenance, and confidence score.
        """
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf":
            text, engine, conf = self.extract_from_pdf(file_bytes)
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            text, engine, conf = self.extract_from_image(file_bytes)
        elif ext in ["txt", "csv", "json"]:
            try:
                text = file_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                text = file_bytes.decode("latin-1", errors="replace").strip()
            engine = "plain_text"
            conf = 1.0
        else:
            text, engine, conf = self.extract_from_image(file_bytes)

        return DocumentExtractionResult(
            filename=filename,
            extracted_text=text,
            char_count=len(text),
            engine_used=engine,
            confidence=conf,
            source_type="ocr_extraction",
            is_mock=(settings.MEDGEMMA_MODE == "mock")
        )

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        Backward-compatible method returning plain extracted text string.
        """
        res = self.extract_with_metadata(file_bytes, filename)
        return res.extracted_text


ocr_service = MedicalDocumentOCRService()
