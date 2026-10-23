import io
import os
import re
import logging
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from backend.config import settings
from backend.schemas.multilingual import DocumentExtractionResult, StructuredMedicalFact
from backend.services.safety import safety_service

logger = logging.getLogger("careflow.ocr")


class MedicalDocumentOCRService:
    """
    Extracts text from uploaded medical documents (prescription photos, lab images, PDFs).
    Performs medical document classification, structured clinical fact extraction,
    and automated PII redaction.
    """

    def __init__(self):
        self._paddle_ocr = None

    @classmethod
    def detect_medical_document(cls, text: str) -> Tuple[bool, float, str]:
        """
        Classifies whether extracted text represents a clinical or laboratory document.
        Returns: (is_medical, confidence_score, detected_category)
        """
        if not text or len(text.strip()) < 10:
            return False, 0.0, "non_medical_or_empty"

        t_low = text.lower()
        score = 0
        doc_type = "general_medical_report"

        lab_markers = [
            "hemoglobin", "hb", "wbc", "rbc", "platelet", "glucose", "creatinine",
            "urea", "bilirubin", "serum", "lipid", "cholesterol", "pathology", "cbc",
            "investigation", "reference range", "normal range", "specimen", "fasting"
        ]
        vital_markers = ["heart rate", "pulse", "bp", "blood pressure", "spo2", "temp", "temperature", "systolic", "diastolic", "bpm", "mmhg"]
        rx_markers = ["rx", "prescription", "capsule", "tablet", "syrup", "dosage", "sig:", "mg", "dr.", "doctor", "clinic", "hospital", "dispense"]

        lab_matches = sum(1 for m in lab_markers if m in t_low)
        vital_matches = sum(1 for m in vital_markers if m in t_low)
        rx_matches = sum(1 for m in rx_markers if m in t_low)

        score = lab_matches + vital_matches + rx_matches

        if lab_matches >= 2:
            doc_type = "laboratory_report"
        elif rx_matches >= 2:
            doc_type = "prescription_or_clinical_notes"
        elif vital_matches >= 2:
            doc_type = "vitals_monitoring_sheet"

        is_medical = score >= 2
        confidence = min(round(0.4 + (score * 0.1), 2), 0.98) if is_medical else 0.2
        return is_medical, confidence, doc_type

    @classmethod
    def extract_structured_medical_facts(cls, text: str) -> Tuple[List[StructuredMedicalFact], Dict[str, Any]]:
        """
        Extracts validated clinical parameters, units, reference ranges, report dates,
        and continuous vitals for the LightGBM urgency classifier.
        Flags uncertain or implausible values for human review.
        """
        facts: List[StructuredMedicalFact] = []
        vitals: Dict[str, Any] = {}

        if not text:
            return facts, vitals

        # 1. Report Date Extraction
        report_date = None
        date_match = re.search(r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b", text)
        if date_match:
            report_date = date_match.group(1)

        # 2. Blood Pressure (Systolic / Diastolic)
        bp_match = re.search(r"\b(?:blood\s*pressure|bp|systolic/diastolic)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmHg)?\b", text, re.IGNORECASE)
        if not bp_match:
            bp_match = re.search(r"\b(?:systolic|sbp)\s*[:=]?\s*(\d{2,3})\s*(?:mmHg)?\b", text, re.IGNORECASE)
        if bp_match:
            try:
                sys_val = float(bp_match.group(1))
                is_unreliable = sys_val < 50 or sys_val > 260
                facts.append(StructuredMedicalFact(
                    measurement_name="Systolic Blood Pressure",
                    numeric_value=sys_val,
                    raw_value=bp_match.group(0),
                    unit="mmHg",
                    reference_range="90 - 120 mmHg",
                    report_date=report_date,
                    confidence=0.95,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Physiologically implausible systolic BP" if is_unreliable else None
                ))
                vitals["systolic_bp"] = int(sys_val)
            except Exception:
                pass

        # 3. Heart Rate / Pulse
        hr_match = re.search(r"\b(?:heart\s*rate|pulse|hr)\s*[:=]?\s*(\d{2,3})\s*(?:bpm|beats/min)?\b", text, re.IGNORECASE)
        if hr_match:
            try:
                hr_val = float(hr_match.group(1))
                is_unreliable = hr_val < 35 or hr_val > 230
                facts.append(StructuredMedicalFact(
                    measurement_name="Heart Rate",
                    numeric_value=hr_val,
                    raw_value=hr_match.group(0),
                    unit="bpm",
                    reference_range="60 - 100 bpm",
                    report_date=report_date,
                    confidence=0.95,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Physiologically implausible heart rate" if is_unreliable else None
                ))
                vitals["heart_rate"] = int(hr_val)
            except Exception:
                pass

        # 4. Oxygen Saturation (SpO2)
        spo2_match = re.search(r"\b(?:spo2|oxygen\s*sat|o2\s*sat)\s*[:=]?\s*(\d{2,3})\s*%?\b", text, re.IGNORECASE)
        if spo2_match:
            try:
                spo2_val = float(spo2_match.group(1))
                is_unreliable = spo2_val < 40 or spo2_val > 100
                facts.append(StructuredMedicalFact(
                    measurement_name="Oxygen Saturation (SpO2)",
                    numeric_value=spo2_val,
                    raw_value=spo2_match.group(0),
                    unit="%",
                    reference_range="95 - 100%",
                    report_date=report_date,
                    confidence=0.95,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Implausible SpO2 percentage" if is_unreliable else None
                ))
                vitals["spo2_pct"] = int(spo2_val)
            except Exception:
                pass

        # 5. Temperature
        temp_match = re.search(r"\b(?:temp|temperature)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:F|deg F|C)?\b", text, re.IGNORECASE)
        if temp_match:
            try:
                t_val = float(temp_match.group(1))
                # Convert Celsius if in 35-43 range
                if 34.0 <= t_val <= 44.0:
                    t_val = round((t_val * 9.0 / 5.0) + 32.0, 1)
                is_unreliable = t_val < 85.0 or t_val > 108.0
                facts.append(StructuredMedicalFact(
                    measurement_name="Body Temperature",
                    numeric_value=t_val,
                    raw_value=temp_match.group(0),
                    unit="°F",
                    reference_range="97.0 - 99.0 °F",
                    report_date=report_date,
                    confidence=0.92,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Implausible body temperature" if is_unreliable else None
                ))
                vitals["temp_f"] = t_val
            except Exception:
                pass

        # 6. Hemoglobin (Hb)
        hb_match = re.search(r"\b(?:hemoglobin|hb)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:g/dL|gm/dl)?\b", text, re.IGNORECASE)
        if hb_match:
            try:
                hb_val = float(hb_match.group(1))
                facts.append(StructuredMedicalFact(
                    measurement_name="Hemoglobin",
                    numeric_value=hb_val,
                    raw_value=hb_match.group(0),
                    unit="g/dL",
                    reference_range="12.0 - 16.0 g/dL",
                    report_date=report_date,
                    confidence=0.94,
                    is_uncertain=hb_val < 3.0 or hb_val > 25.0
                ))
            except Exception:
                pass

        # 7. Blood Glucose / Sugar
        glu_match = re.search(r"\b(?:glucose|blood\s*sugar|rbs|fbs)\s*[:=]?\s*(\d{2,3})\s*(?:mg/dL)?\b", text, re.IGNORECASE)
        if glu_match:
            try:
                glu_val = float(glu_match.group(1))
                facts.append(StructuredMedicalFact(
                    measurement_name="Blood Glucose",
                    numeric_value=glu_val,
                    raw_value=glu_match.group(0),
                    unit="mg/dL",
                    reference_range="70 - 110 mg/dL",
                    report_date=report_date,
                    confidence=0.93,
                    is_uncertain=glu_val < 20 or glu_val > 800
                ))
            except Exception:
                pass

        return facts, vitals

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

    def _get_rapid_ocr(self):
        """Lazy loader for RapidOCR (PP-OCRv4 ONNX)."""
        if getattr(self, "_rapid_ocr", None) is not None:
            return self._rapid_ocr
        try:
            from rapidocr_onnxruntime import RapidOCR
            logger.info("Initializing RapidOCR (PP-OCRv4 ONNX) engine...")
            self._rapid_ocr = RapidOCR()
            return self._rapid_ocr
        except Exception as e:
            logger.warning(f"RapidOCR engine initialization failed: {e}")
            return None

    def _extract_with_rapidocr(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Runs RapidOCR on PIL image, returning structured lines and confidence."""
        engine = self._get_rapid_ocr()
        if engine is None:
            return None, 0.0

        try:
            import numpy as np
            img_arr = np.array(image)
            result, _ = engine(img_arr)
            if not result:
                return None, 0.0

            lines = []
            scores = []
            for item in result:
                if len(item) >= 3:
                    text_str = str(item[1]).strip()
                    conf = float(item[2])
                    if text_str:
                        lines.append(text_str)
                        scores.append(conf)

            if not lines:
                return None, 0.0

            avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.92
            return "\n".join(lines), avg_conf
        except Exception as e:
            logger.warning(f"RapidOCR execution failed: {e}")
            return None, 0.0

    def extract_from_pdf(self, file_bytes: bytes) -> Tuple[str, str, float]:
        """Extracts text content from a PDF document using pypdf, with image OCR fallback for scanned PDFs."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            scanned_pages_text = []

            for idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt and txt.strip():
                    pages_text.append(f"--- Page {idx+1} ---\n{txt.strip()}")
                else:
                    # Attempt OCR on embedded scanned page images
                    try:
                        for img_idx, img_file in enumerate(page.images):
                            img_txt, _, _ = self.extract_from_image(img_file.data)
                            if img_txt and not img_txt.startswith("["):
                                scanned_pages_text.append(f"--- Page {idx+1} (Scanned Image {img_idx+1}) ---\n{img_txt.strip()}")
                    except Exception as scan_err:
                        logger.warning(f"Failed to extract images from PDF page {idx+1}: {scan_err}")

            if pages_text:
                return "\n\n".join(pages_text), "pypdf", 0.95
            elif scanned_pages_text:
                return "\n\n".join(scanned_pages_text), "rapidocr_pdf_scan", 0.90
            return "[Note: PDF contains scanned pages without readable text layer.]", "pypdf", 0.5
        except Exception as e:
            logger.error(f"Error parsing PDF document: {e}")
            return f"[Error parsing PDF: {str(e)}]", "pypdf_error", 0.0

    def extract_from_image(self, file_bytes: bytes) -> Tuple[str, str, float]:
        """
        Extracts text from an image with multi-tier engine support:
        1. RapidOCR (PP-OCRv4 ONNX - High Accuracy Table & Lab Parser)
        2. winocr (Windows Native Fallback)
        3. PaddleOCR (Multilingual Fallback)
        4. pytesseract (Legacy Fallback)
        """
        try:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            # Tier 1: RapidOCR (PP-OCRv4 ONNX)
            rapid_text, rapid_conf = self._extract_with_rapidocr(image)
            if rapid_text and len(rapid_text.strip()) >= 5:
                logger.info(f"Successfully extracted {len(rapid_text)} chars using RapidOCR (conf={rapid_conf})")
                return rapid_text.strip(), "rapidocr", rapid_conf

            # Tier 2: Native Windows OCR (winocr)
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

            # Tier 3: PaddleOCR fallback
            paddle_text, paddle_conf = self._extract_with_paddle(image)
            if paddle_text:
                return paddle_text, "paddleocr", paddle_conf

            # Tier 4: pytesseract fallback
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
        Validates document, extracts text, performs PII masking, classifies document type,
        and extracts structured medical facts.
        """
        # 0. Empty input check
        if not file_bytes or len(file_bytes) == 0:
            return DocumentExtractionResult(
                filename=filename,
                extracted_text="",
                char_count=0,
                engine_used="none",
                confidence=0.0,
                is_medical_document=False,
                detected_document_type="empty"
            )

        # 1. Validation check
        is_safe, error_msg = safety_service.validate_file_upload(filename, file_bytes)
        if not is_safe:
            return DocumentExtractionResult(
                filename=filename,
                extracted_text=f"[Upload Error]: {error_msg}",
                char_count=0,
                engine_used="validation_rejected",
                confidence=0.0,
                source_type="ocr_extraction",
                is_medical_document=False,
                medical_document_confidence=0.0,
                detected_document_type="invalid_upload"
            )

        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf":
            raw_text, engine, conf = self.extract_from_pdf(file_bytes)
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            raw_text, engine, conf = self.extract_from_image(file_bytes)
        elif ext in ["txt", "csv", "json"]:
            try:
                raw_text = file_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                raw_text = file_bytes.decode("latin-1", errors="replace").strip()
            engine = "plain_text"
            conf = 1.0
        else:
            raw_text, engine, conf = self.extract_from_image(file_bytes)

        # 2. Medical Document Classification
        is_med, med_conf, doc_type = self.detect_medical_document(raw_text)

        # 3. Structured Fact Extraction (Vitals, Lab tests, Reference ranges, Dates)
        structured_facts, extracted_vitals = self.extract_structured_medical_facts(raw_text)

        # 4. PII Masking on Extracted Clinical Text
        mask_res = safety_service.mask_pii(raw_text)
        safe_text = mask_res.masked_text

        return DocumentExtractionResult(
            filename=filename,
            extracted_text=safe_text,
            char_count=len(safe_text),
            engine_used=engine,
            confidence=conf,
            source_type="ocr_extraction",
            is_mock=(settings.MEDGEMMA_MODE == "mock"),
            is_medical_document=is_med,
            medical_document_confidence=med_conf,
            detected_document_type=doc_type,
            structured_facts=structured_facts,
            extracted_vitals=extracted_vitals,
            pii_redaction_count=mask_res.redaction_count
        )

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        Backward-compatible method returning plain extracted text string.
        """
        res = self.extract_with_metadata(file_bytes, filename)
        return res.extracted_text


ocr_service = MedicalDocumentOCRService()
detect_medical_document = MedicalDocumentOCRService.detect_medical_document
extract_structured_medical_facts = MedicalDocumentOCRService.extract_structured_medical_facts
