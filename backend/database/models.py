from datetime import datetime
import json
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, Float, Boolean
from sqlalchemy.orm import relationship
from backend.database.db import Base

class Case(Base):
    __tablename__ = "cases"

    # Unique non-identifying Case ID, e.g. CF-2026-A1B2
    id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    preferred_language = Column(String(32), default="English", nullable=False)
    chief_complaint = Column(Text, nullable=True)
    
    # JSON-encoded structures
    evidence_state_json = Column(Text, default="{}", nullable=False)
    ai_summary_json = Column(Text, default="{}", nullable=False)
    
    # AI processing status: 'pending', 'ready', 'failed'
    ai_status = Column(String(32), default="pending", nullable=False)
    
    # Priority triage: 'P0', 'P1', 'P2', 'P3', 'ERR'
    priority_category = Column(String(16), default="P3", nullable=False, index=True)
    ml_model_triage = Column(String(16), default="P3", nullable=True)
    ml_confidence = Column(Float, default=0.0, nullable=True)
    priority_reasons_json = Column(Text, default="[]", nullable=False)
    missing_information_json = Column(Text, default="[]", nullable=False)
    uncertainties_json = Column(Text, default="[]", nullable=False)
    triage_rule_version = Column(String(32), default="v1.0.0-prototype", nullable=False)
    # Final Guarded Triage Result (Primary Displayed Signal)
    final_triage = Column(String(16), default="P3", nullable=True)
    rule_based_triage = Column(String(16), default="P3", nullable=True)
    escalation_applied = Column(Boolean, default=False, nullable=True)
    escalation_reason = Column(Text, nullable=True)
    selected_model_provider = Column(String(32), default="finetuned_medgemma", nullable=True)
    decision_path = Column(String(64), default="model_urgency_reasoning", nullable=True)
    
    # Reviewer workflow: 'pending', 'reviewed', 'approved', 'reopened'
    review_status = Column(String(32), default="pending", nullable=False, index=True)
    reviewer_summary = Column(Text, nullable=True)
    reviewer_priority = Column(String(16), nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    # Relationships
    messages = relationship("Message", back_populates="case", cascade="all, delete-orphan", order_by="Message.created_at")
    evidence_items = relationship("Evidence", back_populates="case", cascade="all, delete-orphan", order_by="Evidence.created_at")
    audit_events = relationship("AuditEvent", back_populates="case", cascade="all, delete-orphan", order_by="AuditEvent.created_at")

    def to_dict(self):
        ml_prio = getattr(self, "ml_model_triage", None) or self.priority_category
        ml_conf = float(getattr(self, "ml_confidence", 0.0) or 0.0)
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "preferred_language": self.preferred_language,
            "chief_complaint": self.chief_complaint,
            "evidence_state": json.loads(self.evidence_state_json or "{}"),
            "ai_summary": json.loads(self.ai_summary_json or "{}"),
            "ai_status": self.ai_status,
            "priority_category": self.priority_category,
            "final_triage": getattr(self, "final_triage", None) or self.priority_category,
            "rule_based_priority": getattr(self, "rule_based_triage", None) or self.priority_category,
            "rule_based_triage": getattr(self, "rule_based_triage", None) or self.priority_category,
            "ml_model_priority": ml_prio,
            "ml_model_triage": ml_prio,
            "ml_confidence": ml_conf,
            "escalation_applied": bool(getattr(self, "escalation_applied", False)),
            "escalation_reason": getattr(self, "escalation_reason", None),
            "selected_model_provider": getattr(self, "selected_model_provider", None) or "finetuned_medgemma",
            "priority_reasons": json.loads(self.priority_reasons_json or "[]"),
            "missing_information": json.loads(self.missing_information_json or "[]"),
            "uncertainties": json.loads(self.uncertainties_json or "[]"),
            "triage_rule_version": self.triage_rule_version,
            "decision_path": getattr(self, "decision_path", None) or "model_urgency_reasoning",
            "review_status": self.review_status,
            "reviewer_summary": self.reviewer_summary,
            "reviewer_priority": self.reviewer_priority,
            "reviewer_notes": self.reviewer_notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # 'patient' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Provenance and Modality Metadata
    raw_transcript = Column(Text, nullable=True)  # Original unedited ASR transcript if voice
    source_type = Column(String(32), default="patient_typed", nullable=True)  # 'patient_typed', 'voice_transcript', 'ocr_extraction'
    source_language = Column(String(16), nullable=True)
    transcription_confidence = Column(Float, nullable=True)
    is_confirmed = Column(Boolean, default=True, nullable=True)

    case = relationship("Case", back_populates="messages")
    evidence_items = relationship("Evidence", back_populates="source_message")

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "role": self.role,
            "content": self.content,
            "raw_transcript": self.raw_transcript,
            "source_type": self.source_type or "patient_typed",
            "source_language": self.source_language,
            "transcription_confidence": self.transcription_confidence,
            "is_confirmed": self.is_confirmed if self.is_confirmed is not None else True,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    field = Column(String(64), nullable=False, index=True)
    value_json = Column(Text, nullable=False)
    status = Column(String(32), default="reported", nullable=False)  # 'reported', 'unknown', 'conflicting', 'uncertain'
    source_message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    source_quote = Column(Text, nullable=True)
    source_type = Column(String(32), default="patient_message", nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    case = relationship("Case", back_populates="evidence_items")
    source_message = relationship("Message", back_populates="evidence_items")

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "field": self.field,
            "value": json.loads(self.value_json) if self.value_json else None,
            "status": self.status,
            "source_message_id": self.source_message_id,
            "source_quote": self.source_quote,
            "source_type": self.source_type or "patient_message",
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    old_value_json = Column(Text, nullable=True)
    new_value_json = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)
    actor_role = Column(String(32), nullable=False)  # 'patient', 'ai_system', 'triage_engine', 'reviewer'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    case = relationship("Case", back_populates="audit_events")

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "event_type": self.event_type,
            "old_value": json.loads(self.old_value_json) if self.old_value_json else None,
            "new_value": json.loads(self.new_value_json) if self.new_value_json else None,
            "reason": self.reason,
            "actor_role": self.actor_role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
