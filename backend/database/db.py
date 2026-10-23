from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

# SQLite connection args for multithreaded server (check_same_thread=False)
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

# Enforce foreign key constraints in SQLite
if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

import logging
from sqlalchemy import text

logger = logging.getLogger("careflow.database")

def _migrate_sqlite_schema():
    """Applies lightweight schema migrations for existing SQLite databases."""
    try:
        with engine.begin() as conn:
            # Check cases table
            res_cases = conn.execute(text("PRAGMA table_info(cases)")).fetchall()
            existing_case_cols = {row[1] for row in res_cases}
            if existing_case_cols:
                if "ml_model_triage" not in existing_case_cols:
                    logger.info("Migrating cases table: adding ml_model_triage column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN ml_model_triage VARCHAR(16) DEFAULT 'P3'"))
                if "final_triage" not in existing_case_cols:
                    logger.info("Migrating cases table: adding final_triage column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN final_triage VARCHAR(16) DEFAULT 'P3'"))
                if "rule_based_triage" not in existing_case_cols:
                    logger.info("Migrating cases table: adding rule_based_triage column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN rule_based_triage VARCHAR(16) DEFAULT 'P3'"))
                if "escalation_applied" not in existing_case_cols:
                    logger.info("Migrating cases table: adding escalation_applied column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN escalation_applied BOOLEAN DEFAULT 0"))
                if "escalation_reason" not in existing_case_cols:
                    logger.info("Migrating cases table: adding escalation_reason column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN escalation_reason TEXT"))
                if "selected_model_provider" not in existing_case_cols:
                    logger.info("Migrating cases table: adding selected_model_provider column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN selected_model_provider VARCHAR(32) DEFAULT 'finetuned_medgemma'"))

            # Check messages table
            res = conn.execute(text("PRAGMA table_info(messages)")).fetchall()
            existing_msg_cols = {row[1] for row in res}
            if existing_msg_cols:
                if "source_type" not in existing_msg_cols:
                    logger.info("Migrating messages table: adding source_type column")
                    conn.execute(text("ALTER TABLE messages ADD COLUMN source_type VARCHAR(32) DEFAULT 'patient_typed'"))
                if "source_language" not in existing_msg_cols:
                    logger.info("Migrating messages table: adding source_language column")
                    conn.execute(text("ALTER TABLE messages ADD COLUMN source_language VARCHAR(16)"))
                if "raw_transcript" not in existing_msg_cols:
                    logger.info("Migrating messages table: adding raw_transcript column")
                    conn.execute(text("ALTER TABLE messages ADD COLUMN raw_transcript TEXT"))
                if "transcription_confidence" not in existing_msg_cols:
                    logger.info("Migrating messages table: adding transcription_confidence column")
                    conn.execute(text("ALTER TABLE messages ADD COLUMN transcription_confidence FLOAT"))
                if "is_confirmed" not in existing_msg_cols:
                    logger.info("Migrating messages table: adding is_confirmed column")
                    conn.execute(text("ALTER TABLE messages ADD COLUMN is_confirmed BOOLEAN DEFAULT 1"))

            # Check evidence table
            res_ev = conn.execute(text("PRAGMA table_info(evidence)")).fetchall()
            existing_ev_cols = {row[1] for row in res_ev}
            if existing_ev_cols:
                if "source_type" not in existing_ev_cols:
                    logger.info("Migrating evidence table: adding source_type column")
                    conn.execute(text("ALTER TABLE evidence ADD COLUMN source_type VARCHAR(32) DEFAULT 'patient_message'"))
                if "confidence" not in existing_ev_cols:
                    logger.info("Migrating evidence table: adding confidence column")
                    conn.execute(text("ALTER TABLE evidence ADD COLUMN confidence FLOAT"))

            # Check cases table
            res_cases = conn.execute(text("PRAGMA table_info(cases)")).fetchall()
            existing_case_cols = {row[1] for row in res_cases}
            if existing_case_cols:
                if "decision_path" not in existing_case_cols:
                    logger.info("Migrating cases table: adding decision_path column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN decision_path VARCHAR(64) DEFAULT 'model_urgency_reasoning'"))
                if "ml_model_triage" not in existing_case_cols:
                    logger.info("Migrating cases table: adding ml_model_triage column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN ml_model_triage VARCHAR(16) DEFAULT 'P3'"))
                if "ml_confidence" not in existing_case_cols:
                    logger.info("Migrating cases table: adding ml_confidence column")
                    conn.execute(text("ALTER TABLE cases ADD COLUMN ml_confidence FLOAT DEFAULT 0.0"))


    except Exception as e:
        logger.warning(f"Notice during SQLite schema check/migration: {e}")

def init_db():
    """Initializes tables in database and applies automatic migrations."""
    Base.metadata.create_all(bind=engine)
    if settings.DATABASE_URL.startswith("sqlite"):
        _migrate_sqlite_schema()
