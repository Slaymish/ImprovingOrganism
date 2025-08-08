try:
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
    from sqlalchemy.orm import declarative_base, sessionmaker  # modern import path
    SQLALCHEMY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SQLALCHEMY_AVAILABLE = False
    create_engine = Column = Integer = String = Text = DateTime = Float = object  # type: ignore
    sessionmaker = lambda *a, **k: None  # type: ignore
    def declarative_base():  # type: ignore
        class Dummy: pass
        return Dummy
from datetime import datetime
from typing import List, Optional
import os
from .config import settings
import os as _os

# Optional import of VectorMemory so tests can patch src.memory_module.VectorMemory
try:  # pragma: no cover
    from .vector_memory import VectorMemory  # type: ignore
except Exception:  # broad except to avoid import chain failures during lightweight tests
    VectorMemory = None  # type: ignore

Base = declarative_base()

class MemoryEntry(Base):
    __tablename__ = "memory"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    content = Column(Text)
    entry_type = Column(String(50))  # 'prompt', 'output', 'feedback'
    session_id = Column(String(100))  # for grouping related entries
    score = Column(Float, nullable=True)  # for feedback entries

# Ensure the data directory exists
db_path = settings.db_url.replace("sqlite:///", "")
db_dir = os.path.dirname(db_path)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

if SQLALCHEMY_AVAILABLE:
    engine = create_engine(settings.db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    try:
        Base.metadata.create_all(engine)
    except Exception:
        pass
else:
    engine = None
    SessionLocal = lambda: None  # type: ignore

class MemoryModule:
    def __init__(self):
        """Initialize memory module with optional SQLAlchemy session and vector memory."""
        self.session = SessionLocal() if callable(SessionLocal) else None
        # Provide vector memory helper if available (skip in lightweight self-learning mode)
        if _os.getenv('LIGHTWEIGHT_SELF_LEARNING'):
            self.vector_memory = None
        else:
            self.vector_memory = VectorMemory() if VectorMemory else None

    # ---- Context management -------------------------------------------------
    def close(self):  # pragma: no cover - trivial
        """Close underlying session if present."""
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass

    def __enter__(self):  # pragma: no cover - syntactic sugar
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        self.close()

    def write(self, content: str, entry_type: str, session_id: Optional[str] = None, score: Optional[float] = None):
        # --- Data hygiene filters (lightweight, rule-based) ---
        if not content or not content.strip():  # skip empty
            return
        text = content.strip()
        # Length bounds (very short or excessively long)
        if len(text) < 5 or len(text) > 5000:  # pragmatic guard rails
            return
        # Repetition / low information density heuristic
        lowered = text.lower()
        unique_chars = len(set(lowered))
        # Allow short strings; flag only if extremely low diversity relative to length
        if len(lowered) > 30 and unique_chars < len(lowered) * 0.1:
            return
        # Excessive single token repetition (e.g., 'hello hello hello ...')
        tokens = lowered.split()
        if tokens:
            most_common = max(tokens.count(t) for t in set(tokens))
            if most_common > 0.6 * len(tokens):
                return
        # Optional: basic profanity / toxicity placeholder (extensible)
        banned = {"toxic_placeholder_word"}
        if any(b in lowered for b in banned):
            return

        entry = MemoryEntry(
            content=text,
            entry_type=entry_type,
            session_id=session_id,
            score=score
        )
        if self.session is not None:
            self.session.add(entry)
            self.session.commit()
        if self.vector_memory:
            # Mirror data into vector memory for semantic operations
            try:
                ts = entry.timestamp.isoformat() if hasattr(entry, 'timestamp') else None
                self.vector_memory.add_entry(content=content, entry_type=entry_type, session_id=session_id, score=score, timestamp=ts)
            except Exception:
                pass
    
    def store_entry(self, content: str, entry_type: str, session_id: Optional[str] = None, score: Optional[float] = None):
        """Alias for write method for compatibility"""
        self.write(content, entry_type, session_id, score)
    
    def read_all(self) -> List[MemoryEntry]:
        if self.session is None:
            return []
        return self.session.query(MemoryEntry).order_by(MemoryEntry.timestamp).all()
    
    def read_by_type(self, entry_type: str, limit: Optional[int] = None) -> List[MemoryEntry]:
        if self.session is None:
            return []
        query = self.session.query(MemoryEntry).filter(
            MemoryEntry.entry_type == entry_type
        ).order_by(MemoryEntry.timestamp)
        
        if limit:
            query = query.limit(limit)
            
        return query.all()
    
    def get_entries_by_type(self, entry_type: str, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Alias for read_by_type for compatibility"""
        return self.read_by_type(entry_type, limit)
    
    def read_by_session(self, session_id: str) -> List[MemoryEntry]:
        if self.session is None:
            return []
        return self.session.query(MemoryEntry).filter(
            MemoryEntry.session_id == session_id
        ).order_by(MemoryEntry.timestamp).all()
    
    def get_feedback_entries(self, min_score: Optional[float] = None) -> List[MemoryEntry]:
        if self.session is None:
            return []
        query = self.session.query(MemoryEntry).filter(
            MemoryEntry.entry_type == "feedback"
        )
        if min_score is not None:
            query = query.filter(MemoryEntry.score >= min_score)
        return query.order_by(MemoryEntry.timestamp.desc()).all()

    # ---- Semantic / vector helpers -----------------------------------------
    def search_semantic(self, query: str, limit: int = 5, entry_types: Optional[List[str]] = None):
        """Search vector memory (semantic search) if available.

        Provided mainly for compatibility with tests and higher-level retrieval.
        Returns empty list gracefully if vector backend not configured.
        """
        if not self.vector_memory:
            return []
        try:
            return self.vector_memory.search(query, limit, entry_types)
        except Exception:
            return []
    
    def get_training_data(self, limit: int = 100) -> List[tuple]:
        """Get prompt-output pairs for training"""
        feedback_entries = self.get_feedback_entries(min_score=3.0)  # Only good feedback
        training_pairs = []
        
        for feedback in feedback_entries[-limit:]:  # Get recent good feedback
            # Parse feedback content to extract prompt and output
            if " -> " in feedback.content and "FEEDBACK:" in feedback.content:
                try:
                    parts = feedback.content.split(" -> ")
                    prompt = parts[0].replace("FEEDBACK: ", "").strip()
                    output_part = parts[1].split(" | score=")[0].strip()
                    training_pairs.append((prompt, output_part, feedback.score))
                except:
                    continue
        
        return training_pairs
