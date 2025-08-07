from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Optional
import os
from .config import settings

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

engine = create_engine(settings.db_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

class MemoryModule:
    def __init__(self):
        self.session = SessionLocal()

    def write(self, content: str, entry_type: str, session_id: Optional[str] = None, score: Optional[float] = None):
        entry = MemoryEntry(
            content=content,
            entry_type=entry_type,
            session_id=session_id,
            score=score
        )
        self.session.add(entry)
        self.session.commit()
    
    def store_entry(self, content: str, entry_type: str, session_id: Optional[str] = None, score: Optional[float] = None):
        """Alias for write method for compatibility"""
        self.write(content, entry_type, session_id, score)
    
    def read_all(self) -> List[MemoryEntry]:
        return self.session.query(MemoryEntry).order_by(MemoryEntry.timestamp).all()
    
    def read_by_type(self, entry_type: str, limit: Optional[int] = None) -> List[MemoryEntry]:
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
        return self.session.query(MemoryEntry).filter(
            MemoryEntry.session_id == session_id
        ).order_by(MemoryEntry.timestamp).all()
    
    def get_feedback_entries(self, min_score: Optional[float] = None) -> List[MemoryEntry]:
        query = self.session.query(MemoryEntry).filter(
            MemoryEntry.entry_type == "feedback"
        )
        if min_score is not None:
            query = query.filter(MemoryEntry.score >= min_score)
        return query.order_by(MemoryEntry.timestamp.desc()).all()
    
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
