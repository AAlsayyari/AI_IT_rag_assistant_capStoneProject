"""
Database module for the Ticketing System Analytics (v0.3).

Implements a normalized relational schema with three linked tables:
  - users:         Organizational context (college, role)
  - tickets:       Issue classification and severity metadata
  - chat_sessions: Interaction metrics, transcript, and resolution data
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ---------------------------------------------------------------------------
# Database path – stored alongside the v0.3 project root
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent.parent / "analytics.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False)

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class User(Base):
    """Stores organizational context to avoid duplicating attributes across tickets."""
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    college = Column(String, nullable=True)
    role = Column(String, nullable=True)  # e.g. student, faculty

    tickets = relationship("Ticket", back_populates="user")


class Ticket(Base):
    """Captures the overarching issue context."""
    __tablename__ = "tickets"

    ticket_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=True)
    external_ticket_ref = Column(String, nullable=True)
    ticket_class = Column(String, nullable=True)    # e.g. Hardware, LMS, Network
    sub_class = Column(String, nullable=True)        # e.g. Password Reset, Wi-Fi Certificate
    severity = Column(String, nullable=True)          # e.g. low, medium, high, critical
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="tickets")
    sessions = relationship("ChatSession", back_populates="ticket")


class ChatSession(Base):
    """Records the actual interaction metrics and links to the parent ticket."""
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), nullable=False)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    ticket_summary = Column(Text, nullable=True)
    issue_resolved = Column(Boolean, nullable=True)
    sentiment = Column(String, nullable=True)
    raw_transcript = Column(Text, nullable=True)

    ticket = relationship("Ticket", back_populates="sessions")


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables if they do not yet exist."""
    Base.metadata.create_all(ENGINE)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(bind=ENGINE)


# ---------------------------------------------------------------------------
# Data‑access helpers
# ---------------------------------------------------------------------------

def upsert_user(user_id: str, college: str | None = None, role: str | None = None) -> User:
    """Insert or update a user record (UPSERT)."""
    with SessionLocal() as session:
        user = session.get(User, user_id)
        if user is None:
            user = User(user_id=user_id, college=college, role=role)
            session.add(user)
        else:
            if college is not None:
                user.college = college
            if role is not None:
                user.role = role
        session.commit()
        session.refresh(user)
        return user


def insert_ticket(
    user_id: str | None,
    ticket_class: str | None,
    sub_class: str | None,
    severity: str | None,
    external_ticket_ref: str | None = None,
) -> Ticket:
    """Create a new ticket row and return the ORM object."""
    with SessionLocal() as session:
        ticket = Ticket(
            user_id=user_id,
            external_ticket_ref=external_ticket_ref,
            ticket_class=ticket_class,
            sub_class=sub_class,
            severity=severity,
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket


def insert_chat_session(
    ticket_id: str,
    start_time: datetime,
    end_time: datetime,
    ticket_summary: str | None,
    issue_resolved: bool | None,
    sentiment: str | None,
    raw_transcript: str | None,
) -> ChatSession:
    """Create a new chat_session row linked to a ticket."""
    with SessionLocal() as session:
        chat = ChatSession(
            ticket_id=ticket_id,
            start_time=start_time,
            end_time=end_time,
            ticket_summary=ticket_summary,
            issue_resolved=issue_resolved,
            sentiment=sentiment,
            raw_transcript=raw_transcript,
        )
        session.add(chat)
        session.commit()
        session.refresh(chat)
        return chat
