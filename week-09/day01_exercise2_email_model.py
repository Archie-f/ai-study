from dataclasses import dataclass
from datetime import datetime


@dataclass
class EmailMetadata:
    """Metadata about an email: what's the email about, who sent it, when, and which thread it belongs to."""
    subject: str
    sender: str
    received_date: datetime
    thread_id: str | None = None


@dataclass
class Email:
    """A single loaded email: its metadata plus body text paragraphs."""
    metadata: EmailMetadata
    body_paragraphs: list[str]