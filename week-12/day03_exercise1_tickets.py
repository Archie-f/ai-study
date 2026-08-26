from dataclasses import dataclass


@dataclass
class TicketResolution:
    ticket_id: str
    category: str
    draft_reply: str


def classify_ticket(ticket_text: str) -> str:
    ...  # returns a category label


def draft_reply(ticket_text: str, category: str) -> str:
    ...  # returns a draft reply string


def resolve_ticket(ticket_id: str, ticket_text: str) -> TicketResolution:
    """Classify a ticket, draft a reply, and bundle both into a TicketResolution.

    Args:
        ticket_id: The ticket's identifier.
        ticket_text: The raw ticket text to classify and reply to.

    Returns:
        A TicketResolution carrying ticket_id, the classified category,
        and the drafted reply.
    """
    category_label = classify_ticket(ticket_text)
    return TicketResolution(
        ticket_id=ticket_id,
        category=category_label,
        draft_reply=draft_reply(ticket_text, category_label),
    )