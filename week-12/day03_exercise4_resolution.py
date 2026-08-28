from dataclasses import dataclass


@dataclass
class TicketResolution:
    ticket_id: str
    category: str
    draft_reply: str


def format_ticket_resolution(resolution: TicketResolution) -> str:
    """Format a TicketResolution as printable, human-readable text.

    Args:
        resolution: The resolution to format.

    Returns:
        Two lines: "<ticket_id> [<category>]" then the draft reply,
        joined by a newline.
    """
    return f"{resolution.ticket_id} [{resolution.category}]\n{resolution.draft_reply}"