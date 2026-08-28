from abc import abstractmethod, ABC


class Notifier(ABC):
    @abstractmethod
    def send(self, ticket_id: str, message: str) -> bool:
        """Send a notification about a ticket. Returns True on success."""


class LogNotifier(Notifier):
    """Notifier that logs to stdout instead of sending anything real."""

    def send(self, ticket_id: str, message: str) -> bool:
        print(f"[{ticket_id}] {message}")
        return True


if __name__ == "__main__":
    notifier = LogNotifier()
    notifier.send("ticket_01", "Hello world!")
