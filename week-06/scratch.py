# Exercise 1 — Optional Collaborator
# A ticket system has this function:
# def close_ticket(ticket_id: str, notifier=None) -> None:
#     """Mark a ticket closed, optionally notifying someone."""
#     # Your implementation here
#
# Write the body so that: (a) it always prints f"Ticket {ticket_id} closed", and (b) if a notifier object
# with a .send(text) method is passed, it also calls notifier.send(f"Ticket {ticket_id} was closed").
# Then write two calls: one with no notifier, one with a small object you define yourself that has a .send() method.
from unittest.mock import MagicMock


class Notifier:
    def send(self, text: str) -> None:
        print(text)

def close_ticket(ticket_id: str, notifier: Notifier | None = None) -> None:
    """Mark a ticket closed, optionally notifying someone."""
    print(f"Ticket {ticket_id} closed")
    if notifier is not None:
        notifier.send(f"Ticket {ticket_id} was closed")


# Exercise 2 — Optional Field on a Dataclass
# You have this dataclass, used by a warehouse system:
# @dataclass
# class ShipmentRow:
#     order_id: str
#     weight_kg: float
#     # Your fields here
#
# Add an optional field customs_fee: float | None = None (a shipment only has a customs fee if it crosses a border).
# Then write a function total_customs(rows: list[ShipmentRow]) -> float that sums only
# the customs_fee values that are present, treating missing ones as 0 — without ever writing 0 into the row itself.

from dataclasses import dataclass

@dataclass
class ShipmentRow:
    order_id: str
    weight_kg: float
    customs_fee: float | None = None

def total_customs(rows: list[ShipmentRow]) -> float:
    """Sums the customs_fee values that are present, treating missing ones as 0."""
    total = 0.0
    for row in rows:
        if row.customs_fee is not None:
            total += row.customs_fee

    return total


# Exercise 3 — Defensive Wrapping
# Write a function fetch_weather(city: str, client) -> dict | None that calls client.get_forecast(city)
# and returns its result. If the call raises any exception, catch it, print a one-line message
# naming the city and the error, and return None instead of letting the exception escape.
# Then write a small test scenario: a fake client whose get_forecast() always raises RuntimeError,
# and show that fetch_weather() returns None rather than crashing.

def fetch_weather(city: str, client) -> dict | None:
    """Calls client.get_forecast(city) and returns its result"""
    try:
        return client.get_forecast(city)
    except Exception as e:
        print(f"The weather forecast for {city} could not be fetched. Error: {e}")
        return None

def test_faulty_fetch_weather():
    client = MagicMock()
    client.get_forecast.side_effect = RuntimeError(f"Could not receive the forecast.")

    result = fetch_weather("New York City", client)
    print(result)


# Exercise 4 — Conditional Column
# Write print_inventory(items: list[dict]) -> None where each item has name and quantity,
# and optionally eta (a string, backorder arrival estimate). Print a table with Name and Quantity columns always,
# and a Backorder ETA column only if at least one item in the list has an eta.
# Rows for items without an eta should show a dash in that column when the column exists.

def print_inventory(items: list[dict]) -> None:
    name = "Name"
    quantity = "Quantity"
    backorder_eta = "Backorder ETA"

    has_eta = any(item.get("eta") is not None for item in items)

    header = f"{name:<10} | {quantity:<10}"
    if has_eta:
        header += f" | {backorder_eta}"
    print(header)
    print("-" * 40)

    for item in items:
        row = f"{item['name']:<10} | {item['quantity']:<10}"
        if has_eta:
            row += f" | {item.get("eta", "-")}"
        print(row)


if __name__ == "__main__":
    print("Exercise 1: _______________")
    tickets: list[str] = [
        "Ticket00123",
        "Ticket00124",
        "Ticket00125",
        "Ticket00126",
        "Ticket00127",
        "Ticket00128"
    ]

    shipments: list[ShipmentRow] = [
        ShipmentRow(order_id=tickets[0], weight_kg=5.1, customs_fee=13.0),
        ShipmentRow(order_id=tickets[1], weight_kg=2.4, customs_fee=7.0),
        ShipmentRow(order_id=tickets[2], weight_kg=3.0),
        ShipmentRow(order_id=tickets[3], weight_kg=4.7, customs_fee=10.0),
        ShipmentRow(order_id=tickets[4], weight_kg=3.2),
        ShipmentRow(order_id=tickets[5], weight_kg=3.0),
    ]

    notifier_obj = Notifier()
    close_ticket(tickets[0])
    close_ticket(tickets[1], notifier_obj)

    print("-" * 20)
    print("Exercise 2: _______________")

    print(f"Total custom fee: ${total_customs(shipments)}")

    print("-" * 20)
    print("Exercise 3: _______________")
    test_faulty_fetch_weather()

    print("-" * 20)
    print("Exercise 4: _______________")
    inventory: list[dict[str, str]] = [
        {"name": "Shoe", "quantity": "30"},
        {"name": "T-shirt", "quantity": "75"},
        {"name": "Jeans", "quantity": "42", "eta": "3"},
        {"name": "Hat", "quantity": "37"},
    ]

    print_inventory(inventory)
