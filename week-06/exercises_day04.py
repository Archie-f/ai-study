# Exercise 1 — Reconstruct a Different Shape
# You have a JSON file containing a list of dicts shaped like {"ticket_id": "T-104", "priority": "high", "resolved": true}.
# Write load_tickets(path: str) -> list[Ticket] where Ticket is a dataclass
# with fields ticket_id: str, priority: str, resolved: bool.
# Then explain in one sentence what would happen if one dict in the file was missing the "resolved" key,
# and how you would prevent a crash (hint: dataclass field defaults, or .get() before construction).
#
# Expected output: a working load_tickets() function,
# plus your one-sentence explanation of the missing-key failure mode.
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Ticket:
    ticket_id: str
    priority: str
    resolved: bool = False

# When one dict in the file is missing the "resolved" key (T-105 in the "tickets_2026-07-04T15:12:33.json" file),
# TypeError: Ticket.__init__() missing 1 required positional argument: 'resolved' is raised.
# To avoid this crash, we can define a default value as in Line 21 (resolved: bool = False)

tickets: list[dict[str, Any]] = [
        {"ticket_id": "T-104", "priority": "high", "resolved": True},
        {"ticket_id": "T-105", "priority": "high"},
        {"ticket_id": "T-106", "priority": "low", "resolved": True},
        {"ticket_id": "T-107", "priority": "medium", "resolved": False},
        {"ticket_id": "T-108", "priority": "medium", "resolved": True},
        {"ticket_id": "T-109", "priority": "low", "resolved": False}
    ]

def create_tickets(ticket_list: list[dict[str, Any]]) -> Path:
    ticket_directory: Path = Path(__file__).resolve().parent / "scratch_tickets"
    ticket_directory.mkdir(parents=True, exist_ok=True)

    time_stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    file_name = ticket_directory / f"tickets_{time_stamp}.json"

    with open(file_name, "w") as f:
        json.dump(ticket_list, f, indent=2)

    print(f"Tickets saved to {file_name}")
    return file_name

def load_tickets(path: str) -> list[Ticket]:
    with open(path, "r") as f:
        raw = json.load(f)

    ticket_list: list[Ticket] = [Ticket(**ticket) for ticket in raw]
    return ticket_list

# Exercise 2 — Two-Level Grouping
# You have a list of SurveyResponse objects, each with .department: str and .satisfaction: int (1-5).
# Write a function that returns a dict[str, float] mapping each department to its average satisfaction score,
# rounded to 2 decimal places.
# Use a domain and variable names of your own — do not reuse orders/fulfillment_rate from the notes.
#
# Expected output: dict[str, float], e.g. {"engineering": 4.20, "sales": 3.67}.

@dataclass
class SurveyResponse:
    department: str
    satisfaction: int

survey_responses: list[dict[str, Any]] = [
    {"department": "HR", "satisfaction": 5},
    {"department": "Finance", "satisfaction": 2},
    {"department": "Sales", "satisfaction": 3},
    {"department": "IT", "satisfaction": 4},
    {"department": "Software", "satisfaction": 4},
    {"department": "Transportation", "satisfaction": 1},
    {"department": "HR", "satisfaction": 2},
    {"department": "Finance", "satisfaction": 4},
    {"department": "Sales", "satisfaction": 2},
    {"department": "IT", "satisfaction": 5},
    {"department": "Software", "satisfaction":1},
    {"department": "Transportation", "satisfaction": 4},
    {"department": "HR", "satisfaction": 3},
    {"department": "Finance", "satisfaction": 5},
    {"department": "Sales", "satisfaction": 4},
    {"department": "IT", "satisfaction": 1},
    {"department": "Software", "satisfaction": 3},
    {"department": "Transportation", "satisfaction": 3},
]


def create_survey_responses(survey_response_list: list[dict[str, Any]]) -> Path:
    directory: Path = Path(__file__).resolve().parent / "scratch_responses"
    directory.mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    file_name = directory / f"survey_responses_{time_stamp}.json"

    with open(file_name, "w") as f:
        json.dump(survey_response_list, f, indent=2)

    return file_name

def load_survey_responses(path: str) -> list[SurveyResponse]:
    with open(path) as f:
        data = json.load(f)

    responses_list: list[SurveyResponse] = [SurveyResponse(**d) for d in data]
    return responses_list

def classify_by_department(responses: list[SurveyResponse]) -> dict[str, list[int]]:
    classified: dict[str, list[int]] = defaultdict(list)
    for r in responses:
        classified[r.department].append(r.satisfaction)

    return dict(classified)

def get_average_satisfaction_values(responses: list[SurveyResponse]) -> dict[str, float]:
    return {k: round(sum(v) / len(v), 2) for k, v in classify_by_department(responses).items()}


# Exercise 3 — Failure-Only CI Report
# You are writing print_build_report(builds: list[BuildResult]) where BuildResult has
# .project: str, .passed: bool, and .duration_seconds: float.
# Print one summary line with overall pass count, then for every failing build print the project name and duration.
# Additionally: if more than 3 builds failed, append a final line reading "Investigate before merging."
# Do not reuse print_health_report's variable names.
#
# Expected output: one summary line, one line per failing build, and the conditional warning line only when failures > 3.

@dataclass
class BuildResult:
    project: str
    passed: bool
    duration_seconds: float

build_list: list[BuildResult] = [
    BuildResult(project="ticketing", passed=True, duration_seconds=29.3),
    BuildResult(project="survey", passed=False, duration_seconds=49.0),
    BuildResult(project="customer reviews", passed=False, duration_seconds=19.0),
    BuildResult(project="authentication", passed=True, duration_seconds=12.4),
    BuildResult(project="billing api", passed=False, duration_seconds=88.1),
    BuildResult(project="ticketing", passed=True, duration_seconds=25.8),
    BuildResult(project="search service", passed=True, duration_seconds=34.2),
    BuildResult(project="survey", passed=True, duration_seconds=41.5),
    BuildResult(project="notifications", passed=False, duration_seconds=15.0),
    BuildResult(project="analytics", passed=True, duration_seconds=102.7)
]

def print_build_report(builds: list[BuildResult]):
    passed = sum(1 for build in builds if build.passed)
    failed = sum(1 for build in builds if not build.passed)

    print(f"Passed: {passed} of {len(builds)} builds passed.")

    if failed > 0:
        print("Failed builds: ")
    for b in builds:
        if not b.passed:
            print(f"    - Project: {b.project:<17}  Duration: {b.duration_seconds}")

    if failed > 3:
        print(f"⚠️ Investigate before merging. {failed} builds failed.")





if __name__ == "__main__":
    file_path = create_tickets(tickets)

    ticket_list: list[Ticket] = load_tickets(str(file_path))
    print("Tickets from json file:")
    for ticket in ticket_list:
        print(f"* {ticket.ticket_id}: {ticket}")

    print("-" * 20)

    file = str(create_survey_responses(survey_responses))
    res = load_survey_responses(file)
    print(get_average_satisfaction_values(res))

    print_build_report(build_list)

