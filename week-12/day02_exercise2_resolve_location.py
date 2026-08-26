def build_job_summary(company_name: str, city: str | None, remote: bool) -> str:
    location = resolve_job_location(city, remote, company_name)
    return f"{company_name} -- {location}"


def build_job_citation(company_name: str, city: str | None, remote: bool) -> dict:
    location = resolve_job_location(city, remote, company_name)
    return {"company": company_name, "location": location}


def resolve_job_location(city: str | None, remote: bool, company_name: str) -> str:
    """Resolve a job's display location from its raw fields.

    Args:
        city: The job's city, if known.
        remote: Whether the job is remote.
        company_name: Fallback when neither city nor remote applies.

    Returns:
        "Remote" if remote is True, else city if set, else company_name. Strips leading/trailing whitespace.
    """
    if remote:
        return "Remote"
    elif city and city.strip():
        return city.strip()
    else:
        return company_name.strip()


if __name__ == "__main__":
    test_cases = [
        # 1. Standard Hybrid/On-site case
        {"city": "New York", "remote": False, "company_name": "Google"},

        # 2. Standard Fully Remote case (City specified but remote is True)
        {"city": "San Francisco", "remote": True, "company_name": "GitLab"},

        # 3. Pure Remote case (No city specified)
        {"city": None, "remote": True, "company_name": "Automattic"},

        # 4. Edge Case: Missing city and not remote (Potential data error/HQ default test)
        {"city": None, "remote": False, "company_name": "Stripe"},

        # 5. Empty string for city with remote False
        {"city": "", "remote": False, "company_name": "Netflix"},

        # 6. Empty string for city with remote True
        {"city": "", "remote": True, "company_name": "Buffer"},

        # 7. International location format
        {"city": "Berlin, Germany", "remote": False, "company_name": "Siemens"},

        # 8. Special characters in company name
        {"city": "Austin", "remote": False, "company_name": "OpenAI (US), Inc."},

        # 9. Whitespace city name (Testing input sanitization)
        {"city": "   Chicago   ", "remote": False, "company_name": "McDonalds"},

        # 10. Regional or multiple city data string
        {"city": "Dallas/Fort Worth", "remote": False, "company_name": "American Airlines"},

        # 11. Unicode characters in city name
        {"city": "São Paulo", "remote": False, "company_name": "Nubank"},

        # 12. Fully whitespace city name
        {"city": "     ", "remote": False, "company_name": "CampBunny"}
    ]

    for test in test_cases:
        company_name = test["company_name"]
        city = test["city"]
        remote = test["remote"]
        print(build_job_summary(company_name, city, remote))
        print(build_job_citation(company_name, city, remote))
