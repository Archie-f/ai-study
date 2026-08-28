from dataclasses import dataclass


@dataclass
class WeatherReport:
    station_name: str
    text: str


def build_weather_context(reports: list[WeatherReport]) -> str:
    """Join weather reports into one labeled context string.

    Each report becomes a block:
        [Station N: <station_name>]
        <text>

    Args:
        reports: Weather reports to format, in the order given.

    Returns:
        A single string with one labeled block per report, blocks
        separated by a blank line.
    """
    weather_context = [
        f"[Station {index:02d}: {weather_report.station_name}]\n{weather_report.text}"
        for index, weather_report in enumerate(reports, start=1)
    ]
    return "\n\n".join(weather_context)

if __name__ == "__main__":
    weather_reports = [
        WeatherReport(station_name="Oslo-Blindern", text="Clear skies with a light breeze from the south."),
        WeatherReport(station_name="Bergen-Florida", text="Heavy rainfall expected to persist throughout the evening."),
        WeatherReport(station_name="Trondheim-Værnes", text="Overcast with patchy fog reducing visibility on runways."),
        WeatherReport(station_name="Stavanger-Sola", text="Strong gales along the coast; high wave warnings active."),
        WeatherReport(station_name="Tromsø-Langnes", text="Light snow flurries with temperatures dropping below freezing."),
        WeatherReport(station_name="Kristiansand-Kjevik", text="Mild and sunny conditions perfect for outdoor activities."),
        WeatherReport(station_name="Bodø-Airport", text="Localized thunderstorms shifting rapidly toward the east."),
        WeatherReport(station_name="Ålesund-Vigra", text="Dense sea fog rolling in, driving temperatures down sharply."),
        WeatherReport(station_name="Svalbard-Lufthavn", text="Extreme cold with high winds creating severe wind chill."),
        WeatherReport(station_name="Fauske-Center", text="Calm and clear night with excellent visibility across the area.")
    ]

    report = build_weather_context(weather_reports)
    print(report)
