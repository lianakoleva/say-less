"""Weather tool backed by Open-Meteo (no API key required)."""

from typing import Any, Dict, Optional

import httpx

from ..base import ToolBase

WEATHER_CODES = {
    0: "clear skies",
    1: "mainly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with hail",
    99: "a severe thunderstorm with hail",
}


def _describe_code(code: Optional[int]) -> str:
    try:
        return WEATHER_CODES.get(int(code), "mixed conditions")
    except (TypeError, ValueError):
        return "mixed conditions"


class WeatherTool(ToolBase):
    """Look up current weather and a short forecast."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "weather"

    def requires_internet(self) -> bool:
        return True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the current weather or a short forecast for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["current", "forecast"],
                            "description": "Current conditions or a multi-day forecast",
                        },
                        "location": {
                            "type": "string",
                            "description": "City or place name",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Forecast days (1-7)",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    async def _geocode(
        self, client: httpx.AsyncClient, location: str
    ) -> Optional[Dict[str, Any]]:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        return results[0] if results else None

    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action not in ("current", "forecast", "get", "weather"):
            return {"success": False, "error": f"Unknown action: {action}"}

        location = str(
            params.get("location")
            or self.settings.weather_location
            or "San Francisco"
        ).strip()
        if not location:
            return {"success": False, "error": "Which city should I check?"}

        days = params.get("days") or (3 if action == "forecast" else 1)
        try:
            days = max(1, min(7, int(days)))
        except (TypeError, ValueError):
            days = 1

        units = (self.settings.weather_units or "fahrenheit").lower()
        temperature_unit = "celsius" if units.startswith("c") else "fahrenheit"
        degree = "C" if temperature_unit == "celsius" else "F"
        wind_unit = "kmh" if temperature_unit == "celsius" else "mph"
        wind_label = "kilometers per hour" if wind_unit == "kmh" else "miles per hour"

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                place = await self._geocode(client, location)
                if not place:
                    return {
                        "success": False,
                        "error": f"I couldn't find a place called {location}.",
                    }

                response = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": place["latitude"],
                        "longitude": place["longitude"],
                        "current": (
                            "temperature_2m,apparent_temperature,"
                            "relative_humidity_2m,precipitation,weather_code,"
                            "wind_speed_10m"
                        ),
                        "daily": (
                            "temperature_2m_max,temperature_2m_min,"
                            "precipitation_probability_max,weather_code"
                        ),
                        "temperature_unit": temperature_unit,
                        "wind_speed_unit": wind_unit,
                        "timezone": "auto",
                        "forecast_days": days,
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return {
                "success": False,
                "error": f"I couldn't reach the weather service: {exc}",
            }

        current = data.get("current") or {}
        daily = data.get("daily") or {}
        name = place.get("name") or location
        region = place.get("admin1")
        country = place.get("country")
        pretty_place = name
        if region and region != name:
            pretty_place += f", {region}"
        elif country:
            pretty_place += f", {country}"

        temperature = current.get("temperature_2m")
        condition = _describe_code(current.get("weather_code"))
        feels = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")

        def rounded(value: Any) -> Any:
            try:
                return round(float(value))
            except (TypeError, ValueError):
                return value

        temperature = rounded(temperature)
        feels = rounded(feels)
        humidity = rounded(humidity)
        wind = rounded(wind)

        summary = f"It's {temperature}°{degree} with {condition} in {pretty_place}."
        if feels is not None and abs(float(feels) - float(temperature or feels)) >= 3:
            summary += f" It feels like {feels}°{degree}."
        if humidity is not None:
            summary += f" Humidity is {humidity}%."
        if wind is not None:
            summary += f" Wind is {wind} {wind_label}."

        highs = daily.get("temperature_2m_max") or []
        lows = daily.get("temperature_2m_min") or []
        rain = daily.get("precipitation_probability_max") or []
        codes = daily.get("weather_code") or []
        dates = daily.get("time") or []

        if highs and lows:
            summary += (
                f" Today's high is {rounded(highs[0])}°{degree} and the low is "
                f"{rounded(lows[0])}°{degree}"
            )
            if rain:
                summary += f", with a {rounded(rain[0])}% chance of precipitation"
            summary += "."

        forecast = []
        for index in range(1, min(len(dates), days)):
            day = "Tomorrow" if index == 1 else dates[index]
            piece = (
                f"{day}: {_describe_code(codes[index]) if index < len(codes) else 'mixed conditions'}, "
                f"high {rounded(highs[index])}°{degree}, low {rounded(lows[index])}°{degree}"
            )
            if index < len(rain) and rain[index] is not None:
                piece += f", {rounded(rain[index])}% chance of precipitation"
            forecast.append(piece)
        if forecast:
            summary += " " + ". ".join(forecast) + "."

        return {
            "success": True,
            "location": pretty_place,
            "summary": summary,
            "temperature": temperature,
            "units": degree,
            "condition": condition,
        }
