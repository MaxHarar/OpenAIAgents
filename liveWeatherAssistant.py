import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import requests
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()

GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/direct"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT_SECONDS = 10
MISSING_KEY_MESSAGE = "Error: OPENWEATHERMAP_API_KEY is not set, so weather data is unavailable."


@dataclass
class WeatherInfo:
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    pressure: int
    location_name: str
    rain_1h: Optional[float] = None
    visibility: Optional[int] = None


def describe_request_error(error: requests.exceptions.RequestException) -> str:
    """Describe a failed request without str(error), which can contain the API key in the URL."""
    if error.response is not None:
        return f"OpenWeatherMap returned HTTP {error.response.status_code}"
    return f"the request failed ({type(error).__name__})"


@function_tool
def get_coordinates(location: str) -> str:
    """Look up the latitude and longitude of a place by name.

    Args:
        location: Place name, optionally with a state and country code, such as "Paris,FR".
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return MISSING_KEY_MESSAGE

    try:
        response = requests.get(
            GEOCODING_URL,
            params={"q": location, "limit": 5, "appid": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        matches = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error looking up coordinates: {describe_request_error(e)}"

    if not matches:
        return f"No locations found for {location!r}."

    lines = []
    for match in matches:
        place = ", ".join(
            part for part in (match["name"], match.get("state"), match.get("country")) if part
        )
        lines.append(f"- {place}: lat={match['lat']}, lon={match['lon']}")
    return "Matching locations:\n" + "\n".join(lines)


@function_tool
def get_weather(lat: float, lon: float) -> str:
    """Get the current weather for a specified location using OpenWeatherMap API.

    Args:
        lat: Latitude of the location (-90 to 90).
        lon: Longitude of the location (-180 to 180).
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return MISSING_KEY_MESSAGE

    try:
        response = requests.get(
            WEATHER_URL,
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        weather_info = WeatherInfo(
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            humidity=data["main"]["humidity"],
            description=data["weather"][0]["description"],
            wind_speed=data["wind"]["speed"],
            pressure=data["main"]["pressure"],
            location_name=data["name"],
            visibility=data.get("visibility"),
            rain_1h=data.get("rain", {}).get("1h"),
        )
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {describe_request_error(e)}"
    except (KeyError, IndexError) as e:
        return f"Error reading weather data: the response is missing {e}"

    report_lines = [
        f"Weather in {weather_info.location_name}:",
        f"- Temperature: {weather_info.temperature}°C (feels like {weather_info.feels_like}°C)",
        f"- Conditions: {weather_info.description}",
        f"- Humidity: {weather_info.humidity}%",
        f"- Wind speed: {weather_info.wind_speed} m/s",
        f"- Pressure: {weather_info.pressure} hPa",
    ]
    if weather_info.visibility is not None:
        report_lines.append(f"- Visibility: {weather_info.visibility} m")
    if weather_info.rain_1h is not None:
        report_lines.append(f"- Rain (last hour): {weather_info.rain_1h} mm")
    return "\n".join(report_lines)


# Create a weather assistant
weather_assistant = Agent(
    name="Weather Assistant",
    instructions="""You are a weather assistant that can provide current weather information.

    When asked about the weather in a named place, call get_coordinates first, then call
    get_weather with the coordinates it returns. Never supply coordinates from memory.
    If get_coordinates returns several different places and the user's request doesn't
    make clear which one they mean, ask for clarification (e.g., Paris, France vs. Paris, Texas).
    If a tool returns an error, tell the user you couldn't get the weather and why.
    Never invent weather data.

    Provide friendly commentary along with the weather data, such as clothing suggestions
    or activity recommendations based on the conditions.
    """,
    tools=[get_coordinates, get_weather],
)


async def main():
    simple_request = await Runner.run(weather_assistant, "What are your capabilities?")
    print(simple_request.final_output)
    print("-" * 70)

    location = ""
    while not location:
        location = input("Which location would you like the weather for? ").strip()

    conversation = f"What's the weather like in {location} right now?"
    while True:
        result = await Runner.run(weather_assistant, conversation)
        print(result.final_output)
        print("-" * 70)

        # The assistant may ask which place you meant, so let the user answer it.
        follow_up = input("Your reply (press Enter to quit): ").strip()
        if not follow_up:
            break
        conversation = result.to_input_list() + [{"role": "user", "content": follow_up}]


if __name__ == "__main__":
    asyncio.run(main())
