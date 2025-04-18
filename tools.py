from smolagents import DuckDuckGoSearchTool
from smolagents import Tool
import random
from huggingface_hub import list_models
import requests
from typing import Dict, Optional, Any


# Initialize the DuckDuckGo search tool
#search_tool = DuckDuckGoSearchTool()

class LocationInfoTool(Tool):
    name = "location_info"
    description = "Get the current location from the user's IP address via a free API"
    inputs = {
        "ip":{
            "type": "string",
            "description": "The IP address to get location info"
        }
    }
    output_type = "object"
    
    def forward(self, ip: str):
        try:
            response = requests.get(f'http://ip-api.com/json/{ip}')
            data = response.json()

            if data['status'] == 'success':
                return data
            else:
                print("Failed to retrieve location: ", data.get("message", "Unknown Error"))
                return None
        except Exception as e:
            print("An error occured: ", e)
            return None


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches today's weather snapshot (condition, average °C, daily precipitation, max wind) for the given coordinates."
    inputs = {
        "lat": {
            "type": "number",
            "description": "Lateral coordinate of the location"
        },
        "lon": {
            "type": "number",
            "description": "Longitudinal coordinate of the location"
        }
    }
    output_type = "string"

    # --- internal helpers -------------------------------------------------
    _CODE_MAP: Dict[int, str] = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Freezing drizzle", 57: "Dense freezing drizzle",
        61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains",
        80: "Light rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Light snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm + light hail", 99: "Thunderstorm + heavy hail"
    }

    def _fetch_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """" Call Open-Meteor and return the first (today's) daily record """
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,"
            "weathercode,precipitation_sum,windspeed_10m_max"
            "&timezone=auto"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        daily = response.json().get("daily")
        if not daily:
            return None
        return {k: v[0] for k, v in daily.items()}

    def forward(self, lat: float, lon: float):
        try:
            today = self._fetch_weather(lat, lon)
            if today is None:
                return None
            
            avg_t = (today["temperature_2m_max"] + today["temperature_2m_min"]) / 2
            code = int(today["weathercode"])
            desc = self._CODE_MAP.get(code, "Unknown condition")
            precip = today["precipitation_sum"]
            wind = today["windspeed_10m_max"]

            return (
                f"Weather @{lat: 4f}, {lon: 4f}: {desc}, "
                f"avg {avg_t: .1f} °C, precip {precip} mm, max wind {wind} km/h"
            )
        except Exception as e:
            return f"Error retrieving weather: {e}"


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
            
            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"

