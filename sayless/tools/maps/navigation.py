"""Navigation and travel-time tool.

Uses the Google Directions API when a key is configured, and falls back to
OpenStreetMap Nominatim + OSRM (both free, no key) otherwise. If the user has
not said where they are starting from, the origin is approximated from their
public IP address.
"""

import asyncio
import re
import shlex
import subprocess
from typing import Any, Dict, Optional, Tuple

import httpx

from ..base import ToolBase
from ..model_tools import normalize_address

USER_AGENT = "SayLess/0.1 (local voice assistant)"
_COORDS = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


class NavigationTool(ToolBase):
    """Get directions and travel time to a destination."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "navigation"
        self._google_error: Optional[str] = None

    def requires_internet(self) -> bool:
        # Routing reports its own network errors. Returning False here avoids
        # the generic queue check, which can fail (for example when a
        # connectivity probe is blocked) even though the routing APIs work.
        return False

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "navigation",
                "description": "Get directions or travel time to a destination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["directions", "eta"],
                            "description": "Directions or just the travel time",
                        },
                        "origin": {
                            "type": "string",
                            "description": "Starting address or place",
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination address or place",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["driving", "walking", "bicycling", "transit"],
                            "description": "Travel mode",
                        },
                    },
                    "required": ["action", "destination"],
                },
            },
        }

    # ── Geolocation ─────────────────────────────────────────────────────────

    async def _ip_location(
        self, client: httpx.AsyncClient
    ) -> Optional[Dict[str, Any]]:
        """Approximate the current location from the public IP address."""
        try:
            response = await client.get("https://ipapi.co/json/")
            if response.status_code == 200:
                data = response.json()
                lat = data.get("latitude")
                lon = data.get("longitude")
                if lat is not None and lon is not None:
                    label = ", ".join(
                        part for part in (data.get("city"), data.get("region")) if part
                    )
                    return {
                        "lat": float(lat),
                        "lon": float(lon),
                        "label": label or "your current location",
                    }
        except Exception:
            pass

        try:
            response = await client.get("https://ipinfo.io/json")
            if response.status_code == 200:
                data = response.json()
                location = str(data.get("loc") or "")
                match = _COORDS.match(location)
                if match:
                    label = ", ".join(
                        part for part in (data.get("city"), data.get("region")) if part
                    )
                    return {
                        "lat": float(match.group(1)),
                        "lon": float(match.group(2)),
                        "label": label or "your current location",
                    }
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_coords(value: str) -> Optional[Tuple[float, float]]:
        match = _COORDS.match(str(value or ""))
        if not match:
            return None
        return float(match.group(1)), float(match.group(2))

    # ── Geocoding and routing ───────────────────────────────────────────────

    @staticmethod
    def _address_variants(query: str) -> list[str]:
        """Return likely Nominatim spellings for a spoken address."""
        variants: list[str] = []

        def add(value: str) -> None:
            value = str(value or "").strip()
            if value and value not in variants:
                variants.append(value)

        add(query)
        normalized = normalize_address(query)
        add(normalized)
        if normalized and "," not in normalized:
            tokens = normalized.split()
            # Try a comma before the trailing city/state ("1201 Main St, SF").
            for cut in (1, 2):
                if len(tokens) >= cut + 3:
                    add(" ".join(tokens[:-cut]) + ", " + " ".join(tokens[-cut:]))
        return variants

    async def _geocode(
        self, client: httpx.AsyncClient, query: str
    ) -> Optional[Dict[str, Any]]:
        coords = self._parse_coords(query)
        if coords:
            return {"lat": coords[0], "lon": coords[1], "label": query}

        for index, candidate in enumerate(self._address_variants(query)):
            if index:
                # Nominatim asks for at most one request per second.
                await asyncio.sleep(1)
            try:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": candidate, "format": "json", "limit": 1},
                    headers={"User-Agent": USER_AGENT},
                )
                response.raise_for_status()
                results = response.json()
            except Exception:
                continue
            if not results:
                continue
            item = results[0]
            return {
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "label": item.get("display_name") or candidate,
            }
        return None

    async def _google_route(
        self,
        client: httpx.AsyncClient,
        origin: str,
        destination: str,
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        key = self.settings.google_maps_api_key
        if not key:
            return None
        self._google_error = None
        try:
            response = await client.get(
                "https://maps.googleapis.com/maps/api/directions/json",
                params={
                    "origin": origin,
                    "destination": destination,
                    "mode": mode,
                    "key": key,
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._google_error = f"request failed: {exc}"
            return None

        if data.get("status") != "OK" or not data.get("routes"):
            detail = data.get("error_message") or data.get("status") or "no route"
            self._google_error = str(detail)
            return None

        route = data["routes"][0]
        leg = route.get("legs", [{}])[0]
        duration = leg.get("duration", {}).get("value")
        distance = leg.get("distance", {}).get("value")
        if duration is None:
            return None
        return {
            "success": True,
            "provider": "google",
            "duration_minutes": max(1, round(duration / 60)),
            "distance_meters": distance,
            "origin_label": leg.get("start_address") or origin,
            "destination_label": leg.get("end_address") or destination,
            "via": route.get("summary"),
            "steps": [
                step.get("html_instructions", "")
                for step in leg.get("steps", [])
            ][:8],
        }

    async def _osrm_route(
        self,
        client: httpx.AsyncClient,
        origin: str,
        destination: str,
        mode: str,
        origin_fix: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        origin_place = origin_fix or await self._geocode(client, origin)
        dest_place = await self._geocode(client, destination)
        if not origin_place or not dest_place:
            return None

        profile = {
            "driving": "driving",
            "walking": "foot",
            "bicycling": "bike",
            "transit": "driving",
        }.get(mode, "driving")
        coords = (
            f"{origin_place['lon']},{origin_place['lat']};"
            f"{dest_place['lon']},{dest_place['lat']}"
        )
        try:
            response = await client.get(
                f"https://router.project-osrm.org/route/v1/{profile}/{coords}",
                params={"overview": "false", "steps": "false"},
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return None

        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        route = data["routes"][0]
        duration = route.get("duration")
        distance = route.get("distance")
        if duration is None:
            return None
        return {
            "success": True,
            "provider": "osrm",
            "duration_minutes": max(1, round(duration / 60)),
            "distance_meters": distance,
            "origin_label": origin_place.get("label") or origin,
            "destination_label": dest_place.get("label") or destination,
        }

    # ── Formatting ──────────────────────────────────────────────────────────

    def _distance_text(self, meters: Optional[float]) -> Optional[str]:
        if meters is None:
            return None
        if str(self.settings.navigation_units).lower().startswith("k"):
            return f"{meters / 1000:.1f} km"
        miles = meters / 1609.344
        if miles < 0.2:
            return f"{int(miles * 5280)} feet"
        return f"{miles:.1f} miles"

    async def _launch(
        self, origin: str, destination: str, label: str
    ) -> bool:
        command = self.settings.navigation_command
        if not command:
            return False
        expanded = command.format(
            origin=shlex.quote(origin),
            destination=shlex.quote(destination),
            label=shlex.quote(label),
        )
        try:
            subprocess.run(
                expanded,
                shell=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return True
        except Exception:
            return False

    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action not in ("directions", "eta", "navigate", "route"):
            return {"success": False, "error": f"Unknown action: {action}"}

        destination = str(params.get("destination") or "").strip()
        if not destination:
            return {"success": False, "error": "Where should I navigate to?"}
        destination_label = str(params.get("destination_label") or destination)
        mode = str(params.get("mode") or "driving").lower()

        origin = str(
            params.get("origin") or self.settings.current_location or ""
        ).strip()
        allow_ip = bool(params.get("use_ip_location"))
        origin_fix: Optional[Dict[str, Any]] = None

        if not origin and not allow_ip:
            return {
                "success": False,
                "error": (
                    "Where are you starting from? Give me a city or address."
                ),
            }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                if not origin:
                    location = await self._ip_location(client)
                    if location:
                        origin = f"{location['lat']},{location['lon']}"
                        origin_fix = dict(location)
                        origin_fix["label"] = (
                            f"{location.get('label') or 'your location'} "
                            "(estimated)"
                        )
                        origin_label = origin_fix["label"]
                    else:
                        return {
                            "success": False,
                            "error": (
                                "I couldn't estimate your location. "
                                "Tell me a city or address to start from."
                            ),
                        }
                else:
                    origin_label = origin

                result = None
                if self.settings.google_maps_api_key:
                    result = await self._google_route(
                        client, origin, destination, mode
                    )
                if result is None:
                    result = await self._osrm_route(
                        client, origin, destination, mode, origin_fix
                    )
        except Exception as exc:
            return {"success": False, "error": f"Navigation failed: {exc}"}

        if result is not None and self._google_error:
            result["google_error"] = self._google_error

        if not result:
            message = f"I couldn't find a route to {destination_label}."
            if destination and destination != destination_label:
                message = (
                    f"I couldn't find a route to {destination_label}. "
                    f"I tried the address {destination}. "
                    "Try saying the street number and city again."
                )
            elif destination:
                message = (
                    f"I couldn't find {destination}. "
                    "Try adding the city or checking the street number."
                )
            return {"success": False, "error": message}

        # The public OSRM server only routes by car. Estimate walking, biking,
        # and transit from the route distance/duration instead of returning a
        # car time labelled as a walk.
        if result.get("provider") == "osrm" and mode in {
            "walking",
            "bicycling",
            "transit",
        }:
            meters = result.get("distance_meters")
            car_minutes = result.get("duration_minutes")
            if mode == "walking" and meters:
                result["duration_minutes"] = max(1, round(meters / 1.35 / 60))
            elif mode == "bicycling" and meters:
                result["duration_minutes"] = max(1, round(meters / 4.5 / 60))
            elif mode == "transit" and car_minutes:
                result["duration_minutes"] = max(1, round(car_minutes * 1.4))
            result["estimated"] = True

        distance = self._distance_text(result.get("distance_meters"))
        minutes = result.get("duration_minutes")
        estimated = bool(result.get("estimated"))
        label_lower = str(destination_label or "").strip().lower()
        if label_lower == "home":
            destination_phrase = "get home"
        elif label_lower == "work":
            destination_phrase = "get to work"
        else:
            destination_phrase = f"get to {destination_label}"
        mode_label = {
            "walking": "On foot, ",
            "bicycling": "By bike, ",
            "transit": "By transit, ",
        }.get(mode, "")
        base = "it'll take" if mode_label else "It'll take"
        summary = (
            f"{mode_label}{base} about {minutes} minutes"
            + (" (estimated)" if estimated else "")
            + f" to {destination_phrase}"
        )
        provider_origin = result.get("origin_label")
        if (
            provider_origin
            and provider_origin != origin
            and provider_origin != destination
        ):
            summary += f" from {provider_origin}"
        elif origin_label and origin_label != "your current location":
            summary += f" from {origin_label}"
        if distance:
            summary += f", about {distance}"
        if result.get("via"):
            summary += f" via {result['via']}"
        summary += "."

        launched = await self._launch(origin, destination, destination_label)
        result.update(
            {
                "summary": summary,
                "origin": origin,
                "origin_label": origin_label,
                "destination": destination,
                "destination_label": destination_label,
                "distance": distance,
                "launched": launched,
            }
        )
        return result
