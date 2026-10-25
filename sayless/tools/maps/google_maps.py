"""Google Maps integration."""

from typing import Dict, Any
from ..base import ToolBase


class GoogleMapsTool(ToolBase):
    """Google Maps tool."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "google_maps"
        self.api_key = self.settings.google_maps_api_key
    
    def requires_internet(self) -> bool:
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "google_maps",
                "description": "Get directions, search places, and get maps information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["directions", "search", "geocode", "reverse_geocode"],
                            "description": "Action to perform"
                        },
                        "origin": {
                            "type": "string",
                            "description": "Starting location (for directions)"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination location (for directions)"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query (for search)"
                        },
                        "address": {
                            "type": "string",
                            "description": "Address (for geocode)"
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Google Maps action."""
        try:
            if action == "directions":
                origin = params.get("origin")
                destination = params.get("destination")
                if not origin or not destination:
                    return {"success": False, "error": "Origin and destination required"}
                # Try to use googlemaps library
                try:
                    import googlemaps
                    if not self.api_key:
                        return {"success": False, "error": "Google Maps API key not configured"}
                    gmaps = googlemaps.Client(key=self.api_key)
                    directions = gmaps.directions(origin, destination)
                    return {"success": True, "directions": directions}
                except ImportError:
                    return {"success": False, "error": "googlemaps library not installed. Install with: pip install googlemaps"}
            elif action == "search":
                query = params.get("query")
                if not query:
                    return {"success": False, "error": "Query required"}
                try:
                    import googlemaps
                    if not self.api_key:
                        return {"success": False, "error": "Google Maps API key not configured"}
                    gmaps = googlemaps.Client(key=self.api_key)
                    places = gmaps.places(query)
                    return {"success": True, "places": places}
                except ImportError:
                    return {"success": False, "error": "googlemaps library not installed"}
            elif action == "geocode":
                address = params.get("address")
                if not address:
                    return {"success": False, "error": "Address required"}
                try:
                    import googlemaps
                    if not self.api_key:
                        return {"success": False, "error": "Google Maps API key not configured"}
                    gmaps = googlemaps.Client(key=self.api_key)
                    result = gmaps.geocode(address)
                    return {"success": True, "result": result}
                except ImportError:
                    return {"success": False, "error": "googlemaps library not installed"}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
