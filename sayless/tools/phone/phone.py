"""Phone app integration."""
from typing import Dict, Any
from ..base import ToolBase


class PhoneTool(ToolBase):
    """Phone app for making calls."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "phone"

    def requires_internet(self) -> bool:
        return True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "phone",
                "description": "Make phone calls",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["call", "dial"],
                            "description": "Action to perform",
                        },
                        "number": {
                            "type": "string",
                            "description": "Phone number to call",
                        },
                        "contact": {
                            "type": "string",
                            "description": "Contact name to call",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if action in ["call", "dial"]:
                number = params.get("number") or params.get("contact")
                if not number:
                    return {"success": False, "error": "Number or contact required"}
                return {"success": False, "error": "Phone app integration not configured"}
            return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
