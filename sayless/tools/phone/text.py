"""Text app integration."""
from typing import Dict, Any
from ..base import ToolBase
from .actions import run_phone_action


class TextTool(ToolBase):
    """Text app for sending messages."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "text"

    def requires_internet(self) -> bool:
        # Texts are handled by a local command or the local log.
        return False

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "text",
                "description": "Send text messages",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["send", "message"],
                            "description": "Action to perform",
                        },
                        "number": {
                            "type": "string",
                            "description": "Phone number to text",
                        },
                        "contact": {
                            "type": "string",
                            "description": "Contact name to text",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message text",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if action in ["send", "message"]:
                number = params.get("number") or params.get("contact")
                message = params.get("message")
                if not number or not message:
                    return {"success": False, "error": "Number/contact and message required"}
                return await run_phone_action("text", number, message)
            return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
