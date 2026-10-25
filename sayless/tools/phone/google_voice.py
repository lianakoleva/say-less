"""Google Voice integration."""

from typing import Dict, Any
from ..base import ToolBase


class GoogleVoiceTool(ToolBase):
    """Google Voice tool for calling and texting."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "google_voice"
    
    def requires_internet(self) -> bool:
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "google_voice",
                "description": "Make calls and send texts using Google Voice",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["call", "text", "get_messages"],
                            "description": "Action to perform"
                        },
                        "number": {
                            "type": "string",
                            "description": "Phone number (for call/text)"
                        },
                        "message": {
                            "type": "string",
                            "description": "Message text (for text)"
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Google Voice action."""
        try:
            import os
            if not (os.environ.get("GOOGLE_VOICE_USERNAME") or os.environ.get("GV_USERNAME") or os.environ.get("GOOGLE_VOICE_EMAIL")):
                return {
                    "success": False,
                    "error": "Google Voice is not set up. To use phone/text via Google Voice, set GOOGLE_VOICE_USERNAME and GOOGLE_VOICE_PASSWORD (or use the official Google Voice app). You can also configure it in your environment or .env file. See docs/README for setup."
                }
            if action == "call":
                number = params.get("number") or params.get("contact")
                if not number:
                    return {"success": False, "error": "Number or contact required for call"}
                return {
                    "success": False,
                    "error": "Google Voice is not authenticated. Run 'sayless setup' or set GOOGLE_VOICE_USERNAME and GOOGLE_VOICE_PASSWORD in your environment/.env. Then try again."
                }
            elif action == "text":
                number = params.get("number") or params.get("contact")
                message = params.get("message")
                if not number or not message:
                    return {"success": False, "error": "Number/contact and message required for text"}
                return {
                    "success": False,
                    "error": "Google Voice is not authenticated. Run 'sayless setup' or set GOOGLE_VOICE_USERNAME and GOOGLE_VOICE_PASSWORD in your environment/.env. Then try again."
                }
            elif action == "get_messages":
                return {
                    "success": False,
                    "error": "Google Voice is not authenticated. Run 'sayless setup' or set GOOGLE_VOICE_USERNAME and GOOGLE_VOICE_PASSWORD in your environment/.env."
                }
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
