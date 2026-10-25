"""Notes tool for LLM."""

from typing import Dict, Any, List
from ..base import ToolBase
from ...data.notes import NotesManager


class NotesTool(ToolBase):
    """Notes management tool."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "notes"
        self.notes = NotesManager()
    
    def requires_internet(self) -> bool:
        return False
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "notes",
                "description": "Manage notes - create, read, search, update, delete notes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "get", "search", "list", "update", "delete", "get_by_tag"],
                            "description": "Action to perform"
                        },
                        "title": {
                            "type": "string",
                            "description": "Note title"
                        },
                        "content": {
                            "type": "string",
                            "description": "Note content"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Note tags"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "note_id": {
                            "type": "string",
                            "description": "Note ID"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Tag to filter by"
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if action == "create":
                title = params.get("title")
                content = params.get("content", "")
                if not title:
                    return {"success": False, "error": "Title required"}
                note_id = await self.notes.create_note(title, content, params.get("tags"))
                return {"success": True, "note_id": note_id}
            elif action == "get":
                note_id = params.get("note_id")
                if not note_id:
                    return {"success": False, "error": "note_id required"}
                note = await self.notes.get_note(note_id)
                if not note:
                    return {"success": False, "error": "Note not found"}
                return {"success": True, "note": note}
            elif action == "search":
                query = params.get("query")
                if not query:
                    return {"success": False, "error": "Query required"}
                results = await self.notes.search_notes(query)
                return {"success": True, "results": results, "count": len(results)}
            elif action == "list":
                notes_list = await self.notes.list_notes()
                return {"success": True, "notes": notes_list, "count": len(notes_list)}
            elif action == "get_by_tag":
                tag = params.get("tag")
                if not tag:
                    return {"success": False, "error": "Tag required"}
                results = await self.notes.get_notes_by_tag(tag)
                return {"success": True, "results": results, "count": len(results)}
            elif action == "update":
                note_id = params.get("note_id")
                if not note_id:
                    return {"success": False, "error": "note_id required"}
                updates = {k: v for k, v in params.items() if k not in ["action", "note_id"]}
                success = await self.notes.update_note(note_id, updates)
                return {"success": success}
            elif action == "delete":
                note_id = params.get("note_id")
                if not note_id:
                    return {"success": False, "error": "note_id required"}
                success = await self.notes.delete_note(note_id)
                return {"success": success}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
