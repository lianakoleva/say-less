"""Contacts tool for LLM."""

from typing import Dict, Any
from ..base import ToolBase
from ...data.contacts import ContactsManager


class ContactsTool(ToolBase):
    """Contacts management tool."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "contacts"
        self.contacts = ContactsManager()
    
    def requires_internet(self) -> bool:
        return False
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "contacts",
                "description": "Manage contacts - add, search, list, get contact information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "get", "search", "list", "update", "delete"],
                            "description": "Action to perform"
                        },
                        "name": {
                            "type": "string",
                            "description": "Contact name"
                        },
                        "phone": {
                            "type": "string",
                            "description": "Phone number"
                        },
                        "email": {
                            "type": "string",
                            "description": "Email address"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "contact_id": {
                            "type": "string",
                            "description": "Contact ID"
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if action == "add":
                name = params.get("name")
                if not name:
                    return {"success": False, "error": "Name required"}
                contact_id = await self.contacts.add_contact(
                    name=name,
                    phone=params.get("phone"),
                    email=params.get("email")
                )
                return {"success": True, "contact_id": contact_id}
            elif action == "get":
                name = params.get("name")
                contact_id = params.get("contact_id")
                if name:
                    contact = await self.contacts.get_contact_by_name(name)
                elif contact_id:
                    contact = await self.contacts.get_contact(contact_id)
                else:
                    return {"success": False, "error": "Name or contact_id required"}
                if not contact:
                    return {"success": False, "error": "Contact not found"}
                return {"success": True, "contact": contact}
            elif action == "search":
                query = params.get("query") or params.get("name")
                if not query:
                    return {"success": False, "error": "Query required"}
                results = await self.contacts.search_contacts(query)
                return {"success": True, "results": results, "count": len(results)}
            elif action == "list":
                contacts_list = await self.contacts.list_contacts()
                return {"success": True, "contacts": contacts_list, "count": len(contacts_list)}
            elif action == "update":
                contact_id = params.get("contact_id")
                if not contact_id:
                    return {"success": False, "error": "contact_id required"}
                updates = {k: v for k, v in params.items() if k not in ["action", "contact_id"]}
                success = await self.contacts.update_contact(contact_id, updates)
                return {"success": success}
            elif action == "delete":
                contact_id = params.get("contact_id")
                if not contact_id:
                    return {"success": False, "error": "contact_id required"}
                success = await self.contacts.delete_contact(contact_id)
                return {"success": success}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
