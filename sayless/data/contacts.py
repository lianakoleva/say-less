"""Contacts management."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from .json_store import JSONStore


class ContactsManager:
    """Manage contacts."""
    
    def __init__(self):
        self.store = JSONStore("contacts")
    
    async def add_contact(self, name: str, phone: Optional[str] = None, 
                         email: Optional[str] = None, **kwargs) -> str:
        """Add a contact."""
        contact_id = name.lower().replace(" ", "_")
        contact = {
            "id": contact_id,
            "name": name,
            "phone": phone,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **kwargs
        }
        await self.store.save(contact_id, contact)
        return contact_id
    
    async def get_contact(self, contact_id: str) -> Optional[Dict[str, Any]]:
        """Get a contact by ID."""
        return await self.store.load(contact_id)
    
    async def get_contact_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a contact by name."""
        contact_id = name.lower().replace(" ", "_")
        return await self.get_contact(contact_id)
    
    async def search_contacts(self, query: str) -> List[Dict[str, Any]]:
        """Search contacts by name, phone, or email."""
        all_contacts = await self.store.load_all()
        query_lower = query.lower()
        results = []
        for contact in all_contacts.values():
            if (query_lower in contact.get("name", "").lower() or
                query_lower in str(contact.get("phone", "")).lower() or
                query_lower in str(contact.get("email", "")).lower()):
                results.append(contact)
        return results
    
    async def list_contacts(self) -> List[Dict[str, Any]]:
        """List all contacts."""
        all_contacts = await self.store.load_all()
        return list(all_contacts.values())
    
    async def update_contact(self, contact_id: str, updates: Dict[str, Any]) -> bool:
        """Update a contact."""
        contact = await self.get_contact(contact_id)
        if not contact:
            return False
        contact.update(updates)
        contact["updated_at"] = datetime.now().isoformat()
        return await self.store.save(contact_id, contact)
    
    async def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact."""
        return await self.store.delete(contact_id)
