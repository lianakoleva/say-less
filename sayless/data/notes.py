"""Notes management."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from .json_store import JSONStore


class NotesManager:
    """Manage notes."""
    
    def __init__(self):
        self.store = JSONStore("notes")
    
    async def create_note(self, title: str, content: str, tags: Optional[List[str]] = None) -> str:
        """Create a note."""
        note_id = str(uuid.uuid4())
        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        await self.store.save(note_id, note)
        return note_id
    
    async def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Get a note by ID."""
        return await self.store.load(note_id)
    
    async def search_notes(self, query: str) -> List[Dict[str, Any]]:
        """Search notes by title or content."""
        all_notes = await self.store.load_all()
        query_lower = query.lower()
        results = []
        for note in all_notes.values():
            if (query_lower in note.get("title", "").lower() or
                query_lower in note.get("content", "").lower() or
                any(query_lower in tag.lower() for tag in note.get("tags", []))):
                results.append(note)
        return results
    
    async def list_notes(self) -> List[Dict[str, Any]]:
        """List all notes."""
        all_notes = await self.store.load_all()
        # Sort by updated_at descending
        notes = list(all_notes.values())
        notes.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return notes
    
    async def update_note(self, note_id: str, updates: Dict[str, Any]) -> bool:
        """Update a note."""
        note = await self.get_note(note_id)
        if not note:
            return False
        note.update(updates)
        note["updated_at"] = datetime.now().isoformat()
        return await self.store.save(note_id, note)
    
    async def delete_note(self, note_id: str) -> bool:
        """Delete a note."""
        return await self.store.delete(note_id)
    
    async def get_notes_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get notes by tag."""
        all_notes = await self.store.load_all()
        results = []
        for note in all_notes.values():
            if tag in note.get("tags", []):
                results.append(note)
        return results
