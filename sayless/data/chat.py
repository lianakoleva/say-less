"""Chat history management."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from .json_store import JSONStore


class ChatManager:
    """Manage chat conversations."""
    
    def __init__(self):
        self.store = JSONStore("chats")
    
    async def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation."""
        conv_id = str(uuid.uuid4())
        conversation = {
            "id": conv_id,
            "title": title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        await self.store.save(conv_id, conversation)
        return conv_id
    
    async def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation."""
        return await self.store.load(conv_id)
    
    async def add_message(self, conv_id: str, role: str, content: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a message to a conversation."""
        conversation = await self.get_conversation(conv_id)
        if not conversation:
            return False
        
        message = {
            "id": str(uuid.uuid4()),
            "role": role,  # user, assistant, system
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        conversation["messages"].append(message)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Auto-generate title from first user message if not set
        if conversation["title"].startswith("Conversation") and role == "user" and len(conversation["messages"]) == 1:
            # Take first 50 chars as title
            conversation["title"] = content[:50] + ("..." if len(content) > 50 else "")
        
        return await self.store.save(conv_id, conversation)
    
    async def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        all_chats = await self.store.load_all()
        conversations = []
        for conv in all_chats.values():
            # Return summary without full messages
            conversations.append({
                "id": conv["id"],
                "title": conv["title"],
                "message_count": len(conv.get("messages", [])),
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"]
            })
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations
    
    async def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        return await self.store.delete(conv_id)
    
    async def clear_conversation(self, conv_id: str) -> bool:
        """Clear messages in a conversation."""
        conversation = await self.get_conversation(conv_id)
        if not conversation:
            return False
        conversation["messages"] = []
        conversation["updated_at"] = datetime.now().isoformat()
        return await self.store.save(conv_id, conversation)
    
    async def get_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """Get messages from a conversation."""
        conversation = await self.get_conversation(conv_id)
        if not conversation:
            return []
        return conversation.get("messages", [])
    
    async def export_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Export full conversation."""
        return await self.get_conversation(conv_id)
