"""Queue manager for offline tasks."""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..config import get_settings


class QueueManager:
    """Manage offline task queue in dotfile folder."""
    
    def __init__(self):
        self.settings = get_settings()
        self.queue_dir = self.settings.queue_dir
        self.queue_file = self.queue_dir / "tasks.json"
        self._ensure_queue_file()
    
    def _ensure_queue_file(self):
        """Ensure queue file exists."""
        if not self.queue_file.exists():
            self._write_queue([])
    
    def _read_queue(self) -> List[Dict[str, Any]]:
        """Read queue from file."""
        try:
            with open(self.queue_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _write_queue(self, tasks: List[Dict[str, Any]]):
        """Write queue to file."""
        with open(self.queue_file, 'w') as f:
            json.dump(tasks, f, indent=2, default=str)
    
    async def enqueue(self, tool_name: str, action: str, params: Dict[str, Any], 
                      requires_internet: bool = True, priority: int = 0) -> str:
        """Enqueue a task."""
        tasks = self._read_queue()
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "tool_name": tool_name,
            "action": action,
            "params": params,
            "requires_internet": requires_internet,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "attempts": 0
        }
        tasks.append(task)
        # Sort by priority (higher first)
        tasks.sort(key=lambda x: x.get("priority", 0), reverse=True)
        self._write_queue(tasks)
        return task_id
    
    async def get_pending(self) -> List[Dict[str, Any]]:
        """Get all pending tasks."""
        tasks = self._read_queue()
        return [t for t in tasks if t["status"] == "pending"]
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """Get all tasks."""
        return self._read_queue()
    
    async def mark_completed(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed."""
        tasks = self._read_queue()
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                task["result"] = result
                break
        self._write_queue(tasks)
    
    async def mark_failed(self, task_id: str, error: str):
        """Mark task as failed."""
        tasks = self._read_queue()
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = "failed"
                task["failed_at"] = datetime.now().isoformat()
                task["error"] = error
                task["attempts"] = task.get("attempts", 0) + 1
                break
        self._write_queue(tasks)
    
    async def clear_completed(self):
        """Clear completed tasks."""
        tasks = self._read_queue()
        tasks = [t for t in tasks if t["status"] != "completed"]
        self._write_queue(tasks)
    
    async def clear_all(self):
        """Clear all tasks."""
        self._write_queue([])
    
    def get_queue_path(self) -> Path:
        """Get queue directory path."""
        return self.queue_dir
