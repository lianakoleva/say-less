"""Tool Manager."""

from typing import Dict, Any, List, Optional
from .base import ToolBase
from ..queue import QueueManager


class ToolManager:
    """Manage tools and integrations."""
    
    def __init__(self):
        self.tools: Dict[str, ToolBase] = {}
        self.queue = QueueManager()
    
    def register_tool(self, name: str, tool: ToolBase):
        """Register a tool."""
        self.tools[name] = tool
    
    async def execute_tool(self, tool_name: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool, queuing if it requires internet and offline."""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool {tool_name} not found"}
        
        tool = self.tools[tool_name]
        
        # Check if tool requires internet
        if tool.requires_internet():
            # Check internet connectivity
            if not await self._check_internet():
                # Queue the task
                task_id = await self.queue.enqueue(
                    tool_name=tool_name,
                    action=action,
                    params=params,
                    requires_internet=True
                )
                return {
                    "success": True,
                    "queued": True,
                    "task_id": task_id,
                    "message": "Task queued - will execute when internet is available"
                }
        
        # Execute directly
        try:
            result = await tool.execute(action, params)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _check_internet(self) -> bool:
        """Check internet connectivity."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get("https://www.google.com", follow_redirects=True)
                return response.status_code < 400
        except Exception:
            return False
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for LLM function calling)."""
        schemas = []
        for name, tool in self.tools.items():
            schema = tool.get_schema()
            schema["name"] = name
            schemas.append(schema)
        return schemas
    
    async def process_queue(self):
        """Process queued tasks when internet is available."""
        if not await self._check_internet():
            return
        
        pending_tasks = await self.queue.get_pending()
        for task in pending_tasks:
            try:
                tool_name = task["tool_name"]
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    result = await tool.execute(task["action"], task["params"])
                    await self.queue.mark_completed(task["id"], result)
                else:
                    await self.queue.mark_failed(task["id"], "Tool not found")
            except Exception as e:
                await self.queue.mark_failed(task["id"], str(e))
