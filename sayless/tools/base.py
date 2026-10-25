"""Base tool class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..config import get_settings


class ToolBase(ABC):
    """Base class for tools/integrations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool action."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        pass
    
    @abstractmethod
    def requires_internet(self) -> bool:
        """Check if this tool requires internet."""
        pass
