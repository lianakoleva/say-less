"""Base classes for SayLess components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..config import get_settings


class Component(ABC):
    """Base class for all components (STT, LLM)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()
        self._enabled = self._get_enabled()
        self._initialized = False
    
    @property
    def enabled(self) -> bool:
        """Check if component is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable component."""
        self._enabled = value
    
    @abstractmethod
    def _get_enabled(self) -> bool:
        """Get whether this component is enabled from settings."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component. Returns True if successful."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized
