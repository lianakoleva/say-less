"""Base UI interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class UIBase(ABC):
    """Base class for UI implementations (CLI, TUI, etc.)."""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def start(self) -> None:
        """Start the UI."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the UI."""
        pass
    
    @abstractmethod
    async def show_message(self, message: str, role: str = "system") -> None:
        """Show a message."""
        pass
    
    @abstractmethod
    async def get_input(self, prompt: str = "") -> Optional[str]:
        """Get user input."""
        pass
    
    @abstractmethod
    async def show_status(self, status: str) -> None:
        """Show status information."""
        pass
    
    @abstractmethod
    async def show_list(self, items: List[Dict[str, Any]], title: str = "") -> None:
        """Show a list of items."""
        pass
    
    @abstractmethod
    async def show_error(self, error: str) -> None:
        """Show error message."""
        pass
