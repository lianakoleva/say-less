"""TUI UI implementation (placeholder for future)."""

from typing import Optional, Dict, Any, List
from .base import UIBase


class TUIUI(UIBase):
    """TUI UI implementation - placeholder."""
    
    def __init__(self):
        super().__init__()
        # TODO: Implement with textual or other TUI framework
    
    async def start(self) -> None:
        raise NotImplementedError("TUI not yet implemented")
    
    async def stop(self) -> None:
        pass
    
    async def show_message(self, message: str, role: str = "system") -> None:
        pass
    
    async def get_input(self, prompt: str = "") -> Optional[str]:
        pass
    
    async def show_status(self, status: str) -> None:
        pass
    
    async def show_list(self, items: List[Dict[str, Any]], title: str = "") -> None:
        pass
    
    async def show_error(self, error: str) -> None:
        pass
