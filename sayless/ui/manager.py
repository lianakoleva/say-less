"""UI Manager."""

from typing import Optional
from .base import UIBase
from .cli_ui import CLIUI


class UIManager:
    """Manage UI implementation."""
    
    def __init__(self, ui_type: str = "cli"):
        self.ui_type = ui_type
        self._ui: Optional[UIBase] = None
    
    async def initialize(self) -> UIBase:
        """Initialize UI."""
        if self.ui_type == "cli":
            self._ui = CLIUI()
        elif self.ui_type == "tui":
            try:
                from .tui_ui import TUIUI
                self._ui = TUIUI()
            except Exception:
                print("TUI not available, falling back to CLI")
                self._ui = CLIUI()
        else:
            self._ui = CLIUI()
        
        await self._ui.start()
        return self._ui
    
    @property
    def ui(self) -> UIBase:
        if not self._ui:
            raise RuntimeError("UI not initialized")
        return self._ui
    
    async def cleanup(self):
        if self._ui:
            await self._ui.stop()
