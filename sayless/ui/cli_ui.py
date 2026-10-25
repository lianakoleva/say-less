"""CLI UI implementation."""

from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from .base import UIBase


class CLIUI(UIBase):
    """CLI UI implementation using Rich."""
    
    def __init__(self):
        super().__init__()
        self.console = Console()
        self._running = False
    
    async def start(self) -> None:
        self._running = True
    
    async def stop(self) -> None:
        self._running = False
    
    async def show_message(self, message: str, role: str = "system") -> None:
        if role == "user":
            self.console.print(f"[bold cyan]You:[/bold cyan] {message}")
        elif role == "assistant":
            self.console.print(Panel(message, title="[bold green]Assistant[/bold green]", border_style="green"))
        elif role == "system":
            self.console.print(f"[dim]{message}[/dim]")
        else:
            self.console.print(message)
    
    async def get_input(self, prompt: str = "") -> Optional[str]:
        try:
            if prompt:
                return self.console.input(f"[bold cyan]{prompt}[/bold cyan] ").strip()
            else:
                return input().strip()
        except (EOFError, KeyboardInterrupt):
            return None
    
    async def show_status(self, status: str) -> None:
        self.console.print(f"[yellow]Status:[/yellow] {status}")
    
    async def show_list(self, items: List[Dict[str, Any]], title: str = "") -> None:
        if not items:
            self.console.print("[yellow]No items to display[/yellow]")
            return
        
        if title:
            self.console.print(f"\n[bold]{title}[/bold]")
        
        table = Table()
        # Add columns based on first item
        if items:
            for key in items[0].keys():
                if key not in ['metadata']:  # Skip complex nested
                    table.add_column(key.replace('_', ' ').title(), style="cyan")
        
        for item in items:
            row = []
            for key in items[0].keys():
                if key not in ['metadata']:
                    val = item.get(key, "")
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    elif isinstance(val, str) and len(val) > 50:
                        val = val[:47] + "..."
                    row.append(str(val))
            table.add_row(*row)
        
        self.console.print(table)
    
    async def show_error(self, error: str) -> None:
        self.console.print(f"[bold red]Error:[/bold red] {error}")
    
    async def show_panel(self, content: str, title: str = ""):
        self.console.print(Panel(content, title=title if title else None))
    
    async def clear(self):
        self.console.clear()
