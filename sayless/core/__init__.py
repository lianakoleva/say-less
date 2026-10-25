"""Core functionality for SayLess."""

from .base import Component

__all__ = ["Assistant", "Component"]


def __getattr__(name: str):
    """Import Assistant lazily to avoid circular imports with llm/stt."""
    if name == "Assistant":
        from .assistant import Assistant

        return Assistant
    raise AttributeError(name)
