"""Dummy STT engine for testing."""

from typing import Optional, Union
import numpy as np
from ..base import STTBase


class DummySTT(STTBase):
    """Dummy STT engine that returns placeholder text."""
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def cleanup(self) -> None:
        pass
    
    async def transcribe(self, audio: Union[np.ndarray, bytes, str]) -> Optional[str]:
        """Return dummy transcription."""
        return "hello sayless"
    
    async def listen(self, duration: Optional[float] = None, start_prompt: bool = True) -> Optional[str]:
        """Return dummy transcription."""
        return "hello sayless"
