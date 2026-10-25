"""Base STT class."""

from abc import abstractmethod
from typing import Optional, Union
import numpy as np
from ..core.base import Component


class STTBase(Component):
    """Base class for STT engines."""
    
    def _get_enabled(self) -> bool:
        return self.settings.stt_enabled
    
    @abstractmethod
    async def transcribe(self, audio: Union[np.ndarray, bytes, str]) -> Optional[str]:
        """Transcribe audio to text."""
        pass
    
    @abstractmethod
    async def listen(
        self, duration: Optional[float] = None, start_prompt: bool = True
    ) -> Optional[str]:
        """Listen from microphone and transcribe."""
        pass
