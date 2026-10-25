"""STT Manager to handle multiple engines."""

from typing import Optional, Dict, Any, Union
import numpy as np
from .base import STTBase
from ..config import get_settings


class STTManager:
    """Manage STT engines."""
    
    def __init__(self):
        self.settings = get_settings()
        self._engine: Optional[STTBase] = None
    
    async def initialize(self, engine_type: Optional[str] = None) -> bool:
        """Initialize STT engine."""
        engine_type = engine_type or self.settings.stt_engine
        
        if engine_type in ("faster-whisper", "faster_whisper"):
            try:
                from .engines.faster_whisper import FasterWhisperSTT
                self._engine = FasterWhisperSTT()
            except ImportError as exc:
                print(
                    f"faster-whisper is not installed ({exc}). "
                    "Run: pip install 'sayless[stt]'"
                )
                return False
        elif engine_type == "dummy":
            from .engines.dummy import DummySTT
            self._engine = DummySTT()
        else:
            print(
                f"Unknown STT engine '{engine_type}'. "
                "Only 'faster-whisper' is supported."
            )
            return False
        
        if not self._engine.enabled:
            return False
        
        return await self._engine.initialize()
    
    async def transcribe(self, audio: Union[np.ndarray, bytes, str]) -> Optional[str]:
        """Transcribe audio."""
        if not self._engine or not self._engine.is_initialized:
            return None
        return await self._engine.transcribe(audio)
    
    async def listen(self, duration: Optional[float] = None, start_prompt: bool = True) -> Optional[str]:
        """Listen from microphone."""
        if not self._engine or not self._engine.is_initialized:
            return None
        return await self._engine.listen(duration, start_prompt=start_prompt)
    
    @property
    def enabled(self) -> bool:
        return self._engine.enabled if self._engine else False
    
    async def cleanup(self):
        if self._engine:
            await self._engine.cleanup()
