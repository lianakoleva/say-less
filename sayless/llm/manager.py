"""LLM Manager."""

from typing import Optional, Dict, Any, List, Union
from .base import LLMBase
from ..config import get_settings


class LLMManager:
    """Manage the LiteRT-LM engine."""

    def __init__(self):
        self.settings = get_settings()
        self._engine: Optional[LLMBase] = None

    def _fail(self, error: Optional[str]) -> bool:
        print("\nERROR: The language model could not be initialized.")
        if error:
            print(f"Reason: {error}")
        print("\nTo fix this, run:")
        print("  sayless setup")
        print("\nOr start the LiteRT-LM server manually:")
        print(
            f"  python -m sayless.llm.server --model {self.settings.llm_model_path}"
        )
        print("\nOr set SAYLESS_LLM_ENABLED=false to run without an LLM.\n")
        return False

    async def initialize(self, engine_type: Optional[str] = None) -> bool:
        """Initialize the LLM engine."""
        engine_type = engine_type or self.settings.llm_engine

        if engine_type not in ("litert-lm", "litert", "gemma"):
            return self._fail(
                f"Unknown LLM engine '{engine_type}'. Only 'litert-lm' is supported."
            )

        try:
            from .engines.litert_lm import LiteRTLMLLM

            self._engine = LiteRTLMLLM()
        except Exception as e:
            return self._fail(str(e))

        if not self._engine.enabled:
            return False

        result = await self._engine.initialize()
        if not result:
            return self._fail(getattr(self._engine, "error", None))
        return True

    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        if not self._engine or not self._engine.is_initialized:
            return None
        return await self._engine.generate(prompt, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        if not self._engine or not self._engine.is_initialized:
            return None
        return await self._engine.chat(messages, **kwargs)

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not self._engine or not self._engine.is_initialized:
            return None
        return await self._engine.generate_with_tools(prompt, tools, **kwargs)

    @property
    def enabled(self) -> bool:
        return self._engine.enabled if self._engine else False

    async def cleanup(self):
        if self._engine:
            await self._engine.cleanup()
