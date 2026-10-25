"""Dummy LLM engine."""

from typing import Optional, Dict, Any, List, Union
from ..base import LLMBase


class DummyLLM(LLMBase):
    """Dummy LLM engine."""
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def cleanup(self) -> None:
        pass
    
    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        return f"I heard you say: {prompt}. How can I help?"
    
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Optional[str]:
        if messages:
            last = messages[-1].get("content", "")
            return f"I heard you say: {last}. How can I help?"
        return "Hello! How can I help you?"
    
    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        return {
            "response": f"I heard you say: {prompt}. I can help with that.",
            "tool_calls": []
        }
