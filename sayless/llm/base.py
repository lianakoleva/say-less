"""Base LLM class."""

from abc import abstractmethod
from typing import Optional, Dict, Any, List, Union
from ..core.base import Component


class LLMBase(Component):
    """Base class for LLM engines."""
    
    def _get_enabled(self) -> bool:
        return self.settings.llm_enabled
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Optional[str]:
        """Chat with message history."""
        pass
    
    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate response with tool calling."""
        pass
