"""Base data storage."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class DataStore(ABC):
    """Base class for data storage."""
    
    @abstractmethod
    async def save(self, key: str, data: Any) -> bool:
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        pass
