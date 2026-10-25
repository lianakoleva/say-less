"""JSON file storage."""

import json
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import DataStore
from ..config import get_settings


class JSONStore(DataStore):
    """JSON file-based storage."""
    
    def __init__(self, subdir: str = "data"):
        self.settings = get_settings()
        self.store_dir = self.settings.data_dir / subdir
        self.store_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        # Sanitize key
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.store_dir / f"{safe_key}.json"
    
    async def save(self, key: str, data: Any) -> bool:
        try:
            path = self._get_path(key)
            async with aiofiles.open(path, 'w') as f:
                await f.write(json.dumps(data, indent=2, default=str))
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        try:
            path = self._get_path(key)
            if not path.exists():
                return None
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
            return json.loads(content)
        except Exception as e:
            print(f"Load error: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        try:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False
    
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        try:
            keys = []
            for path in self.store_dir.glob("*.json"):
                key = path.stem
                if prefix is None or key.startswith(prefix):
                    keys.append(key)
            return keys
        except Exception:
            return []
    
    async def load_all(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Load all items with optional prefix."""
        keys = await self.list_keys(prefix)
        result = {}
        for key in keys:
            data = await self.load(key)
            if data is not None:
                result[key] = data
        return result
