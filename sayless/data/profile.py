"""Small persistent notes about the user."""

from typing import Any, Dict, Optional

from .json_store import JSONStore


class UserProfile:
    """User preferences stored in ``~/.sayless/profile/user.json``.

    This is intentionally small: it holds things the assistant has learned
    from the user (for example the default weather city) so it does not have
    to ask again on the next run.
    """

    def __init__(self):
        self.store = JSONStore("profile")

    async def get_all(self) -> Dict[str, Any]:
        data = await self.store.load("user")
        return data if isinstance(data, dict) else {}

    async def get(self, key: str, default: Optional[Any] = None) -> Any:
        data = await self.get_all()
        return data.get(key, default)

    async def set(self, key: str, value: Any) -> bool:
        data = await self.get_all()
        data[key] = value
        return await self.store.save("user", data)

    async def delete(self, key: str) -> bool:
        data = await self.get_all()
        if key not in data:
            return False
        data.pop(key, None)
        return await self.store.save("user", data)
