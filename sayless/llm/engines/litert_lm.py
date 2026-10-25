"""LiteRT-LM LLM engine.

Talks to the local OpenAI-compatible LiteRT-LM server that runs Gemma on the
CoralBoard CPU. Start it with:

    python -m sayless.llm.server

The ``mobile_actions`` LiteRT-LM model does not use native OpenAI tool calling.
Instead the tools are described in the prompt and the model replies with:

    {"function_call": {"name": "...", "arguments": {...}}}

This module builds that prompt and parses the reply.
"""

import json
import re
from typing import Optional, Dict, Any, List, Union

import httpx

from ..base import LLMBase


FUNCTION_CALL_INSTRUCTIONS = """You are a local voice assistant with access to tools.
When the user requests an action, respond with ONLY a JSON object in this exact format:
{"function_call": {"name": "function_name", "arguments": {"arg_name": "arg_value"}}}

Rules:
- Output raw JSON only. No markdown, no code fences, no extra text.
- Use a function from the available list and fill its arguments.
- If no function is needed, reply normally in plain English."""

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class LiteRTLMLLM(LLMBase):
    """Client for the local LiteRT-LM OpenAI-compatible server."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.client: Optional[httpx.AsyncClient] = None
        self.model = self.config.get("model") or self.settings.llm_model_name
        self.error: Optional[str] = None

    @property
    def base_url(self) -> str:
        return self.settings.llm_base_url.rstrip("/")

    async def initialize(self) -> bool:
        """Verify that the LiteRT-LM server is reachable."""
        self.error = None
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

        try:
            response = await self.client.get("/models")
            if response.status_code == 404:
                # Some server builds only expose /health at the root.
                root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
                response = await self.client.get(f"{root}/health")
            response.raise_for_status()
        except Exception:
            self.error = (
                f"Could not reach the LiteRT-LM server at {self.base_url}. "
                "Start it with: python -m sayless.llm.server"
            )
            await self.cleanup()
            return False

        self._initialized = True
        return True

    async def cleanup(self) -> None:
        if self.client is not None:
            await self.client.aclose()
        self.client = None
        self._initialized = False

    def _payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.settings.llm_temperature),
            "top_p": kwargs.get("top_p", self.settings.llm_top_p),
            "top_k": kwargs.get("top_k", self.settings.llm_top_k),
            "max_tokens": kwargs.get("max_tokens", self.settings.llm_max_tokens),
        }
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools
        return payload

    async def _complete(self, messages: List[Dict[str, Any]], **kwargs) -> Optional[Dict[str, Any]]:
        """Send a chat request and return the assistant message."""
        if not self.client or not self._initialized:
            return None
        try:
            response = await self.client.post(
                "/chat/completions", json=self._payload(messages, **kwargs)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]
        except Exception as e:
            print(f"LLM chat error: {e}")
            return None

    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Optional[str]:
        message = await self._complete(messages, **kwargs)
        if message is None:
            return None
        return message.get("content") or ""

    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    @staticmethod
    def _describe_tools(tools: List[Dict[str, Any]]) -> str:
        lines = []
        for tool in tools:
            function = tool.get("function", tool)
            if not isinstance(function, dict):
                continue
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters") or {}
            properties = parameters.get("properties") or {}
            required = set(parameters.get("required") or [])
            args = ", ".join(
                f"{key}{'' if key in required else '=None'}: "
                f"{spec.get('type', 'string') if isinstance(spec, dict) else 'string'}"
                for key, spec in properties.items()
            )
            lines.append(f"- {name}({args}): {description}")
        return "\n".join(lines)

    @staticmethod
    def _last_user_text(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role", "user") not in ("user", "tool"):
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if str(content).strip():
                return str(content).strip()
        return ""

    @classmethod
    def _tool_prompt(cls, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
        return (
            f"{FUNCTION_CALL_INSTRUCTIONS}\n\n"
            f"Available functions:\n{cls._describe_tools(tools)}\n\n"
            f"User request: {cls._last_user_text(messages)}\n\n"
            "Respond now."
        )

    @staticmethod
    def _parse_function_call(text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Parse the model's JSON function call, if any."""
        candidates = [text.strip()]
        candidates.extend(match.strip() for match in _FENCED_JSON.findall(text))
        match = _JSON_OBJECT.search(text)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue

            call = data.get("function_call", data)
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if not name:
                continue
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            return str(name), arguments

        return None

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate a reply and parse the model's JSON ``function_call``."""
        messages = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": str(prompt)}
        ]
        tool_prompt = self._tool_prompt(messages, tools)
        message = await self._complete(
            [{"role": "user", "content": tool_prompt}], **kwargs
        )
        if message is None:
            return None

        text = (message.get("content") or "").strip()
        call = self._parse_function_call(text)

        valid_names = {
            tool.get("function", tool).get("name")
            for tool in tools
            if isinstance(tool.get("function", tool), dict)
        }
        if call is None or call[0] not in valid_names:
            # The model sometimes answers non-tool questions with JSON anyway.
            # Retry without the tool prompt so the user gets a normal reply.
            if "function_call" in text or text.lstrip().startswith("{"):
                plain = await self.chat(messages, **kwargs)
                if plain and plain.strip():
                    text = plain.strip()
            return {"response": text, "tool_calls": []}

        name, arguments = call
        return {
            "response": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }
