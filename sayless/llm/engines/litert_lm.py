"""LiteRT-LM LLM engine.

Talks to the local OpenAI-compatible LiteRT-LM server that runs Gemma on the
CoralBoard CPU. Start it with:

    python -m sayless.llm.server

FunctionGemma models (such as ``mobile_actions_q8_ekv1024.litertlm``) receive
their tool declarations through LiteRT-LM and reply with native
``<start_function_call>`` tokens, which the server turns into OpenAI-style
``tool_calls``. Older chat models without function-calling support fall back to
a JSON prompt:

    {"function_call": {"name": "...", "arguments": {...}}}
"""

import json
import re
from typing import Optional, Dict, Any, List, Union

import httpx

from ..base import LLMBase
from ..function_call import parse_call
from ...tools.model_tools import DEVELOPER_INSTRUCTION

JSON_FUNCTION_CALL_INSTRUCTIONS = """You are a local voice assistant with access to tools.
When the user requests an action, respond with ONLY a JSON object in this exact format:
{"function_call": {"name": "function_name", "arguments": {"arg_name": "arg_value"}}}

Rules:
- Output raw JSON only. No markdown, no code fences, no extra text.
- Use a function from the available list and fill its arguments.
- If no function is needed, reply normally in plain English."""

_SPECIAL_TOKENS = re.compile(
    r"<end_of_turn>|<start_of_turn>|<eos>|<bos>|<end_function_call>|"
    r"<start_function_call>|<start_function_response>|<end_function_response>"
)


class LiteRTLMLLM(LLMBase):
    """Client for the local LiteRT-LM OpenAI-compatible server."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.client: Optional[httpx.AsyncClient] = None
        self.model = self.config.get("model") or self.settings.llm_model_name
        self.error: Optional[str] = None
        # None means the server did not report a capability; assume native
        # function calling since FunctionGemma is the target model.
        self.supports_tools: Optional[bool] = None

    @property
    def base_url(self) -> str:
        return self.settings.llm_base_url.rstrip("/")

    async def initialize(self) -> bool:
        """Verify that the LiteRT-LM server is reachable."""
        self.error = None
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "supports_tools" in data:
                    self.supports_tools = bool(data["supports_tools"])
            else:
                # Some server builds only expose /health at the root.
                root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
                response = await self.client.get(f"{root}/health")
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "supports_tools" in data:
                        self.supports_tools = bool(data["supports_tools"])
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
        return self._clean(message.get("content") or "")

    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    @staticmethod
    def _clean(text: str) -> str:
        """Drop chat-template special tokens from model text."""
        return _SPECIAL_TOKENS.sub("", str(text)).strip()

    # ── JSON fallback for models without native function calling ────────────

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
            f"{JSON_FUNCTION_CALL_INSTRUCTIONS}\n\n"
            f"Available functions:\n{cls._describe_tools(tools)}\n\n"
            f"User request: {cls._last_user_text(messages)}\n\n"
            "Respond now."
        )

    @staticmethod
    def _parse_function_call(text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Parse a native marker block or a JSON ``function_call``."""
        return parse_call(text)

    @staticmethod
    def _valid_names(tools: List[Dict[str, Any]]) -> set:
        names = set()
        for tool in tools:
            function = tool.get("function", tool)
            if isinstance(function, dict) and function.get("name"):
                names.add(function["name"])
        return names

    async def _json_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Prompt-driven JSON calls for chat models without tool support."""
        tool_prompt = self._tool_prompt(messages, tools)
        message = await self._complete(
            [{"role": "user", "content": tool_prompt}], **kwargs
        )
        if message is None:
            return None

        text = self._clean(message.get("content") or "")
        call = parse_call(text)
        if call is None or call[0] not in self._valid_names(tools):
            # The model sometimes answers non-tool questions with JSON anyway.
            # Retry without the tool prompt so the user gets a normal reply.
            if "function_call" in text or text.lstrip().startswith("{"):
                plain = await self.chat(messages, **kwargs)
                if plain:
                    text = plain
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

    @staticmethod
    def _looks_like_tool_refusal(text: str) -> bool:
        """FunctionGemma refuses when it sees no tool declarations."""
        lower = text.lower()
        return (
            "i am functiongemma" in lower
            or ("only assist" in lower and "function call" in lower)
            or ("i can only" in lower and "function" in lower)
        )

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate a reply and return any requested tool calls."""
        messages = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": str(prompt)}
        ]

        # FunctionGemma refuses action requests unless the developer
        # instruction is present. Callers normally don't set a system message.
        if not any(m.get("role") == "system" for m in messages):
            messages = [
                {"role": "system", "content": DEVELOPER_INSTRUCTION},
                *messages,
            ]

        # Native path: LiteRT-LM renders the FunctionGemma developer block from
        # the tool schemas and parses the model's call markers for us.
        message = await self._complete(messages, tools=tools, **kwargs)
        if message is None:
            return None

        text = self._clean(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return {"response": "", "tool_calls": tool_calls}

        # Some LiteRT-LM builds return the raw call tokens as content. Parse
        # before stripping special tokens so the markers are still present.
        call = parse_call(message.get("content") or "")
        if call is not None and call[0] in self._valid_names(tools):
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

        # A model without native function calling either answers normally or
        # ignores the schemas. Fall back to the JSON prompt when the reply is
        # empty or FunctionGemma's "no functions declared" refusal.
        if not text or self._looks_like_tool_refusal(text):
            fallback = await self._json_tool_call(messages, tools, **kwargs)
            if fallback and (fallback.get("tool_calls") or fallback.get("response")):
                return fallback

        return {"response": text, "tool_calls": []}
