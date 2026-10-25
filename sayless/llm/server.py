"""OpenAI-compatible HTTP server for LiteRT-LM (Gemma on CoralBoard).

Run:

    python -m sayless.llm.server
    python -m sayless.llm.server --model /path/to/mobile_actions_q8_ekv1024.litertlm

Only the small subset SayLess needs is implemented:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import litert_lm

DEFAULT_MODEL = "mobile_actions_q8_ekv1024.litertlm"


def _resolve_model_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        local = Path.cwd() / candidate
        if local.is_file():
            return local.resolve()
    raise FileNotFoundError(
        f"LiteRT-LM model not found: {path}\n"
        "Pass --model /path/to/mobile_actions_q8_ekv1024.litertlm "
        "or set SAYLESS_LLM_MODEL_PATH."
    )


def _message_text(message: Any) -> str:
    """Extract plain text from a LiteRT-LM Message."""
    if message is None:
        return ""
    if hasattr(message, "to_json"):
        data = message.to_json()
    elif isinstance(message, dict):
        data = message
    else:
        return str(message)

    content = data.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


class _SchemaTool(litert_lm.Tool):
    """A tool schema with no implementation.

    The server only needs the schema so LiteRT-LM can decode a tool call. The
    actual execution happens on the client, which is why the conversation is
    created with ``automatic_tool_calling=False``.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def get_tool_description(self) -> dict[str, Any]:
        if "function" in self.spec:
            return self.spec
        return {"type": "function", "function": self.spec}

    def execute(self, param: dict[str, Any]) -> str:
        return "Tool execution is handled by the SayLess client."


def _openai_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Normalize LiteRT-LM tool calls to the OpenAI response shape."""
    if not raw_calls:
        return []
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function", call)
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
            }
        )
    return calls


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


class LiteRTEngine:
    """Serialized wrapper around a single LiteRT-LM engine."""

    def __init__(self, model_path: Path, threads: int | None = None) -> None:
        self.litert_lm = litert_lm
        backend = litert_lm.Backend.CPU(thread_count=threads)
        print(f"Loading {model_path} into LiteRT-LM...", flush=True)
        self.engine = litert_lm.Engine(str(model_path), backend=backend)
        self.model_path = model_path
        self.model_name = model_path.name
        self.lock = threading.Lock()
        print("LiteRT-LM engine ready.", flush=True)

    def _history(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Convert OpenAI messages to LiteRT-LM message dicts."""
        system_parts: list[str] = []
        history: list[dict[str, Any]] = []
        call_names: dict[str, str] = {}

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                system_parts.append(_text_of(content))
                continue

            if role == "tool":
                name = message.get("name") or call_names.get(
                    str(message.get("tool_call_id", "")), ""
                )
                history.append(
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_response",
                                "name": name,
                                "response": content,
                            }
                        ],
                    }
                )
                continue

            entry: dict[str, Any] = {
                "role": "assistant" if role in ("assistant", "model") else "user",
                "content": _text_of(content),
            }
            tool_calls = message.get("tool_calls")
            if tool_calls:
                entry["tool_calls"] = tool_calls
                for call in tool_calls:
                    if isinstance(call, dict):
                        function = call.get("function", {})
                        if isinstance(function, dict):
                            call_names[str(call.get("id", ""))] = function.get(
                                "name", ""
                            )
            history.append(entry)

        return system_parts, history

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        litert_lm = self.litert_lm

        system_parts, history = self._history(messages)

        if not history:
            raise ValueError("messages must include at least one user message")

        prompt = history[-1]
        prior = history[:-1]

        sampler = litert_lm.SamplerConfig(
            temperature=temperature, top_p=top_p, top_k=top_k
        )
        tool_objects = [_SchemaTool(tool) for tool in (tools or [])]

        with self.lock:
            conversation = self.engine.create_conversation(
                messages=prior or None,
                system_message="\n".join(p for p in system_parts if p) or None,
                tools=tool_objects or None,
                automatic_tool_calling=False,
                sampler_config=sampler,
                max_output_tokens=max_tokens,
            )
            with conversation:
                response = conversation.send_message(prompt)
                data = (
                    response.to_json()
                    if hasattr(response, "to_json")
                    else dict(response)
                )

        return {
            "content": _message_text(response).strip(),
            "tool_calls": _openai_tool_calls(data.get("tool_calls")),
        }


class Handler(BaseHTTPRequestHandler):
    engine: LiteRTEngine

    server_version = "SayLessLiteRT/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/healthz"):
            self._send(
                200,
                {"status": "ok", "model": self.engine.model_name},
            )
            return
        if path in ("/v1/models", "/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.model_name,
                            "object": "model",
                            "owned_by": "sayless",
                        }
                    ],
                },
            )
            return
        self._send(404, {"error": {"message": f"unknown path: {path}"}})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": f"unknown path: {path}"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            messages = request.get("messages") or []
            if not messages:
                raise ValueError("no messages provided")

            result = self.engine.chat(
                messages,
                tools=request.get("tools") or None,
                temperature=float(request.get("temperature", 0.7)),
                top_p=float(request.get("top_p", 0.95)),
                top_k=int(request.get("top_k", 64)),
                max_tokens=int(request.get("max_tokens", 512)),
            )
            tool_calls = result["tool_calls"]
            message: dict[str, Any] = {
                "role": "assistant",
                "content": result["content"],
            }
            if tool_calls:
                message["tool_calls"] = tool_calls
            self._send(
                200,
                {
                    "id": "chatcmpl-litert",
                    "object": "chat.completion",
                    "model": request.get("model", self.engine.model_name),
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool_calls else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": -1,
                        "completion_tokens": -1,
                        "total_tokens": -1,
                    },
                },
            )
        except Exception as exc:
            self._send(500, {"error": {"message": str(exc)}})


def main() -> None:
    from ..config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="SayLess LiteRT-LM server")
    parser.add_argument(
        "--model",
        default=str(settings.llm_model_path),
        help="Path to the .litertlm model",
    )
    parser.add_argument("--host", default=settings.llm_server_host)
    parser.add_argument("--port", type=int, default=settings.llm_server_port)
    parser.add_argument(
        "--threads",
        type=int,
        default=settings.llm_num_threads,
        help="CPU threads for LiteRT-LM",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model)
    engine = LiteRTEngine(model_path, threads=args.threads)

    Handler.engine = engine
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    def _stop(*_: Any) -> None:
        # BaseServer.shutdown() must run in another thread; calling it from a
        # signal handler on the serve_forever() thread deadlocks.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(
        f"LiteRT-LM server listening on http://{args.host}:{args.port}/v1 "
        f"({engine.model_name})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        engine.engine.close()


if __name__ == "__main__":
    main()
