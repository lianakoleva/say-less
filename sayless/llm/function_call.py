"""Parsing for model-generated function calls.

FunctionGemma emits calls with dedicated marker tokens::

    <start_function_call>call:create_contact{name:<escape>Mom<escape>,
    phone_number:<escape>1224-879-1662<escape>}<end_function_call>

Some other LiteRT-LM models reply with a JSON object instead::

    {"function_call": {"name": "notes", "arguments": {...}}}

:func:`parse_call` accepts either shape.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

START_FUNCTION_CALL = "<start_function_call>"
END_FUNCTION_CALL = "<end_function_call>"

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_CALL_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _parse_value(text: str, pos: int) -> Tuple[Any, int]:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text):
        raise ValueError("Missing argument value")
    ch = text[pos]
    if ch == "{":
        return _parse_mapping(text, pos)
    if ch == "[":
        return _parse_sequence(text, pos)
    if text.startswith("<escape>", pos):
        end = text.index("<escape>", pos + 8)
        return text[pos + 8 : end], end + 8
    end = pos
    while end < len(text) and text[end] not in ",}]:":
        end += 1
    token = text[pos:end].strip()
    if token == "true":
        return True, end
    if token == "false":
        return False, end
    try:
        return int(token), end
    except ValueError:
        pass
    try:
        return float(token), end
    except ValueError:
        pass
    if token in ("null", "None"):
        return None, end
    raise ValueError(f"Invalid argument value: {token!r}")


def _parse_mapping(text: str, pos: int) -> Tuple[dict, int]:
    result: dict = {}
    pos += 1  # {
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] == "}":
        return result, pos + 1
    while pos < len(text):
        sep = text.index(":", pos)
        key = text[pos:sep].strip()
        value, pos = _parse_value(text, sep + 1)
        if not _CALL_NAME.fullmatch(key) or key in result:
            raise ValueError(f"Invalid or duplicate argument name: {key!r}")
        result[key] = value
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        if text[pos] == ",":
            pos += 1
            continue
        if text[pos] == "}":
            return result, pos + 1
        raise ValueError(
            f"Malformed function call arguments near {text[pos:pos + 20]!r}"
        )
    raise ValueError("Unterminated argument mapping")


def _parse_sequence(text: str, pos: int) -> Tuple[list, int]:
    items: list = []
    pos += 1  # [
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] == "]":
        return items, pos + 1
    while pos < len(text):
        value, pos = _parse_value(text, pos)
        items.append(value)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        if text[pos] == ",":
            pos += 1
            continue
        if text[pos] == "]":
            return items, pos + 1
        raise ValueError(
            f"Malformed function call arguments near {text[pos:pos + 20]!r}"
        )
    raise ValueError("Unterminated argument sequence")


def parse_native_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse a single ``<start_function_call>`` block."""
    if not text or text.count(START_FUNCTION_CALL) != 1:
        return None
    start = text.index(START_FUNCTION_CALL) + len(START_FUNCTION_CALL)
    end = text.find(END_FUNCTION_CALL, start)
    if end == -1:
        end = len(text)
    body = text[start:end].strip()
    if not body.startswith("call:"):
        return None
    name, sep, obj = body[len("call:") :].partition("{")
    name = name.strip()
    if not sep or not _CALL_NAME.fullmatch(name):
        return None
    try:
        arguments, pos = _parse_mapping("{" + obj, 0)
    except (ValueError, RecursionError):
        return None
    # Tolerate trailing chat-template tokens after the closing brace.
    tail = ("{" + obj)[pos:].strip()
    if tail and not tail.startswith("<"):
        return None
    return name, arguments


def _as_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def parse_json_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse a JSON ``function_call`` object, including prose/fenced replies."""
    if not text:
        return None
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
        return str(name), _as_arguments(call.get("arguments", {}))
    return None


def parse_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse native FunctionGemma markers first, then JSON."""
    return parse_native_call(text) or parse_json_call(text)
