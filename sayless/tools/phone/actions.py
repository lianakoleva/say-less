"""Local phone/text provider.

CoralBoard has no cellular modem, so calls and texts are handled in one of two
ways:

* ``SAYLESS_PHONE_CALL_COMMAND`` / ``SAYLESS_PHONE_TEXT_COMMAND`` run an
  external app or script. ``{number}`` and ``{message}`` are substituted.
* Otherwise the action is logged to ``~/.sayless/phone.log`` and reported as
  simulated, so the full assistant flow can still be tested.
"""

import asyncio
import json
import re
import shlex
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

from ...config import get_settings


def _validate_number(number: Optional[str]) -> Optional[str]:
    if not number:
        return None
    value = str(number).strip()
    if len(re.sub(r"\D", "", value)) < 7:
        return None
    return value


def _run_command(command: str, number: str, message: Optional[str]) -> Dict[str, Any]:
    expanded = command.format(
        number=shlex.quote(number),
        message=shlex.quote(message or ""),
    )
    try:
        completed = subprocess.run(
            expanded,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return {"success": False, "error": f"Phone command failed: {exc}"}

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return {
            "success": False,
            "error": f"Phone command exited with {completed.returncode}"
            + (f": {detail}" if detail else "."),
        }
    return {"success": True, "provider": "command"}


def _log_action(action: str, number: str, message: Optional[str]) -> Dict[str, Any]:
    settings = get_settings()
    log_path = settings.data_dir / "phone.log"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "number": number,
    }
    if message is not None:
        entry["message"] = message
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception as exc:
        return {"success": False, "error": f"Could not log phone action: {exc}"}
    return {"success": True, "provider": "log", "simulated": True}


async def run_phone_action(
    action: str, number: Optional[str], message: Optional[str] = None
) -> Dict[str, Any]:
    """Run a ``call`` or ``text`` action."""
    settings = get_settings()
    valid_number = _validate_number(number)
    if not valid_number:
        return {
            "success": False,
            "error": "I need a phone number to do that.",
        }

    if action == "call":
        command = settings.phone_call_command
    elif action == "text":
        command = settings.phone_text_command
        if not str(message or "").strip():
            return {"success": False, "error": "I need the message to send."}
    else:
        return {"success": False, "error": f"Unknown phone action: {action}"}

    if command:
        result = await asyncio.to_thread(
            _run_command, command, valid_number, message
        )
    else:
        result = await asyncio.to_thread(
            _log_action, action, valid_number, message
        )

    result["number"] = valid_number
    if message is not None:
        result["message"] = message
    result["action"] = action
    return result
