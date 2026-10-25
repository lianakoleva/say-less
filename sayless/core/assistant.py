"""Main Assistant class."""

import asyncio
import os
from typing import Optional, Dict, Any, List
from ..config import get_settings
from ..stt import STTManager
from ..llm import LLMManager
from ..tools import ToolManager
from ..tools.phone.google_voice import GoogleVoiceTool
from ..tools.phone.phone import PhoneTool
from ..tools.phone.text import TextTool
from ..tools.maps.google_maps import GoogleMapsTool
from ..tools.contacts.tool import ContactsTool
from ..tools.notes.tool import NotesTool
from ..queue import QueueManager
from ..data.chat import ChatManager
from ..data.contacts import ContactsManager
from ..data.notes import NotesManager


class Assistant:
    """Main SayLess assistant."""
    
    def __init__(self):
        self.settings = get_settings()
        self.stt = STTManager()
        self.llm = LLMManager()
        self.tools = ToolManager()
        self.queue = QueueManager()
        self.chat = ChatManager()
        self.contacts = ContactsManager()
        self.notes = NotesManager()
        self._running = False
        self._active_conversation_id = None
        self.quiet = os.environ.get('SAYLESS_QUIET', '0') == '1'
    
    def _log(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)
    
    async def initialize(self) -> bool:
        """Initialize all components."""
        self._log("Initializing SayLess...")
        
        # Initialize STT
        if self.settings.stt_enabled:
            stt_ok = await self.stt.initialize()
            self._log(f"STT: {'OK' if stt_ok else 'FAILED'}")
            if not stt_ok:
                print("ERROR: STT failed to initialize")
                return False
        else:
            self._log("STT: DISABLED")
        
        # Initialize LLM
        if self.settings.llm_enabled:
            llm_ok = await self.llm.initialize()
            self._log(f"LLM: {'OK' if llm_ok else 'FAILED'}")
            if not llm_ok:
                return False
        else:
            self._log("LLM: DISABLED")
        
        # Register tools
        if self.settings.enable_phone_tools:
            self.tools.register_tool("google_voice", GoogleVoiceTool())
            self.tools.register_tool("phone", PhoneTool())
            self.tools.register_tool("text", TextTool())
            self._log("Tool google_voice: REGISTERED")
            self._log("Tool phone: REGISTERED")
            self._log("Tool text: REGISTERED")
        
        if self.settings.enable_maps_tools:
            self.tools.register_tool("google_maps", GoogleMapsTool())
            self._log("Tool google_maps: REGISTERED")
        
        # Register local tools
        self.tools.register_tool("contacts", ContactsTool())
        self._log("Tool contacts: REGISTERED")
        self.tools.register_tool("notes", NotesTool())
        self._log("Tool notes: REGISTERED")
        
        self._log(f"Queue: {self.queue.get_queue_path()}")
        self._log("Initialization complete!")
        return True
    
    async def process_voice(self, start_prompt: bool = True) -> Optional[str]:
        """Listen and process voice input."""
        if not self.stt.enabled:
            return None
        
        for attempt in range(2):
            try:
                # Listen (Enter starts and stops the recording)
                text = await self.stt.listen(start_prompt=start_prompt and attempt == 0)
                if text and str(text).strip():
                    cleaned = str(text).strip()
                    if not self.quiet:
                        print(f"You said: {cleaned}")
                    else:
                        print(f"YOU: {cleaned}", flush=True)
                    return await self.process_text(cleaned)
                if attempt == 0:
                    if not self.quiet:
                        print("(No speech detected, trying again...)")
                else:
                    if not self.quiet:
                        print("(No speech detected)")
                    else:
                        print("(No speech detected)", flush=True)
            except Exception as e:
                if not self.quiet:
                    print(f"(Listening error: {e})")
                break
        return None
    
    async def _summarize_tool_results(
        self,
        messages: List[Dict[str, Any]],
        executed_calls: List[Any],
    ) -> Optional[str]:
        """Ask the model for a one-line confirmation after running tools.

        A plain ``user`` follow-up is used instead of native tool-response
        messages so this works with every LiteRT-LM chat template, including
        base Gemma models that do not define a ``tool`` role.
        """
        import json

        results = []
        for call, result in executed_calls:
            func = call.get("function", call) if isinstance(call, dict) else {}
            name = func.get("name") or "tool"
            payload = (
                result
                if isinstance(result, str)
                else json.dumps(result, default=str)
            )
            results.append(f"{name} -> {payload}")

        followup = list(messages)
        followup.append({"role": "assistant", "content": "Calling tools."})
        followup.append(
            {
                "role": "user",
                "content": (
                    "Tool results:\n"
                    + "\n".join(results)
                    + "\nRespond to the user in one short sentence of plain English. "
                    "Do not output JSON."
                ),
            }
        )

        try:
            reply = await self.llm.chat(followup)
        except Exception as e:
            if not self.quiet:
                print(f"(Could not summarize tool results: {e})")
            return None
        return reply.strip() if reply else None

    async def _ensure_conversation(self, conversation_id: Optional[str] = None) -> str:
        if conversation_id is not None:
            return conversation_id
        if not hasattr(self, "_active_conversation_id") or self._active_conversation_id is None:
            self._active_conversation_id = await self.chat.create_conversation()
        return self._active_conversation_id
    
    async def process_text(self, text: str, conversation_id: Optional[str] = None) -> Optional[str]:
        """Process text input."""
        if not self.quiet:
            print(f"You: {text}")
        else:
            print(f"YOU: {text}", flush=True)
        
        conversation_id = await self._ensure_conversation(conversation_id)
        await self.chat.add_message(conversation_id, "user", text)
        history = await self.chat.get_messages(conversation_id)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        
        tools_schema = self.tools.get_all_schemas()
        if tools_schema and self.llm.enabled:
            try:
                response = await self.llm.generate_with_tools(messages, tools_schema)
                if response:
                    llm_response = response.get("response", "")
                    tool_calls = response.get("tool_calls", [])
                    
                    # Handle tool calls
                    tool_results = []
                    executed_calls = []
                    if tool_calls:
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                func = tool_call.get("function", {}) or {}
                                tool_name = func.get("name") or tool_call.get("name")
                                args = func.get("arguments", {}) or tool_call.get("arguments", {}) or {}
                            else:
                                func = getattr(tool_call, "function", None)
                                if func is not None:
                                    tool_name = getattr(func, "name", None)
                                    args = getattr(func, "arguments", {}) or {}
                                else:
                                    tool_name = getattr(tool_call, "name", None)
                                    args = getattr(tool_call, "arguments", {}) or {}
                            if isinstance(args, str):
                                import json
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            
                            if tool_name:
                                if not self.quiet:
                                    print(f"Calling tool: {tool_name}")
                                else:
                                    print(f"Calling tool: {tool_name}", flush=True)
                                # For contacts tool, we need to pass action
                                if tool_name == "contacts" and "action" not in args:
                                    if "add" in str(text).lower() or "name" in str(text).lower() or any(k in args for k in ["name", "phone", "email"]):
                                        args["action"] = "add"
                                if tool_name in ["google_voice", "text", "phone"] and "action" not in args:
                                    lower = str(text).lower()
                                    if "text" in lower or "message" in lower or "send" in lower:
                                        if tool_name == "google_voice":
                                            args["action"] = "text"
                                        elif tool_name == "text":
                                            args["action"] = "send"
                                        elif tool_name == "phone":
                                            args["action"] = "call"
                                    elif "call" in lower or "dial" in lower:
                                        if tool_name == "google_voice":
                                            args["action"] = "call"
                                        elif tool_name == "phone":
                                            args["action"] = "call"
                                        elif tool_name == "text":
                                            args["action"] = "send"
                                # Route phone/text through google_voice
                                if tool_name == "phone":
                                    gv_args = dict(args)
                                    gv_args["action"] = "call"
                                    result = await self.tools.execute_tool("google_voice", gv_args.get("action", "call"), gv_args)
                                elif tool_name == "text":
                                    gv_args = dict(args)
                                    gv_args["action"] = "text"
                                    if "message" not in gv_args and "msg" in gv_args:
                                        gv_args["message"] = gv_args["msg"]
                                    result = await self.tools.execute_tool("google_voice", gv_args.get("action", "text"), gv_args)
                                else:
                                    result = await self.tools.execute_tool(tool_name, args.get("action", ""), args)
                                tool_results.append(result)
                                executed_calls.append((tool_call, result))
                                if not self.quiet:
                                    print(f"Tool result: {result}")
                    
                    if llm_response:
                        if not self.quiet:
                            print(f"Assistant: {llm_response}")
                        else:
                            print(f"ASSISTANT: {llm_response}", flush=True)
                        await self.chat.add_message(conversation_id, "assistant", llm_response)
                        return llm_response
                    elif tool_calls:
                        # Check if any tool failed
                        failed = False
                        error_msg = None
                        for r in tool_results:
                            if isinstance(r, dict) and r.get("success") is False and r.get("error"):
                                failed = True
                                error_msg = r.get("error")
                                break
                        if failed and error_msg:
                            confirmation = error_msg
                        else:
                            confirmation = await self._summarize_tool_results(
                                messages, executed_calls
                            ) or "Done."
                        if not self.quiet:
                            print(f"Assistant: {confirmation}")
                        else:
                            print(f"ASSISTANT: {confirmation}", flush=True)
                        await self.chat.add_message(conversation_id, "assistant", confirmation)
                        return confirmation
            except Exception as e:
                if not self.quiet:
                    print(f"Tool generation failed: {e}")
        
        if self.llm.enabled:
            try:
                response = await self.llm.chat(messages)
                if response:
                    if not self.quiet:
                        print(f"Assistant: {response}")
                    else:
                        print(f"ASSISTANT: {response}", flush=True)
                    await self.chat.add_message(conversation_id, "assistant", response)
                    return response
            except Exception as e:
                if not self.quiet:
                    print(f"LLM chat failed: {e}")
        
        # Fallback response
        fallback = (
            f"I heard you say: {text}. The LiteRT-LM server may not be running. "
            "Start it with 'python -m sayless.llm.server' or run 'sayless setup'."
        )
        if not self.quiet:
            print(f"Assistant: {fallback}")
        else:
            print(f"ASSISTANT: {fallback}", flush=True)
        await self.chat.add_message(conversation_id, "assistant", fallback)
        return fallback
    
    async def process_queue(self):
        """Process offline queue."""
        await self.tools.process_queue()
    
    async def cleanup(self):
        """Cleanup all components."""
        await self.stt.cleanup()
        await self.llm.cleanup()
