"""Main Assistant class."""

import os
import re
from typing import Optional, Dict, Any, List
from ..config import get_settings
from ..stt import STTManager
from ..llm import LLMManager
from ..tools import ToolManager, model_tools
from ..tools.phone.google_voice import GoogleVoiceTool
from ..tools.phone.phone import PhoneTool
from ..tools.phone.text import TextTool
from ..tools.maps.google_maps import GoogleMapsTool
from ..tools.maps.navigation import NavigationTool
from ..tools.contacts.tool import ContactsTool
from ..tools.notes.tool import NotesTool
from ..tools.weather.tool import WeatherTool
from ..queue import QueueManager
from ..data.chat import ChatManager
from ..data.contacts import ContactsManager
from ..data.notes import NotesManager
from ..data.profile import UserProfile


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
        self.profile = UserProfile()
        self._running = False
        self._active_conversation_id = None
        self._last_contact_name: Optional[str] = None
        self._weather_location: Optional[str] = None
        self._home_address: Optional[str] = None
        self._work_address: Optional[str] = None
        self._current_location: Optional[str] = None
        self._pending: Optional[Dict[str, Any]] = None
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

        # Seed demo contacts on a fresh install.
        if self.settings.seed_contacts:
            added = await self.contacts.seed_demo_contacts()
            if added:
                self._log(f"Seeded {added} demo contacts")

        saved_location = await self.profile.get("weather_location")
        self._weather_location = self.settings.weather_location or (
            str(saved_location) if saved_location else None
        )

        saved_home = await self.profile.get("home_address")
        saved_work = await self.profile.get("work_address")
        self._home_address = (
            model_tools.normalize_address(saved_home) if saved_home else None
        )
        self._work_address = (
            model_tools.normalize_address(saved_work) if saved_work else None
        )
        # A current location is only used when the user configured one
        # explicitly. It is never inferred, stored, or assumed between trips.
        self._current_location = self.settings.current_location or None
        if await self.profile.get("current_location"):
            # Older builds saved an inferred location. Drop it so it cannot be
            # mistaken for a starting point the user actually gave.
            await self.profile.delete("current_location")
        # Repair addresses saved before spoken numbers and homophones were
        # cleaned up (for example "1 2 0 1 Tennessee straight San Francisco").
        if self._home_address and self._home_address != str(saved_home):
            await self.profile.set("home_address", self._home_address)
        if self._work_address and self._work_address != str(saved_work):
            await self.profile.set("work_address", self._work_address)

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
            self.tools.register_tool("navigation", NavigationTool())
            self._log("Tool google_maps: REGISTERED")
            self._log("Tool navigation: REGISTERED")

        if self.settings.enable_weather_tools:
            self.tools.register_tool("weather", WeatherTool())
            self._log("Tool weather: REGISTERED")

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

    # ── Output helpers ──────────────────────────────────────────────────────

    async def _respond(self, conversation_id: str, message: str) -> str:
        if not self.quiet:
            print(f"Assistant: {message}")
        else:
            print(f"ASSISTANT: {message}", flush=True)
        await self.chat.add_message(conversation_id, "assistant", message)
        return message

    # ── Phone target resolution ─────────────────────────────────────────────

    async def _resolve_phone_target(
        self, params: Dict[str, Any], user_text: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve a call/text recipient to a real phone number.

        Prefer what the user actually said (a spoken number or a saved contact
        name) over the model's argument, because the fine-tune sometimes
        hallucinates a number such as 123-456-7890.
        """
        target = params.get("number")
        label = params.get("contact_name")

        text_phone = model_tools.extract_phone(user_text)
        if text_phone:
            contact = None
            if label:
                contact = await self.contacts.find_contact(label)
            elif target and not model_tools.looks_like_phone(target):
                contact = await self.contacts.find_contact(target)
            else:
                contact = await self.contacts.find_contact_in_text(user_text)
            return text_phone, (
                (contact or {}).get("name")
                or label
                or (target if target and not model_tools.looks_like_phone(target) else None)
            )

        contact = await self.contacts.find_contact_in_text(user_text)
        if contact and contact.get("phone"):
            return str(contact["phone"]), contact.get("name")

        if target:
            if model_tools.looks_like_phone(target):
                return str(target), label
            contact = await self.contacts.find_contact(target)
            if contact and contact.get("phone"):
                return str(contact["phone"]), contact.get("name")
            if model_tools.is_pronoun(target):
                target = self._last_contact_name or target
                contact = await self.contacts.find_contact(target)
                if contact and contact.get("phone"):
                    return str(contact["phone"]), contact.get("name")
            return None, label or str(target)

        if self._last_contact_name and re.search(
            r"\b(?:her|him|them|their|she|he)\b", user_text, re.IGNORECASE
        ):
            contact = await self.contacts.find_contact(self._last_contact_name)
            if contact and contact.get("phone"):
                return str(contact["phone"]), contact.get("name")

        return None, label

    # ── Execution ───────────────────────────────────────────────────────────

    async def _execute_model_call(
        self, call: Dict[str, Any], user_text: str = ""
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """Run one model function call and return (confirmation, result)."""
        function = call.get("function", call) if isinstance(call, dict) else {}
        model_name = function.get("name") or call.get("name") or ""
        arguments = function.get("arguments", {}) or call.get("arguments", {}) or {}
        if isinstance(arguments, str):
            import json
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        if not self.quiet:
            print(f"Calling tool: {model_name}")
        else:
            print(f"Calling tool: {model_name}", flush=True)

        canonical = model_tools.canonical_name(model_name)
        if canonical == "list_tools":
            result = {"success": True}
            return model_tools.help_text(), result

        resolved = model_tools.resolve_tool_call(
            model_name,
            arguments,
            user_text,
            last_contact=self._last_contact_name,
        )
        if resolved is None:
            result = {
                "success": False,
                "error": (
                    "I couldn't work out that request. "
                    "Try 'what can you do?' to hear what I can handle."
                ),
            }
            return str(result["error"]), result

        tool_name, action, params = resolved
        target_label = params.get("contact_name")

        if tool_name in ("phone", "text"):
            number, label = await self._resolve_phone_target(params, user_text)
            if not number:
                result = {
                    "success": False,
                    "error": model_tools.missing_contact_message(
                        label or params.get("number"), action
                    ),
                }
                return str(result["error"]), result
            params["number"] = number
            target_label = label or target_label

        result = await self.tools.execute_tool(tool_name, action, params)
        if not self.quiet:
            print(f"Tool result: {result}")

        if tool_name == "contacts" and result.get("success"):
            contact = result.get("contact")
            if isinstance(contact, dict) and contact.get("name"):
                self._last_contact_name = str(contact["name"])

        return model_tools.confirmation(
            model_name, arguments, result, target_label=target_label
        ), result

    # ── Deterministic routing ───────────────────────────────────────────────

    @staticmethod
    def _clean_location(text: str) -> Optional[str]:
        """Pull a city name out of an answer like \"I'm in Seattle\"."""
        location = str(text or "").strip()
        location = re.sub(
            r"^(?:actually|no|nope|sorry|okay|ok)[, ]+", "", location, flags=re.I
        )
        location = re.sub(
            r"^(?:i'?m in|i am in|i live in|i'?m currently in|"
            r"i am currently in|my current location is|my location is|"
            r"i'?m at|i am at|it'?s|the city of|city of|"
            r"for|in|at|near|around)\s+",
            "",
            location,
            flags=re.I,
        )
        location = re.sub(
            r"\b(?:please|today|tonight|tomorrow|right now|currently|now)\b",
            " ",
            location,
            flags=re.I,
        )
        location = re.sub(r"\s+", " ", location).strip(" .,!?;:\"'")
        return location or None

    @staticmethod
    def _clean_address(text: str) -> Optional[str]:
        """Pull an address out of an answer like \"I live at 42 Main St\"."""
        address = str(text or "").strip()
        address = re.sub(
            r"^(?:it'?s|that'?s|i live at|i am at|i'?m at|"
            r"my (?:home|house|work|office)(?: address)? is|"
            r"(?:home|work)(?: address)? is|the address is|address is)\s+",
            "",
            address,
            flags=re.I,
        )
        address = re.sub(r"\s+", " ", address).strip(" .,!?;:\"'")
        return model_tools.normalize_address(address) or None

    def _saved_address(self, place: str) -> Optional[str]:
        return self._home_address if place == "home" else self._work_address

    async def _save_address(self, place: str, address: str) -> None:
        if place == "home":
            self._home_address = address
        else:
            self._work_address = address
        await self.profile.set(f"{place}_address", address)

    async def _prepare_navigation(
        self, arguments: Dict[str, Any], conversation_id: str
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Resolve saved places, or ask for an address and remember it.

        Returns ``(arguments, response)``. When ``response`` is set the request
        is waiting on the user and should not run yet.
        """
        args = dict(arguments)
        place = model_tools.place_key(args.get("destination"))
        if place:
            saved = self._saved_address(place)
            if not saved:
                self._pending = {
                    "kind": "address",
                    "place": place,
                    "arguments": args,
                }
                return args, await self._respond(
                    conversation_id,
                    f"What's your {place} address? I'll remember it for next time.",
                )
            args["destination"] = saved
            args["destination_label"] = place

        origin_place = model_tools.place_key(args.get("origin"))
        if origin_place:
            saved_origin = self._saved_address(origin_place)
            if saved_origin:
                args["origin"] = saved_origin
            else:
                args.pop("origin", None)

        if not args.get("origin") and not args.get("use_ip_location"):
            if self._current_location:
                args["origin"] = self._current_location
            else:
                self._pending = {"kind": "origin", "arguments": args}
                return args, await self._respond(
                    conversation_id,
                    "Where are you starting from? Give me a city or address, "
                    "or say 'use my current location' to estimate it.",
                )

        if not args.get("mode"):
            self._pending = {"kind": "travel_mode", "arguments": args}
            return args, await self._respond(
                conversation_id,
                "Should I use driving, walking, or transit?",
            )
        return args, None

    async def _finish_navigation(
        self,
        arguments: Dict[str, Any],
        conversation_id: str,
        user_text: str,
        context: Optional[str] = None,
    ) -> str:
        """Resolve the origin and travel mode if needed, then navigate."""
        args = dict(arguments)
        prefix = f"Got it: {context}. " if context else ""

        if not args.get("origin") and not args.get("use_ip_location"):
            if self._current_location:
                args["origin"] = self._current_location
            else:
                self._pending = {"kind": "origin", "arguments": args}
                return await self._respond(
                    conversation_id,
                    prefix
                    + "Where are you starting from? Give me a city or address, "
                    "or say 'use my current location' to estimate it.",
                )

        if not args.get("mode"):
            self._pending = {"kind": "travel_mode", "arguments": args}
            return await self._respond(
                conversation_id,
                prefix + "Should I use driving, walking, or transit?",
            )
        return await self._run_deterministic(
            "navigate", args, user_text, conversation_id
        )

    async def _clarify(self, conversation_id: str, text: str) -> Optional[str]:
        """Handle a pending clarification from the previous turn."""
        pending = self._pending
        if not pending:
            return None

        if re.match(
            r"^(?:never ?mind|forget it|cancel|no thanks|nothing)\b",
            str(text or "").strip(),
            re.I,
        ):
            self._pending = None
            return await self._respond(conversation_id, "Okay, cancelled.")

        if model_tools.looks_like_question(text):
            # The user is asking about the clarification, not answering it.
            return None

        if pending["kind"] == "weather_location":
            location = self._clean_location(text)
            if not location:
                return await self._respond(
                    conversation_id, "Which city should I check the weather for?"
                )
            self._pending = None
            self._weather_location = location
            await self.profile.set("weather_location", location)
            args = {"days": pending.get("days", 1), "location": location}
            return await self._run_deterministic(
                "get_weather", args, text, conversation_id
            )

        if pending["kind"] == "origin":
            parsed = model_tools.location_request(text)
            if parsed and parsed.get("negated"):
                return await self._respond(
                    conversation_id,
                    "Okay, I won't use that. Where are you starting from? "
                    "Give me a city or address, or say 'use my current "
                    "location' to estimate it.",
                )
            if re.search(
                r"\b(?:use\s+)?(?:my\s+)?(?:current\s+)?location\b|"
                r"\bfrom\s+here\b|^\s*here\s*[.!]?$",
                str(text or ""),
                re.I,
            ):
                self._pending = None
                args = dict(pending.get("arguments") or {})
                if self._current_location:
                    args["origin"] = self._current_location
                    args.pop("use_ip_location", None)
                else:
                    args["use_ip_location"] = True
                return await self._finish_navigation(
                    args, conversation_id, text
                )
            location = (parsed or {}).get("place") or self._clean_location(text)
            if not location:
                return await self._respond(
                    conversation_id,
                    "Where are you starting from? Give me a city or address, "
                    "or say 'use my current location' to estimate it.",
                )
            self._pending = None
            args = dict(pending.get("arguments") or {})
            args["origin"] = location
            return await self._finish_navigation(args, conversation_id, text)

        if pending["kind"] == "address":
            place = pending.get("place") or "home"
            address = self._clean_address(text)
            if not address:
                return await self._respond(
                    conversation_id,
                    f"What's your {place} address?",
                )
            self._pending = None
            await self._save_address(place, address)
            args = dict(pending.get("arguments") or {})
            args["destination"] = address
            args["destination_label"] = place
            return await self._finish_navigation(
                args, conversation_id, text, context=address
            )

        if pending["kind"] == "travel_mode":
            mode = model_tools.travel_mode(text)
            if not mode:
                return await self._respond(
                    conversation_id,
                    "Should I use driving, walking, or transit?",
                )
            self._pending = None
            args = dict(pending.get("arguments") or {})
            args["mode"] = mode
            return await self._finish_navigation(args, conversation_id, text)

        if pending["kind"] == "note_items":
            title = pending.get("title") or "List"
            content = model_tools.clean_note_items(text)
            if not content:
                return await self._respond(
                    conversation_id, "What should I add to the note?"
                )
            self._pending = None
            return await self._run_deterministic(
                "make_note",
                {"title": title, "content": content},
                text,
                conversation_id,
            )

        self._pending = None
        if pending["kind"] == "contact_name":
            name = model_tools.extract_contact_name(text)
            if not name:
                name = re.sub(
                    r"^(?:it'?s|this is|for|the name is|her name is|his name is)\s+",
                    "",
                    text.strip(),
                    flags=re.I,
                )
                name = name.strip(" .,!?\"'")
            args: Dict[str, Any] = {"first_name": name}
            if pending.get("phone"):
                args["phone_number"] = pending["phone"]
            if pending.get("email"):
                args["email"] = pending["email"]
            return await self._run_deterministic(
                "create_contact", args, text, conversation_id
            )

        if pending["kind"] == "text_message":
            args = {
                "phone_number": pending.get("target"),
                "message": text,
                "name": pending.get("target"),
            }
            return await self._run_deterministic(
                "send_text", args, text, conversation_id
            )
        return None

    async def _run_deterministic(
        self,
        name: str,
        arguments: Dict[str, Any],
        user_text: str,
        conversation_id: str,
    ) -> str:
        confirmation, _ = await self._execute_model_call(
            {"name": name, "arguments": arguments}, user_text=user_text
        )
        return await self._respond(conversation_id, confirmation or "Done.")

    async def _try_deterministic(
        self, text: str, conversation_id: str
    ) -> Optional[str]:
        """Handle clear contact/phone/weather/help requests before the LLM."""
        pending_result = await self._clarify(conversation_id, text)
        if pending_result is not None:
            return pending_result

        route = model_tools.deterministic_call(
            text, last_contact=self._last_contact_name
        )
        if route is None:
            return None

        name, arguments = route

        if name == "set_current_location":
            if arguments.get("negated"):
                return await self._respond(
                    conversation_id,
                    "Understood. I won't assume where you are. I'll ask where "
                    "you're starting from each time you ask for directions.",
                )
            place = self._clean_location(arguments.get("place")) or str(
                arguments.get("place") or ""
            ).strip()
            if not place:
                return await self._respond(
                    conversation_id, "Where are you starting from?"
                )
            return await self._respond(
                conversation_id,
                f"Got it, you're in {place}. I won't assume it later — I'll "
                "ask where you're starting from each time you ask for "
                "directions.",
            )

        if name == "set_address":
            place = str(arguments.get("place") or "home").lower()
            address = self._clean_address(arguments.get("address"))
            if not address:
                return await self._respond(
                    conversation_id, f"What's your {place} address?"
                )
            await self._save_address(place, address)
            return await self._respond(
                conversation_id, f"Saved your {place} address: {address}."
            )

        if name == "get_weather":
            arguments = dict(arguments)
            location = arguments.get("location")
            if not location:
                if self._weather_location:
                    arguments["location"] = self._weather_location
                else:
                    self._pending = {
                        "kind": "weather_location",
                        "days": arguments.get("days", 1),
                    }
                    return await self._respond(
                        conversation_id,
                        "Which city should I check the weather for? "
                        "I'll remember it for next time.",
                    )
            elif not self._weather_location:
                # First explicit location becomes the default for next time.
                self._weather_location = str(location)
                await self.profile.set("weather_location", str(location))

        if name == "navigate":
            arguments, waiting = await self._prepare_navigation(
                arguments, conversation_id
            )
            if waiting is not None:
                return waiting

        if name == "make_note":
            arguments = dict(arguments)
            if not str(arguments.get("content") or "").strip():
                title = str(arguments.get("title") or "List")
                self._pending = {"kind": "note_items", "title": title}
                if title.lower() == "list":
                    prompt = "What should I put on the list?"
                elif title.lower() == "note":
                    prompt = "What should the note say?"
                else:
                    prompt = f"What should I add to {title}?"
                return await self._respond(conversation_id, prompt)

        if name == "create_contact":
            contact_name = arguments.get("first_name") or arguments.get("name")
            if not contact_name:
                phone = arguments.get("phone_number")
                email = arguments.get("email")
                if phone or email:
                    self._pending = {
                        "kind": "contact_name",
                        "phone": phone,
                        "email": email,
                    }
                    return await self._respond(
                        conversation_id, "Who is this contact for?"
                    )

        if name == "send_text" and not str(arguments.get("message") or "").strip():
            target = arguments.get("name") or arguments.get("phone_number")
            self._pending = {"kind": "text_message", "target": target}
            return await self._respond(
                conversation_id, f"What should I say to {target or 'them'}?"
            )

        return await self._run_deterministic(name, arguments, text, conversation_id)

    # ── Conversation ────────────────────────────────────────────────────────

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

        # Contact, phone, weather, and help requests are handled locally.
        # FunctionGemma often picks the wrong function for these, so routing
        # them deterministically is both faster and more reliable.
        deterministic = await self._try_deterministic(text, conversation_id)
        if deterministic is not None:
            return deterministic

        history = await self.chat.get_messages(conversation_id)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]

        tools_schema = model_tools.schemas()
        if tools_schema and self.llm.enabled:
            try:
                response = await self.llm.generate_with_tools(messages, tools_schema)
                if response:
                    llm_response = response.get("response", "")
                    tool_calls = response.get("tool_calls", [])

                    if tool_calls:
                        confirmations = []
                        for tool_call in tool_calls:
                            confirmation, _ = await self._execute_model_call(
                                tool_call, user_text=text
                            )
                            if confirmation:
                                confirmations.append(confirmation)

                        confirmation = " ".join(confirmations) or "Done."
                        return await self._respond(conversation_id, confirmation)

                    if llm_response:
                        return await self._respond(conversation_id, llm_response)
            except Exception as e:
                if not self.quiet:
                    print(f"Tool generation failed: {e}")

        if self.llm.enabled:
            try:
                response = await self.llm.chat(messages)
                if response:
                    return await self._respond(conversation_id, response)
            except Exception as e:
                if not self.quiet:
                    print(f"LLM chat failed: {e}")

        # Fallback response
        fallback = model_tools.help_text()
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
