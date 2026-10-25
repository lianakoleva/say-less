"""Flat tool schemas and deterministic routing for the FunctionGemma model.

The ``mobile_actions`` fine-tune was trained on small, single-purpose functions
such as ``create_contact(first_name, last_name, phone_number, email)``. SayLess
tools are grouped (``contacts`` with an ``action`` argument), so this module
provides the flat schemas the model sees and translates its calls back to tool
actions.

The model is also weak at intent selection: "save Judy's number" can come back
as ``make_note`` or ``search_notes``. Contact and phone requests are therefore
routed deterministically from the user's words first, and the model is used for
everything else.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

#: FunctionGemma's trained developer instruction. Without it the checkpoint
#: refuses action requests (\"I can only assist with requests that require a
#: function call\").
DEVELOPER_INSTRUCTION = (
    "You are a model that can do function calling with the following functions\n"
    "Use create_contact to save a contact. Put a phone number in phone_number, "
    "never in email. Put an email address in email. If only one name is given, "
    "put it in first_name and leave last_name empty. Use list_contacts when the "
    "user asks who their contacts are. When the user asks to call or text "
    "someone, use call_phone or send_text with the contact's name. Use "
    "get_weather for weather or temperature questions, and list_tools when "
    "the user asks what you can do."
)

#: Function schemas sent to the model. These follow the mobile-actions
#: fine-tune's trained shapes as closely as possible.
MODEL_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_contact",
            "description": "Creates a contact in the phone's contact list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email address of the contact.",
                    },
                    "first_name": {
                        "type": "string",
                        "description": "The first name of the contact.",
                    },
                    "last_name": {
                        "type": "string",
                        "description": "The last name of the contact.",
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "The phone number of the contact.",
                    },
                },
                "required": ["first_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_contacts",
            "description": (
                "Lists all saved contacts and their phone numbers and email "
                "addresses. Use this when the user asks who their contacts are "
                "or wants contact information."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_text",
            "description": "Sends a text message to a phone number or a saved contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "The recipient's phone number or contact name.",
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send.",
                    },
                },
                "required": ["phone_number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_phone",
            "description": "Calls a phone number or a saved contact by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "The phone number or contact name to call.",
                    },
                },
                "required": ["phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": (
                "Gets directions or travel time to a destination such as home "
                "or work. Use this for questions like 'how long will it take "
                "me to get home?' or 'directions to the airport'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination address or place.",
                    },
                    "origin": {
                        "type": "string",
                        "description": "The starting address or place.",
                    },
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Gets the current weather or a short forecast for a city. Use "
                "this whenever the user asks about the weather, temperature, "
                "rain, or snow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or place to check.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of forecast days (1-7).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_note",
            "description": "Saves a note with a title and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The note title."},
                    "content": {"type": "string", "description": "The note body."},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "Lists all saved notes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Searches saved notes for a word or phrase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tools",
            "description": (
                "Shows what SayLess can do in plain language. Use this when the "
                "user asks what you can do, for help, or for a list of capabilities."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_PHONE_RE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{5,}\d(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NAME = r"[A-Za-z][A-Za-z'.\-]*(?:\s+[A-Za-z][A-Za-z'.\-]*){0,3}"
_SAVE_VERB = r"(?:save|add|create|store|remember|update|put|set)"
_PRONOUNS = {"her", "his", "him", "them", "their", "she", "he", "it", "they"}
_NAME_STOPWORDS = {
    "a", "an", "the", "my", "our", "new", "contact", "contacts", "phone",
    "number", "email", "mobile", "cell", "this", "that", "it", "me", "us",
    "someone", "somebody", "please", "for", "to", "with", "at", "and",
}

HELP_TEXT = (
    "I can save and update contacts using just a phone number or just an "
    "email, list your contacts and their details, call or text a saved "
    "contact by name, check the weather for a city, get directions or travel "
    "time to home, work, or another place by driving, walking, or transit, "
    "and create, list, or search notes and lists. Try saying \"Who are my "
    "contacts?\", \"Send Judy a text saying I'm running late.\", \"What's the "
    "weather in Seattle?\", \"How long will it take me to get home?\", or "
    "\"Make a list of groceries: milk and eggs.\""
)


def help_text() -> str:
    """Return the human-readable capabilities message."""
    return HELP_TEXT


def schemas() -> List[Dict[str, Any]]:
    """Return a copy of the model-facing tool schemas."""
    return [
        {"type": "function", "function": dict(schema["function"])}
        for schema in MODEL_TOOL_SCHEMAS
    ]


def _first(args: Dict[str, Any], *names: str) -> Optional[Any]:
    """Return the first present, non-empty argument among ``names``."""
    for name in names:
        value = args.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def looks_like_phone(value: Any) -> bool:
    """True for values that are mostly a phone number, not a name."""
    text = str(value or "").strip()
    if not text or "@" in text:
        return False
    number = digits(text)
    if len(number) < 7:
        return False
    return len(number) >= len(re.sub(r"[\s().+-]", "", text)) * 0.8


def looks_like_email(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and _EMAIL_RE.fullmatch(text.strip(".,")))


def extract_phone(text: str) -> Optional[str]:
    """Find a spoken phone number in the user's utterance."""
    for match in _PHONE_RE.finditer(text or ""):
        candidate = match.group().strip(" .,")
        if len(digits(candidate)) >= 7:
            return candidate
    return None


def extract_email(text: str) -> Optional[str]:
    """Find an email address, including \"name at example dot com\"."""
    source = text or ""
    direct = _EMAIL_RE.search(source)
    if direct:
        return direct.group().strip(".,")
    spoken = re.sub(r"\s+at\s+", "@", source, flags=re.I)
    spoken = re.sub(r"\s+dot\s+", ".", spoken, flags=re.I)
    match = _EMAIL_RE.search(spoken)
    if match:
        return match.group().strip(".,")
    # A bare domain such as \"gmail.com\" after a name is usually an email.
    domain = re.search(r"\b([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)\b", source)
    return domain.group(1) if domain else None


def _clean_name(raw: Optional[str], last_contact: Optional[str] = None) -> Optional[str]:
    """Turn a raw name fragment into a usable contact name."""
    if not raw:
        return last_contact
    name = str(raw).strip(" \t\r\n.,!?;:\"'")
    name = re.sub(r"'s$", "", name, flags=re.I)
    name = re.sub(
        rf"^(?:{_SAVE_VERB}|please)\s+", "", name, flags=re.I
    ).strip()
    name = re.sub(r"^(?:a|an|the|new|my|our|contact)\s+", "", name, flags=re.I)
    name = re.sub(r"\s+(?:to|for|with|at)$", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip(" .,!?:;\"'")

    if not name:
        return last_contact
    if name.lower() in _PRONOUNS:
        return last_contact
    if name.lower() in _NAME_STOPWORDS:
        return None
    if re.fullmatch(r"[\d\s().+\-]+", name):
        return None

    name = re.sub(
        r"\s+(?:phone|number|email|contact|mobile|cell)$", "", name, flags=re.I
    ).strip()
    if not name or name.lower() in _NAME_STOPWORDS:
        return None
    if name.islower() or name.isupper():
        name = name.title()
    return name


def extract_contact_name(text: str, last_contact: Optional[str] = None) -> Optional[str]:
    """Extract the person's name from a contact-related utterance."""
    source = str(text or "").strip()
    lower = source.lower()
    if not source:
        return last_contact
    if re.match(r"^(?:her|his|their|its)\b", lower) or re.search(
        r"\b(?:her|his|their)\s+(?:phone|number|email|contact)", lower
    ):
        if last_contact:
            return last_contact

    phone = extract_phone(source)
    email = extract_email(source)
    cleaned = source
    if phone:
        cleaned = cleaned.replace(phone, " ")
    if email:
        cleaned = cleaned.replace(email, " ")

    patterns = [
        # Save Judy's phone number...
        rf"{_SAVE_VERB}\s+(?:a\s+|an\s+|the\s+)?(?:new\s+)?(?:contact\s+)?"
        rf"(?:for\s+|called\s+|named\s+)?(?P<name>{_NAME}?)\s*(?:'s)?\s+"
        rf"(?:phone|number|email|mobile|cell|contact)\b",
        # Save my mom to contacts...
        rf"{_SAVE_VERB}\s+(?:a\s+|an\s+|the\s+)?(?:new\s+)?(?:contact\s+)?"
        rf"(?P<name>{_NAME}?)\s+to\s+(?:my\s+)?contacts?\b",
        # Create a contact for Judy...
        rf"{_SAVE_VERB}\s+(?:a\s+|an\s+)?(?:contact\s+)?"
        rf"(?:for\s+|called\s+|named\s+)(?P<name>{_NAME}?)"
        rf"(?:\s+(?:with|at)\b|$|[,.])",
        # Judy's number is...
        rf"(?P<name>{_NAME}?)\s*(?:'s)?\s+(?:phone|number|email|mobile|cell)"
        rf"\s+(?:is|:)\s*",
        # Judy at gmail.com
        rf"(?P<name>{_NAME}?)\s+(?:at|@)\s+[A-Za-z0-9.\-]+\.[A-Za-z]{{2,}}",
        # This number is for Judy
        rf"(?:this|the)\s+(?:number|phone|email)\s+(?:is\s+)?"
        rf"(?:for|belongs\s+to)\s+(?P<name>{_NAME}?)",
        # Save 224-... for Judy
        rf"(?:for|to)\s+(?P<name>{_NAME}?)\s*[.!?]*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.I)
        if match:
            name = _clean_name(match.group("name"), last_contact)
            if name:
                return name

    # Save Judy 224-... -> after removing the number: "Save Judy".
    match = re.match(
        rf"^{_SAVE_VERB}\s+(?:a\s+|an\s+)?(?:contact\s+)?(?P<name>{_NAME})$",
        cleaned.strip(" .,!?"),
        re.I,
    )
    if match:
        return _clean_name(match.group("name"), last_contact)
    return None


def _clean_target(value: Optional[str], last_contact: Optional[str] = None) -> Optional[str]:
    """Clean a call/text recipient."""
    if not value:
        return last_contact
    target = str(value).strip(" \t\r\n.,!?;:\"'")
    target = re.sub(r"^(?:the|my|to)\s+", "", target, flags=re.I)
    target = re.sub(
        r"'s\s+(?:phone|number|mobile|cell)$", "", target, flags=re.I
    )
    target = re.sub(
        r"\s+(?:phone|number|mobile|cell)$", "", target, flags=re.I
    ).strip()
    target = re.sub(r"\b(?:for me|now|please)\b", " ", target, flags=re.I)
    target = re.sub(r"\s+", " ", target).strip(" .,!?;:\"'")
    if target.lower() in _PRONOUNS:
        return last_contact or target
    return target or last_contact


def _clean_message(value: Optional[str]) -> str:
    message = str(value or "").strip()
    message = re.sub(
        r"^(?:saying|that|to say|with the message|message)\s*[:\-]?\s*",
        "",
        message,
        flags=re.I,
    )
    return message.strip().strip("\"'").strip()


def _looks_like_question(lower: str) -> bool:
    return bool(
        re.match(r"^(?:what|who|where|when|why|how|do|does|did|can|could|would|is|are)\b", lower)
        or lower.endswith("?")
    )


def looks_like_question(text: str) -> bool:
    """Public helper for clarification handling."""
    return _looks_like_question(str(text or "").strip().lower())


def is_pronoun(value: Any) -> bool:
    """True when ``value`` is a pronoun such as \"her\" or \"them\"."""
    return str(value or "").strip().lower() in _PRONOUNS


def _is_help_request(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:what can you do|what do you do|how can you help|help me|"
                  r"what are your (?:capabilities|tools|commands)|list (?:your )?tools|"
                  r"what can i say|show me what you can do)\b", lower)
        or re.search(r"\b(?:what|which)\s+(?:tools|functions|commands|capabilities)\b", lower)
        or re.search(r"\b(?:what|which)\s+can\s+(?:you|sayless)\s+do\b", lower)
        or lower.strip() in {"help", "?", "commands", "capabilities"}
    )


def _is_list_contacts(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:who are|list|show|what are)\b.*\bcontacts?\b", lower)
        or "contact list" in lower
        or "saved contacts" in lower
        or "my contacts" in lower
        or re.search(r"\bcontacts?\b.*\b(?:info|information|details)\b", lower)
        or re.search(r"\b(?:what|which) contacts\b", lower)
        or "their info" in lower
        or "their information" in lower
    )


def _lookup_contact(lower: str) -> Optional[str]:
    """Return the name in \"what's Judy's number?\" style questions."""
    match = re.search(
        r"(?:what(?:'s| is)|whats|tell me|do you know|show me)\s+"
        r"(?P<name>.+?)(?:'s)?\s+"
        r"(?:phone|number|email|contact\s+(?:info|information|details)|info|details)",
        lower,
    )
    if not match:
        match = re.search(
            r"(?:what(?:'s| is)|whats)\s+(?P<name>.+?)\s+contact\s+"
            r"(?:info|information|details)",
            lower,
        )
    return _clean_name(match.group("name")) if match else None


def _navigation_request(text: str) -> Optional[Dict[str, str]]:
    """Return navigation arguments for a directions/travel-time request."""
    lower = text.lower()
    nav_intent = bool(
        re.search(
            r"\b(?:navigate|navigation|directions|route|take me|drive me|"
            r"get me|commute|eta)\b",
            lower,
        )
        or re.search(r"\bhow far\b", lower)
        or re.search(
            r"\bhow (?:long|far)\b.*\b(?:get|go|drive|travel|commute|walk|"
            r"bike|cycle)\b",
            lower,
        )
        or (
            re.search(r"\b(?:home|work)\b", lower)
            and re.search(r"\b(?:get|go|drive|take|navigate|directions|route)\b", lower)
        )
    )
    if not nav_intent:
        return None

    origin = None
    destination = None

    match = re.search(
        r"\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)\s*[.!?]*$",
        text,
        re.I,
    )
    if match:
        origin = match.group("origin")
        destination = match.group("destination")
    else:
        patterns = [
            r"\b(?:how long|how far)\b.*?\b(?:get|go|drive|travel|commute|"
            r"walk|bike|cycle)\s+(?:to\s+)?(?P<destination>.+?)\s*[.!?]*$",
            r"\b(?:navigate|navigation|directions|route|take me|drive me|get me)"
            r"\s+(?:to\s+)?(?P<destination>.+?)\s*[.!?]*$",
            r"\bhow far is\s+(?P<destination>.+?)\s*[.!?]*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                destination = match.group("destination")
                break

    if not destination:
        if re.search(r"\bhome\b", lower):
            destination = "home"
        elif re.search(r"\bwork\b", lower):
            destination = "work"

    def clean(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        value = re.sub(
            r"\b(?:please|right now|now|today|for me)\b",
            " ",
            value,
            flags=re.I,
        )
        value = re.sub(
            r"\s+(?:by|on|via)\s+(?:car|driving|foot|walking|transit|"
            r"public transit|bus|train|subway|bike|bicycle|bicycling|cycling)\b",
            "",
            value,
            flags=re.I,
        )
        value = re.sub(
            r"\s+\b(?:driving|walking|transit|bicycling|cycling)\b$",
            "",
            value,
            flags=re.I,
        )
        value = re.sub(r"\s+", " ", value).strip(" .,!?;:\"'")
        return value or None

    destination = clean(destination)
    origin = clean(origin)
    if destination:
        # "How long will it take me to get home from Oakland?"
        trailing = re.search(
            r"^(?P<destination>.+?)\s+from\s+(?P<origin>.+)$",
            destination,
            re.I,
        )
        if trailing:
            destination = trailing.group("destination").strip()
            if not origin:
                origin = clean(trailing.group("origin"))
    if not destination:
        return None
    args = {"destination": destination}
    if origin:
        if origin.lower() in {
            "here",
            "my location",
            "my current location",
            "current location",
        }:
            args["use_ip_location"] = True
        else:
            args["origin"] = origin
    mode = travel_mode(text)
    if mode:
        args["mode"] = mode
    return args


def location_request(text: str) -> Optional[Dict[str, Any]]:
    """Parse "my current location is San Francisco" / "I'm in Oakland"."""
    source = str(text or "").strip()
    source = re.sub(
        r"^(?:actually|no|nope|sorry|okay|ok)[, ]+", "", source, flags=re.I
    )
    negated = re.match(
        r"^(?:my\s+)?(?:current\s+)?location\s+is\s+not\s+"
        r"(?:in\s+|at\s+)?(?P<place>.+)$",
        source,
        re.I,
    ) or re.match(
        r"^i(?:'m| am)\s+not\s+(?:in|at)\s+(?P<place>.+)$",
        source,
        re.I,
    )
    if negated:
        return {"negated": True, "place": negated.group("place")}

    patterns = [
        r"^(?:please\s+)?(?:set\s+)?(?:my\s+)?(?:current\s+)?location\s+"
        r"(?:is|to|=)\s+(?P<place>.+)$",
        r"^(?:i'?m|i am)\s+(?:currently\s+)?(?:in|at|near)\s+"
        r"(?P<place>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, source, re.I)
        if not match:
            continue
        place = clean_note_items(match.group("place"))
        place = re.sub(r"^the (?:city|town) of\s+", "", place, flags=re.I)
        place = place.strip(" .,!?;:\"'")
        if re.search(
            r"\b(?:mood|hurry|trouble|need|looking|middle|middle of)\b",
            place,
            re.I,
        ):
            return None
        if place:
            return {"place": place}
    return None


def _address_request(text: str) -> Optional[Dict[str, str]]:
    """Parse "my home is 42 Main St" / "change my work address to ..."."""
    source = str(text or "").strip()
    patterns = [
        r"^(?:please\s+)?(?:set|save|update|change|remember)\s+(?:my\s+)?"
        r"(?P<place>home|work)(?:\s+address)?\s+(?:to|is|as)\s+"
        r"(?P<address>.+)$",
        r"^(?:my\s+)?(?P<place>home|work)(?:\s+address)?\s+(?:is|:)\s+"
        r"(?P<address>.+)$",
        r"^i\s+(?P<verb>live|work)\s+(?:at|in)\s+(?P<address>.+)$",
    ]
    for index, pattern in enumerate(patterns):
        match = re.match(pattern, source, re.I)
        if not match:
            continue
        groups = match.groupdict()
        if index == 2:
            place = "home" if groups.get("verb", "").lower() == "live" else "work"
        else:
            place = groups.get("place", "home").lower()
        address = normalize_address(groups.get("address") or "")
        if address:
            return {"place": place, "address": address}
    return None


def _weather_request(text: str) -> Optional[Tuple[Optional[str], int]]:
    """Return ``(location, days)`` for a weather request, or ``None``."""
    lower = text.lower()
    if not re.search(
        r"\b(?:weather|forecast|temperature|raining|rain|snow|sunny|cloudy|"
        r"humid|windy|degrees|hot outside|cold outside)\b",
        lower,
    ):
        return None
    if _is_help_request(lower):
        return None

    location = None
    match = re.search(
        r"\b(?:in|at|for|near|around)\s+(?P<location>.+)$", text, re.I
    )
    if match:
        location = match.group("location")
    else:
        # "Seattle weather" / "weather Seattle"
        match = re.search(
            r"^(?P<location>[A-Za-z][A-Za-z .'-]{1,40}?)\s+(?:weather|forecast)$",
            text,
            re.I,
        )
        if not match:
            match = re.search(
                r"^(?:weather|forecast)\s+(?:in\s+|for\s+)?(?P<location>.+)$",
                text,
                re.I,
            )
        if match:
            location = match.group("location")

    if location:
        location = re.sub(
            r"\b(?:today|tonight|tomorrow|right now|currently|this week|"
            r"this weekend|please|now)\b",
            " ",
            location,
            flags=re.I,
        )
        location = re.sub(r"\s+", " ", location).strip(" .,!?;:\"'")
        if location.lower() in _NAME_STOPWORDS or not location:
            location = None

    days = 1
    if re.search(r"\b(?:forecast|this week|next few days|weekend)\b", lower):
        days = 3
    if re.search(r"\bthis week\b", lower):
        days = 7
    if re.search(r"\btomorrow\b", lower) and days == 1:
        days = 2
    return location, days


def _call_target(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(number_or_name, label)`` for a call request."""
    phone = extract_phone(text)
    if phone:
        label = extract_contact_name(text)
        return phone, label
    target = re.sub(
        r"^(?:please\s+)?(?:call|dial|ring|phone)\s+", "", text, flags=re.I
    )
    target = _clean_target(target)
    return target, target


def _text_request(text: str) -> Tuple[Optional[str], str]:
    """Return ``(recipient, message)`` for a text request."""
    patterns = [
        r"^(?:please\s+)?send\s+(?:a\s+|an\s+)?(?:text|message|sms)\s+"
        r"(?:to\s+)?(?P<target>.+?)(?:\s+(?:saying|that|to say)\b(?P<rest>.*))?$",
        r"^(?:please\s+)?send\s+(?P<target>.+?)\s+(?:a\s+|an\s+)?"
        r"(?:text|message|sms)\b(?P<rest>.*)$",
        r"^(?:please\s+)?(?:text|message|sms)\s+"
        r"(?P<target>[A-Za-z][A-Za-z'.\-]*)\s+(?P<rest>.+)$",
        r"^(?:please\s+)?(?:text|message|sms)\s+(?P<target>.+?)"
        r"(?:\s+(?:saying|that|to say)\b(?P<rest>.*))?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, re.I)
        if not match:
            continue
        target = _clean_target(match.group("target"))
        if ":" in target:
            target = target.split(":", 1)[0].strip()
        rest = match.groupdict().get("rest") or ""
        message = _clean_message(rest)
        if not message:
            # "Send Judy a text: I'm running late." / "Text Judy: running late"
            after = re.split(r"[:;]", text, maxsplit=1)
            if len(after) == 2:
                message = _clean_message(after[1])
        if not message:
            match_msg = re.search(
                r"(?:saying|that|to say|message)\s+(?P<message>.+)$", text, re.I
            )
            if match_msg:
                message = _clean_message(match_msg.group("message"))
        return target, message
    return None, ""


_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_STREET_WORDS = {
    r"\bstraight\b": "Street",
    r"\bsteet\b": "Street",
    r"\bstr\b": "Street",
    r"\bave\b|\bav\b": "Avenue",
    r"\bblvd\b": "Boulevard",
    r"\bdr\b": "Drive",
    r"\bct\b": "Court",
    r"\bln\b": "Lane",
    r"\brd\b": "Road",
    r"\bpl\b": "Place",
    r"\bter\b": "Terrace",
    r"\bpkwy\b": "Parkway",
}


def normalize_address(value: Any) -> str:
    """Clean up a spoken address before geocoding.

    Speech-to-text often returns house numbers as separate digits ("1 2 0 1")
    and turns "Street" into "straight". Both make Nominatim fail even though
    the address is correct.
    """
    address = str(value or "").strip()
    if not address:
        return ""

    # "1 2 0 1" -> "1201". Require at least three separated digits so years
    # or "1 2" style fragments are left alone.
    address = re.sub(
        r"(?<![\d.])(?:\d\s+){2,}\d(?![\d.])",
        lambda match: re.sub(r"\s+", "", match.group(0)),
        address,
    )

    # Spelled-out house numbers: "one two oh one" -> "1201"
    words = re.split(r"(\W+)", address)
    converted = []
    for token in words:
        replacement = _SPOKEN_DIGITS.get(token.lower())
        converted.append(replacement if replacement is not None else token)
    address = "".join(converted)

    # Collapse digits again in case they were spelled out.
    address = re.sub(
        r"(?<![\d.])(?:\d\s+){2,}\d(?![\d.])",
        lambda match: re.sub(r"\s+", "", match.group(0)),
        address,
    )

    for pattern, replacement in _STREET_WORDS.items():
        address = re.sub(pattern, replacement, address, flags=re.I)

    address = re.sub(r"\s+", " ", address).strip(" .,!?;:\"'")
    return address


def normalize_travel_mode(value: Any) -> Optional[str]:
    """Map a model-supplied travel mode to a navigation mode."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"driving", "drive", "car", "by car"}:
        return "driving"
    if text in {"walking", "walk", "foot", "on foot", "pedestrian"}:
        return "walking"
    if text in {
        "bicycling",
        "biking",
        "bike",
        "cycling",
        "bicycle",
    }:
        return "bicycling"
    if text in {
        "transit",
        "public transit",
        "public_transit",
        "public transport",
        "bus",
        "train",
        "subway",
        "metro",
    }:
        return "transit"
    return travel_mode(text)


def travel_mode(text: Any) -> Optional[str]:
    """Return a navigation mode mentioned in the text, if any."""
    lower = str(text or "").lower()
    if re.search(r"\b(?:walk|walking|on foot|by foot|pedestrian)\b", lower):
        return "walking"
    if re.search(
        r"\b(?:bike|biking|bicycling|cycling|bicycle|by bike)\b", lower
    ):
        return "bicycling"
    if re.search(
        r"\b(?:transit|public transport|public transit|bus|train|subway|"
        r"metro|light rail)\b",
        lower,
    ):
        return "transit"
    if re.search(r"\b(?:drive|driving|car|by car)\b", lower):
        return "driving"
    return None


def place_key(value: Any) -> Optional[str]:
    """Return ``"home"``/``"work"`` for a saved-place destination."""
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:my|the)\s+", "", text)
    text = re.sub(r"\s+(?:address|place|location)$", "", text)
    text = text.strip(" .,!?;:\"'")
    if text in {"home", "house", "apartment", "flat"}:
        return "home"
    if text in {"work", "office", "job", "workplace"}:
        return "work"
    return None


def clean_note_items(text: Any) -> str:
    """Turn a dictated answer into note content."""
    value = str(text or "").strip()
    value = re.sub(
        r"^(?:it'?s|they'?re|i need|i want|add|put|include|write|the items are)\s+",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip(" .,!?;:\"'")
    return value


def _default_note_title(text: str, kind: str) -> str:
    if re.search(r"\bgrocer", text, re.I):
        return "Groceries"
    if re.search(r"\bshopping\b", text, re.I):
        return "Shopping list"
    if re.search(r"\bto[- ]?do\b", text, re.I):
        return "To-do list"
    if re.search(r"\bpacking\b", text, re.I):
        return "Packing list"
    if re.search(r"\b(?:wish|gift)\b", text, re.I):
        return "Wish list"
    return "Note" if kind == "note" else "List"


def _clean_note_title(value: Any, fallback: str) -> str:
    title = str(value or "").strip()
    title = re.sub(
        r"^(?:called|named|titled|of|for|about|my|the)\s+", "", title, flags=re.I
    )
    title = re.sub(r"\s+(?:list|note)$", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title).strip(" .,!?;:\"'")
    if not title or title.lower() in {"a", "an", "the", "new"}:
        return fallback
    normalized = title.lower()
    if normalized in {"grocery", "groceries"}:
        return "Groceries"
    if normalized in {"shopping"}:
        return "Shopping list"
    if normalized in {"to do", "todo", "to-do"}:
        return "To-do list"
    if normalized in {"packing"}:
        return "Packing list"
    if normalized in {"wish", "gift", "gifts"}:
        return "Wish list"
    return title[0].upper() + title[1:]


def _note_request(text: str) -> Optional[Dict[str, Any]]:
    """Parse note/list requests such as \"make a list of groceries\".

    Returns an action dict with either ``action`` (list/search) or
    ``title``/``content`` for a new note.
    """
    source = str(text or "").strip()
    lower = source.lower()
    if not source or not re.search(r"\b(?:notes?|lists?)\b", lower):
        return None
    if re.search(r"\bcontacts?\b", lower) or _is_help_request(lower):
        return None

    # "list my notes", "what are my notes?"
    if re.match(
        r"^(?:please\s+)?(?:list|show|read|see|display|tell me|"
        r"what(?:'s| is| are))\b.*\bnotes?\b",
        lower,
    ):
        return {"action": "list_notes"}

    # "search my notes for milk"
    search = re.search(
        r"\b(?:search|find|look through|look in)\b.*?\bnotes?\b"
        r"\s*(?:for|about)?\s*(?P<query>.+)$",
        source,
        re.I,
    )
    if search:
        query = clean_note_items(search.group("query"))
        if query:
            return {"action": "search_notes", "query": query}

    # "what's on my grocery list?"
    on_list = re.search(
        r"\bwhat(?:'s| is)?\s+on\s+(?:my|the)\s+(?P<title>[\w' -]+?)\s+list\b",
        source,
        re.I,
    )
    if on_list:
        query = _clean_note_title(on_list.group("title"), "List")
        if query.lower() == "groceries":
            query = "grocer"
        return {"action": "search_notes", "query": query}

    # "add milk to my grocery list"
    add_to = re.match(
        r"^(?:please\s+)?(?:add|put)\s+(?P<content>.+?)\s+(?:to|on)\s+"
        r"(?:my|the)\s+(?P<title>[\w' -]+?)\s+(?:list|note)$",
        source,
        re.I,
    )
    if add_to:
        kind = "note" if "note" in lower else "list"
        title = _clean_note_title(
            add_to.group("title"), _default_note_title(source, kind)
        )
        return {"title": title, "content": clean_note_items(add_to.group("content"))}

    create = re.match(
        r"^(?:please\s+)?(?:make|create|start|save|write|jot down|note down|"
        r"take|note)\b(?P<body>.*)$",
        source,
        re.I,
    )
    if not create:
        return None

    body = re.sub(
        r"^\s*(?:me\s+)?(?:a\s+|an\s+)?(?:new\s+|quick\s+)?",
        "",
        create.group("body"),
        flags=re.I,
    ).strip()
    body = re.sub(r"^(?:down|out)\s+", "", body, flags=re.I).strip()

    kind = "note" if re.search(r"\bnotes?\b", lower) else "list"
    default_title = _default_note_title(source, kind)
    title: Optional[str] = None
    content = ""

    # Leading title: "grocery list with milk and eggs"
    leading = re.match(
        r"^(?P<title>[\w' -]{1,40}?)\s+(?P<kind>list|note)\b(?P<rest>.*)$",
        body,
        re.I,
    )
    # Keyword first: "list of groceries: milk, eggs", "note called Budget"
    keyword = None if leading else re.match(
        r"^(?P<kind>list|note)\b(?P<rest>.*)$", body, re.I
    )

    if leading or keyword:
        match = leading or keyword
        assert match is not None
        kind = match.group("kind").lower()
        rest = (match.group("rest") or "").strip()
        hinted = match.groupdict().get("title")
        if hinted:
            title = _clean_note_title(hinted, default_title)
        elif default_title in {"Groceries", "Shopping list", "To-do list"}:
            title = default_title

        named = re.match(
            r"^(?:called|named|titled)\s+(?P<title>[^:]+?)"
            r"(?:(?:,|:|\s+(?:with|containing|saying|that says))\s*(?P<rest>.*))?$",
            rest,
            re.I,
        )
        if named:
            title = _clean_note_title(named.group("title"), default_title)
            content = clean_note_items(named.group("rest") or "")
        else:
            # "note that I need milk" / "note saying bring snacks".
            said = re.match(
                r"^(?:that says|saying|that)\b\s*(?P<content>.+)$", rest, re.I
            )
            if said:
                content = clean_note_items(said.group("content"))
            else:
                rest = re.sub(
                    r"^(?:of|for|about|containing|with|saying|that says|that|:|-)\b\s*",
                    "",
                    rest,
                    flags=re.I,
                ).strip()

                if ":" in rest:
                    head, tail = rest.split(":", 1)
                    head = _clean_note_title(head, "")
                    if head:
                        title = head
                    content = clean_note_items(tail)
                elif leading and rest:
                    # "grocery list with milk and eggs" -> title already known.
                    content = clean_note_items(
                        re.sub(r"^(?:of|with|containing)\s+", "", rest, flags=re.I)
                    )
                elif rest:
                    # A plain phrase is either a title ("list of groceries") or
                    # dictated items ("list of milk and eggs").
                    if re.search(r"\b(?:and|,)\b", rest):
                        content = clean_note_items(rest)
                    else:
                        title = _clean_note_title(rest, default_title)
    else:
        content = clean_note_items(body)

    title = title or default_title
    return {"title": title, "content": content}


def deterministic_call(
    text: str, last_contact: Optional[str] = None
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Route clear contact/phone/weather/help requests without asking the model.

    Returns a model-style ``(function_name, arguments)`` pair, or ``None`` when
    the request is not clearly one of these actions.
    """
    source = str(text or "").strip()
    lower = source.lower()
    if not source:
        return None

    phone = extract_phone(source)
    email = extract_email(source)
    name = extract_contact_name(source, last_contact)

    if _is_help_request(lower):
        return "list_tools", {}

    navigation = _navigation_request(source)
    if navigation is not None:
        return "navigate", navigation

    location = location_request(source)
    if location is not None:
        return "set_current_location", location

    address = _address_request(source)
    if address is not None:
        return "set_address", address

    weather = _weather_request(source)
    if weather is not None:
        location, days = weather
        args: Dict[str, Any] = {"days": days}
        if location:
            args["location"] = location
        return "get_weather", args

    # Call: "call Judy", "dial 224-...", "phone mom".
    if re.match(r"^(?:please\s+)?(?:call|dial|ring)\b", lower) or re.match(
        r"^phone\s+(?!number\b|is\b|was\b)", lower
    ):
        target, label = _call_target(source)
        if target:
            args: Dict[str, Any] = {"phone_number": target}
            if label and label != target:
                args["name"] = label
            return "call_phone", args

    # Text: "text Judy saying...", "send Judy a text...".
    if re.match(
        r"^(?:please\s+)?(?:send\s+.+\s+(?:a\s+)?(?:text|message|sms)|"
        r"send\s+(?:a\s+)?(?:text|message|sms)|text\s+\S|message\s+\S)",
        lower,
    ):
        target, message = _text_request(source)
        if target:
            return "send_text", {
                "phone_number": target,
                "message": message,
                "name": target,
            }

    # "What's Judy's number?" / "Tell me Judy's email."
    lookup = _lookup_contact(lower)
    if lookup:
        return "get_contact", {"name": lookup}

    if _is_list_contacts(lower):
        return "list_contacts", {}

    note = _note_request(source)
    if note is not None:
        action = note.get("action")
        if action == "list_notes":
            return "list_notes", {}
        if action == "search_notes":
            return "search_notes", {"query": note["query"]}
        return "make_note", {
            "title": note.get("title") or "List",
            "content": note.get("content") or "",
        }

    save_intent = bool(
        re.search(rf"\b{_SAVE_VERB}\b", lower)
        or re.search(r"\b(?:phone|number|email)\s+(?:is|:)\b", lower)
        or (email and not _looks_like_question(lower))
        or (phone and re.search(r"\b(?:for|belongs to)\b", lower))
    )
    if save_intent:
        args = {}
        if name:
            args["first_name"] = name
        if phone:
            args["phone_number"] = phone
        if email:
            args["email"] = email
        if args:
            return "create_contact", args
    return None


#: Alternate names a fine-tune may emit for the same action.
_ALIASES = {
    "add_contact": "create_contact",
    "save_contact": "create_contact",
    "new_contact": "create_contact",
    "update_contact": "create_contact",
    "get_contacts": "list_contacts",
    "search_contacts": "list_contacts",
    "show_contacts": "list_contacts",
    "send_message": "send_text",
    "send_sms": "send_text",
    "text_contact": "send_text",
    "make_call": "call_phone",
    "call_contact": "call_phone",
    "dial": "call_phone",
    "create_note": "make_note",
    "add_note": "make_note",
    "save_note": "make_note",
    "get_notes": "list_notes",
    "find_notes": "search_notes",
    "help": "list_tools",
    "capabilities": "list_tools",
}


def canonical_name(name: str) -> str:
    return _ALIASES.get(name, name)


def _contact_name(
    arguments: Dict[str, Any],
    user_text: str,
    last_contact: Optional[str] = None,
) -> Optional[str]:
    """Build a contact name, dropping a last name the model invented."""
    first = _text(_first(arguments, "first_name", "given_name"))
    last = _text(_first(arguments, "last_name", "family_name", "surname"))
    full = _text(_first(arguments, "name", "contact", "full_name"))

    if not first and full:
        parts = full.split(None, 1)
        first = parts[0]
        if len(parts) > 1 and not last:
            last = parts[1]

    if first and first.lower() in _PRONOUNS and last_contact:
        first = last_contact
        last = None

    if first and looks_like_phone(first) and not full:
        # The model put the number in the name slot. Try the utterance.
        first = extract_contact_name(user_text, last_contact)
        if not first:
            return None

    # The fine-tune often invents a last name (e.g. \"Smith\"). Only keep it
    # when the user actually said it.
    if last and user_text and last.lower() not in user_text.lower():
        last = None
    if last and first and last.lower() == first.lower():
        last = None

    if not first and not last:
        return None
    name = " ".join(part for part in (first, last) if part)
    # If the model supplied the full name in first_name, keep it as-is.
    return name.strip() or None


def resolve_tool_call(
    name: str,
    arguments: Optional[Dict[str, Any]],
    user_text: str = "",
    last_contact: Optional[str] = None,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Map a model function call to ``(tool, action, params)``.

    ``user_text`` is what the user actually said. It is used to recover phone
    numbers and emails the model placed in the wrong slot or dropped.
    """
    if not isinstance(arguments, dict):
        return None
    args = {key: value for key, value in arguments.items() if value is not None}
    name = canonical_name(name)

    if name == "list_tools":
        return "help", "show", {}

    if name == "create_contact":
        contact_name = _contact_name(args, user_text, last_contact)
        if not contact_name:
            return None

        phone = _text(
            _first(args, "phone_number", "phone", "number", "mobile")
        )
        email = _text(_first(args, "email", "email_address"))
        text_phone = extract_phone(user_text)
        text_email = extract_email(user_text)

        # The model sometimes puts a phone number in the email slot.
        if not phone and looks_like_phone(email):
            phone, email = email, None
        if not phone and text_phone:
            phone = text_phone
        if email and looks_like_phone(email):
            email = None
        if not email and text_email:
            email = text_email
        if phone and looks_like_email(phone):
            phone, email = None, phone

        params: Dict[str, Any] = {"name": contact_name}
        if phone:
            params["phone"] = phone
        if email:
            params["email"] = email
        return "contacts", "add", params

    if name == "get_contact":
        contact_name = _text(
            _first(args, "name", "contact", "first_name", "full_name")
        ) or extract_contact_name(user_text, last_contact)
        if not contact_name:
            return None
        return "contacts", "get", {"name": contact_name}

    if name == "list_contacts":
        query = _text(_first(args, "query", "search"))
        if query and not looks_like_phone(query) and query.lower() not in {
            "object", "all", "contacts", "contact"
        }:
            return "contacts", "search", {"query": query}
        return "contacts", "list", {}

    if name == "send_text":
        target = _text(
            _first(args, "phone_number", "phone", "number", "contact", "name")
        )
        label = _text(_first(args, "name", "contact"))
        message = _text(_first(args, "message", "text", "msg"))
        if not message:
            return None
        return "text", "send", {
            "number": target,
            "message": message,
            "contact_name": label,
        }

    if name == "call_phone":
        target = _text(
            _first(args, "phone_number", "phone", "number", "contact", "name")
        )
        label = _text(_first(args, "name", "contact"))
        return "phone", "call", {
            "number": target,
            "contact_name": label,
        }

    if name == "navigate":
        destination = _text(
            _first(args, "destination", "address", "to", "place", "where")
        )
        if not destination:
            navigation = _navigation_request(user_text)
            if navigation:
                destination = navigation.get("destination")
        if not destination:
            return None
        origin = _text(_first(args, "origin", "from", "start"))
        label = _text(_first(args, "destination_label", "label"))
        mode = normalize_travel_mode(
            _first(args, "mode", "travel_mode", "transportation", "method")
        ) or travel_mode(user_text)
        nav_params: Dict[str, Any] = {"destination": destination}
        if origin:
            nav_params["origin"] = origin
        if label:
            nav_params["destination_label"] = label
        if mode:
            nav_params["mode"] = mode
        if args.get("use_ip_location"):
            nav_params["use_ip_location"] = True
        return "navigation", "directions", nav_params

    if name == "get_weather":
        location = _text(_first(args, "location", "city", "place"))
        if not location:
            weather = _weather_request(user_text)
            if weather:
                location = weather[0]
        days = _first(args, "days", "forecast_days")
        try:
            days = int(days) if days is not None else 1
        except (TypeError, ValueError):
            days = 1
        action = "forecast" if days > 1 else "current"
        weather_params: Dict[str, Any] = {"days": max(1, min(7, days))}
        if location:
            weather_params["location"] = location
        return "weather", action, weather_params

    if name == "make_note":
        title = _text(_first(args, "title", "name"))
        if not title:
            return None
        content = _text(_first(args, "content", "body", "text")) or ""
        return "notes", "create", {"title": title, "content": content}

    if name == "list_notes":
        return "notes", "list", {}

    if name == "search_notes":
        query = _text(_first(args, "query", "text"))
        if not query:
            return None
        return "notes", "search", {"query": query}

    return None


def _describe_contacts(contacts: List[Dict[str, Any]]) -> str:
    """Describe contacts in a natural spoken sentence."""
    if not contacts:
        return "You don't have any saved contacts yet."
    if len(contacts) == 1:
        contact = contacts[0]
        name = contact.get("name") or "Unnamed"
        details = []
        if contact.get("phone"):
            details.append(f"phone {contact['phone']}")
        if contact.get("email"):
            details.append(f"email {contact['email']}")
        if not details:
            return f"You have one saved contact: {name}."
        return f"You have one saved contact: {name}, " + " and ".join(details) + "."

    parts = []
    for contact in contacts:
        name = contact.get("name") or "Unnamed"
        details = []
        if contact.get("phone"):
            details.append(str(contact["phone"]))
        if contact.get("email"):
            details.append(str(contact["email"]))
        parts.append(f"{name} ({', '.join(details)})" if details else name)
    return f"You have {len(contacts)} saved contacts: " + "; ".join(parts) + "."


def _describe_contact(contact: Dict[str, Any]) -> str:
    name = contact.get("name") or "that contact"
    details = []
    if contact.get("phone"):
        details.append(f"phone {contact['phone']}")
    if contact.get("email"):
        details.append(f"email {contact['email']}")
    if not details:
        return f"I have {name} saved, but no phone number or email yet."
    return f"{name}: " + " and ".join(details) + "."


def confirmation(
    name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
    target_label: Optional[str] = None,
) -> str:
    """Build a short spoken confirmation without another model call.

    Calling the model again just to phrase ``{\"success\": true}`` doubles the
    latency on the CoralBoard and can make it answer with a canned refusal.
    """
    if isinstance(result, dict) and result.get("success") is False:
        return str(result.get("error") or "That didn't work.")
    if isinstance(result, dict) and result.get("queued"):
        message = str(
            result.get("message") or "I queued that until the internet is back."
        )
        return message if message.endswith(".") else message + "."

    args = arguments if isinstance(arguments, dict) else {}
    name = canonical_name(name)
    contact = _text(_first(args, "name", "contact", "full_name"))
    if not contact:
        contact = _text(_first(args, "first_name", "last_name")) or "the contact"
    result_contact = result.get("contact") if isinstance(result, dict) else None
    if isinstance(result_contact, dict) and result_contact.get("name"):
        contact = result_contact["name"]
    title = _text(_first(args, "title", "name")) or "your note"
    target = target_label or _text(
        _first(args, "phone_number", "phone", "number", "contact", "name")
    )
    number = result.get("number") if isinstance(result, dict) else None
    simulated = bool(result.get("simulated")) if isinstance(result, dict) else False

    if name == "list_tools":
        return help_text()
    if name == "navigate":
        if isinstance(result, dict) and result.get("summary"):
            return str(result["summary"])
        destination = result.get("destination_label") if isinstance(result, dict) else None
        if destination:
            return f"Here are the directions to {destination}."
        return "I couldn't get directions right now."
    if name == "get_weather":
        if isinstance(result, dict) and result.get("summary"):
            return str(result["summary"])
        location = result.get("location") if isinstance(result, dict) else None
        if location:
            return f"Here is the weather for {location}."
        return "I couldn't get the weather right now."
    if name == "create_contact":
        if isinstance(result, dict) and result.get("updated"):
            return f"Updated {contact} in contacts."
        return f"Saved {contact} to contacts."
    if name == "get_contact":
        found = result.get("contact") if isinstance(result, dict) else None
        if isinstance(found, dict):
            return _describe_contact(found)
        return f"I don't have a contact named {contact}."
    if name == "list_contacts":
        contacts = result.get("contacts") if isinstance(result, dict) else None
        return _describe_contacts(list(contacts or []))
    if name == "send_text":
        where = f" to {target}" if target else ""
        at = f" at {number}" if number and target != number else ""
        suffix = " (simulated)" if simulated else ""
        return f"Sent your text{where}{at}{suffix}."
    if name == "call_phone":
        where = target or number or "the number"
        at = f" at {number}" if number and number != where else ""
        suffix = " (simulated)" if simulated else ""
        return f"Calling {where}{at}{suffix}."
    if name == "make_note":
        return f"Saved the note {title}."
    if name == "list_notes":
        notes = result.get("notes") if isinstance(result, dict) else None
        if not notes:
            return "You don't have any saved notes."
        titles = [str(note.get("title") or "Untitled") for note in notes]
        return "Your notes: " + "; ".join(titles) + "."
    if name == "search_notes":
        count = result.get("count") if isinstance(result, dict) else None
        if count is not None:
            return f"Found {count} matching notes."
        return "Here are the matching notes."
    return "Done."


def missing_contact_message(target: Optional[str], action: str) -> str:
    """Human-readable error when a call/text recipient has no number."""
    who = str(target or "that contact").strip() or "that contact"
    return (
        f"I don't have a phone number for {who}. "
        f"Save it first, then ask me to {action} {who}."
    )
