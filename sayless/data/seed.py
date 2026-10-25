"""Seed contacts so SayLess works out of the box."""

from typing import Dict, List, Optional

# These are demo contacts. They make it possible to try listing and texting
# immediately, before the user has added anyone. Texting uses the local
# phone provider (a log file by default, or SAYLESS_PHONE_TEXT_COMMAND).
SEED_CONTACTS: List[Dict[str, Optional[str]]] = [
    {
        "name": "Judy Sellers",
        "phone": "+1 224-875-9162",
        "email": "judy@example.com",
    },
    {
        "name": "Mom",
        "phone": "+1 224-879-1662",
        "email": None,
    },
    {
        "name": "Alex Rivera",
        "phone": "+1 415-555-0134",
        "email": "alex.rivera@example.com",
    },
    {
        "name": "Priya Patel",
        "phone": "+1 646-555-0198",
        "email": "priya@example.com",
    },
    {
        "name": "Sam Chen",
        "phone": "+1 312-555-0177",
        "email": None,
    },
]
