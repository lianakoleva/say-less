"""Contacts management."""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from .json_store import JSONStore


def normalize_name(name: Optional[str]) -> str:
    """Normalize a name for matching (case, possessives, punctuation)."""
    text = str(name or "").strip().lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^a-z0-9@.+\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contact_id(name: str) -> str:
    normalized = normalize_name(name)
    return normalized.replace(" ", "_") or "contact"


class ContactsManager:
    """Manage contacts.

    A contact only needs a name plus at least one of a phone number or an
    email address. Contacts created from a name alone are allowed too, so the
    user can add the number or email in a follow-up turn.
    """

    def __init__(self):
        self.store = JSONStore("contacts")

    async def add_contact(self, name: str, phone: Optional[str] = None,
                         email: Optional[str] = None, **kwargs) -> str:
        """Add a contact, merging into an existing contact with the same name.

        This is an upsert: saving a phone number and then an email address with
        the same name updates one contact instead of creating duplicates or
        wiping the previously saved field.
        """
        existing = await self.find_contact(name)
        now = datetime.now().isoformat()

        if existing:
            contact = dict(existing)
            # Keep the more complete name (for example "Judy Smith" over
            # "Judy"), and never lose an existing phone or email.
            current_name = str(contact.get("name") or "")
            if len(normalize_name(name).split()) > len(normalize_name(current_name).split()):
                contact["name"] = name
            if phone:
                contact["phone"] = phone
            if email:
                contact["email"] = email
            for key, value in kwargs.items():
                if value is not None:
                    contact[key] = value
            contact["updated_at"] = now
            await self.store.save(contact["id"], contact)
            return contact["id"]

        contact_id = _contact_id(name)
        contact = {
            "id": contact_id,
            "name": name,
            "phone": phone,
            "email": email,
            "created_at": now,
            "updated_at": now,
            **kwargs
        }
        await self.store.save(contact_id, contact)
        return contact_id

    async def get_contact(self, contact_id: str) -> Optional[Dict[str, Any]]:
        """Get a contact by ID."""
        return await self.store.load(contact_id)

    async def get_contact_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a contact by exact (normalized) name."""
        contact = await self.store.load(_contact_id(name))
        if contact:
            return contact
        return await self.find_contact(name)

    async def find_contact(self, reference: Optional[str]) -> Optional[Dict[str, Any]]:
        """Find a contact by name, first name, contact ID, or phone number.

        Returns ``None`` when the reference is ambiguous rather than guessing
        between two contacts with the same first name.
        """
        target = normalize_name(reference)
        if not target:
            return None

        contacts = await self.list_contacts()
        digit_target = re.sub(r"\D", "", str(reference or ""))

        # Exact name / ID.
        for contact in contacts:
            if normalize_name(contact.get("name")) == target:
                return contact
            if normalize_name(contact.get("id")) == target.replace(" ", "_"):
                return contact

        # Phone number.
        if len(digit_target) >= 7:
            for contact in contacts:
                if re.sub(r"\D", "", str(contact.get("phone") or "")) == digit_target:
                    return contact

        # Full name starts with the reference ("Judy" -> "Judy Smith").
        prefix_matches = [
            contact for contact in contacts
            if normalize_name(contact.get("name")).startswith(target + " ")
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

        # First-name match, but only when it is unambiguous.
        first_matches = [
            contact for contact in contacts
            if normalize_name(contact.get("name")).split()[:1] == target.split()[:1]
        ]
        if len(first_matches) == 1:
            return first_matches[0]
        return None

    async def find_contact_in_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Find a saved contact whose name appears in ``text``."""
        source = str(text or "")
        if not source:
            return None
        contacts = await self.list_contacts()
        best: Optional[Dict[str, Any]] = None
        best_length = 0
        for contact in contacts:
            name = str(contact.get("name") or "").strip()
            if not name:
                continue
            candidates = {name, name.split()[0] if name.split() else name}
            for candidate in candidates:
                if len(candidate) < 2:
                    continue
                if re.search(rf"\b{re.escape(candidate)}\b", source, re.IGNORECASE):
                    if len(name) > best_length:
                        best = contact
                        best_length = len(name)
                    break
        return best

    async def search_contacts(self, query: str) -> List[Dict[str, Any]]:
        """Search contacts by name, phone, or email."""
        all_contacts = await self.store.load_all()
        query_lower = str(query or "").lower()
        query_digits = re.sub(r"\D", "", query_lower)
        results = []
        for contact in sorted(
            all_contacts.values(), key=lambda item: str(item.get("name", "")).lower()
        ):
            name = str(contact.get("name", "")).lower()
            phone = str(contact.get("phone") or "")
            email = str(contact.get("email") or "").lower()
            phone_digits = re.sub(r"\D", "", phone)
            if (
                query_lower in name
                or query_lower in phone
                or query_lower in email
                or (len(query_digits) >= 3 and query_digits in phone_digits)
            ):
                results.append(contact)
        return results

    async def list_contacts(self) -> List[Dict[str, Any]]:
        """List all contacts, sorted by name."""
        all_contacts = await self.store.load_all()
        return sorted(
            all_contacts.values(), key=lambda item: str(item.get("name", "")).lower()
        )

    async def update_contact(self, contact_id: str, updates: Dict[str, Any]) -> bool:
        """Update a contact."""
        contact = await self.get_contact(contact_id)
        if not contact:
            return False
        for key, value in updates.items():
            if value is not None:
                contact[key] = value
        contact["updated_at"] = datetime.now().isoformat()
        return await self.store.save(contact_id, contact)

    async def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact."""
        return await self.store.delete(contact_id)

    async def seed_demo_contacts(self) -> int:
        """Add any missing demo contacts without overwriting existing data.

        Returns the number of contacts added. Running this repeatedly is safe.
        """
        from .seed import SEED_CONTACTS

        added = 0
        for entry in SEED_CONTACTS:
            name = str(entry["name"])
            existing = await self.find_contact(name)
            if existing:
                # Fill in missing contact details without replacing anything
                # the user already saved.
                updates = {}
                if not existing.get("phone") and entry.get("phone"):
                    updates["phone"] = entry["phone"]
                if not existing.get("email") and entry.get("email"):
                    updates["email"] = entry["email"]
                if updates:
                    await self.update_contact(existing["id"], updates)
                    added += 1
                continue
            await self.add_contact(
                name=name,
                phone=entry.get("phone"),
                email=entry.get("email"),
                seeded=True,
            )
            added += 1
        return added
