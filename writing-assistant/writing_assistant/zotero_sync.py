"""Utilities for syncing structured article metadata into Zotero."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pyzotero import zotero


class ZoteroSyncError(RuntimeError):
    """Raised when a Zotero operation fails."""


@dataclass
class ZoteroSyncResult:
    key: str
    select_uri: str
    web_url: Optional[str]
    existed: bool


FIELD_ALIASES = {
    "doi": "DOI",
    "publicationTitle": "publicationTitle",
    "conferenceName": "conferenceName",
    "proceedingsTitle": "proceedingsTitle",
    "bookTitle": "bookTitle",
    "institution": "institution",
    "abstractNote": "abstractNote",
    "text_summary": "abstractNote",
}


def _build_zotero_client() -> zotero.Zotero:
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    api_key = os.getenv("ZOTERO_API_KEY")

    if not library_id or not api_key:
        raise ZoteroSyncError(
            "Zotero credentials missing. Set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY in the environment."
        )

    try:
        return zotero.Zotero(library_id, library_type, api_key)
    except Exception as exc:  # pragma: no cover - network/auth failure
        raise ZoteroSyncError(f"Failed to initialise Zotero client: {exc}") from exc


def _normalise_creators(creators: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for creator in creators:
        if not isinstance(creator, dict):
            continue
        creator_type = creator.get("creatorType") or "author"
        first = creator.get("firstName", "")
        last = creator.get("lastName", "")
        name = creator.get("name")
        if name and not (first or last):
            cleaned.append({"creatorType": creator_type, "name": name})
        else:
            cleaned.append(
                {
                    "creatorType": creator_type,
                    "firstName": first,
                    "lastName": last,
                }
            )
    return cleaned


def _normalise_tags(tags: Any) -> List[Dict[str, str]]:
    if not tags:
        return []
    if isinstance(tags, list):
        normalised: List[Dict[str, str]] = []
        for tag in tags:
            if isinstance(tag, dict) and tag.get("tag"):
                normalised.append({"tag": str(tag["tag"])})
            elif isinstance(tag, str) and tag.strip():
                normalised.append({"tag": tag.strip()})
        return normalised
    if isinstance(tags, str) and tags.strip():
        return [{"tag": tags.strip()}]
    return []


def _choose_existing_item(
    client: zotero.Zotero,
    doi: Optional[str],
    title: Optional[str],
) -> Optional[Dict[str, Any]]:
    try:
        if doi:
            client.add_parameters(q=doi, qmode="everything", itemType="-attachment", limit=1)
            matches = client.items()
            if matches:
                return matches[0]
        if title:
            client.add_parameters(q=title, qmode="titleCreatorYear", itemType="-attachment", limit=1)
            matches = client.items()
            if matches:
                return matches[0]
    except Exception:
        return None
    return None


def _build_web_url(key: str) -> Optional[str]:
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    if not library_id:
        return None
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user").lower()
    if library_type not in {"user", "group"}:
        library_type = "user"
    base = "users" if library_type == "user" else "groups"
    return f"https://www.zotero.org/{base}/{library_id}/items/{key}"


def sync_structured_item(structured: Dict[str, Any]) -> ZoteroSyncResult:
    items = structured.get("items")
    if not isinstance(items, list) or not items:
        raise ZoteroSyncError("Structured payload does not contain any Zotero item entries.")

    entry = items[0]
    if not isinstance(entry, dict):
        raise ZoteroSyncError("Zotero item entry is malformed.")

    item_type = entry.get("itemType") or "journalArticle"

    client = _build_zotero_client()

    try:
        template = client.item_template(item_type)
    except Exception as exc:
        raise ZoteroSyncError(f"Unable to retrieve Zotero template for '{item_type}': {exc}") from exc

    payload = {key: value for key, value in template.items() if key not in {"creators", "tags"}}

    for key, value in entry.items():
        normalised_key = FIELD_ALIASES.get(key, key)
        if normalised_key in payload and value not in (None, ""):
            payload[normalised_key] = value

    creators = _normalise_creators(entry.get("creators", []))
    if creators:
        payload["creators"] = creators

    tags = _normalise_tags(entry.get("tags"))
    if tags:
        payload["tags"] = tags

    if abstract := structured.get("context", {}).get("summary"):
        payload.setdefault("abstractNote", abstract)

    doi = entry.get("doi") or entry.get("DOI")
    title = entry.get("title")

    existing = _choose_existing_item(client, doi=doi, title=title)
    if existing:
        key = existing.get("key") or existing.get("data", {}).get("key")
        if not key:
            raise ZoteroSyncError("Existing Zotero item lacks a key identifier.")
        select_uri = f"zotero://select/items/{key}"
        return ZoteroSyncResult(
            key=key,
            select_uri=select_uri,
            web_url=_build_web_url(key),
            existed=True,
        )

    try:
        response = client.create_items([payload])
    except Exception as exc:  # pragma: no cover - network failure
        raise ZoteroSyncError(f"Failed to create Zotero item: {exc}") from exc

    success = response.get("success") if isinstance(response, dict) else None
    if not success:
        raise ZoteroSyncError(f"Zotero creation returned no success payload: {response}")
    key = next(iter(success.values()))
    select_uri = f"zotero://select/items/{key}"
    return ZoteroSyncResult(
        key=key,
        select_uri=select_uri,
        web_url=_build_web_url(key),
        existed=False,
    )


__all__ = ["sync_structured_item", "ZoteroSyncResult", "ZoteroSyncError"]
