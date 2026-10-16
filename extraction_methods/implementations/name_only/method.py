from __future__ import annotations

from typing import Any


class NameOnlyExtractionMethod:
    method_id = "name_only"
    requires_entity_type = False

    def extraction_instructions(self) -> str:
        return (
            "Extract important entities and explicit relationships directly from the document. "
            "Do not assign entity types. Assign concise, consistent relationship types and use "
            "the same relationship type for the same concept across chunks. Do not invent extra "
            "items or infer unsupported facts. Keep entity names, relationship types, and "
            "descriptions in the source language. For English source text, write these fields "
            "in English and preserve original names and terminology; do not translate them into "
            "Chinese. Return entity objects with name and description, and relationship objects "
            "with source, target, type, and description."
        )

    def output_shape(self) -> str:
        return (
            '{"entities":[{"name":"...","description":"..."}],'
            '"relationships":[{"source":"...","target":"...",'
            '"type":"...","description":"..."}]}'
        )

    def entity_key(self, name: str, entity_type: str) -> tuple[str, ...]:
        normalized_name = " ".join(name.casefold().split())
        return (normalized_name,)

    def make_entity(self, item: dict[str, Any], name: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": str(item.get("description", "")).strip(),
        }
