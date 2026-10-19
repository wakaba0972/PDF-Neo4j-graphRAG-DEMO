from __future__ import annotations

from typing import Any


class TypedNameExtractionMethod:
    method_id = "typed_name"
    requires_entity_type = True

    def extraction_instructions(self) -> str:
        return (
            "Extract important entities and explicit relationships directly from the document. "
            "Assign concise, consistent types to entities and relationships; use the same type "
            "for the same concept across chunks. Do not invent extra items or infer unsupported facts. "
            "Keep entity names, relationship types, and descriptions in the source language. "
            "For English source text, write these fields in English and preserve original names "
            "and terminology; do not translate them into Chinese. Return entity objects with "
            "name, type, and description, and relationship objects with source, target, type, "
            "and description."
        )

    def output_shape(self) -> str:
        return (
            '{"entities":[{"name":"...","type":"...","description":"..."}],'
            '"relationships":[{"source":"...","target":"...",'
            '"type":"...","description":"..."}]}'
        )

    def entity_key(self, name: str, entity_type: str) -> tuple[str, ...]:
        return entity_type.casefold(), name.casefold()

    def make_entity(self, item: dict[str, Any], name: str) -> dict[str, Any]:
        return {
            "name": name,
            "type": str(item.get("type", "")).strip(),
            "description": str(item.get("description", "")).strip(),
        }
