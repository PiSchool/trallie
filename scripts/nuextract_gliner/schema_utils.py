"""Helpers to load input text and build extraction configs from JSON schema files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_text_from_path(path: Path) -> str:
    """Load document text from a UTF-8 text file or extract text from a PDF."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        out = "\n".join(parts).strip()
        if not out:
            raise ValueError(f"No extractable text in PDF: {path}")
        return out
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def attributes_from_schema(schema: dict[str, Any]) -> list[str]:
    """Accepts trallie-style ``{"domain": "...", "attributes": [...]}`` or a bare list."""
    if isinstance(schema, list):
        return [str(x) for x in schema]
    attrs = schema.get("attributes")
    if not isinstance(attrs, list):
        raise ValueError("Schema must be a list of names or a dict with an 'attributes' list.")
    return [str(x) for x in attrs]


def nuextract_template_from_attributes(names: list[str], field_type: str = "string") -> dict[str, Any]:
    """
    Build a flat NuExtract template: each field maps to a supported type.

    See NuMind docs: ``string`` and ``verbatim-string`` are common choices.
    """
    if field_type not in {"string", "verbatim-string"}:
        raise ValueError("field_type must be 'string' or 'verbatim-string'")
    return {name: field_type for name in names}
