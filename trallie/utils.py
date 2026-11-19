import json
import re
import time


def post_process_response(response: str) -> str:
    """
    Removes <think>...</think> content (reasoning models) from the response.
    """
    return re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL).strip()


def normalize_attribute_name(name: str) -> str:
    """
    Canonical attribute name so that 'intendedUse', 'intended_use' and
    'Intended Use' are counted as the same attribute.
    """
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    name = re.sub(r"[_\-]+", " ", name)
    return " ".join(name.lower().split())


def format_schema(schema) -> str:
    """Serialise a schema (list of names or {name: description}) for a prompt."""
    if isinstance(schema, (dict, list)):
        return json.dumps(schema, ensure_ascii=False)
    return str(schema)


def backoff(attempt: int, max_delay: float = 30.0) -> None:
    """Sleep before retrying: 1s, 2s, 4s, ... capped at max_delay."""
    time.sleep(min(2 ** attempt, max_delay))
