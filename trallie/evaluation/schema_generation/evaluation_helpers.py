import pandas as pd
import json
from typing import List, Tuple

def group_descriptions(descriptions, group_size):
    """
    Groups descriptions into batches of a given size.
    """
    return [descriptions[i:i + group_size] for i in range(0, len(descriptions), group_size)]

def validate_json(output):
    """
    Validates if the given string is a valid JSON.
    """
    try:
        json.loads(output)
        return True
    except json.JSONDecodeError:
        return False

def extract_keys(schema, parent_key=None):
    """
    Recursively extract all keys from a JSON schema.
    """
    keys = set()
    for key, value in schema.items():
        full_key = f"{parent_key}.{key.lower()}" if parent_key else key.lower()
        keys.add(full_key)
        if isinstance(value, dict):
            keys.update(extract_keys(value, full_key))
    return keys
