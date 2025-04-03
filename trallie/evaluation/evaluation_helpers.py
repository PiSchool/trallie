import pandas as pd
import json
from typing import List, Tuple
from datasets import Dataset
import difflib
from collections import defaultdict


################### SCHEMA GENERATION HELPERS ###########################
def group_descriptions(descriptions, group_size):
    """
    Groups descriptions into batches of a given size.
    """
    return [
        descriptions[i : i + group_size]
        for i in range(0, len(descriptions), group_size)
    ]


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


def prepare_eval_input(data_collection, sentences_per_group=20):
    eval_input = []
    current_group = []
    record_counter = 1  # Start record count at 1

    for example in data_collection["validation"]:
        tokens = example["tokens"]
        # Create the sentence with the "Record x:" format
        sentence = f"Record {record_counter}: " + " ".join(tokens)
        current_group.append(sentence)
        record_counter += 1  # Increment record counter for each sentence

        # Check if we reached the desired number of sentences in the group
        if len(current_group) >= sentences_per_group:
            # Join sentences in the group with a newline
            eval_input.append("\n".join(current_group))
            current_group = []  # Reset for the next group

    # Add any remaining sentences as the last group if not empty
    if current_group:
        eval_input.append("\n".join(current_group))

    return eval_input


def filter_duplicate_text_columns(dataset, column_name):
    """
    Filters duplicate text columns in a dataset.

    Args:
      dataset: The input dataset (Hugging Face Datasets format).
      column_name: The name of the text column to filter.

    Returns:
      A new dataset with duplicate text entries removed from the specified column.
    """

    # Convert the dataset column to a list
    text_column = dataset[column_name]

    # Efficiently find unique entries and their indices
    unique_texts = []
    unique_indices = []
    seen_texts = set()
    for index, text in enumerate(text_column):
        if text not in seen_texts:
            unique_texts.append(text)
            unique_indices.append(index)
            seen_texts.add(text)

    # Create a new dataset with only the unique entries
    filtered_dataset = Dataset.from_dict({column_name: unique_texts})

    return filtered_dataset


def remove_duplicate_chunks(text):
    """Remove duplicate chunks of text while preserving order."""
    words = text.split()
    seen = set()
    filtered_words = []

    for word in words:
        if word not in seen:
            seen.add(word)
            filtered_words.append(word)

    return " ".join(filtered_words)


def group_text_by_file_name(dataset):
    grouped = defaultdict(list)

    for row in dataset["validaion"]:  # Fixed typo
        grouped[row["file_name"]].append(row["text"])  # Group texts by file name

    # Remove duplicate phrases and return a list of deduplicated full texts
    return [remove_duplicate_chunks(" ".join(texts)) for texts in grouped.values()]


########### SCHEMA FILLING HELPERS #####################
def group_text_and_values_by_file_name(dataset):
    grouped = {}

    for row in dataset["validaion"]:  # Fixed typo
        file_name = row["file_name"]
        if file_name not in grouped:
            grouped[file_name] = {"keys": [], "values": [], "full_text": []}

        grouped[file_name]["keys"].append(row["key"])
        grouped[file_name]["values"].append(row["value"])
        grouped[file_name]["full_text"].append(row["text"])  # Collect all text chunks

    # Remove duplicate phrases and return cleaned data
    return [
        {
            "file_name": key,
            **value,
            "full_text": remove_duplicate_chunks(
                " ".join(value["full_text"])
            ),  # Clean duplicate chunks
        }
        for key, value in grouped.items()
    ]
