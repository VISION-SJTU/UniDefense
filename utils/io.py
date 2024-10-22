"""
Input/output utils.
"""

import json


def load_from_json(filename: str):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.endswith(".json")
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: str, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.endswith(".json")
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)