import json
import random
from pathlib import Path
from typing import Any


def random_boolean_choice() -> bool:
    return random.choice([True, False])


def invert_residue_to_codon_mapping(mapping: dict[Any, list[Any]]) -> dict[Any, Any]:
    reversed = {}
    for k, v in mapping.items():
        for j in v:
            reversed[j] = k
    return reversed


def json_load(json_file: str | Path) -> Any:
    with open(json_file) as f:
        data = json.load(f)
    return data


def uniform_sample(vals: list[Any]) -> Any:
    return random.sample(vals, 1)[0]


def sample_list_with_probs(vals: list[Any], probs: list[float]) -> Any:
    return random.choices(vals, weights=probs, k=1)[0]
