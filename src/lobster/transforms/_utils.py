import json
import random
from typing import Any, Dict, List


def random_boolean_choice() -> bool:
    return random.choice([True, False])


def invert_residue_to_codon_mapping(mapping: Dict[Any, List[Any]]) -> Dict[Any, Any]:
    reversed = {}
    for k, v in mapping.items():
        for j in v:
            reversed[j] = k
    return reversed


def json_load(json_file) -> Any:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def uniform_sample(vals: List[Any]) -> Any:
    return random.sample(vals, 1)[0]


def sample_list_with_probs(vals: List[Any], probs: List[float]) -> Any:
    return random.choices(vals, weights=probs, k=1)[0]
