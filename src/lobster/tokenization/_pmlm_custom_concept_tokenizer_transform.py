import importlib.resources
import re
from typing import Any, Dict, List, Union

import torch
from transformers.tokenization_utils_base import (
    BatchEncoding,
)

from lobster.transforms import Transform

VOCAB_PATH = importlib.resources.files("lobster") / "assets" / "uniref_tokenzier"


def extract_unirefswissport_fields(input_string):
    # Define placeholders for results and current content
    input_string = input_string[1:]
    concepts = input_string.split("$$$")

    concepts_value = {}
    for concept in concepts:
        name, value = concept.split("=")
        concepts_value[name] = value
    return concepts_value


def build_lookup_table(file_path, concept_name=""):
    lookup_table = {}
    concept_names = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            # Strip line breaks and leading/trailing whitespace
            stripped_line = line.strip()
            if stripped_line not in lookup_table:
                lookup_table[stripped_line] = line_number - 1
                concept_names.append(concept_name + " " + stripped_line)
    return lookup_table, line_number, concept_names


class UnirefDescriptorTransform(Transform):
    def __init__(
        self,
    ):
        self.cluster_name_lookup, self.num_cluster_name, self.cluster_name_concepts = build_lookup_table(
            VOCAB_PATH / "cluster_name_unique_values.txt", "cluster_name"
        )
        self.members_lookup, self.num_members, self.members_concepts = build_lookup_table(
            VOCAB_PATH / "members_unique_values.txt", "members"
        )
        self.taxon_id_lookup, self.num_taxon_id, self.taxon_concepts = build_lookup_table(
            VOCAB_PATH / "taxon_id_unique_values.txt", "taxon"
        )
        self.concepts_names = ["cluster_name", "members", "taxon_id"]
        self.concepts_type = ["cat", "cat", "cat"]
        self.concept_max_values = [self.num_cluster_name + 1, self.num_members + 1, self.num_taxon_id + 1]
        self.concept_emb = [10, 2, 5]
        super().__init__()

    def transform(
        self,
        text: Union[str, List[str], List[int]],
        parameters: dict[str, Any],
    ) -> BatchEncoding:
        tokenized = {}
        # Remove '>' before splitting
        text = text[1:]
        # Split the string to isolate the unique identifier and the rest
        unique_identifier, rest = text.split(" ", 1)

        # Find the numeric value for 'n='
        members = re.search(r"n=(\d+)", rest).group(1)
        # Extract the ClusterName, assuming it ends before ' n='
        cluster_name = rest.split(" n=")[0]
        # Find the TaxID by locating it between 'TaxID=' and ' RepID='
        taxon_id = re.search(r"TaxID=(\d+)", rest).group(1)

        if cluster_name in self.cluster_name_lookup:
            # here we add one because I reserve "0" for a value that has not been seen before
            tokenized["cluster_name"] = torch.tensor([self.cluster_name_lookup[cluster_name] + 1])
        else:
            tokenized["cluster_name"] = torch.tensor([0])

        if members in self.members_lookup:
            # here we add one because I reserve "0" for a value that has not been seen before
            tokenized["members"] = torch.tensor([self.members_lookup[members] + 1])
        else:
            tokenized["members"] = torch.tensor([0])

        if taxon_id in self.taxon_id_lookup:
            # here we add one because I reserve "0" for a value that has not been seen before
            tokenized["taxon_id"] = torch.tensor([self.taxon_id_lookup[taxon_id] + 1])
        else:
            tokenized["taxon_id"] = torch.tensor([0])

        tokenized["all_concepts"] = torch.cat(
            (tokenized["cluster_name"], tokenized["members"], tokenized["taxon_id"]), dim=0
        )
        return tokenized

    def _reverse_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, str):
            return text[::-1]
        elif isinstance(text, list):
            return [t[::-1] for t in text]

    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        return self.transform(input, parameters)

    def validate(self, flat_inputs: list[Any]) -> None:
        pass

    def _check_inputs(self, inputs: List[Any]) -> None:
        pass


class UnirefSwissPortDescriptorTransform(Transform):
    def __init__(
        self,
    ):
        self.cluster_name_lookup, self.num_cluster_name, cluster_name_concepts = build_lookup_table(
            VOCAB_PATH / "cluster_name_unique_values_100.txt", "cluster_name"
        )
        self.biological_process_lookup, self.num_biological_process, biological_process_concepts = build_lookup_table(
            VOCAB_PATH / "biological_process_unique_values_100.txt", "biological_process"
        )
        self.cellular_component_lookup, self.num_cellular_component, cellular_component_concepts = build_lookup_table(
            VOCAB_PATH / "cellular_component_unique_values_100.txt", "cellular_component"
        )
        self.molecular_function_lookup, self.num_molecular_function, molecular_function_concepts = build_lookup_table(
            VOCAB_PATH / "molecular_function_unique_values_100.txt", "molecular_function"
        )
        self.organism_lookup, self.num_organism, organism_concepts = build_lookup_table(
            VOCAB_PATH / "organism_unique_values_100.txt", "organism"
        )
        self.taxon_lookup, self.num_taxon, taxon_concepts = build_lookup_table(
            VOCAB_PATH / "taxon_unique_values_100.txt", "taxon"
        )

        self.concepts_names = [
            "cluster_name",
            "Gene_Ontology(biological_process)",
            "Gene_Ontology(cellular_component)",
            "Gene_Ontology(molecular_function)",
            "Organism",
            "taxon",
        ]
        self.concept_size = [
            self.num_cluster_name,
            self.num_biological_process,
            self.num_cellular_component,
            self.num_molecular_function,
            self.num_organism,
            self.num_taxon,
        ]
        self.concept_lookup = [
            self.cluster_name_lookup,
            self.biological_process_lookup,
            self.cellular_component_lookup,
            self.molecular_function_lookup,
            self.organism_lookup,
            self.taxon_lookup,
        ]
        self.emd_size = sum(self.concept_size) // 10
        self.concepts_names_full = (
            cluster_name_concepts
            + biological_process_concepts
            + cellular_component_concepts
            + molecular_function_concepts
            + organism_concepts
            + taxon_concepts
        )
        # print(self.organism_lookup)
        super().__init__()

    def transform(
        self,
        text: Union[str, List[str], List[int]],
        parameters: dict[str, Any],
    ) -> BatchEncoding:
        concepts_value = extract_unirefswissport_fields(text)
        tokenized = {}
        # print(concepts_value)
        for i in range(len(self.concepts_names)):
            tokenized[self.concepts_names[i]] = torch.zeros(self.concept_size[i])
            tokenized[self.concepts_names[i]][self.concept_lookup[i][concepts_value[self.concepts_names[i]]]] = 1
            if i == 0:
                tokenized["all_concepts"] = tokenized[self.concepts_names[i]]
            else:
                # print(tokenized["all_concepts"].shape)

                tokenized["all_concepts"] = torch.cat(
                    (tokenized["all_concepts"], tokenized[self.concepts_names[i]]), dim=0
                )

        return tokenized

    def _reverse_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, str):
            return text[::-1]
        elif isinstance(text, list):
            return [t[::-1] for t in text]

    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        return self.transform(input, parameters)

    def validate(self, flat_inputs: list[Any]) -> None:
        pass

    def _check_inputs(self, inputs: List[Any]) -> None:
        pass

    def _return_unk(self, tokenized):
        for i in range(len(self.concepts_names)):
            tokenized[self.concepts_names[i]] = torch.zeros(1, self.concept_size[i])
            tokenized[self.concepts_names[i]][0, self.concept_lookup[i][0]] = 1
            if i == 0:
                tokenized["all_concepts"] = tokenized[self.concepts_names[i]]
            else:
                tokenized["all_concepts"] = torch.cat(
                    (tokenized["all_concepts"], tokenized[self.concepts_names[i]]), dim=0
                )

        return tokenized


CUSTOM_TOKENIZER = {
    "UnirefDescriptorTransform": UnirefDescriptorTransform,
    "UnirefSwissPortDescriptorTransform": UnirefSwissPortDescriptorTransform,
}
