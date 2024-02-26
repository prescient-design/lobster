import os
import tempfile
from typing import Any

import torch
from prescient.transforms import Transform

from lobster.model import PrescientPLMFold


class FoldseekTransform(Transform):
    """
    Transforms a structure (PDB) into a discretized 3Di sequence.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (AA_seq, 3Di_struc_seq, combined_seq).

    """

    def __init__(
        self,
        foldseek: str,
        lobster_fold_model_name: str = "esmfold_v1",
        linker_length: int = 25,
    ):
        super().__init__()
        assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
        self._foldseek = foldseek
        self._lobster_fold_model_name = lobster_fold_model_name
        self._linker_length = linker_length
        self._model = PrescientPLMFold(model_name=self._lobster_fold_model_name)
        self._model.eval()
        self._model.model.trunk.set_chunk_size(64)

    def transform(self, sequences: list[str], chains: list = None) -> dict:

        pdb_file = self._lobster_fold_transform(sequences)
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".pdb"
        ) as pdb_temp_file, tempfile.NamedTemporaryFile(
            delete=True, suffix=".tsv"
        ) as tsv_temp_file:
            with open(pdb_temp_file.name, "w") as f:
                f.write(pdb_file)

            path_to_pdb = pdb_temp_file.name

            cmd = f"{self._foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path_to_pdb} {tsv_temp_file.name}"
            os.system(cmd)

            seq_dict = {}
            name = os.path.basename(path_to_pdb)
            with open(tsv_temp_file.name, "r") as file_handle:
                for _i, line in enumerate(file_handle):
                    # print(line)
                    desc, seq, struc_seq = line.split("\t")[:3]

                    name_chain = desc.split(" ")[0]
                    chain = name_chain.replace(name, "").split("_")[-1]

                    if chains is None or chain in chains:
                        if chain not in seq_dict:
                            combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                            seq_dict[chain] = (seq, struc_seq, combined_seq)

        return seq_dict

    def _lobster_fold_transform(self, sequences: list[str]) -> str:
        # TODO: currently only supports monomer and dimer
        if len(sequences) > 1:  # dimer
            linker = "G" * self._linker_length
            sequence = f"{linker}".join(sequences)
            print(sequence)
        else:  # monomer
            sequence = sequences[0]
        tokenized_input = self._model.tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        # add a large offset to the position IDs of the second chain
        if len(sequences) > 1:
            with torch.no_grad():
                position_ids = torch.arange(len(sequence), dtype=torch.long)
                position_ids[len(sequence) + len(linker) :] += 512
                tokenized_input["position_ids"] = position_ids.unsqueeze(0)
                output = self._model.model(**tokenized_input)

                # remove the poly-G linker from the output, so we can display the structure as fully independent chains
                linker_mask = torch.tensor(
                    [1] * len(sequence) + [0] * len(linker) + [1] * len(sequence)
                )[None, :, None]
                # output['atom37_atom_exists'] = output['atom37_atom_exists'] * linker_mask.to(output['atom37_atom_exists'].device)
                output["atom37_atom_exists"] = output["atom37_atom_exists"] * linker_mask
        else:
            with torch.no_grad():
                output = self._model.model(tokenized_input)

        pdb_file = self._model.model.output_to_pdb(output)[0]

        return pdb_file

    def validate(self, flat_inputs: list[Any]) -> None:
        pass
