import functools
from typing import Literal

import einops
import torch
from transformers import EsmForMaskedLM, EsmTokenizer
from beignet.constants import STANDARD_RESIDUES

from ._utils import single_position_masked_sequences, batched


@functools.cache
def get_esm_cached(
    mlm_model_name: Literal[
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
    ] = "facebook/esm2_t6_8M_UR50D",
):
    assert mlm_model_name in {
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
    }
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = EsmTokenizer.from_pretrained(mlm_model_name, clean_up_tokenization_spaces=False)
    model = EsmForMaskedLM.from_pretrained(mlm_model_name)
    model.to(device)
    model.eval()

    return model, tokenizer


def esm_aa_naturalness(sequence: str, model, tokenizer, batch_size: int = 16) -> dict:
    L = len(sequence)
    vocab = tokenizer.get_vocab()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    assert set(sequence) <= set(STANDARD_RESIDUES)

    vocab_id_to_aa_id = {vocab[aa]: i for i, aa in enumerate(sorted(STANDARD_RESIDUES))}
    amino_acid_vocab_ids = list(vocab_id_to_aa_id.keys())

    encoded = tokenizer(sequence, return_tensors="pt")
    encoded_aa_id = torch.as_tensor(
        [vocab_id_to_aa_id[x] for x in encoded["input_ids"].squeeze(0)[1:-1].cpu().tolist()], device=model.device
    )

    masked_sequences = single_position_masked_sequences(sequence)

    with torch.inference_mode():
        logits = torch.cat(
            [
                model(**{k: v.to(device) for k, v in tokenizer(input, return_tensors="pt").items()}).logits
                for input in batched(masked_sequences, batch_size)
            ],
            dim=0,
        )
        logits = torch.diagonal(logits[:, 1:-1, :], dim1=0, dim2=1)
        logits = einops.rearrange(logits, "token length -> length token", length=L)

        logits = logits[:, torch.as_tensor(amino_acid_vocab_ids, device=model.device)]
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        wt_logp = logp[torch.arange(L), encoded_aa_id]
        naturalness = torch.exp(wt_logp.mean()).item()

    return {
        "logp": logp.cpu().tolist(),
        "wt_logp": wt_logp.cpu().tolist(),
        "naturalness": naturalness,
        "encoded": encoded_aa_id.cpu().tolist(),
    }
