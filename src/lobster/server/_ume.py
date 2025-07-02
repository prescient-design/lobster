import functools

import einops
import torch
from beignet.constants import STANDARD_RESIDUES

from lobster.model import UME

from ._utils import single_position_masked_sequences, batched


@functools.cache
def get_ume_cached():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ume = UME.from_pretrained("ume-mini-base-12M")
    ume.to(device)
    ume.eval()

    return ume


def ume_aa_naturalness(sequence: str, model, batch_size: int = 16) -> dict:
    L = len(sequence)
    tokenizer = model.get_tokenizer("amino_acid")
    vocab = tokenizer.get_vocab()

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
                model.model.decoder(model.embed(tokenizer(input, return_tensors="pt"), aggregate=False))
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

    return {"logp": logp, "wt_logp": wt_logp, "naturalness": naturalness, "encoded": encoded_aa_id.cpu().tolist()}
