from typing import Tuple, Union

import pandas as pd
import torch

from lobster.data import _PRESCIENT_AVAILABLE
from lobster.model import LobsterPMLM

if _PRESCIENT_AVAILABLE:
    import prescient.metrics.functional as pmf
    import prescient.transforms.functional as ptf


def fix_liabilities(
    fv_heavy: str,
    fv_light: str,
    model_name: str = None,
    ckpt_path: str = None,
    return_sequences=True,
) -> Union[Tuple[str, str], pd.DataFrame]:
    """Fix liabilities of a sequence.

    Parameters
    ----------
    fv_heavy: str
        Heavy chain sequence.
    fv_light: str
        Light chain sequence.
    model_name: str
        Lobster model name to use. If None, the default is esm2_t6_8M_UR50D
    ckpt_path: str
        Path to a checkpoint to load the model from.
    return_sequences: bool
        If True, return the fixed sequences.
        If False, return a dataframe with the fixed liabilities.

    Returns
    -------
    fv_heavy_fixed, fv_light_fixed: Tuple[str, str]
        Fixed heavy and light chain sequences.
    liability_fix_df: pd.DataFrame
        Dataframe with fixed liabilities. Columns are:
            - liability: The liability name.
            - chain: The chain where the liability is located.
            - aho_idx: The AHo index of the liability.
            - top1: The top 1 mutation.
            - top2: The top 2 mutation.
            - top3: The top 3 mutation.
    """
    # Load the model
    if model_name:
        model = LobsterPMLM(model_name=model_name)
    elif ckpt_path:
        model = LobsterPMLM.load_from_checkpoint(ckpt_path, strict=False)
    else:
        model = LobsterPMLM(model_name="esm2_t6_8M_UR50D")
    model.eval()

    fv_heavy_aho, fv_light_aho = ptf.anarci_numbering([fv_heavy, fv_light])
    liabilities_bool = pmf.liabilities(fv_heavy_aho, fv_light_aho, return_indices=False)
    liabilities_idx = pmf.liabilities(fv_heavy_aho, fv_light_aho, return_indices=True)

    liability_list = []
    fv_heavy_aho_fixed = list(fv_heavy_aho)
    fv_light_aho_fixed = list(fv_light_aho)
    # Fix liabilities
    for lbool, lidx in zip(liabilities_bool, liabilities_idx):
        if lbool[1]:
            idx = lidx[1][0]  # in AHo numbering
            chain = lidx[0].split("_")[0]
            if chain == "heavy":
                masked_sequence = list(fv_heavy_aho)
            else:
                masked_sequence = list(fv_light_aho)
            masked_sequence[idx] = "<mask>"
            masked_sequence = [m for m in masked_sequence if m != "-"]  # ungap
            idx_mask = masked_sequence.index("<mask>")  # no AHo
            with torch.inference_mode():
                masked_encoded = torch.tensor([model.tokenizer.encode(masked_sequence)])
                h = model.model(input_ids=masked_encoded, output_hidden_states=True)[
                    "hidden_states"
                ][-1]
                logits = model.model.lm_head(h)
                mutation_list = model.tokenizer.decode(
                    torch.topk(logits, 20, dim=-1).indices[0][idx_mask + 1]
                ).split(" ")  # +1 for <cls> token
            data = {"liability": lbool[0], "chain": chain, "aho_idx": idx} | {
                f"top{k}": v for k, v in enumerate(mutation_list, 1)
            }
            if return_sequences:
                if chain == "heavy":
                    for m in mutation_list:
                        if m != fv_heavy_aho[idx]:
                            fv_heavy_aho_fixed[
                                idx
                            ] = m  # replace with novel top1 mutation
                            break
                else:
                    for m in mutation_list:
                        if m != fv_light_aho[idx]:
                            fv_light_aho_fixed[
                                idx
                            ] = m  # replace with novel top1 mutation
                            break
            liability_list.append(data)

    liability_fix_df = pd.DataFrame(liability_list)

    if return_sequences:
        return "".join(fv_heavy_aho_fixed).replace("-", ""), "".join(
            fv_light_aho_fixed
        ).replace("-", "")
    else:
        return liability_fix_df
