from typing import Literal, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lobster.data import _PRESCIENT_AVAILABLE, DataFrameDatasetInMemory
from lobster.model._utils import model_typer

from ._graft_germlines import GraftGermline

if _PRESCIENT_AVAILABLE:
    import prescient.metrics.functional as pmf
    import prescient.transforms.functional as ptf

S3_PATH_TO_GERMLINES = (
    "s3://prescient-data-dev/sandbox/wanga84/preferred_absolve_germlines_aho.csv"
)
S3_PATH_TO_PKL_DF = "s3://prescient-data-dev/sandbox/wanga84/humanization/preferred_absolve_germlines_aho_tokenized.pkl"


def substitute_chars(str1, str2, mask):
    return "".join(s2 if m else s1 for s1, s2, m in zip(str1, str2, mask))


def humanize(
    fv_heavy: str,
    fv_light: str,
    fv_heavy_mask: Optional[torch.Tensor] = None,
    fv_light_mask: Optional[torch.Tensor] = None,
    keep_vernier_from_animal: Literal["human", "mouse", "rabbit"] = None,
    model_name: str = None,
    ckpt_path: str = None,
    model_type: Literal["LobsterPMLM", "LobsterPCLM"] = "LobsterPCLM",
    return_sequences: bool = True,
    naturalness: bool = False,
    smoke: bool = False,
) -> Union[Tuple[str, str], pd.DataFrame]:
    """Humanize sequences.

    If full masks are provided for both chains, all unmasked positions will be
    humanized. Otherwise, specify the keep_vernier_from_animal parameter to
    keep or humanize the vernier zone (species dependent).

    Parameters
    ----------
    fv_heavy: str
        Heavy chain sequence.
    fv_light: str
        Light chain sequence.
    fv_heavy_mask: Optional[torch.Tensor]  [bool][149]
        Mask of the heavy chain sequence. The mask is True for
        positions that *should* be humanized and False for positions
        that should not be mutated. Fully specifies positions.
    fv_light_mask: Optional[torch.Tensor]  [bool][148]
        Mask of the light chain sequence. The mask is True for
        positions that *should* be humanized and False for positions
        that should not be mutated. Fully specifies positions.
    keep_vernier_from_animal: Literal["human", "mouse", "rabbit"]
        Keep the vernier zone from a specific animal. Allowed
        species from prescient.constants.VERNIER_ZONES
    ckpt_path: str
        Path to a checkpoint to load the model from.
    model_name: str
        Name of the model to use.
    model_type: str
        Type of the model to use.
    return_sequences: bool
        If True, return the humanized sequences.
        If False, return a dataframe with all possible grafts scored.
    smoke: bool
        If True, run a smoke test.

    Returns
    -------
    best_fv_heavy, best_fv_light: Tuple[str, str]
        The highest log likelihood humanized sequences.
    df : pd.DataFrame
        A dataframe with all possible grafts scored.
        Columns are 'fv_heavy', 'fv_light', 'fv_heavy_likelihood', 'fv_light_likelihood'.
    """
    # Load the model
    model_cls = model_typer[model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name is not None:
        model = model_cls(
            model_name=model_name,
        )  # load a specific model, e.g. ESM2-8M
    if ckpt_path is not None:
        model = model_cls.load_from_checkpoint(
            ckpt_path, strict=False
        )  # load specific pre-trained chkpt
    model.eval()

    fv_heavy_aho, fv_light_aho = ptf.anarci_numbering([fv_heavy, fv_light])

    germline_df = pd.read_csv(S3_PATH_TO_GERMLINES)
    # germline_df = pd.read_pickle(S3_PATH_TO_PKL_DF)
    germline_df = germline_df.drop_duplicates(subset=["fv_heavy_aho", "fv_light_aho"])

    if smoke:
        germline_df = germline_df.sample(5)  # smoke test

    tokenization_transform = model._transform_fn

    # graft germlines with and without constant vernier zones
    graft_germlines = GraftGermline(germline_df, tokenization_transform)
    grafted_df, _, _ = graft_germlines.graft(fv_heavy_aho, fv_light_aho)

    graft_germlines_const_vernier = GraftGermline(
        germline_df, tokenization_transform, keep_vernier_from_animal="rabbit"
    )
    grafted_df_const_vernier, _, _ = graft_germlines_const_vernier.graft(
        fv_heavy_aho, fv_light_aho
    )

    grafts = pd.concat([grafted_df, grafted_df_const_vernier])
    grafts = grafts.dropna(subset=["fv_heavy_aho", "fv_light_aho"])
    grafts["fv_heavy"] = grafts["fv_heavy_aho"].apply(lambda x: x.replace("-", ""))
    grafts["fv_light"] = grafts["fv_light_aho"].apply(lambda x: x.replace("-", ""))
    unique_fv_heavy_grafts = grafts.drop_duplicates(subset=["fv_heavy"])
    unique_fv_light_grafts = grafts.drop_duplicates(subset=["fv_light"])
    unique_fv_heavy_grafts["edist_to_seed_fv_heavy"] = unique_fv_heavy_grafts[
        "fv_heavy"
    ].apply(lambda x: pmf._edit_distance._edit_distance(x, fv_heavy))
    unique_fv_light_grafts["edist_to_seed_fv_light"] = unique_fv_light_grafts[
        "fv_light"
    ].apply(lambda x: pmf._edit_distance._edit_distance(x, fv_light))
    fv_heavy_humanized = unique_fv_heavy_grafts["fv_heavy"].tolist()
    fv_light_humanized = unique_fv_light_grafts["fv_light"].tolist()

    if fv_heavy_mask is not None and fv_light_mask is not None:
        humanized_fv_heavy_aho_list = [
            substitute_chars(fv_heavy_aho, germline_fv_heavy_aho, fv_heavy_mask)
            for germline_fv_heavy_aho in germline_df["fv_heavy_aho"]
        ]
        humanized_fv_light_aho_list = [
            substitute_chars(fv_light_aho, germline_fv_light_aho, fv_light_mask)
            for germline_fv_light_aho in germline_df["fv_light_aho"]
        ]
        fv_heavy_humanized = [
            s.replace("-", "") for s in humanized_fv_heavy_aho_list
        ]  # ungap
        fv_light_humanized = [
            s.replace("-", "") for s in humanized_fv_light_aho_list
        ]  # ungap
        unique_fv_heavy_grafts["fv_heavy"] = fv_heavy_humanized
        unique_fv_light_grafts["fv_light"] = fv_light_humanized

    unique_fv_heavy_grafts_dataset = DataFrameDatasetInMemory(
        unique_fv_heavy_grafts, transform_fn=model._transform_fn, columns=["fv_heavy"]
    )
    unique_fv_light_grafts_dataset = DataFrameDatasetInMemory(
        unique_fv_light_grafts, transform_fn=model._transform_fn, columns=["fv_light"]
    )
    fv_heavy_batch_likelihoods = []
    fv_light_batch_likelihoods = []

    if model_type == "LobsterPCLM":
        for batch in DataLoader(
            unique_fv_heavy_grafts_dataset, batch_size=64, shuffle=False
        ):
            fv_heavy_batch_likelihoods.extend(
                model.batch_to_log_likelihoods(batch).to("cpu")
            )

        for batch in DataLoader(
            unique_fv_light_grafts_dataset, batch_size=64, shuffle=False
        ):
            fv_light_batch_likelihoods.extend(
                model.batch_to_log_likelihoods(batch).to("cpu")
            )
        fv_heavy_humanized_likelihoods = torch.stack(fv_heavy_batch_likelihoods)
        fv_light_humanized_likelihoods = torch.stack(fv_light_batch_likelihoods)

    elif model_type == "LobsterPMLM":  # slow
        fv_heavy_humanized_likelihoods = model.naturalness(fv_heavy_humanized)
        fv_light_humanized_likelihoods = model.naturalness(fv_light_humanized)

    fv_heavy_data = {}
    fv_light_data = {}
    if naturalness:  # slow
        fv_heavy_humanized_naturalness = pmf.naturalness(
            fv_heavy_humanized, device=device
        )
        fv_light_humanized_naturalness = pmf.naturalness(
            fv_light_humanized, device=device
        )
        fv_heavy_data[
            "fv_heavy_naturalness"
        ] = fv_heavy_humanized_naturalness.cpu().numpy()
        fv_light_data[
            "fv_light_naturalness"
        ] = fv_light_humanized_naturalness.cpu().numpy()

    fv_heavy_data["fv_heavy"] = fv_heavy_humanized
    fv_light_data["fv_light"] = fv_light_humanized
    fv_heavy_data["fv_heavy_likelihood"] = fv_heavy_humanized_likelihoods
    fv_light_data["fv_light_likelihood"] = fv_light_humanized_likelihoods

    fv_heavy_df = pd.DataFrame(fv_heavy_data)
    fv_light_df = pd.DataFrame(fv_light_data)

    best_fv_heavy = fv_heavy_df.loc[fv_heavy_df["fv_heavy_likelihood"].idxmax()][
        "fv_heavy"
    ]
    best_fv_light = fv_light_df.loc[fv_light_df["fv_light_likelihood"].idxmax()][
        "fv_light"
    ]

    if return_sequences:
        return best_fv_heavy, best_fv_light
    else:
        return pd.concat([fv_heavy_df, fv_light_df], axis=1)
