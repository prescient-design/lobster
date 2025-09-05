"""Code for calculating (normalized) RDKit descriptors.

The normalization logic is taken from the `descriptastorus` [1]_ package.

.. [1] https://github.com/bp-kelley/descriptastorus/blob/master/descriptastorus/descriptors/dists.py
"""

import logging
import numpy as np
import scipy.stats
import torch
from torch import Tensor

import lobster
from lobster.constants import RDKIT_DESCRIPTOR_DISTRIBUTIONS

logger = logging.getLogger(__name__)


def _validate_descriptors_available(descriptor_list: list[str], available_descriptors: set[str]) -> None:
    """Validate that all requested descriptors are available in RDKit."""
    missing_descriptors = [name for name in descriptor_list if name not in available_descriptors]

    if missing_descriptors:
        raise ValueError(
            f"The following descriptors are not available in RDKit: {missing_descriptors}. "
            f"Available descriptors: {sorted(available_descriptors)}"
        )


def _validate_descriptors_have_distributions(descriptor_list: list[str]) -> None:
    """Validate that all requested descriptors have normalization distributions."""
    missing_distributions = [name for name in descriptor_list if name not in RDKIT_DESCRIPTOR_DISTRIBUTIONS]

    if missing_distributions:
        raise ValueError(
            f"The following descriptors do not have normalization distributions available: {missing_distributions}. "
            f"Descriptors with distributions: {sorted(RDKIT_DESCRIPTOR_DISTRIBUTIONS.keys())}"
        )


def _smiles_to_descriptors(smiles_seq: str) -> dict[str, float] | None:
    """Convert SMILES to molecule and calculate all descriptors."""
    lobster.ensure_package("rdkit", group="mgm")

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles_seq)

    if mol is None:
        logger.warning(f"SMILES sequence `{smiles_seq}` is invalid. Returning None.")
        return None

    return Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)


def _filter_descriptors(
    descs: dict[str, float], descriptor_list: list[str] | None, require_distributions: bool = False
) -> list[str]:
    """Filter and validate descriptor list."""
    if descriptor_list is not None:
        _validate_descriptors_available(descriptor_list, set(descs.keys()))
        if require_distributions:
            _validate_descriptors_have_distributions(descriptor_list)
        return descriptor_list
    else:
        return list(RDKIT_DESCRIPTOR_DISTRIBUTIONS.keys()) if require_distributions else list(descs.keys())


def smiles_to_rdkit_descs(smiles_seq: str, descriptor_list: list[str] | None = None) -> Tensor | None:
    """Get the full set of RDKit descriptors for a SMILES sequence.

    Parameters
    ----------
    smiles_seq : str
        SMILES sequence to get descriptors for
    descriptor_list : list[str] | None, optional
        List of specific descriptor names to calculate. If None, calculates all available descriptors.
        Defaults to None.

    Returns
    -------
    Tensor | None
        RDKit descriptors for the SMILES sequence, or ``None`` if the SMILES sequence is invalid. Descriptors that were
        not calculable are replaced with ``nan``.
    """
    descs = _smiles_to_descriptors(smiles_seq)

    if descs is None:
        return None

    descriptors = _filter_descriptors(descs, descriptor_list, require_distributions=False)
    out = [descs[name] for name in descriptors]

    return torch.tensor(out, dtype=torch.float32)


# NOTE(degraff): This function can be made more eficient by taking batches of inputs, if needed.
def smiles_to_normalized_rdkit_descs(
    smiles_seq: str, invert: bool = False, descriptor_list: list[str] | None = None
) -> Tensor | None:
    r"""Get the normalized RDKit descriptors for a SMILES sequence.

    A normalized RDKit descriptor is either (a) the CDF of the clamped input value under a given descriptor's
    distribution (i.e., the probability integral transform):

    .. math::
        x_j = F_{X_j}\big[\max(x_\text{min}, \min(x, x_\text{max}))\big]

    or (b) the inverse Gaussian CDF of (a):

    .. math::
        \Phi^{-1}\left(F_{X_j}\big[\max(x_\text{min}, \min(x, x_\text{max}))\big]\right)

    where :math:`F_{X_j}` is the CDF of the descriptor :math:`X_j`; :math:`x` is the input value;
    :math:`x_\text{min}` and :math:`x_\text{max}` are the minimum and maximum values of the
    descriptor, respectively; and :math:`\Phi^{-1}` is the inverse Gaussian CDF.

    .. important::
        The returned list of descriptors will be a **subset** of the full set of RDKit descriptors based on the keys of
        the :data:`lobster.constants.RDKIT_DESCRIPTOR_DISTRIBUTIONS` dictionary. That is, the returned list is not
        necessarily the same or in the same order as the list returned from :func:`smiles_to_rdkit_descs`.

    Parameters
    ----------
    smiles_seq : str
        SMILES sequence to get descriptors for
    invert : bool, optional
        Whether to invert the normalized descriptor. Defaults to ``False``.
    descriptor_list : list[str] | None, optional
        List of specific descriptor names to calculate and normalize. If None, uses all descriptors available
        in RDKIT_DESCRIPTOR_DISTRIBUTIONS. Defaults to None.

    Returns
    -------
    Tensor | None
        Normalized RDKit descriptors for the SMILES sequence, or ``None`` if the SMILES sequence is invalid.
        Descriptors that were not calculable are replaced with ``nan``.

    See Also
    --------
    :data:`lobster.constants.RDKIT_DESCRIPTOR_DISTRIBUTIONS`
        The distributions used to normalize the RDKit descriptors.
    """
    descs = _smiles_to_descriptors(smiles_seq)

    if descs is None:
        return None

    descriptors = _filter_descriptors(descs, descriptor_list, require_distributions=True)

    xs = []
    for name in descriptors:
        x = descs[name]

        dist, (x_min, x_max, *_) = RDKIT_DESCRIPTOR_DISTRIBUTIONS[name]

        p = dist.cdf(np.clip(x, x_min, x_max))
        if invert:
            # Clip p to avoid -inf/+inf from ppf at exactly 0/1
            p_clipped = np.clip(p, 1e-8, 1 - 1e-8)
            xs.append(scipy.stats.norm.ppf(p_clipped))
        else:
            xs.append(p)

    return torch.tensor(xs, dtype=torch.float32)
