"""Code for calculating (normalized) RDKit descriptors.

The normalization logic is taken from the `descriptastorus` [1]_ package.

.. [1] https://github.com/bp-kelley/descriptastorus/blob/master/descriptastorus/descriptors/dists.py
"""

import numpy as np
import scipy.stats
import torch
from torch import Tensor

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


from lobster.constants import RDKIT_DESCRIPTOR_DISTRIBUTIONS


def smiles_to_rdkit_descs(smiles_seq: str) -> Tensor | None:
    """Get the full set of RDKit descriptors for a SMILES sequence.

    Parameters
    ----------
    smiles_seq : str
        SMILES sequence to get descriptors for

    Returns
    -------
    list[float] | None
        RDKit descriptors for the SMILES sequence, or ``None`` if the SMILES sequence is invalid. Descriptors that were
        not calculable are replaced with ``nan``.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is not installed. Please install it with `uv sync --extra mgm`.")

    mol = Chem.MolFromSmiles(smiles_seq)
    if mol is None:
        return None

    out = list(Descriptors.CalcMolDescriptors(mol, missingVal=np.nan).values())

    return torch.tensor(out)


# NOTE(degraff): This function can be made more eficient by taking batches of inputs, if needed.
def smiles_to_normalized_rdkit_descs(smiles_seq: str, invert: bool = False) -> list[float] | None:
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
    Returns
    -------
    list[float] | None
        Normalized RDKit descriptors for the SMILES sequence, or ``None`` if the SMILES sequence is invalid.
        Descriptors that were not calculable are replaced with ``nan``.

    See Also
    --------
    :data:`lobster.constants.RDKIT_DESCRIPTOR_DISTRIBUTIONS`
        The distributions used to normalize the RDKit descriptors.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is not installed. Please install it with `uv sync --extra mgm`.")

    mol = Chem.MolFromSmiles(smiles_seq)

    if mol is None:
        return None

    descs = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)

    xs = []

    for name, x in descs.items():
        try:
            dist, (x_min, x_max, *_) = RDKIT_DESCRIPTOR_DISTRIBUTIONS[name]
        except KeyError:
            continue

        p = dist.cdf(np.clip(x, x_min, x_max))
        xs.append(p if not invert else scipy.stats.norm.ppf(p))

    return torch.tensor(xs)
