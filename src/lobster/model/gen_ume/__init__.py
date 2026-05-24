"""Deprecated alias for :mod:`lobster.model.leflur`.

The package was renamed from ``gen_ume`` to ``leflur`` in preparation for the
LeFlur release. This module re-exports the renamed classes under their
historical names so existing checkpoints (whose ``hyper_parameters._target_``
points at ``lobster.model.gen_ume...``) and external scripts continue to load.

New code should import from ``lobster.model.leflur`` directly.
"""

from __future__ import annotations

import warnings

from lobster.model.leflur import (
    AuxiliaryRegressionTaskHead,
    AuxiliaryTask,
    BondMatrixEmbedding,
    BondMatrixLoss,
    BondMatrixPredictionHead,
    LeFlurProteinLigandEncoderModule as ProteinLigandEncoderModule,
    LeFlurProteinLigandLightningModule as ProteinLigandEncoderLightningModule,
    LeFlurSequenceStructureEncoderLightningModule as UMESequenceStructureEncoderLightningModule,
    LeFlurSequenceStructureEncoderModule as UMESequenceStructureEncoderModule,
)

warnings.warn(
    "lobster.model.gen_ume is deprecated; import from lobster.model.leflur instead. "
    "The old class names (UMESequenceStructureEncoder*, ProteinLigandEncoder*) "
    "remain available here as aliases for backwards compatibility with existing "
    "checkpoints, but new code should use the LeFlur* names from lobster.model.leflur.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "UMESequenceStructureEncoderModule",
    "UMESequenceStructureEncoderLightningModule",
    "AuxiliaryTask",
    "AuxiliaryRegressionTaskHead",
    "ProteinLigandEncoderModule",
    "ProteinLigandEncoderLightningModule",
    "BondMatrixEmbedding",
    "BondMatrixPredictionHead",
    "BondMatrixLoss",
]
