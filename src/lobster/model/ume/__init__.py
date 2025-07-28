from ._ume import UME as UMEModule
from ._ume_lightning_module import UMELightningModule

# For backward compatibility, UME refers to the Lightning module
UME = UMELightningModule

__all__ = ["UME", "UMEModule", "UMELightningModule"]
