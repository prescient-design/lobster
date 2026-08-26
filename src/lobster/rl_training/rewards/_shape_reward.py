"""Back-compat shim: moved to :mod:`plm_design_rl.rewards._shape_reward`."""

import sys

from plm_design_rl.rewards import _shape_reward as _m

sys.modules[__name__] = _m
