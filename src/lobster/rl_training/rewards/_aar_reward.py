"""Back-compat shim: moved to :mod:`plm_design_rl.rewards._aar_reward`."""

import sys

from plm_design_rl.rewards import _aar_reward as _m

sys.modules[__name__] = _m
