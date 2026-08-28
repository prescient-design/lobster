"""Back-compat shim: moved to :mod:`plm_design_rl.rewards._rog_reward`."""

import sys

from plm_design_rl.rewards import _rog_reward as _m

sys.modules[__name__] = _m
