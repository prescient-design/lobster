"""Back-compat shim: moved to :mod:`plm_design_rl.rewards._sc_clash_reward`."""

import sys

from plm_design_rl.rewards import _sc_clash_reward as _m

sys.modules[__name__] = _m
