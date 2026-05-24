"""Per-mode generation pipelines for :mod:`lobster.cmdline.generate`.

Each module here implements one generation mode dispatched from
:func:`lobster.cmdline.generate.generate`:

- :mod:`._unconditional` — de-novo joint sequence/structure generation
  (+ self-reflection helpers)
- :mod:`._inverse_folding` — sequence generation conditioned on a fixed structure
- :mod:`._forward_folding` — structure prediction conditioned on a fixed sequence
- :mod:`._inpainting` — masked redesign within a fixed structural scaffold
- :mod:`._binders` — binder design against a target chain

The protein-ligand modes (forward folding, inverse folding, de-novo ligand-
conditioned generation) live in :mod:`lobster.cmdline._ligand_conditioned_runner`
and are dispatched directly from ``generate.py``; they are intentionally not
duplicated here.
"""

from ._binders import _generate_binders
from ._forward_folding import _generate_forward_folding
from ._inpainting import _generate_inpainting
from ._inverse_folding import _generate_inverse_folding
from ._unconditional import _generate_unconditional

__all__ = [
    "_generate_binders",
    "_generate_forward_folding",
    "_generate_inpainting",
    "_generate_inverse_folding",
    "_generate_unconditional",
]
