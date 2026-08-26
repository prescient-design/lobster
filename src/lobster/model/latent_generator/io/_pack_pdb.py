"""LigandMPNN side-chain packer engine (in-process, no PDB/CIF round-trip).

LeFlur's :func:`writepdb` emits **backbone-only** structures (idealized O + Cβ, no real
side chains). Every downstream consumer that needs all-atom geometry — the GRPO
shape-complementarity / all-atom-clash rewards, the offline SC-clash eval, an optional
all-atom output from ``lobster_generate`` — must therefore repack side chains out of
band. This module is that engine: it rebuilds every side chain from the N/CA/C rigid
frame + LigandMPNN-predicted χ torsions (``repack_everything=True``), operating on **raw
coordinate clouds** so no file ever touches disk.

The engine is deliberately dependency-light: numpy at module scope, with ``torch`` /
LigandMPNN (``sc_utils`` / ``data_utils``, which want ``openfold`` + ``prody``) imported
lazily inside :class:`Repacker`. Importing this module therefore stays cheap even where
those heavy deps are absent (e.g. a numpy-only clash scorer that only needs the atom14
element table).

CPU-viable: the packer is a small net (3 denoising steps × 4 von-Mises samples) and runs
at ~0.66 s/design on 8 CPU threads (all outputs valid) vs 0.234 s/design on an a10g — so
the promoted packing step and the SC-clash / AAR reward pools default to CPU and scale by
adding CPU workers rather than contending for the GPU queue (memory
``ligandmpnn-packer-cpu-viable``).

Public API
----------
``pack_structure(chains, ...)`` / ``write_packed_pdb(...)``
    N-chain entry points: pack a list of ``(backbone (L,3,3) [N,CA,C], seq_str)`` chains
    into a packed atom14 cloud, and write it to a full-atom PDB.
``pack_complex(...)`` / ``write_packed_complex_pdb(...)``
    2-chain (antigen + binder) back-compat wrappers used by the GRPO reward glue.
``Repacker``
    Loads the packer checkpoint once; ``pack_structure`` / ``pack_complex`` /
    ``heavy_cloud`` reuse it.
``heavy_cloud(X14, X_m, S)``
    Packed atom14 block → ``(xyz (M,3), elements list)`` heavy-atom cloud.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

# LigandMPNN lives outside the package; the blessed venv carries its deps (prody, openfold,
# torch). Path (+ SC packer checkpoint) are env-overridable. The default mirrors
# ``lobster.metrics.pylon_client.LIGANDMPNN_DEFAULT_PATH`` but is kept local so importing
# this light engine never drags in that module's heavy top-level imports (~45 s).
LIGANDMPNN_PATH = os.environ.get("LIGANDMPNN_PATH", "/cv/home/lisanzas/LigandMPNN")
SC_CKPT = os.environ.get(
    "LIGANDMPNN_SC_CKPT",
    os.path.join(LIGANDMPNN_PATH, "model_params", "ligandmpnn_sc_v_32_002_16.pt"),
)


def _ensure_ligandmpnn_on_path() -> None:
    """Add ``LIGANDMPNN_PATH`` to ``sys.path`` so ``sc_utils`` / ``data_utils`` import."""
    if LIGANDMPNN_PATH not in sys.path:
        sys.path.insert(0, LIGANDMPNN_PATH)


# atom14 name table for each residue, in packed-atom14 order (N, CA, C, O, CB, ...).
# Copied verbatim from LigandMPNN ``data_utils.write_full_PDB`` (it is a *local* dict,
# not importable). ``pack_side_chains`` emits X in exactly this order and marks valid
# atoms in X_m, so element = name[:1] for the atoms where X_m == 1 (matching the PDB
# writer's ``element_name_list += [nm[:1] for nm in names[sel]]``).
RESTYPE_NAME_TO_ATOM14_NAMES: dict[str, list[str]] = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

# Precomputed element (name[:1]) per residue x atom14 slot; "" where the slot is unused.
_RESTYPE_ATOM14_ELEMENTS: dict[str, list[str]] = {
    aa: [nm[:1] for nm in names] for aa, names in RESTYPE_NAME_TO_ATOM14_NAMES.items()
}

# MPNN alphabet ints (0-19, X=20) -> 3-letter names, matching data_utils.restype_* so the
# packer sees the same sequence the client encoded.
_INT_TO_AA1 = "ACDEFGHIKLMNPQRSTVWY"  # restype_int_to_str[0..19]
_AA1_TO_INT = {a: i for i, a in enumerate(_INT_TO_AA1)}
_AA1_TO_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
}
_AA3_BY_INT = {i: _AA1_TO_AA3[a] for a, i in _AA1_TO_INT.items()}
_AA3_BY_INT[20] = "UNK"


def _seq_to_ints(seq: str) -> np.ndarray:
    """1-letter AA string -> MPNN alphabet int array (unknown residues -> X=20)."""
    return np.array([_AA1_TO_INT.get(c, 20) for c in seq], dtype=np.int64)


def _ensure_prody_importable() -> None:
    """Let ``import data_utils`` succeed when prody is absent (e.g. the worktree venv).

    ``data_utils`` does ``from prody import *`` then ``confProDy(verbosity="none")`` at
    module top, but the only symbol we reach from it — ``featurize`` — uses no prody at
    runtime (the prody-dependent parse/write helpers live in other functions we never
    call; we write PDBs with the local :func:`write_packed_pdb`). When prody genuinely
    isn't installed, install a minimal stub whose star-import binds the one module-level
    name ``data_utils`` actually calls (``confProDy``) to a no-op, with a module
    ``__getattr__`` no-op as a safety net for any other bare prody reference. Guarded: a
    real prody (present in the SC-reward pool's venv) is never overridden.
    """
    try:
        import prody  # noqa: F401
    except ImportError:
        stub = types.ModuleType("prody")

        def _noop(*_args, **_kwargs):
            return None

        # `from prody import *` binds only names in __all__; data_utils uses confProDy().
        stub.confProDy = _noop
        stub.__all__ = ["confProDy"]

        # Any other bare prody symbol reached at import time resolves to a no-op too,
        # but dunders raise AttributeError so stdlib introspection (inspect/importlib
        # probing __file__, __path__, ...) still sees a normal module.
        def _stub_getattr(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _noop

        stub.__getattr__ = _stub_getattr
        sys.modules["prody"] = stub


def _synth_oxygen(bb: np.ndarray) -> np.ndarray:
    """Synthesize the backbone carbonyl O for one *chain* from its N/CA/C trace.

    ``bb`` is ``(n, 3, 3)`` in ``[N, CA, C]`` order. For residue ``i`` the O sits in the
    peptide plane, on the bisector of the (CA_i->C_i) and (N_{i+1}->C_i) bonds at 1.23 Å
    (the C=O bond length); the terminal residue (no next N) falls back to the C-CA
    direction. Returns ``(n, 3)``. This is the standard idealized-O placement; it only
    affects the backbone psi torsion in the packer, never the rebuilt side chains.
    """
    n = bb.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    N, CA, C = bb[:, 0], bb[:, 1], bb[:, 2]
    v1 = CA - C
    v1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
    n_next = np.roll(N, -1, axis=0)  # N_{i+1}; last row wraps (fixed up below)
    v2 = n_next - C
    v2 = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
    bis = -(v1 + v2)
    bis = bis / (np.linalg.norm(bis, axis=1, keepdims=True) + 1e-8)
    O = C + 1.23 * bis
    # Terminal residue: no next N -> place along C-CA.
    vt = C[-1] - CA[-1]
    vt = vt / (np.linalg.norm(vt) + 1e-8)
    O[-1] = C[-1] + 1.23 * vt
    return O


def write_packed_pdb(
    save_path: str | os.PathLike,
    X14: np.ndarray,
    X_m: np.ndarray,
    S: np.ndarray,
    chain_sizes: list[int] | tuple[int, ...] | np.ndarray,
    chain_letters: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Write a packed multi-chain structure (atom14 -> full-atom PDB).

    Consumes exactly the ``(X14, X_m, S, chain_sizes)`` tuple that
    :func:`pack_structure` returns. Emits standard PDB ``ATOM`` records directly from the
    module's local atom14 name table (``RESTYPE_NAME_TO_ATOM14_NAMES``) — the same names /
    elements / ordering LigandMPNN's ``write_full_PDB`` uses, but without importing
    ``data_utils`` (which does ``from prody import *`` and calls real prody at write time).
    Chains occupy contiguous residue blocks in ``chain_sizes`` order; residues are
    renumbered 1.. per chain.

    Parameters
    ----------
    save_path : str | os.PathLike
        Output ``.pdb`` path.
    X14 : np.ndarray
        ``(L, 14, 3)`` packed atom14 coordinates.
    X_m : np.ndarray
        ``(L, 14)`` valid-atom mask.
    S : np.ndarray
        ``(L,)`` MPNN-alphabet sequence ints (``0..19`` = ``ACDEFGHIKLMNPQRSTVWY``).
    chain_sizes : sequence of int
        Residue count per chain, in order; must sum to ``L``.
    chain_letters : sequence of str, optional
        Chain ids, one per block. Defaults to ``A, B, C, ...`` (wrapping past ``Z``).
    """
    X14 = np.asarray(X14, dtype=np.float64)
    X_m = np.asarray(X_m)
    S = np.asarray(S, dtype=np.int64)
    L = int(X14.shape[0])
    chain_sizes = [int(c) for c in chain_sizes]
    if sum(chain_sizes) != L:
        raise ValueError(f"chain_sizes sum {sum(chain_sizes)} != L {L}")
    if chain_letters is None:
        chain_letters = [chr(ord("A") + (k % 26)) for k in range(len(chain_sizes))]
    if len(chain_letters) != len(chain_sizes):
        raise ValueError("chain_letters and chain_sizes length mismatch")

    # Per-residue (chain letter, per-chain resseq) from the contiguous block layout.
    res_chain: list[str] = []
    res_seq: list[int] = []
    for letter, size in zip(chain_letters, chain_sizes):
        res_chain.extend([letter] * size)
        res_seq.extend(range(1, size + 1))

    lines: list[str] = []
    serial = 1
    prev = None  # (chain, resseq, aa3) of the last residue written
    for i in range(L):
        aa3 = _AA3_BY_INT.get(int(S[i]), "UNK")
        names = RESTYPE_NAME_TO_ATOM14_NAMES[aa3]
        chain = res_chain[i]
        resseq = res_seq[i]
        if prev is not None and chain != prev[0]:
            lines.append(f"TER   {serial:>5d}      {prev[2]:>3s} {prev[0]:1s}{prev[1]:>4d}\n")
            serial += 1
        for a in range(14):
            if float(X_m[i, a]) < 0.5:
                continue
            name = names[a]
            if not name:
                continue
            x, y, z = X14[i, a]
            atom_field = (" " + name) if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:>5d} {atom_field:<4s} {aa3:>3s} {chain:1s}{resseq:>4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {name[0]:>2s}\n"
            )
            serial += 1
        prev = (chain, resseq, aa3)
    if prev is not None:
        lines.append(f"TER   {serial:>5d}      {prev[2]:>3s} {prev[0]:1s}{prev[1]:>4d}\n")
    lines.append("END\n")
    Path(save_path).write_text("".join(lines))


def write_packed_complex_pdb(
    save_path: str | os.PathLike,
    X14: np.ndarray,
    X_m: np.ndarray,
    S: np.ndarray,
    n_ag: int,
    chain_letters: tuple[str, str] = ("A", "B"),
) -> None:
    """2-chain (antigen + binder) back-compat wrapper over :func:`write_packed_pdb`.

    Consumes the ``(X14, X_m, S, n_ag)`` tuple that :meth:`Repacker.pack_complex` returns;
    the antigen block (chain ``chain_letters[0]``) occupies the first ``n_ag`` residues,
    the binder (chain ``chain_letters[1]``) the rest.
    """
    n_ag = int(n_ag)
    n_bd = int(np.asarray(S).shape[0]) - n_ag
    write_packed_pdb(save_path, X14, X_m, S, [n_ag, n_bd], list(chain_letters))


def heavy_cloud(X14: np.ndarray, X_m: np.ndarray, S: np.ndarray):
    """Packed atom14 block -> ``(xyz (M,3), elements list)`` heavy-atom cloud."""
    xyz_list, elem_list = [], []
    for i in range(X14.shape[0]):
        aa3 = _AA3_BY_INT.get(int(S[i]), "UNK")
        elems = _RESTYPE_ATOM14_ELEMENTS[aa3]
        sel = X_m[i].astype(bool)
        if not sel.any():
            continue
        xyz_list.append(X14[i][sel])
        elem_list.extend([elems[j] for j in range(14) if sel[j]])
    if not xyz_list:
        return np.empty((0, 3), dtype=np.float64), []
    return np.concatenate(xyz_list, axis=0).astype(np.float64), elem_list


class Repacker:
    """Loads the LigandMPNN side-chain packer once; repacks designs in process.

    Parameters
    ----------
    device : str
        torch device string (``"cpu"`` / ``"cuda"``). The packer is CPU-viable
        (~0.66 s/design @ 8 threads); default the pool to CPU and scale by worker count.
    sc_ckpt : str, optional
        Path to the LigandMPNN side-chain packer checkpoint. Defaults to :data:`SC_CKPT`.
    """

    def __init__(self, device: str, sc_ckpt: str = SC_CKPT):
        _ensure_ligandmpnn_on_path()
        import torch
        from sc_utils import Packer

        self.torch = torch
        self.device = torch.device(device)
        # Packer hyperparameters are fixed by the checkpoint (run.py:95).
        model_sc = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=self.device,
            num_mix=3,
        )
        ckpt = torch.load(sc_ckpt, map_location=self.device)
        model_sc.load_state_dict(ckpt["model_state_dict"])
        self.model_sc = model_sc.to(self.device).eval()

    def build_input_dict(self, chains: list[tuple[np.ndarray, str]]) -> tuple[dict, list[int]]:
        """Assemble the LigandMPNN featurize input for an N-chain complex from raw clouds.

        Each chain is ``(backbone (L_c, 3, 3) [N, CA, C], seq_str)``. Synthesizes the
        backbone O per chain (never across the inter-chain gap), concatenates the chains in
        order, and builds the per-residue ``S`` / ``chain_labels`` / ``R_idx`` tensors.
        Returns ``(input_dict, chain_sizes)``; the same input dict feeds both the
        side-chain packer and a ProteinMPNN AAR/consistency scorer (both use
        LigandMPNN ``featurize``).
        """
        import torch

        chain_sizes = [int(np.asarray(bb).shape[0]) for bb, _ in chains]
        X_blocks, S_blocks, lab_blocks, ridx_blocks = [], [], [], []
        for k, (bb, seq) in enumerate(chains):
            bb = np.asarray(bb, dtype=np.float64)
            n = bb.shape[0]
            # [N, CA, C, O]; O synthesized within this chain only.
            X_blocks.append(np.concatenate([bb, _synth_oxygen(bb)[:, None, :]], axis=1))  # (n,4,3)
            S_blocks.append(_seq_to_ints(seq))
            lab_blocks.append(np.full(n, k, dtype=np.int64))
            ridx_blocks.append(np.arange(n, dtype=np.int64))
        X = np.concatenate(X_blocks, axis=0).astype(np.float32)  # (L,4,3)
        S = np.concatenate(S_blocks)  # (L,)
        chain_labels = np.concatenate(lab_blocks)
        R_idx = np.concatenate(ridx_blocks)
        L = int(X.shape[0])

        dev = self.device
        input_dict = {
            "X": torch.from_numpy(X).to(dev),
            "mask": torch.ones(L, dtype=torch.float32, device=dev),
            "S": torch.from_numpy(S).to(dev),
            "R_idx": torch.from_numpy(R_idx).to(dev),
            "chain_labels": torch.from_numpy(chain_labels).to(dev),
            "chain_mask": torch.ones(L, dtype=torch.float32, device=dev),
            # No ligand: empty (0,3)/(0,) ligand tensors trigger featurize's no-ligand
            # early-return in get_nearest_neighbours, which synthesizes correctly-shaped
            # zero ligand context. Passing (L,16,*) here instead makes .repeat() see a
            # 4-D tensor and crash.
            "Y": torch.zeros(0, 3, dtype=torch.float32, device=dev),
            "Y_t": torch.zeros(0, dtype=torch.int32, device=dev),
            "Y_m": torch.zeros(0, dtype=torch.float32, device=dev),
        }
        return input_dict, chain_sizes

    def pack_structure(
        self,
        chains: list[tuple[np.ndarray, str]],
        num_denoising_steps: int = 3,
        num_samples: int = 4,
    ):
        """Repack an N-chain complex in process; return ``(X14, X_m, S, chain_sizes)``.

        Chains are packed together (so inter-chain context informs the χ torsions) and
        returned as one contiguous atom14 cloud in input order. ``X14`` is ``(L, 14, 3)``
        packed coords, ``X_m`` ``(L, 14)`` the valid-atom mask, ``S`` ``(L,)`` MPNN ints,
        ``chain_sizes`` the per-chain residue counts (split points).
        """
        import torch

        _ensure_prody_importable()  # data_utils does `from prody import *` at module top
        from data_utils import featurize
        from sc_utils import pack_side_chains

        input_dict, chain_sizes = self.build_input_dict(chains)
        with torch.no_grad():
            feat = featurize(
                input_dict,
                cutoff_for_score=8.0,
                use_atom_context=False,
                number_of_ligand_atoms=16,
                model_type="ligand_mpnn",
            )
            feat = pack_side_chains(
                feat,
                self.model_sc,
                num_denoising_steps=num_denoising_steps,
                num_samples=num_samples,
                repack_everything=True,
                num_context_atoms=16,
            )
        X14 = feat["X"][0].detach().cpu().numpy()  # (L,14,3)
        X_m = feat["X_m"][0].detach().cpu().numpy()  # (L,14)
        S = input_dict["S"].detach().cpu().numpy()
        return X14, X_m, S, chain_sizes

    def pack_complex(
        self,
        ag_bb: np.ndarray,
        ag_seq: str,
        bd_bb: np.ndarray,
        bd_seq: str,
        num_denoising_steps: int = 3,
        num_samples: int = 4,
    ):
        """2-chain (antigen + binder) back-compat wrapper over :meth:`pack_structure`.

        Returns ``(X14, X_m, S, n_ag)`` — ``(L,14,3)`` coords, ``(L,14)`` valid mask,
        ``(L,)`` MPNN sequence ints, and the antigen block length (split point). The
        antigen is chain 0, the binder chain 1, exactly as the SC reward glue expects.
        """
        X14, X_m, S, chain_sizes = self.pack_structure(
            [(ag_bb, ag_seq), (bd_bb, bd_seq)], num_denoising_steps, num_samples
        )
        return X14, X_m, S, chain_sizes[0]

    def _heavy_cloud(self, X14: np.ndarray, X_m: np.ndarray, S: np.ndarray):
        """Deprecated alias for the module-level :func:`heavy_cloud` (kept for callers)."""
        return heavy_cloud(X14, X_m, S)


# One cached Repacker per (device, checkpoint) so the module-level convenience wrappers
# (and the post-generation packing pass, which packs many structures per call) reuse a
# single loaded net (~3.5 s to build) instead of reloading it per design.
_REPACKER_CACHE: dict[tuple[str, str], Repacker] = {}


def get_repacker(device: str = "cpu", sc_ckpt: str = SC_CKPT) -> Repacker:
    """Return a process-cached :class:`Repacker` for ``(device, sc_ckpt)`` (built once)."""
    key = (str(device), str(sc_ckpt))
    if key not in _REPACKER_CACHE:
        _REPACKER_CACHE[key] = Repacker(device, sc_ckpt=sc_ckpt)
    return _REPACKER_CACHE[key]


def pack_structure(
    chains: list[tuple[np.ndarray, str]],
    device: str = "cpu",
    num_denoising_steps: int = 3,
    num_samples: int = 4,
    sc_ckpt: str = SC_CKPT,
):
    """Repack an N-chain complex; return ``(X14, X_m, S, chain_sizes)``.

    Module-level convenience wrapper over :meth:`Repacker.pack_structure` backed by a
    per-``(device, sc_ckpt)`` cached packer. Each chain is
    ``(backbone (L_c, 3, 3) [N, CA, C], seq_str)``. Defaults to CPU (the packer is
    CPU-viable; scale by adding CPU workers).
    """
    return get_repacker(device, sc_ckpt).pack_structure(chains, num_denoising_steps, num_samples)


def pack_complex(
    ag_bb: np.ndarray,
    ag_seq: str,
    bd_bb: np.ndarray,
    bd_seq: str,
    device: str = "cpu",
    num_denoising_steps: int = 3,
    num_samples: int = 4,
    sc_ckpt: str = SC_CKPT,
):
    """2-chain (antigen + binder) convenience wrapper over :func:`pack_structure`.

    Returns ``(X14, X_m, S, n_ag)``. Backed by the same per-device cached packer.
    """
    return get_repacker(device, sc_ckpt).pack_complex(ag_bb, ag_seq, bd_bb, bd_seq, num_denoising_steps, num_samples)
