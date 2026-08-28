"""Interface ΔΔG-of-binding reward via tmol (Rosetta ``beta_nov2016`` energy).

This is the canonical engine + reward mapper for the physics-complete upgrade of the
numpy ``e_lj`` interface proxy (:mod:`._packed_potentials` / ``scripts/_packed_potentials.py``).
It scores a side-chain-packed two-chain complex with tmol's full ``beta_nov2016`` term set —
including the ``fa_lk`` / ``lk_ball`` Lazaridis–Karplus desolvation physics the numpy proxy
lacks — and reports the interface ΔΔG of binding in two flavors:

**Rigid** (``ddg_*``): translate one chain far enough that all cross-chain neighbor-list
interactions vanish. Every intra-chain term is then identical in the bound and separated
states and cancels exactly, leaving only the **cross-chain pair sum** — the interface
interaction energy (Rosetta InterfaceAnalyzer ``dG_separated`` without unbound relaxation).

**Unbound-relaxed** (``ddg_ub_*``): after separating the chains, Cartesian-minimize the
(now decoupled) pose so each partner relaxes to its *own* unbound minimum before scoring —
``ΔΔG = E(bound,relaxed) − E(A,relaxed) − E(B,relaxed)``. This adds strain-relief-on-unbinding
physics (a real InterfaceAnalyzer ΔΔG with ``pack_separated`` / minimization) and penalizes
interfaces that only look favorable because the bound side chains are strained. Because the
separated chains sit beyond every energy cutoff, one ``run_cart_min`` on the separated pose
minimizes ``E_A + E_B`` with a fully decoupled gradient — it relaxes both partners
independently in a single call, with no chain extraction.

Optionally the bound complex is Cartesian-minimized first (``relax=True``): the packed side
chains carry large ``fa_ljrep`` clashes that dominate the raw score; one LBFGS minimization
relieves them and yields a physically-sensible ΔΔG. ``relax_unbound=True`` additionally
computes the unbound-relaxed flavor (and implies ``relax``).

Packaging
---------
This module keeps the ``plm_design_rl`` reward-module contract: ``torch`` is a hard
dependency (top-level import), but **tmol is imported lazily** so importing this module never
requires it. The first call that needs tmol (:func:`ddg_terms`, :func:`ddg_per_binder_residue`)
raises a clear, actionable error naming the optional extra — ``pip install plm-design-rl[tmol]``
— rather than a bare ``ModuleNotFoundError`` at import time. Like every other term the reward
is bounded to ``[0, 1]`` and contributes exactly ``0`` when its weight is ``0`` (opt-in).

Notes
-----
The rigid ΔΔG **total** and every per-term cross-chain sum are invariant to which chain is
translated (the interaction is symmetric), so chain labelling does not affect the scalar
reward or its offline AUROC. The unbound-relaxed flavor is likewise chain-swap symmetric.
Per-binder-residue attribution (dense per-token credit, :func:`ddg_per_binder_residue`)
distributes the rigid interface total across binder residues by their cross-chain heavy-atom
contact count — a cheap, coordinate-only dense-credit scheme (no extra scoring), the ΔΔG
analogue of the per-token advantage used by the other structure rewards.

This is the engine scoped in ``docs/leflur/grpo_ddg_reward_scope.md`` and validated offline
by ``scripts/_tmol_ddg_compute.py`` / ``scripts/_tmol_ddg_analyze.py`` (head-to-head AUROC vs
``e_lj``). Ship the term as a live reward only if a ΔΔG flavor beats ``e_lj`` (AUROC 0.778).
"""

from __future__ import annotations

import math

import torch

# curated term groupings (beta_nov2016 score-type names as tmol reports them)
_ATR = ("fa_ljatr",)
_REP = ("fa_ljrep",)
_SOL = ("fa_lk",)  # Lazaridis-Karplus isotropic solvation (desolvation on binding)
_LKBALL = ("lk_ball", "lk_ball_iso", "lk_bridge", "lk_bridge_uncpl")  # anisotropic solv.
_ELEC = ("fa_elec",)
_HB = ("hbond",)

# column groups emitted per flavor (mirrors _ddg_groups keys)
_GRP_SUFFIXES = ("total", "atr", "rep", "sol", "lkball", "elec", "hb", "noclash")

# Reward mapping (logistic, more-negative ΔΔG = stronger binding = higher reward).
# Center/scale are in beta_nov2016 REU; defaults are placeholders to be calibrated against
# the offline distribution (scripts/_tmol_ddg_analyze.py) before the weight is turned on.
DDG_CENTER = -20.0  # ΔΔG at which reward = 0.5
DDG_SCALE = 15.0  # logistic width (REU); larger = softer

_TMOL_INSTALL_HINT = (
    "The interface ΔΔG reward requires the optional 'tmol' dependency (uw-ipd/tmol, "
    "Apache-2.0), which is not installed. Install it with:\n"
    "    pip install 'plm-design-rl[tmol]'\n"
    "The wheel lane must match your torch/CUDA build (e.g. +cu128torch2.8 cp312 for GPU, "
    "+cpu for CPU-only). See docs/leflur/grpo_ddg_reward_scope.md."
)

_sfxn_cache: dict[str, object] = {}


def _require_tmol():
    """Import tmol lazily, raising a clear install-the-extra error if it is absent."""
    try:
        import tmol  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(_TMOL_INSTALL_HINT) from e
    return tmol


def _score_function(device: torch.device):
    key = str(device)
    sf = _sfxn_cache.get(key)
    if sf is None:
        _require_tmol()
        from tmol import beta2016_score_function

        sf = beta2016_score_function(device)
        _sfxn_cache[key] = sf
    return sf


def _atom_chain(ps) -> torch.Tensor:
    """(n_atoms,) chain id for every coordinate row, from block offsets."""
    cid = ps.chain_id[0]
    bco = ps.block_coord_offset[0]
    napb = ps.n_ats_per_block[0]
    n_atoms = ps.coords.shape[1]
    atom_chain = torch.full((n_atoms,), -1, dtype=torch.long, device=ps.coords.device)
    for b in range(cid.shape[0]):
        s = int(bco[b])
        n = int(napb[b])
        if n > 0:
            atom_chain[s : s + n] = int(cid[b])
    return atom_chain


def _term_energies(sfxn, ps, coords) -> dict[str, float]:
    """Weighted per-score-type energy for a pose at ``coords``."""
    render = sfxn.render_whole_pose_scoring_module(ps)
    sts = [str(s).split(".")[-1] for s in sfxn.all_score_types()]
    w = sfxn.weights_tensor()
    with torch.no_grad():
        e = render(coords, sum_terms=False, apply_weights=False)[:, 0] * w
    return {sts[i]: float(e[i]) for i in range(len(sts))}


def _relax(sfxn, ps):
    """Cartesian LBFGS minimization; returns a new relaxed PoseStack."""
    _require_tmol()
    from tmol.optimization import run_cart_min

    with torch.enable_grad():
        return run_cart_min(ps, sfxn)


def _with_coords(ps, coords):
    """Clone ``ps`` with ``coords`` substituted (same topology)."""
    ps2 = ps.clone()
    ps2.coords[:] = coords
    return ps2


def _ddg_groups(prefix: str, eb: dict[str, float], es: dict[str, float]) -> dict[str, float]:
    """Per-group weighted ΔΔG (eb − es) under a column ``prefix`` (e.g. ``ddg`` / ``ddg_ub``)."""
    ddg = {k: eb[k] - es.get(k, 0.0) for k in eb}

    def grp(names: tuple[str, ...]) -> float:
        return float(sum(ddg.get(n, 0.0) for n in names))

    total = float(sum(ddg.values()))
    rep = grp(_REP)
    return {
        f"{prefix}_total": total,
        f"{prefix}_atr": grp(_ATR),
        f"{prefix}_rep": rep,
        f"{prefix}_sol": grp(_SOL),
        f"{prefix}_lkball": grp(_LKBALL),
        f"{prefix}_elec": grp(_ELEC),
        f"{prefix}_hb": grp(_HB),
        f"{prefix}_noclash": total - rep,
    }


def ddg_terms(
    pdb_path: str,
    device: str | torch.device = "cpu",
    relax: bool = False,
    relax_unbound: bool = False,
    sep_dist: float = 500.0,
    num_threads: int = 8,
) -> dict[str, float | str]:
    """Interface ΔΔG (per beta_nov2016 term) of a packed two-chain complex.

    Parameters
    ----------
    pdb_path : str
        Side-chain-packed complex PDB (>=2 chains).
    device : str | torch.device
        tmol device.
    relax : bool
        Cartesian-minimize the bound complex before scoring (de-clash). Adds ~35 s/design
        on CPU but converts the clash-dominated raw score into a physical ΔΔG. Implied by
        ``relax_unbound``.
    relax_unbound : bool
        Additionally compute the **unbound-relaxed** flavor (``ddg_ub_*``): relax each
        separated partner independently before scoring the unbound state. Adds a second
        ~25 s/design minimization (InterfaceAnalyzer-style ΔΔG with strain relief).
    sep_dist : float
        Translation (Å) applied to one chain to reach the unbound reference.
    num_threads : int
        torch CPU thread cap.

    Returns
    -------
    dict
        Rigid ΔΔG columns ``ddg_{total,atr,rep,sol,lkball,elec,hb,noclash}`` and, when
        ``relax_unbound``, the unbound-relaxed columns ``ddg_ub_*`` (same suffixes), plus
        ``e_bound``, ``e_sep`` (rigid), ``e_sep_relaxed`` (unbound), chain sizes
        (``n_res_a``/``n_res_b``), ``relaxed``/``relaxed_unbound`` flags, and an ``err``
        string (empty on success).

    Raises
    ------
    ImportError
        On first use if the optional ``tmol`` dependency is not installed (names the
        ``plm-design-rl[tmol]`` extra). Per-design scoring errors are captured in the
        returned ``err`` field instead of raised.
    """
    _require_tmol()
    from tmol import pose_stack_from_pdb

    torch.set_num_threads(num_threads)
    dev = torch.device(device)
    do_relax = relax or relax_unbound

    out: dict[str, float | str] = {}
    for suf in _GRP_SUFFIXES:
        out[f"ddg_{suf}"] = float("nan")
        out[f"ddg_ub_{suf}"] = float("nan")
    out.update(
        e_bound=float("nan"),
        e_sep=float("nan"),
        e_sep_relaxed=float("nan"),
        n_res_a=-1,
        n_res_b=-1,
        relaxed=int(do_relax),
        relaxed_unbound=int(relax_unbound),
        err="",
    )
    try:
        ps = pose_stack_from_pdb(pdb_path, dev)
        sfxn = _score_function(dev)
        if do_relax:
            ps = _relax(sfxn, ps)

        atom_chain = _atom_chain(ps)
        mask = atom_chain == int(atom_chain.max())

        eb = _term_energies(sfxn, ps, ps.coords)

        # rigid separation of the (relaxed) bound state
        cs = ps.coords.clone()
        cs[0, mask, 0] += sep_dist
        es = _term_energies(sfxn, ps, cs)
        out.update(_ddg_groups("ddg", eb, es))
        out["e_bound"] = float(sum(eb.values()))
        out["e_sep"] = float(sum(es.values()))

        # unbound-relaxed: minimize the decoupled separated pose (relaxes each partner)
        if relax_unbound:
            ps_sep = _with_coords(ps, cs)
            ps_sep_r = _relax(sfxn, ps_sep)
            eu = _term_energies(sfxn, ps_sep_r, ps_sep_r.coords)
            out.update(_ddg_groups("ddg_ub", eb, eu))
            out["e_sep_relaxed"] = float(sum(eu.values()))

        cid = ps.chain_id[0]
        uniq, counts = torch.unique(cid, return_counts=True)
        if len(counts) >= 2:
            out["n_res_a"] = int(counts[0])
            out["n_res_b"] = int(counts[-1])
    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def ddg_reward(
    res: dict | None,
    *,
    flavor: str = "ddg",
    center: float = DDG_CENTER,
    scale: float = DDG_SCALE,
    key: str = "total",
) -> float:
    """Scalar ΔΔG reward from a metrics dict: logistic in the (negative-good) ΔΔG ∈ [0, 1].

    More-negative ΔΔG (stronger predicted binding) maps to a higher reward::

        reward = 1 / (1 + exp((ddg − center) / scale))

    so ``ddg = center`` → 0.5, ``ddg → −∞`` → 1, ``ddg → +∞`` → 0. Bounded, monotone, and
    never negative; a missing/failed design (``None``) or a non-finite ΔΔG floors to 0.0.

    Parameters
    ----------
    res : dict | None
        Metrics dict from :func:`ddg_terms`. ``None`` → 0.0.
    flavor : str
        ``"ddg"`` (rigid) or ``"ddg_ub"`` (unbound-relaxed) — selects the column prefix.
    center : float
        ΔΔG (REU) mapped to reward 0.5. Calibrate against the offline distribution
        (``scripts/_tmol_ddg_analyze.py``) before enabling the weight.
    scale : float
        Logistic width (REU); larger = softer transition.
    key : str
        ΔΔG group suffix to read (default ``"total"``; e.g. ``"noclash"`` to reward the
        clash-free interaction energy).

    Returns
    -------
    float
        Reward in ``[0, 1]``.
    """
    if res is None:
        return 0.0
    ddg = res.get(f"{flavor}_{key}")
    if ddg is None:
        return 0.0
    ddg = float(ddg)
    if not math.isfinite(ddg):
        return 0.0
    # numerically stable logistic reward = sigmoid(-z), z = (ddg - center)/scale.
    # Branch on the sign of z so math.exp never sees a large positive argument.
    z = (ddg - center) / max(scale, 1e-6)
    if z >= 0.0:  # weak binding -> reward toward 0
        ez = math.exp(-z)
        r = ez / (1.0 + ez)
    else:  # strong binding -> reward toward 1
        r = 1.0 / (1.0 + math.exp(z))
    return float(min(1.0, max(0.0, r)))


def ddg_per_binder_residue(
    pdb_path: str,
    binder_chain: int = 0,
    device: str | torch.device = "cpu",
    relax: bool = False,
    sep_dist: float = 500.0,
    contact_cutoff: float = 5.0,
    num_threads: int = 8,
) -> dict:
    """Dense per-binder-residue attribution of the rigid interface ΔΔG (contact-weighted).

    The whole-pose scoring module reports only pose totals (no per-block breakdown), so an
    exact per-residue energy decomposition would need one re-scoring per residue. Instead we
    distribute the (validated) rigid interface ΔΔG total across binder residues in proportion
    to each residue's number of cross-chain heavy-atom contacts within ``contact_cutoff`` —
    a cheap, coordinate-only dense-credit scheme (the ΔΔG analogue of the per-token advantage
    used by the other structure rewards). Residues with no interface contact receive 0.

    Parameters
    ----------
    pdb_path : str
        Side-chain-packed complex PDB (>=2 chains).
    binder_chain : int
        Chain id of the binder (its residues receive per-token credit).
    device : str | torch.device
        tmol device.
    relax : bool
        Cartesian-minimize the bound complex before scoring (recommended: packed side
        chains clash).
    sep_dist : float
        Rigid-separation translation (Å).
    contact_cutoff : float
        Heavy-atom distance (Å) defining a cross-chain contact for the attribution weights.
    num_threads : int
        torch CPU thread cap.

    Returns
    -------
    dict
        ``per_res`` (list[float], length = #binder residues, summing to ``ddg_total``),
        ``binder_res_ids`` (block indices), ``ddg_total`` (rigid interface ΔΔG),
        ``n_contacts`` (per-residue cross-chain contact count), ``binder_chain``, and
        ``err`` (empty on success).

    Raises
    ------
    ImportError
        On first use if the optional ``tmol`` dependency is not installed.
    """
    _require_tmol()
    from tmol import pose_stack_from_pdb

    torch.set_num_threads(num_threads)
    dev = torch.device(device)
    out: dict = {
        "per_res": [],
        "binder_res_ids": [],
        "ddg_total": float("nan"),
        "n_contacts": [],
        "binder_chain": int(binder_chain),
        "err": "",
    }
    try:
        ps = pose_stack_from_pdb(pdb_path, dev)
        sfxn = _score_function(dev)
        if relax:
            ps = _relax(sfxn, ps)

        atom_chain = _atom_chain(ps)
        bco = ps.block_coord_offset[0]
        napb = ps.n_ats_per_block[0]
        cid = ps.chain_id[0]
        coords = ps.coords[0]  # (n_atoms, 3)

        # rigid interface ΔΔG total (which chain moves does not matter; move the top chain)
        mask = atom_chain == int(atom_chain.max())
        eb = _term_energies(sfxn, ps, ps.coords)
        cs = ps.coords.clone()
        cs[0, mask, 0] += sep_dist
        es = _term_energies(sfxn, ps, cs)
        ddg_total = float(sum(eb[k] - es.get(k, 0.0) for k in eb))
        out["ddg_total"] = ddg_total

        # per-binder-residue cross-chain contact counts (heavy atoms within cutoff)
        binder_blocks = [b for b in range(cid.shape[0]) if int(cid[b]) == int(binder_chain)]
        ag_mask = (atom_chain != int(binder_chain)) & (atom_chain >= 0)
        ag_xyz = coords[ag_mask]
        c2 = float(contact_cutoff) ** 2
        n_contacts: list[int] = []
        res_ids: list[int] = []
        for b in binder_blocks:
            s = int(bco[b])
            n = int(napb[b])
            res_ids.append(int(b))
            if n <= 0 or ag_xyz.shape[0] == 0:
                n_contacts.append(0)
                continue
            bx = coords[s : s + n]  # (n, 3)
            d2 = ((bx[:, None, :] - ag_xyz[None, :, :]) ** 2).sum(-1)
            n_contacts.append(int((d2 < c2).sum()))

        total_c = float(sum(n_contacts))
        if total_c > 0:
            per_res = [ddg_total * (c / total_c) for c in n_contacts]
        else:
            per_res = [0.0 for _ in n_contacts]
        out["per_res"] = per_res
        out["binder_res_ids"] = res_ids
        out["n_contacts"] = n_contacts
    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def ddg_packed_all(
    pdb_path: str,
    binder_chain: int = 1,
    device: str | torch.device = "cpu",
    relax: bool = False,
    sep_dist: float = 500.0,
    contact_cutoff: float = 5.0,
    num_threads: int = 4,
) -> dict:
    """Rigid interface ΔΔG group columns AND per-binder-residue attribution in ONE pass.

    The efficient combined engine for the live GRPO reward: it renders the bound and rigidly-
    separated poses **once each** (two :func:`_term_energies` calls) and reuses that single
    ``(eb, es)`` pair for both outputs — the whole-interface ΔΔG group columns
    (:func:`_ddg_groups`, driving the scalar ``w_ddg`` reward via :func:`ddg_reward`) and the
    contact-weighted per-binder-residue vector (driving the dense per-token structure arm,
    exactly as :func:`ddg_per_binder_residue`). Calling :func:`ddg_terms` **and**
    :func:`ddg_per_binder_residue` separately would redo the same rigid separation twice (four
    scorings); this does two, halving the per-design tmol cost on the repack worker.

    Relaxation flavors (``relax_unbound``) are intentionally not offered: the live reward is
    single-point rigid (``relax=False``), which on a packer-resolved complex already gives
    ``ddg_noclash ≈ ddg_total`` and costs ≈0.15 s/design versus ≈30 s relaxed. Set
    ``relax=True`` only for the (slow) de-clashed variant.

    Parameters
    ----------
    pdb_path : str
        Side-chain-packed complex PDB (>=2 chains; antigen then binder).
    binder_chain : int
        Chain id whose residues receive per-token credit (packed complex: antigen = 0,
        binder = 1).
    device, relax, sep_dist, contact_cutoff, num_threads
        As in :func:`ddg_terms` / :func:`ddg_per_binder_residue`.

    Returns
    -------
    dict
        The rigid group columns ``ddg_{total,atr,rep,sol,lkball,elec,hb,noclash}``,
        ``e_bound``/``e_sep``, ``n_res_a``/``n_res_b``, ``binder_chain``, ``relaxed``; the
        per-token fields ``per_res`` (list[float], length = #binder residues, summing to
        ``ddg_total``), ``binder_res_ids``, ``n_contacts``; and ``err`` (empty on success).

    Raises
    ------
    ImportError
        On first use if the optional ``tmol`` dependency is not installed (names the
        ``plm-design-rl[tmol]`` extra). Per-design scoring errors are captured in ``err``.
    """
    _require_tmol()
    from tmol import pose_stack_from_pdb

    torch.set_num_threads(num_threads)
    dev = torch.device(device)

    out: dict = {}
    for suf in _GRP_SUFFIXES:
        out[f"ddg_{suf}"] = float("nan")
    out.update(
        per_res=[],
        binder_res_ids=[],
        n_contacts=[],
        binder_chain=int(binder_chain),
        e_bound=float("nan"),
        e_sep=float("nan"),
        n_res_a=-1,
        n_res_b=-1,
        relaxed=int(relax),
        err="",
    )
    try:
        ps = pose_stack_from_pdb(pdb_path, dev)
        sfxn = _score_function(dev)
        if relax:
            ps = _relax(sfxn, ps)

        atom_chain = _atom_chain(ps)
        bco = ps.block_coord_offset[0]
        napb = ps.n_ats_per_block[0]
        cid = ps.chain_id[0]
        coords = ps.coords[0]  # (n_atoms, 3)

        # single rigid separation (which chain moves does not matter — interaction is symmetric)
        mask = atom_chain == int(atom_chain.max())
        eb = _term_energies(sfxn, ps, ps.coords)
        cs = ps.coords.clone()
        cs[0, mask, 0] += sep_dist
        es = _term_energies(sfxn, ps, cs)

        out.update(_ddg_groups("ddg", eb, es))
        ddg_total = float(out["ddg_total"])
        out["e_bound"] = float(sum(eb.values()))
        out["e_sep"] = float(sum(es.values()))

        # contact-weighted per-binder-residue attribution of ddg_total (same scheme as
        # ddg_per_binder_residue), reusing the coordinates already loaded above.
        binder_blocks = [b for b in range(cid.shape[0]) if int(cid[b]) == int(binder_chain)]
        ag_mask = (atom_chain != int(binder_chain)) & (atom_chain >= 0)
        ag_xyz = coords[ag_mask]
        c2 = float(contact_cutoff) ** 2
        n_contacts: list[int] = []
        res_ids: list[int] = []
        for b in binder_blocks:
            s = int(bco[b])
            n = int(napb[b])
            res_ids.append(int(b))
            if n <= 0 or ag_xyz.shape[0] == 0:
                n_contacts.append(0)
                continue
            bx = coords[s : s + n]
            d2 = ((bx[:, None, :] - ag_xyz[None, :, :]) ** 2).sum(-1)
            n_contacts.append(int((d2 < c2).sum()))

        total_c = float(sum(n_contacts))
        out["per_res"] = [ddg_total * (c / total_c) for c in n_contacts] if total_c > 0 else [0.0 for _ in n_contacts]
        out["binder_res_ids"] = res_ids
        out["n_contacts"] = n_contacts

        uniq, counts = torch.unique(cid, return_counts=True)
        if len(counts) >= 2:
            out["n_res_a"] = int(counts[0])
            out["n_res_b"] = int(counts[-1])
    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    import sys

    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    for p in paths:
        r = ddg_terms(p, relax="--relax" in sys.argv, relax_unbound="--relax-unbound" in sys.argv)
        print(p, r)
        print("  reward(rigid) =", ddg_reward(r), "reward(ub) =", ddg_reward(r, flavor="ddg_ub"))
