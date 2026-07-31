"""Standalone GRPO trainer for the LeFlur absorbing-state discrete flow binder policy.

This is a plain training loop (not a Lightning ``training_step``): the policy is a
generative sampler whose reward — Protenix co-folding interface-pTM — is slow,
non-differentiable, and served out-of-process, so the natural structure is an
explicit GRPO loop:

1. **Rollout** a group of ``group_size`` designs for a target with
   :meth:`~LeFlurSequenceStructureEncoderLightningModule.rollout_with_logprobs`
   (production sampler, all CFG/bias/diversity/schedules), capturing per-step
   transitions.
2. **Score** the sampled endpoint sequences with the reward oracle
   (:class:`~lobster.rl_training.ProtenixRewardClient`) and assemble the reward as a
   flat sum of four weighted terms — a per-metric-clipped confidence combo, a
   structure self-consistency TM-score, and two within-group k-mer-Jaccard novelty
   rewards (AA + 3Di) — see ``rewards/README.md``.
3. **Advantage** = group-relative standardization ``(r - mean) / (std + eps)`` with
   a variance floor so zero-variance groups don't explode.
4. **PPO update** — for ``mu`` inner iterations over a random subsample of steps
   (diffu-GRPO step-subsampling), recompute the differentiable log-prob
   (:meth:`~LeFlurSequenceStructureEncoderLightningModule.logprob_over_trajectory`)
   and KL-to-reference, form the clipped surrogate ``+ beta * KL``, and step AdamW
   over the encoder only. The *old* policy log-prob is snapshotted per-step under
   ``no_grad`` before the inner loop, so the importance ratio is against a fixed
   behaviour policy even as the weights move.

The reward is a slow external oracle and the policy holds the LG codec, so this
loop is exercised on GPU via ``lobster_rl_train`` rather than unit-tested; the
differentiable log-prob/KL kernels and the reward/diversity terms it composes are
each unit-tested (``tests/lobster/rl_training/``).

Notes
-----
This v1 targets the single-target overfit milestone and runs on one device. It is
written to be wrapped in DDP later (shard ``group_size`` across ranks, all-gather
rewards before standardizing); that is intentionally left as a follow-up so the
overfit signal can be established on one GPU first.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lobster.rl_training.rewards import (
    ProtenixRewardClient,
    confidence_components,
    continuous_ip,
    jaccard_novelty_group,
    passes,
    reward_from_confidence,
    structure_terms,
)

logger = logging.getLogger(__name__)

# Standard-AA token id -> letter (0-19 amino acids; 20=X, 21/22 gaps). Mirrors
# residue_constants.restype_order_with_x_inv; non-standard ids are remapped to a
# neutral residue before scoring (the sampled binder should be all-standard).
_AA_LETTERS = "ARNDCQEGHILKMFPSTWYV"
# 3Di structural-token alphabet (mini3di), index-aligned with the model's tri track.
_TRI_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


@dataclass
class GRPOTrainerConfig:
    """Hyperparameters for :class:`LeFlurGRPOTrainer`.

    Attributes
    ----------
    group_size : int
        Designs sampled per GRPO group (the "G" over which advantages are
        standardized).
    num_steps : int
        Number of GRPO optimization steps (outer loop).
    rollout_nsteps : int
        Denoising steps per rollout. Reduced (~40) vs production (200) because the
        log-prob recompute cost scales with captured steps (diffu-GRPO).
    steps_per_update : int
        Random step-subsample size used per inner PPO update. ``0`` = use all steps.
    mu : int
        Inner PPO updates per GRPO step (reusing the old-policy snapshot).
    beta : float
        KL-to-reference coefficient.
    eps_clip : float
        PPO ratio clip half-width.
    lr : float
        AdamW learning rate (encoder only).
    adv_eps : float
        Added to the group reward std when standardizing advantages.
    adv_std_floor : float
        Minimum group std; groups flatter than this contribute zero advantage
        (skipped) to avoid amplifying reward noise.
    normalize_advantage : bool
        If ``True`` (GRPO default) advantages are standardized ``(r - mean) /
        (std + eps)``. If ``False`` (Dr. GRPO) they are only mean-centered
        ``r - mean`` — this removes the per-group ``1/std`` reweighting that, in
        low-diversity groups, inflates the gradient of near-identical designs and
        biases the update toward easy (low-variance) groups.
    w_iptm, w_ptm, w_abag_iptm, w_plddt, w_gpde, w_pae_global, w_pae_interface : float
        Confidence-term weights. Each scales the corresponding Protenix metric after
        it is oriented higher-is-better and clipped to ``[0,1]`` (``*tm`` as-is,
        ``plddt/100``, ``1 - gpde/2``, ``1 - pae/32``). Defaults recover the shipped
        M22 reward: ``w_abag_iptm=1.0, w_ptm=0.5`` and every other metric ``0``.
    w_sctm_binder, w_sctm_complex : float
        Structure self-consistency weights on the TM-score between the LeFlur-generated
        backbone and the Protenix-predicted backbone — binder chain and whole complex
        respectively. ``> 0`` triggers ``return_coords`` on the reward client so the
        predicted CA coordinates ride back with the confidences. Default ``0``.
    w_seq_diversity, w_struct_diversity : float
        Weights on the within-group mean pairwise k-mer-Jaccard novelty of the AA
        sequence and the 3Di structural-token string (both already in ``[0,1]``,
        higher = more novel). Default ``0``.
    binder_length_min, binder_length_max : int | None
        When both set, each GRPO step samples one binder length ``L ~ U[min, max]`` and
        holds it constant across the whole group (per-group length sampling); otherwise
        the target's own ``spec.binder_length`` is used. Default ``None`` (fixed length).
    tracks : tuple[str, ...]
        Tracks included in the log-prob ratio / KL (seq-only ablation = just
        ``("sequence_tokens",)``).
    capture_old_lp_inline : bool
        Use the behaviour-policy log-prob captured inline during sampling (exact
        biased logits, zero extra forwards) for ``old_lp``, instead of a post-hoc
        recompute. ``True`` also enables the ``ppo/ratio_init ~ 1`` consistency
        check. Set ``False`` only for rollouts predating inline capture.
    grad_clip : float
        Max global grad norm (``0`` disables).
    rollout_kwargs : dict
        Extra keyword arguments forwarded to ``generate_sample`` (temperatures,
        stochasticity, cfg_weight, biases, diversity penalties, schedules).
    seed : int
        Base RNG seed (step subsampling + target cycling).
    log_every : int
        Log metrics every N GRPO steps.
    ckpt_dir : str | None
        Directory to save periodic policy checkpoints (None = disabled).
    ckpt_every : int
        Save a checkpoint every N GRPO steps (when ``ckpt_dir`` set).
    """

    group_size: int = 16
    num_steps: int = 500
    rollout_nsteps: int = 40
    steps_per_update: int = 4
    mu: int = 2
    beta: float = 0.02
    eps_clip: float = 0.2
    lr: float = 1e-6
    adv_eps: float = 1e-4
    adv_std_floor: float = 1e-3
    normalize_advantage: bool = True
    # Confidence-term weights (README defaults recover shipped M22: abag_iptm + 0.5*ptm).
    w_iptm: float = 0.0
    w_ptm: float = 0.5
    w_abag_iptm: float = 1.0
    w_plddt: float = 0.0
    w_gpde: float = 0.0
    w_pae_global: float = 0.0
    w_pae_interface: float = 0.0
    # Structure self-consistency (TM-score gen-vs-predicted); > 0 => request coords.
    w_sctm_binder: float = 0.0
    w_sctm_complex: float = 0.0
    # Within-group Jaccard-novelty diversity (AA + 3Di token string).
    w_seq_diversity: float = 0.0
    w_struct_diversity: float = 0.0
    # Per-group binder-length sampling (both set => L ~ U[min, max], constant per group).
    binder_length_min: int | None = None
    binder_length_max: int | None = None
    tracks: tuple[str, ...] = ("sequence_tokens", "structure_tokens", "tri_tokens")
    capture_old_lp_inline: bool = True
    grad_clip: float = 1.0
    rollout_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    log_every: int = 1
    ckpt_dir: str | None = None
    ckpt_every: int = 50


@dataclass
class TargetSpec:
    """One binder-design target for GRPO.

    Attributes
    ----------
    target_id : str
        Key into the reward manifest (``complexa_score_targets.csv``).
    antigen_pdb : str
        Path to the target/antigen PDB used to build the composite conditioning.
    target_chain : str
        Chain id of the target within ``antigen_pdb``.
    binder_length : int
        Binder chain length to design.
    epitope_indices : list[int] | None
        Target-local epitope residues (for epitope-anchored init + conditioning).
    """

    target_id: str
    antigen_pdb: str
    target_chain: str
    binder_length: int
    epitope_indices: list[int] | None = None


class LeFlurGRPOTrainer:
    """GRPO fine-tuning loop for the LeFlur binder policy (see module docstring).

    Parameters
    ----------
    model : LeFlurSequenceStructureEncoderLightningModule
        The policy to fine-tune (updated in place; encoder params only).
    reward_client : ProtenixRewardClient
        Blocking client for the Protenix reward pool.
    targets : list[TargetSpec]
        Targets to optimize; cycled round-robin across GRPO steps.
    config : GRPOTrainerConfig
        Hyperparameters.
    device : torch.device
        Device for the policy + rollouts.
    gen_cfg : object
        A ``cfg.generation``-like node with ``.get(name, default)`` (max_length,
        epitope/template flags, etc.) passed through to
        :func:`build_binder_static_cond`.
    ref_module : LeFlurSequenceStructureEncoderLightningModule | None
        Frozen reference policy for the KL term. Defaults to a deep copy of
        ``model`` taken at construction.
    wandb_run : object | None
        Optional wandb run for metric logging.
    """

    def __init__(
        self,
        model,
        reward_client: ProtenixRewardClient,
        targets: list[TargetSpec],
        config: GRPOTrainerConfig,
        device: torch.device,
        gen_cfg,
        ref_module=None,
        wandb_run=None,
    ) -> None:
        if not targets:
            raise ValueError("At least one target is required")
        self.model = model.to(device)
        self.reward_client = reward_client
        self.targets = targets
        self.config = config
        self.device = device
        self.gen_cfg = gen_cfg
        self.wandb_run = wandb_run

        # Frozen reference for the KL regularizer.
        self.ref_module = ref_module if ref_module is not None else copy.deepcopy(model)
        self.ref_module.to(device)
        self.ref_module.requires_grad_(False)
        self.ref_module.eval()

        # Permanent CPU copy of the VIT structure decoder. Initialized once here so
        # reward computation never moves the GPU decoder, avoiding CUDA deadlocks with
        # live rollout state on the GPU stream. Reused by the structure self-consistency
        # term to decode the generated backbone on CPU.
        self._cpu_vit_decoder = copy.deepcopy(model.decoder_factory.decoders["vit_decoder"])
        self._cpu_vit_decoder.cpu()
        self._cpu_vit_decoder.eval()
        self._cpu_vit_decoder.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.model.encoder.parameters(), lr=config.lr)
        self._rng = random.Random(config.seed)
        # Cache the per-(target, binder-length) static conditioning (target chain +
        # composite masks) — only the sampled binder changes across rollouts. Keyed by
        # (target_id, L) since per-group length sampling cycles many lengths per target;
        # a bounded LRU keeps the cache from growing across ~2000 targets × variable L.
        self._static_cache: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._static_cache_cap = 16

    # ---------------------------------------------------------------- conditioning
    def _sample_binder_length(self, spec: TargetSpec) -> int:
        """Binder length for this group: per-group sample when configured, else fixed.

        With both ``binder_length_min`` and ``binder_length_max`` set, draws one
        ``L ~ U[min, max]`` from the trainer RNG and holds it constant across the whole
        group (per the design decision). Otherwise falls back to ``spec.binder_length``.
        """
        cfg = self.config
        if cfg.binder_length_min is not None and cfg.binder_length_max is not None:
            return self._rng.randint(cfg.binder_length_min, cfg.binder_length_max)
        return spec.binder_length

    def _target_static(self, spec: TargetSpec, binder_length: int) -> dict:
        """Build (and LRU-cache) the composite static conditioning for a target + length."""
        key = (spec.target_id, binder_length)
        if key in self._static_cache:
            self._static_cache.move_to_end(key)  # mark most-recently-used
            return self._static_cache[key]

        from lobster.cmdline.generate_modes._binders import (
            _build_scalar_cond_bins,
            build_binder_static_cond,
            load_binder_target,
        )
        from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

        max_length = self.gen_cfg.get("max_length", 512)
        loaded = load_binder_target(spec.antigen_pdb, spec.target_chain, max_length=max_length)
        if loaded is None:
            raise ValueError(f"Failed to load target {spec.target_id} from {spec.antigen_pdb}")

        tok = AminoAcidTokenizerTransform(max_length=max_length)
        comp = build_binder_static_cond(
            binder_length,
            loaded["target_data_filtered"],
            loaded["chains_key"],
            loaded["target_chain_idx"],
            model=self.model,
            gen_cfg=self.gen_cfg,
            device=self.device,
            tokenizer_transform=tok,
            epitope_indices=spec.epitope_indices,
            output_dir=None,
            structure_path=spec.antigen_pdb,
            save_initial=False,
        )
        if comp is None:
            raise ValueError(f"Composite for target {spec.target_id} exceeds max_length {max_length}")
        # Positions of the designed binder chain (for slicing decoded sequences).
        comp["binder_positions"] = comp["chains_ids"][0] == comp["binder_chain_idx"]
        # Optional per-target scalar conditioning bins (concat cond), placed on binder residues.
        comp["scalar_cond_bins"] = _build_scalar_cond_bins(
            self.gen_cfg, comp["chains_ids"][0], comp["binder_chain_idx"], comp["L_total"], self.device
        )
        self._static_cache[key] = comp
        self._static_cache.move_to_end(key)
        while len(self._static_cache) > self._static_cache_cap:
            self._static_cache.popitem(last=False)  # evict least-recently-used
        return comp

    def _build_gen_kwargs(self, comp: dict, group_size: int) -> dict:
        """Assemble ``generate_sample`` kwargs for a group rollout from static cond.

        Every conditioning tensor is expanded along the batch dim to ``group_size``
        so the sampler draws ``G`` independent designs from identical conditioning.
        """

        def _expand(t):
            if t is None:
                return None
            return t.expand(group_size, *t.shape[1:]).contiguous()

        cfg = self.config
        kwargs = dict(
            length=comp["L_total"],
            num_samples=group_size,
            nsteps=cfg.rollout_nsteps,
            inpainting=True,
            input_structure_coords=_expand(comp["coords_res"]),
            input_sequence_tokens=_expand(comp["sequence_tokenized"]),
            input_mask=_expand(comp["mask"]),
            input_indices=_expand(comp["indices"]),
            inpainting_mask_sequence=_expand(comp["mask_sequence"]),
            inpainting_mask_structure=_expand(comp["mask_structure"]),
            chain_ids=_expand(comp["chain_ids_emb"]),
            conditioning_tensor_override=_expand(comp["cond_tensor"]),
            template_structure_tokens=_expand(comp["template_arg"]),
        )
        if comp.get("scalar_cond_bins"):
            kwargs["scalar_cond_bins"] = {k: _expand(v) for k, v in comp["scalar_cond_bins"].items()}
        # Sampler knobs (temperatures, stochasticity, cfg_weight, biases, diversity,
        # schedules) come straight from config.rollout_kwargs, overriding nothing above.
        kwargs.update(cfg.rollout_kwargs)
        return kwargs

    # ------------------------------------------------------------------- decoding
    def _decode_binder_seqs(self, trajectory: dict, comp: dict) -> list[str]:
        """Sampled binder sequences (letters) for each design in the group."""
        aa = self.model.decode_endpoint_aa(trajectory)  # (G, L) standard-AA ids
        pos = comp["binder_positions"].to(aa.device)
        binder_ids = aa[:, pos].cpu().tolist()
        seqs = []
        for row in binder_ids:
            seqs.append("".join(_AA_LETTERS[i] if 0 <= i < 20 else "G" for i in row))
        return seqs

    def _decode_binder_tri(self, trajectory: dict, comp: dict) -> list[str] | None:
        """Per-design 3Di structural-token strings for the binder positions.

        Reads the sampled endpoint 3Di tokens (``final_xt["tri_tokens"]``) — the
        model's own structural alphabet ``"ACDEFGHIKLMNPQRSTVWYX"`` — and maps them
        to letters at the binder positions, mirroring :meth:`_decode_binder_seqs`.
        Returns ``None`` when the checkpoint has no 3Di track (no ``tri_tokens`` in
        the rollout), so the caller falls back to sequence-only diversity.
        """
        final_xt = trajectory.get("final_xt", {})
        tri = final_xt.get("tri_tokens")
        if tri is None:
            return None
        pos = comp["binder_positions"].cpu()
        tri_ids = tri.long()[:, pos].cpu().tolist()
        n = len(_TRI_ALPHABET)
        return ["".join(_TRI_ALPHABET[i] if 0 <= i < n else "X" for i in row) for row in tri_ids]

    # ------------------------------------------------------------- generated coords
    def _decode_backbone_coords(self, trajectory: dict, comp: dict) -> torch.Tensor:
        """Decode the sampled structure tokens to ``(G, L, 3, 3)`` backbone coords on CPU.

        Uses the permanent CPU copy of the VIT decoder (``self._cpu_vit_decoder``,
        built in ``__init__``) so no device moves happen here — avoids a CUDA deadlock
        with live GPU rollout state. The three decoded backbone atoms are ``(N, CA, C)``;
        CA is atom index 1. Reused by the structure self-consistency reward.
        """
        G = self.config.group_size
        struc_tokens = trajectory["final_xt"]["structure_tokens"].long().cpu()  # (G, L)
        n_tok = self.model.quantizer.n_tokens
        mask_cpu = comp["mask"].expand(G, -1).bool().cpu()  # (G, L)
        valid = mask_cpu & (struc_tokens < n_tok)
        with torch.no_grad():
            onehot = (
                torch.nn.functional.one_hot(struc_tokens.clamp(0, n_tok - 1), n_tok).float()
                * valid.unsqueeze(-1).float()
            )  # (G, L, n_tok)
            xyz = self._cpu_vit_decoder(onehot, valid.float())  # (G, L, 3, 3)
        return xyz

    def _structure_terms_for_group(
        self, trajectory: dict, comp: dict, confs: list[dict | None]
    ) -> tuple[list[float], list[float], list[float]]:
        """Per-design structure self-consistency: ``(weighted_term, sctm_binder, sctm_complex)``.

        Decodes the generated backbone once (CA at atom index 1), then for each design
        computes the TM-score of the generated vs Protenix-predicted CA — binder chain
        and whole complex (antigen-then-binder, matching the worker's chain A/B order).
        A missing/length-mismatched pair contributes ``0.0`` (see
        :func:`~lobster.rl_training.rewards.structure_terms`).
        """
        import numpy as np

        cfg = self.config
        gen_ca = self._decode_backbone_coords(trajectory, comp)[:, :, 1, :]  # (G, L, 3)
        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()
        antigen_mask = valid & ~binder_mask
        weighted, sctm_b, sctm_c = [], [], []
        for i, conf in enumerate(confs):
            gca = gen_ca[i].numpy()
            gen_binder = gca[binder_mask]
            gen_complex = np.concatenate([gca[antigen_mask], gca[binder_mask]], axis=0)
            pred_binder = pred_complex = None
            if conf is not None:
                bx = conf.get("binder_xyz")
                ax = conf.get("antigen_xyz")
                if bx is not None:
                    pred_binder = np.asarray(bx, dtype=np.float64)
                if ax is not None and bx is not None:
                    pred_complex = np.concatenate(
                        [np.asarray(ax, dtype=np.float64), np.asarray(bx, dtype=np.float64)], axis=0
                    )
            st = structure_terms(gen_binder, pred_binder, gen_complex, pred_complex)
            sctm_b.append(st["sctm_binder"])
            sctm_c.append(st["sctm_complex"])
            weighted.append(cfg.w_sctm_binder * st["sctm_binder"] + cfg.w_sctm_complex * st["sctm_complex"])
        return weighted, sctm_b, sctm_c

    # -------------------------------------------------------------------- rewards
    def _compute_rewards(
        self,
        target_id: str,
        seqs: list[str],
        tri_seqs: list[str] | None,
        trajectory: dict,
        comp: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Assemble the four-term reward for a group; return ``(rewards (G,), metrics)``.

        ``reward_i = confidence_i + structure_i + seq_diversity_i + struct_diversity_i``,
        each term a weighted, per-metric-clipped (``[0,1]``) contribution (see
        ``rewards/README.md``). Structure coords are fetched from the oracle only when a
        structure weight is non-zero.
        """
        cfg = self.config
        G = len(seqs)
        conf_weights = {
            "w_iptm": cfg.w_iptm,
            "w_ptm": cfg.w_ptm,
            "w_abag_iptm": cfg.w_abag_iptm,
            "w_plddt": cfg.w_plddt,
            "w_gpde": cfg.w_gpde,
            "w_pae_global": cfg.w_pae_global,
            "w_pae_interface": cfg.w_pae_interface,
        }
        need_struct = cfg.w_sctm_binder > 0 or cfg.w_sctm_complex > 0
        confs = self.reward_client.score_group(target_id, seqs, return_coords=need_struct)

        # 1. Confidence term — flat, per-metric-clipped weighted linear combo.
        conf_terms = [reward_from_confidence(c, conf_weights) for c in confs]

        # 2. Structure self-consistency term (only when a structure weight is on).
        if need_struct:
            struct_terms_l, sctm_b, sctm_c = self._structure_terms_for_group(trajectory, comp, confs)
        else:
            struct_terms_l = [0.0] * G
            sctm_b = sctm_c = [0.0] * G

        # 3./4. Diversity terms — mean pairwise k-mer-Jaccard novelty (AA + 3Di).
        seq_nov = jaccard_novelty_group(seqs)
        seq_div_terms = [cfg.w_seq_diversity * v for v in seq_nov]
        struct_nov = jaccard_novelty_group(tri_seqs) if tri_seqs is not None else [0.0] * G
        struct_div_terms = [cfg.w_struct_diversity * v for v in struct_nov]

        rewards = torch.tensor(
            [c + s + sd + td for c, s, sd, td in zip(conf_terms, struct_terms_l, seq_div_terms, struct_div_terms)],
            dtype=torch.float32,
            device=self.device,
        )

        # Per-key mean/std over the designs that actually received that Protenix
        # field (None-scored designs are excluded from the raw-confidence stats).
        def _conf_stat(key: str) -> tuple[float, float]:
            vals = [c[key] for c in confs if c is not None and c.get(key) is not None]
            if not vals:
                return 0.0, 0.0
            t = torch.tensor(vals, dtype=torch.float32)
            return float(t.mean()), (float(t.std(unbiased=False)) if len(vals) > 1 else 0.0)

        ptm_mean, ptm_std = _conf_stat("ptm")
        iptm_mean, iptm_std = _conf_stat("iptm")
        abag_mean, abag_std = _conf_stat("abag_iptm")
        plddt_mean, _ = _conf_stat("plddt")
        gpde_mean, gpde_std = _conf_stat("gpde")
        pae_global_mean, pae_global_std = _conf_stat("pae_global")
        pae_iface_mean, pae_iface_std = _conf_stat("pae_interface")
        scored_frac = float(sum(continuous_ip(c) is not None for c in confs) / G)

        # Per-metric confidence contribution means (only active, present metrics).
        comp_sums: dict[str, float] = {}
        for c in confs:
            for wk, v in confidence_components(c, conf_weights).items():
                comp_sums[wk] = comp_sums.get(wk, 0.0) + v

        metrics = {
            # --- total reward + four-term decomposition (terms sum to reward/mean) ---
            "reward/mean": float(rewards.mean()),
            "reward/std": float(rewards.std(unbiased=False)),
            "reward/max": float(rewards.max()),
            "reward/confidence_term_mean": float(sum(conf_terms) / G),
            "reward/structure_term_mean": float(sum(struct_terms_l) / G),
            "reward/seq_diversity_term_mean": float(sum(seq_div_terms) / G),
            "reward/struct_diversity_term_mean": float(sum(struct_div_terms) / G),
            "reward/struct_sctm_binder_term_mean": float(cfg.w_sctm_binder * sum(sctm_b) / G),
            "reward/struct_sctm_complex_term_mean": float(cfg.w_sctm_complex * sum(sctm_c) / G),
            # --- raw Protenix confidences (confidence module's field names) ---
            "conf/ptm_mean": ptm_mean,
            "conf/ptm_std": ptm_std,
            "conf/iptm_mean": iptm_mean,
            "conf/iptm_std": iptm_std,
            "conf/abag_iptm_mean": abag_mean,
            "conf/abag_iptm_std": abag_std,
            "conf/plddt_mean": plddt_mean,
            "conf/gpde_mean": gpde_mean,
            "conf/gpde_std": gpde_std,
            # PAE (predicted aligned error, Å) from token_pair_pae. global = mean over all
            # token pairs; interface = mean over cross-chain (A↔B) pairs.
            "conf/pae_global_mean": pae_global_mean,
            "conf/pae_global_std": pae_global_std,
            "conf/pae_interface_mean": pae_iface_mean,
            "conf/pae_interface_std": pae_iface_std,
            "conf/scored_frac": scored_frac,
            "conf/pass_rate": float(sum(passes(c) for c in confs) / G),
            # --- structure self-consistency (raw TM-scores) ---
            "struct/sctm_binder": float(sum(sctm_b) / G),
            "struct/sctm_complex": float(sum(sctm_c) / G),
            # --- diversity novelty (raw mean pairwise Jaccard distance) ---
            "diversity/seq_novelty_mean": float(sum(seq_nov) / G),
            "diversity/struct_novelty_mean": float(sum(struct_nov) / G),
            "diversity/unique_frac": float(len(set(seqs)) / G),
        }
        # Per-metric confidence-term means (e.g. reward/conf_ptm_term_mean) — active only.
        for wk, total in comp_sums.items():
            metrics[f"reward/conf_{wk[2:]}_term_mean"] = float(total / G)
        if tri_seqs is not None:
            metrics["diversity/tri_unique_frac"] = float(len(set(tri_seqs)) / len(tri_seqs))
        return rewards, metrics

    @staticmethod
    def _advantages(
        rewards: torch.Tensor, eps: float, std_floor: float, normalize: bool = True
    ) -> tuple[torch.Tensor, float]:
        """Group-relative advantages; returns ``(A, std)``.

        With ``normalize=True`` (GRPO) advantages are standardized by the group std;
        with ``normalize=False`` (Dr. GRPO) they are only mean-centered, dropping the
        ``1/std`` reweighting. ``std`` is always the raw group reward std (used for the
        flat-group skip and logging), independent of the normalization mode.
        """
        std = float(rewards.std(unbiased=False))
        centered = rewards - rewards.mean()
        adv = centered / (rewards.std(unbiased=False) + eps) if normalize else centered
        return adv, std

    @staticmethod
    def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        """Pearson correlation of two 1-D tensors; 0.0 if either is constant."""
        a = a.detach().float()
        b = b.detach().float()
        a = a - a.mean()
        b = b - b.mean()
        denom = a.norm() * b.norm()
        if float(denom) < 1e-12:
            return 0.0
        return float((a * b).sum() / denom)

    # ---------------------------------------------------------------- GRPO update
    def _grpo_step(self, spec: TargetSpec, step_idx: int) -> dict:
        """One GRPO step for a target: rollout -> reward -> mu PPO inner updates."""
        cfg = self.config
        binder_length = self._sample_binder_length(spec)
        comp = self._target_static(spec, binder_length)

        # 1. Rollout (no grad — sampling only; grad enters via log-prob recompute).
        with torch.no_grad():
            trajectory = self.model.rollout_with_logprobs(**self._build_gen_kwargs(comp, cfg.group_size))
        seqs = self._decode_binder_seqs(trajectory, comp)
        tri_seqs = self._decode_binder_tri(trajectory, comp)

        # 2. Reward (four-term) + 3. advantages.
        rewards, metrics = self._compute_rewards(spec.target_id, seqs, tri_seqs, trajectory, comp)
        metrics["rollout/binder_length"] = float(binder_length)
        advantages, std = self._advantages(rewards, cfg.adv_eps, cfg.adv_std_floor, cfg.normalize_advantage)

        if std < cfg.adv_std_floor:
            logger.info("step %d target %s: group std %.4g < floor, skipping update", step_idx, spec.target_id, std)
            metrics["update/skipped_flat_group"] = 1.0
            return metrics

        n_steps = len(trajectory["steps"])
        # 4a. OLD (behaviour) policy per-step log-prob, snapshotted under no_grad so the
        # importance ratio is against fixed weights across all inner updates. Log-prob is
        # additive over steps, so a subset's old log-prob is the sum of its per-step rows.
        # Prefer the INLINE captures (exact biased logits the sampler drew from, zero extra
        # forwards, faithful by construction); fall back to a post-hoc recompute for rollouts
        # without inline log-prob. With inline old_lp, the first inner iteration's ratio must
        # be ~1 (`ppo/ratio_init`) — a live check that the new_lp recompute mirrors the sampler.
        with torch.no_grad():
            if cfg.capture_old_lp_inline:
                old_lp_per_step = self.model.captured_logprob_per_step(trajectory, tracks=cfg.tracks)
            else:
                old_lp_per_step = torch.stack(
                    [
                        self.model.logprob_over_trajectory(trajectory, tracks=cfg.tracks, step_indices=[i])
                        for i in range(n_steps)
                    ]
                )  # (n_steps, G)

        ratios, clipfracs, kls, pg_losses = [], [], [], []
        dlp_means, dlp_corrs = [], []
        for _ in range(cfg.mu):
            subset = self._sample_step_subset(n_steps)
            new_lp = self.model.logprob_over_trajectory(trajectory, tracks=cfg.tracks, step_indices=subset)
            old_lp = old_lp_per_step[subset].sum(dim=0)
            # KL costs a full extra grad-forward over the subset (self + frozen ref). When
            # beta == 0 the term is multiplied by zero and contributes nothing to the loss, so
            # skip it entirely — this roughly halves the backward-graph memory (letting a larger
            # steps_per_update fit) and the compute. Log kl_mean = 0 in that case.
            if cfg.beta > 0:
                kl = self.model.kl_over_trajectory(trajectory, self.ref_module, tracks=cfg.tracks, step_indices=subset)
                kl_mean = kl.mean()
            else:
                kl_mean = torch.zeros((), device=new_lp.device)

            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * advantages
            pg_loss = -torch.min(unclipped, clipped).mean()
            loss = pg_loss + cfg.beta * kl_mean

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), cfg.grad_clip)
                if cfg.grad_clip > 0
                else torch.tensor(0.0)
            )
            self.optimizer.step()

            ratios.append(float(ratio.mean()))
            clipfracs.append(float(((ratio - 1.0).abs() > cfg.eps_clip).float().mean()))
            kls.append(float(kl_mean))
            pg_losses.append(float(pg_loss))
            # Common-mode vs advantage-differential diagnostic. dlp = new_lp - old_lp is the
            # per-design log-prob shift this update produced. Its GROUP MEAN is the common-mode
            # drift (all designs moving together, orthogonal to the mean-centered advantage); its
            # correlation with the advantage is the useful, differential signal. Flat reward with
            # large |dlp_mean| but ~0 dlp_adv_corr = the update is pushing the whole group in
            # lockstep instead of separating winners from losers (the M6-M12 failure mode).
            dlp = (new_lp - old_lp).detach()
            dlp_means.append(float(dlp.mean()))
            dlp_corrs.append(self._pearson(advantages, dlp))

        metrics.update(
            {
                "advantage/std": std,
                "advantage/abs_mean": float(advantages.abs().mean()),
                "advantage/max_abs": float(advantages.abs().max()),
                # First inner-iter ratio: with inline old_lp this is a consistency check on the
                # new_lp recompute (should be ~1.0). Divergence flags a sampler/recompute mismatch.
                "ppo/ratio_init": ratios[0],
                "ppo/ratio_mean": sum(ratios) / len(ratios),
                "ppo/clip_frac": sum(clipfracs) / len(clipfracs),
                "ppo/kl_mean": sum(kls) / len(kls),
                "ppo/kl_term": cfg.beta * (sum(kls) / len(kls)),
                "ppo/pg_loss": sum(pg_losses) / len(pg_losses),
                "ppo/grad_norm": float(grad_norm),
                # Diagnostics: common-mode drift and advantage-differential faithfulness.
                "ppo/dlp_mean": sum(dlp_means) / len(dlp_means),
                "ppo/dlp_adv_corr": sum(dlp_corrs) / len(dlp_corrs),
                "rollout/n_steps": float(n_steps),
            }
        )
        return metrics

    def _sample_step_subset(self, n_steps: int) -> list[int]:
        k = self.config.steps_per_update
        if k <= 0 or k >= n_steps:
            return list(range(n_steps))
        return sorted(self._rng.sample(range(n_steps), k))

    # ------------------------------------------------------------------------ run
    def train(self) -> None:
        """Run the GRPO optimization loop for ``config.num_steps`` steps."""
        cfg = self.config
        logger.info(
            "Starting GRPO: %d steps, group_size=%d, rollout_nsteps=%d, mu=%d, beta=%g, lr=%g, targets=%s",
            cfg.num_steps,
            cfg.group_size,
            cfg.rollout_nsteps,
            cfg.mu,
            cfg.beta,
            cfg.lr,
            [t.target_id for t in self.targets],
        )
        # eval() — grad still flows (encoder params require grad); this only disables dropout /
        # BN-stat updates so the sampling forward and the log-prob recompute are deterministic and
        # bit-identical (the importance ratio depends on it). No train-only layers on the gen path.
        self.model.eval()
        for step in range(cfg.num_steps):
            spec = self.targets[step % len(self.targets)]
            metrics = self._grpo_step(spec, step)
            metrics["step"] = step
            metrics["target"] = spec.target_id
            if step % cfg.log_every == 0:
                logger.info(
                    "step %d [%s] reward/mean=%.4f pass=%.3f ptm=%.3f iptm=%.3f kl=%.4g ratio=%.3f",
                    step,
                    spec.target_id,
                    metrics.get("reward/mean", 0.0),
                    metrics.get("conf/pass_rate", 0.0),
                    metrics.get("conf/ptm_mean", 0.0),
                    metrics.get("conf/abag_iptm_mean", 0.0),
                    metrics.get("ppo/kl_mean", 0.0),
                    metrics.get("ppo/ratio_mean", 1.0),
                )
            if self.wandb_run is not None:
                self.wandb_run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
            if cfg.ckpt_dir and cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0:
                self._save_checkpoint(step + 1)
        if cfg.ckpt_dir:
            self._save_checkpoint(cfg.num_steps)

    def _save_checkpoint(self, step: int) -> None:
        out = Path(self.config.ckpt_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"grpo_step_{step}.ckpt"
        # Lightning-style save so the result reloads via load_from_checkpoint.
        torch.save({"state_dict": self.model.state_dict(), "grpo_step": step}, path)
        logger.info("Saved GRPO checkpoint: %s", path)
