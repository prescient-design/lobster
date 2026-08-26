"""Hydra entry point for GRPO RL fine-tuning of the LeFlur 3Di binder policy.

Wires the config-composed pieces together and hands off to
:class:`lobster.rl_training.LeFlurGRPOTrainer`:

* load the policy checkpoint (same ``resolve_checkpoint`` path as ``lobster_generate``),
* translate ``cfg.generation`` into the production sampler ``rollout_kwargs`` (schedules,
  temperatures, stochasticity, CFG weight, logit biases, diversity penalties) — identical
  knobs to ``lobster_generate`` binder design, so rollouts match eval-time sampling,
* build the target list and the Protenix :class:`~lobster.rl_training.ProtenixRewardClient`,
* assemble :class:`~lobster.rl_training.GRPOTrainerConfig` from ``cfg.grpo`` and run.

The reward oracle is served by an out-of-process Protenix worker pool
(``scripts/protenix_reward_server.py``) over the shared filesystem queue at
``cfg.reward.queue_dir``; this process only submits jobs and blocks for results.

Usage
-----
.. code-block:: bash

    lobster_rl_train --config-name experiment/rl_leflur_binder_grpo_overfit \\
        paths=public reward.targets_csv=/path/to/complexa_score_targets.csv
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from lobster.model.leflur import resolve_checkpoint
from lobster.rl_training import GRPOTrainerConfig, LeFlurGRPOTrainer, ProtenixRewardClient, TargetSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_rollout_kwargs(gen_cfg: DictConfig, device: torch.device) -> dict:
    """Translate ``cfg.generation`` into ``generate_sample`` sampler knobs.

    Mirrors the binder-design sampler setup in
    :mod:`lobster.cmdline.generate_modes._binders` so RL rollouts are drawn from
    the exact production sampler (schedules, temperatures, stochasticity, CFG,
    logit biases, diversity penalties). Per-target static conditioning tensors are
    added later by the trainer; this returns only the target-independent knobs.
    """
    import functools

    from lobster.cmdline.generate_modes._shared import _get_inference_schedule_class
    from lobster.model.leflur._leflur_sequence_structure_encoder_lightning_module import (
        LinearInferenceSchedule,
        LogInferenceSchedule,
    )

    def _build_sched(name, exp, default):
        if not name:
            return default
        cls = _get_inference_schedule_class(name)
        return functools.partial(cls, exponent=float(exp)) if exp is not None else cls

    struct_schedule = _build_sched(
        gen_cfg.get("inference_schedule_struc", None), gen_cfg.get("schedule_exponent", None), LinearInferenceSchedule
    )
    seq_schedule = _build_sched(
        gen_cfg.get("inference_schedule_seq", None), gen_cfg.get("seq_schedule_exponent", None), LogInferenceSchedule
    )
    tri_schedule = _build_sched(
        gen_cfg.get("inference_schedule_tri", None), gen_cfg.get("tri_schedule_exponent", None), None
    )

    # Per-amino-acid additive sequence logit bias (dict AA -> float), same as binder gen.
    seq_logit_bias = None
    seq_bias_cfg = gen_cfg.get("sequence_logit_bias", None)
    if seq_bias_cfg:
        from lobster.tokenization._amino_acid import AA_VOCAB

        seq_logit_bias = torch.zeros(len(AA_VOCAB), device=device)
        for aa, bval in dict(seq_bias_cfg).items():
            if aa in AA_VOCAB:
                seq_logit_bias[AA_VOCAB[aa]] = float(bval)
            else:
                logger.warning("Unknown amino acid %r in sequence_logit_bias, skipping", aa)

    # Per-3Di-state additive logit bias (dict state -> float).
    tri_logit_bias = None
    tri_bias_cfg = gen_cfg.get("tri_logit_bias", None)
    if tri_bias_cfg:
        from lobster.model.latent_generator.utils.mini3di._encoder import ALPHABET

        tri_alphabet = "".join(ALPHABET)  # "ACDEFGHIKLMNPQRSTVWYX"
        tri_logit_bias = torch.zeros(22, device=device)
        for state, bval in dict(tri_bias_cfg).items():
            if state in tri_alphabet:
                tri_logit_bias[tri_alphabet.index(state)] = float(bval)
            else:
                logger.warning("Unknown 3Di state %r in tri_logit_bias, skipping", state)

    return dict(
        inference_schedule_seq=seq_schedule,
        inference_schedule_struc=struct_schedule,
        inference_schedule_tri=tri_schedule,
        tri_time_accel=float(gen_cfg.get("tri_time_accel", 1.0)),
        temperature_seq=gen_cfg.get("temperature_seq", 0.5),
        temperature_struc=gen_cfg.get("temperature_struc", 0.5),
        stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
        stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
        stochasticity_tri=gen_cfg.get("stochasticity_tri", None),
        asynchronous_sampling=bool(gen_cfg.get("asynchronous_sampling", False)),
        cfg_weight=float(gen_cfg.get("cfg_weight", 1.0)),
        sequence_logit_bias=seq_logit_bias,
        sequence_logit_bias_steps=int(gen_cfg.get("sequence_logit_bias_steps", 200)),
        tri_logit_bias=tri_logit_bias,
        sequence_diversity_penalty=float(gen_cfg.get("sequence_diversity_penalty", 0.0)),
        tri_diversity_penalty=float(gen_cfg.get("tri_diversity_penalty", 0.0)),
        encode_target_only=bool(gen_cfg.get("encode_target_only", False)),
    )


def _parse_epitope_indices(raw) -> list[int] | None:
    """Parse epitope indices from a CSV cell or YAML list into ``list[int] | None``.

    Accepts a ``,``/whitespace-separated string (CSV path) or a sequence (YAML list);
    empty / missing → ``None`` (no epitope conditioning).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p for p in raw.replace(",", " ").split() if p != ""]
        return [int(p) for p in parts] if parts else None
    idx = [int(p) for p in raw]
    return idx or None


def _build_targets(cfg: DictConfig) -> list[TargetSpec]:
    """Parse ``cfg.targets`` into :class:`TargetSpec` records.

    ``cfg.targets`` is either an inline YAML list of target dicts (small overfit
    configs) or a string path to a targets CSV with columns ``target_id,
    antigen_pdb, target_chain, binder_length, epitope_indices`` (the pinder-heteromer
    manifest, ~2000 rows — kept out of the YAML). ``epitope_indices`` is a
    ``,``-separated field in the CSV and a list in YAML.
    """
    specs: list[TargetSpec] = []
    targets = cfg.targets
    if isinstance(targets, str):
        import csv

        with open(targets, newline="") as fh:
            for row in csv.DictReader(fh):
                specs.append(
                    TargetSpec(
                        target_id=row["target_id"],
                        antigen_pdb=row["antigen_pdb"],
                        target_chain=row.get("target_chain", "A") or "A",
                        binder_length=int(row["binder_length"]),
                        epitope_indices=_parse_epitope_indices(row.get("epitope_indices")),
                    )
                )
    else:
        for entry in targets:
            specs.append(
                TargetSpec(
                    target_id=entry["target_id"],
                    antigen_pdb=entry["antigen_pdb"],
                    target_chain=entry.get("target_chain", "A"),
                    binder_length=int(entry["binder_length"]),
                    epitope_indices=_parse_epitope_indices(entry.get("epitope_indices")),
                )
            )
    if not specs:
        raise ValueError("cfg.targets must list at least one target (inline list or CSV path)")
    return specs


def _maybe_wandb(cfg: DictConfig):
    """Initialize a wandb run if ``cfg.wandb.enabled``; else return None."""
    wandb_cfg = cfg.get("wandb", None)
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return None
    import wandb

    # Snapshot the config for wandb WITHOUT resolving interpolations: this RL config
    # sets ``output_dir`` directly and never populates Hydra's ``paths`` group, so
    # ``paths.output_dir=${paths.root_dir}`` points at a mandatory-missing (``???``)
    # value. ``resolve=True`` raises InterpolationToMissingValueError on it (and
    # ``throw_on_missing=False`` does NOT suppress interpolation-to-missing errors);
    # the RL knobs we care about (``grpo.*``, ``output_dir``, ``reward.*``) are
    # literals/overrides, not interpolations, so an unresolved snapshot logs fine.
    return wandb.init(
        project=wandb_cfg.get("project", "leflur-grpo"),
        name=wandb_cfg.get("name", None),
        entity=wandb_cfg.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=False),
    )


@hydra.main(version_base=None, config_path="../hydra_config", config_name="experiment/rl_leflur_binder_grpo_overfit")
def rl_train(cfg: DictConfig) -> None:
    """Hydra entry point for the ``lobster_rl_train`` console script (see module docstring)."""
    logger.info("Starting LeFlur GRPO RL fine-tuning")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if cfg.get("seed") is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)

    # Load policy checkpoint (short name / hf:// / https / local path -> concrete file).
    resolved_ckpt = resolve_checkpoint(cfg.model.ckpt_path)
    logger.info("Loading policy from %r -> %s", cfg.model.ckpt_path, resolved_ckpt)
    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls.load_from_checkpoint(str(resolved_ckpt))

    reward_cfg = cfg.reward
    reward_client = ProtenixRewardClient(
        queue_dir=reward_cfg.queue_dir,
        targets_csv=reward_cfg.targets_csv,
        timeout_s=int(reward_cfg.get("timeout_s", 1800)),
        poll_s=float(reward_cfg.get("poll_s", 2.0)),
        cache=bool(reward_cfg.get("cache", True)),
        n_shards=int(reward_cfg.get("n_shards", 1)),
    )

    g = cfg.grpo
    _blen_min = g.get("binder_length_min", None)
    _blen_max = g.get("binder_length_max", None)
    # LigandMPNN-repack reward pool config (serves SC shape-complementarity, all-atom clash,
    # and ProteinMPNN AAR from ONE pool). Read from an optional top-level `shape:` block
    # (mirrors `reward:`); the queue defaults to a `shape_queue/` subdir of the run's
    # output_dir when any repack weight (w_shape / w_sc_clash / w_aar) > 0 and no explicit
    # queue_dir is given. Absent block + all repack weights 0 leaves every repack term inert
    # (no client constructed).
    shape_cfg = cfg.get("shape", None)
    _shape_queue = None
    if shape_cfg is not None:
        _shape_queue = shape_cfg.get("queue_dir", None)
    _need_repack = (
        float(g.get("w_shape", 0.0)) > 0
        or float(g.get("w_sc_clash", 0.0)) > 0
        or float(g.get("w_aar", 0.0)) > 0
        # CHORD SFT distillation also draws the LigandMPNN-designed sequence from the pool.
        or float(g.get("sft_mu", 0.0)) > 0
    )
    if _shape_queue is None and _need_repack:
        _shape_queue = f"{cfg.output_dir}/shape_queue"
    grpo_config = GRPOTrainerConfig(
        group_size=int(g.group_size),
        num_steps=int(g.num_steps),
        rollout_nsteps=int(g.rollout_nsteps),
        steps_per_update=int(g.steps_per_update),
        mu=int(g.mu),
        beta=float(g.beta),
        eps_clip=float(g.eps_clip),
        lr=float(g.lr),
        adv_eps=float(g.get("adv_eps", 1e-4)),
        adv_std_floor=float(g.get("adv_std_floor", 1e-3)),
        normalize_advantage=bool(g.get("normalize_advantage", True)),
        # Confidence-term weights (README defaults recover shipped M22: abag_iptm + 0.5*ptm).
        w_iptm=float(g.get("w_iptm", 0.0)),
        w_ptm=float(g.get("w_ptm", 0.5)),
        w_abag_iptm=float(g.get("w_abag_iptm", 1.0)),
        w_plddt=float(g.get("w_plddt", 0.0)),
        w_gpde=float(g.get("w_gpde", 0.0)),
        w_pae_global=float(g.get("w_pae_global", 0.0)),
        w_pae_interface=float(g.get("w_pae_interface", 0.0)),
        # Structure self-consistency + diversity weights (all off by default).
        w_sctm_binder=float(g.get("w_sctm_binder", 0.0)),
        w_sctm_complex=float(g.get("w_sctm_complex", 0.0)),
        log_struct_diagnostic=bool(g.get("log_struct_diagnostic", False)),
        w_seq_diversity=float(g.get("w_seq_diversity", 0.0)),
        w_struct_diversity=float(g.get("w_struct_diversity", 0.0)),
        # Within-sequence anti-degeneracy: per-design saturating linguistic-complexity reward
        # (reward += w_seq_complexity * clip(LC/lc_full, 0, 1)); off by default.
        w_seq_complexity=float(g.get("w_seq_complexity", 0.0)),
        lc_full=float(g.get("lc_full", 0.7)),
        # Interface-distribution distance reward (Protenix-free shaping); off by default.
        w_aa_dist=float(g.get("w_aa_dist", 0.0)),
        w_3di_dist=float(g.get("w_3di_dist", 0.0)),
        # Track the distribution diagnostics even when both dist weights are 0 (the term
        # then contributes 0 to the reward); requires dist_reference.
        log_dist_diagnostic=bool(g.get("log_dist_diagnostic", False)),
        dist_metric=str(g.get("dist_metric", "tv")),
        dist_reference=g.get("dist_reference", None),
        # Interface-size guardrail (collapse penalty); defaults reproduce prior behaviour.
        dist_min_iface=int(g.get("dist_min_iface", 4)),
        # Interface vs whole-binder distribution blend (0 = interface-only, default).
        dist_binder_frac=float(g.get("dist_binder_frac", 0.0)),
        dist_iface_penalty=float(g.get("dist_iface_penalty", 0.0)),
        # Smooth clash + interface-contact geometry reward (Protenix-free); off by default.
        w_clash_contact=float(g.get("w_clash_contact", 0.0)),
        clash_d_clash=float(g.get("clash_d_clash", 2.2)),
        clash_soft=float(g.get("clash_soft", 0.5)),
        clash_scale=float(g.get("clash_scale", 50.0)),
        contact_d0=float(g.get("contact_d0", 8.0)),
        contact_soft=float(g.get("contact_soft", 1.0)),
        frac_lo=float(g.get("frac_lo", 0.05)),
        frac_peak=float(g.get("frac_peak", 0.16)),
        frac_hi=float(g.get("frac_hi", 0.4)),
        clash_seq_sep=int(g.get("clash_seq_sep", 2)),
        clash_include_cb=bool(g.get("clash_include_cb", True)),
        # Backbone chain-break realism reward (Protenix-free; C–N peptide-bond geometry). A [0,1]
        # regularizer (mean_r·gate) weighted by w_chainbreak; off by default. Supplies the
        # per-residue energy that the per-token chain-break advantage decomposes.
        w_chainbreak=float(g.get("w_chainbreak", 0.0)),
        chainbreak_gate=str(g.get("chainbreak_gate", "count")),
        chainbreak_gate_k=float(g.get("chainbreak_gate_k", 2.0)),
        chainbreak_ideal=float(g.get("chainbreak_ideal", 1.33)),
        chainbreak_tol=float(g.get("chainbreak_tol", 0.10)),
        chainbreak_cap=float(g.get("chainbreak_cap", 2.00)),
        chainbreak_sigma=float(g.get("chainbreak_sigma", 0.50)),
        chainbreak_break_hard=float(g.get("chainbreak_break_hard", 2.0)),
        chainbreak_break_d0=float(g.get("chainbreak_break_d0", 2.0)),
        chainbreak_break_soft=float(g.get("chainbreak_break_soft", 0.10)),
        # Full-atom LigandMPNN-repack rewards (ONE shared pool): SC 3DZD shape-complementarity,
        # all-atom side-chain clash, and ProteinMPNN AAR. All off by default; any weight > 0
        # builds the pool client and requires a shape queue. All run on the CPU worker pool —
        # scale throughput by adding CPU workers (no GPU contention).
        w_shape=float(g.get("w_shape", 0.0)),
        w_sc_clash=float(g.get("w_sc_clash", 0.0)),
        sc_clash_density=bool(g.get("sc_clash_density", False)),
        # Track (compute + log) all-atom SC clash without putting it in the reward (diagnostic).
        log_sc_clash_diagnostic=bool(g.get("log_sc_clash_diagnostic", False)),
        w_aar=float(g.get("w_aar", 0.0)),
        # Per-token (per-residue) backbone-clash advantage routed to the structure track.
        per_token_clash=bool(g.get("per_token_clash", False)),
        w_pt_clash=float(g.get("w_pt_clash", 1.0)),
        # Per-token (per-residue) backbone chain-break advantage routed to the structure track
        # (the exact analog of per_token_clash). Requires w_chainbreak > 0 (its energy source).
        per_token_chainbreak=bool(g.get("per_token_chainbreak", False)),
        w_pt_chainbreak=float(g.get("w_pt_chainbreak", 1.0)),
        # Per-token (per-residue) all-atom interface-potential advantage (e_lj/dsasa/n_hb on the
        # LigandMPNN pack) routed to the structure track. Reuses the R_SC pack via the "pot" want.
        per_token_pot=bool(g.get("per_token_pot", False)),
        w_pt_lj=float(g.get("w_pt_lj", 1.0)),
        w_pt_dsasa=float(g.get("w_pt_dsasa", 1.0)),
        w_pt_hb=float(g.get("w_pt_hb", 1.0)),
        pot_with_sasa=bool(g.get("pot_with_sasa", True)),
        pt_clash_tracks=tuple(g.get("pt_clash_tracks", ("structure_tokens",))),
        # CHORD SFT distillation (dense per-token LigandMPNN sequence supervision blended into
        # the GRPO loss). sft_mu>0 activates it; it rides the "aar" repack path (return_seq),
        # so it needs a shape queue but does NOT require w_aar>0. Off by default.
        sft_mu=float(g.get("sft_mu", 0.0)),
        sft_mu_schedule=g.get("sft_mu_schedule", None),
        sft_use_phi=bool(g.get("sft_use_phi", True)),
        sft_scope=str(g.get("sft_scope", "interface")),
        sft_label=str(g.get("sft_label", "hard")),
        sft_temperature=float(g.get("sft_temperature", 1.0)),
        sft_masked_only=bool(g.get("sft_masked_only", True)),
        sft_reward_gate=g.get("sft_reward_gate", None),
        # Structure CHORD SFT distillation (structural dual of the sequence CHORD term: Protenix
        # folds the policy sequence -> X* -> derive 3Di tau* + LG structure tokens s*, then distill
        # the structure/tri tracks). struct_sft_mu>0 activates it and REQUIRES the Protenix worker
        # pool (reward_client). Off by default.
        struct_sft_mu=float(g.get("struct_sft_mu", 0.0)),
        struct_sft_mu_schedule=g.get("struct_sft_mu_schedule", None),
        struct_sft_w_struct=float(g.get("struct_sft_w_struct", 1.0)),
        struct_sft_w_tri=float(g.get("struct_sft_w_tri", 1.0)),
        struct_sft_use_phi=bool(g.get("struct_sft_use_phi", True)),
        struct_sft_masked_only=bool(g.get("struct_sft_masked_only", True)),
        struct_sft_reward_gate=g.get("struct_sft_reward_gate", None),
        shape_queue_dir=_shape_queue,
        shape_timeout_s=float(shape_cfg.get("timeout_s", 1800.0)) if shape_cfg is not None else 1800.0,
        shape_poll_s=float(shape_cfg.get("poll_s", 2.0)) if shape_cfg is not None else 2.0,
        shape_cache=bool(shape_cfg.get("cache", True)) if shape_cfg is not None else True,
        shape_n_shards=int(shape_cfg.get("n_shards", 1)) if shape_cfg is not None else 1,
        # Per-group binder-length sampling (both set => L ~ U[min, max], constant per group).
        binder_length_min=None if _blen_min is None else int(_blen_min),
        binder_length_max=None if _blen_max is None else int(_blen_max),
        # Multi-target gradient accumulation. Default 10 (training-stabilization baseline);
        # set accum_targets=1 + shuffle_targets=False for the legacy single-target loop.
        accum_targets=int(g.get("accum_targets", 10)),
        shuffle_targets=bool(g.get("shuffle_targets", False)),
        tracks=tuple(g.get("tracks", ("sequence_tokens", "structure_tokens", "tri_tokens"))),
        capture_old_lp_inline=bool(g.get("capture_old_lp_inline", True)),
        grad_clip=float(g.get("grad_clip", 1.0)),
        grad_checkpoint=bool(g.get("grad_checkpoint", False)),
        rollout_kwargs=_build_rollout_kwargs(cfg.generation, device),
        seed=int(g.get("seed", 0)),
        log_every=int(g.get("log_every", 1)),
        ckpt_dir=g.get("ckpt_dir", None),
        ckpt_every=int(g.get("ckpt_every", 50)),
    )

    trainer = LeFlurGRPOTrainer(
        model=model,
        reward_client=reward_client,
        targets=_build_targets(cfg),
        config=grpo_config,
        device=device,
        gen_cfg=cfg.generation,
        wandb_run=_maybe_wandb(cfg),
    )
    trainer.train()
    logger.info("✓ GRPO run complete")


if __name__ == "__main__":
    rl_train()
