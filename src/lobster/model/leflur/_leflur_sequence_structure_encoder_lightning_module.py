import logging
from collections.abc import Sequence
from typing import Literal
from collections.abc import Callable

import torch
import torch.nn as nn
import transformers
from lightning import LightningModule
from torch import Tensor
from tqdm import tqdm

from lobster.constants import Modality, ModalityType

from ._leflur_sequence_structure_encoder import AuxiliaryTask, LeFlurSequenceStructureEncoderModule

# latent generator code:
from lobster.model.latent_generator.cmdline import LatentEncoderDecoder
from lobster.model.latent_generator.cmdline import methods as latent_generator_methods
from lobster.model.latent_generator.utils import apply_random_se3_batched

# Re-route the LeFlur-paired LG codecs (which by default live behind s3:// or
# /cv/... paths) to their HuggingFace mirror so external users can sample
# without internal credentials. Idempotent / safe to call from multiple
# LeFlur Lightning modules.
from .checkpoints import install_paired_lg_codec_overrides

install_paired_lg_codec_overrides()

# bionemo interpolant code — imported after install_paired_lg_codec_overrides()
# so the LG codec path patches are applied before any LG-consuming code runs.
from bionemo.moco.distributions.prior import DiscreteUniformPrior, DiscreteMaskedPrior  # noqa: E402
from bionemo.moco.distributions.time import UniformTimeDistribution  # noqa: E402
from bionemo.moco.interpolants import DiscreteFlowMatcher  # noqa: E402
from bionemo.moco.schedules.inference_time_schedules import (  # noqa: E402
    LinearInferenceSchedule,
    LogInferenceSchedule,
)


logger = logging.getLogger(__name__)


class LeFlurSequenceStructureEncoderLightningModule(LightningModule):
    """PyTorch Lightning module for protein-only LeFlur training and inference.

    Wraps :class:`LeFlurSequenceStructureEncoderModule` with the discrete
    flow-matching interpolant (via ``bionemo.moco``), the paired Latent
    Generator codec for structure tokenization, and the training /
    optimization loop. Three of the published canonical checkpoints
    (``leflur-base``, ``leflur-ted``, plus all of the Tier-2 research
    variants) are loaded into this module via
    :meth:`load_from_checkpoint` with a path resolved by
    :func:`lobster.model.leflur.resolve_checkpoint`.

    The companion :class:`LeFlurProteinLigandLightningModule` extends this
    surface to also handle protein-ligand complexes; see ``leflur-pl``.

    Inference modes (selected via :func:`lobster.cmdline.generate.generate`):

    * **Unconditional** — sample novel sequences + structures.
    * **Forward folding** — sequence → structure.
    * **Inverse folding** — structure → sequence.
    * **Inpainting** — fill missing residues conditioned on a partial complex.

    See ``docs/leflur/`` for a user-level walkthrough, and
    :mod:`lobster.cmdline.generate` for the dispatch wiring.
    """

    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        seed: int = 0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_warmup_steps: int = 20_000,
        num_training_steps: int = 100_000,
        weight_decay: float = 0.0,
        scheduler: str = "constant",
        scheduler_kwargs: dict | None = None,
        encoder_kwargs: dict | None = None,
        ckpt_path: str | None = None,
        # LatentGenerator params
        decode_tokens_during_training: bool = True,
        latent_generator_model_name: str = "LG full attention",
        # generation params
        prior_distribution_seq: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        prior_distribution_struc: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        time_distribution_seq: Callable[..., UniformTimeDistribution] = UniformTimeDistribution,
        time_distribution_struc: Callable[..., UniformTimeDistribution] = UniformTimeDistribution,
        interpolant: Callable[..., DiscreteFlowMatcher] = DiscreteFlowMatcher,
        inference_schedule: Callable[..., LinearInferenceSchedule] = LinearInferenceSchedule,
        use_masked_prior: bool = True,
        inverse_folding: bool = False,
        # Per-residue epitope conditioning. The training loop drops the
        # conditioning Bernoulli-style at this rate per example (CFG-style
        # classifier-free guidance training). Default 0.0 preserves the
        # pre-existing unconditional behaviour bit-for-bit; set >0 only
        # for finetunes on data that emits `epitope_tensor`
        # (e.g. Pinder dimers through `BinderTargetTransform`).
        cond_percentage: float = 0.0,
        # Per-residue TEMPLATE structure-token conditioning. Each step, with prob
        # `template_percentage`, provide leak-free per-chain structure tokens (each
        # chain encoded IN ISOLATION) as an extra conditioning signal; within that,
        # uniformly pick {both chains, one chain, none}, and for each templated chain
        # mask its interface residues 50% of the time (so the template cannot leak
        # the interface). 0.0 (default) = disabled (no template layer -> legacy
        # checkpoints load unchanged).
        template_percentage: float = 0.0,
        # Auxiliary AF3-style DISTOGRAM loss. When > 0 (and the encoder is built with
        # encoder_kwargs.use_distogram_head=true), add a binned Cb-Cb distance
        # cross-entropy on top of the interpolant loss, predicted from the (noised)
        # hidden state. The FULL map (intra + inter chain) is supervised so the signal
        # fires on every example (monomers included), and inter-chain pairs are
        # up-weighted since docking is the sparse, high-value part of the map. 0.0
        # (default) = disabled -> fully backward compatible with existing checkpoints.
        distogram_loss_weight: float = 0.0,
        distogram_inter_chain_weight: float = 5.0,
        distogram_min_bin: float = 2.3125,
        distogram_max_bin: float = 21.6875,
        distogram_num_bins: int = 64,
        # Fresh-finetune-from-pretrained knobs. These are NOT used inside
        # __init__ — they're consumed by `cmdline/train.py` AFTER model
        # instantiation. Declared here purely so Hydra can pass them
        # through `model.pretrained_ckpt=...` without raising on an
        # unexpected kwarg, and so they get saved into the checkpoint's
        # `hyper_parameters` block for reproducibility.
        pretrained_ckpt: str | None = None,
        zero_init_conditioning_on_load: bool = False,
        zero_init_chain_embedding_on_load: bool = False,
        # Per-signal CFG dropout for the scalar (categorical) conditioning. During training each present
        # signal is set to NULL (bin 0) per-example with its dropout prob, so the model learns to run with
        # any subset -> each signal is independently CFG-scalable + optional at inference. float = same rate
        # for all; dict {signal: rate} for per-signal schedules.
        scalar_cond_dropout: float | dict = 0.2,
        # 3Di generative TRACK: a third discrete-flow track (Foldseek 3Di alphabet, 20 states + mask + pad).
        # Adds its own masked-prior DiscreteFlowMatcher + CE loss (weight `tri_loss_weight`). GT 3Di tokens
        # come from batch["3di_states"] (Structure3diTransform). off/0-weight => fully backward compatible.
        use_3di_track: bool = False,
        tri_loss_weight: float = 1.0,
        # Bidirectional hotspots: fold binder-side interface residues (batch["paratope_tensor"]) into the
        # SAME epitope conditioning channel, RARER + SPARSER than epitope so the model never depends on it.
        # Gated on epitope-on examples; when on, mark only a few (<= paratope_max_residues) random paratope
        # residues. off (default) => paratope never added (pure epitope behaviour).
        use_paratope_conditioning: bool = False,
        paratope_cond_percentage: float = 0.2,
        paratope_max_residues: int = 3,
    ):
        self.save_hyperparameters()
        self.cond_percentage = cond_percentage
        self.template_percentage = template_percentage
        # Normalize to a plain float or plain dict (hydra passes a DictConfig, which is NOT `isinstance
        # dict` -> would fall into the float() branch and crash).
        self._scalar_cond_dropout = (
            float(scalar_cond_dropout)
            if isinstance(scalar_cond_dropout, (int, float))
            else {str(k): float(v) for k, v in dict(scalar_cond_dropout).items()}
        )
        self.distogram_loss_weight = distogram_loss_weight
        self.distogram_inter_chain_weight = distogram_inter_chain_weight
        self.distogram_min_bin = distogram_min_bin
        self.distogram_max_bin = distogram_max_bin
        self.distogram_num_bins = distogram_num_bins
        # 3Di track + bidirectional (paratope) hotspot config.
        self.use_3di_track = use_3di_track
        self.tri_loss_weight = tri_loss_weight
        self.use_paratope_conditioning = use_paratope_conditioning
        self.paratope_cond_percentage = paratope_cond_percentage
        self.paratope_max_residues = paratope_max_residues
        # 3Di vocab: 20 Foldseek states (0..19) + mask (20) + pad (21).
        self.num_tri_classes = 22
        self.mask_index_tri = 20
        self.padding_index_tri = 21

        super().__init__()

        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.seed = seed
        self.auxiliary_tasks = auxiliary_tasks
        self.auxiliary_task_loss_fns = {
            "regression": nn.MSELoss(),
        }

        # LatentGenerator params
        self.decode_tokens_during_training = decode_tokens_during_training
        self.structure_latent_encoder_decoder = LatentEncoderDecoder()
        self.structure_latent_encoder_decoder.load_model(
            latent_generator_methods[latent_generator_model_name].model_config.checkpoint,
            latent_generator_methods[latent_generator_model_name].model_config.config_path,
            latent_generator_methods[latent_generator_model_name].model_config.config_name,
            overrides=latent_generator_methods[latent_generator_model_name].model_config.overrides,
        )
        self.quantizer = self.structure_latent_encoder_decoder.model.quantizer
        self.strucure_encoder = self.structure_latent_encoder_decoder.model.encoder
        self.decoder_factory = self.structure_latent_encoder_decoder.model.decoder_factory
        self.loss_factory = self.structure_latent_encoder_decoder.model.loss_factory

        # generation params
        self.inverse_folding = inverse_folding
        self.prior_distribution_seq = prior_distribution_seq
        self.prior_distribution_struc = prior_distribution_struc
        if use_masked_prior:
            self.prior_distribution_seq = DiscreteMaskedPrior(
                num_classes=self.vocab_size, mask_dim=self.mask_token_id, inclusive=True
            )
            self.prior_distribution_struc = DiscreteMaskedPrior(
                num_classes=self.quantizer.n_tokens + 2, mask_dim=self.quantizer.n_tokens, inclusive=True
            )
            prior_seq = self.prior_distribution_seq
            prior_struc = self.prior_distribution_struc

        else:
            prior_seq = self.prior_distribution_seq(num_classes=self.vocab_size)
            prior_struc = self.prior_distribution_struc(num_classes=self.quantizer.n_tokens + 2)
        self.time_distribution_seq = time_distribution_seq
        self.time_distribution_struc = time_distribution_struc
        self.interpolant = interpolant
        self.inference_schedule = inference_schedule
        time_distribution_seq = self.time_distribution_seq()
        time_distribution_struc = self.time_distribution_struc()

        device = next(self.parameters()).device
        interpolant_seq = self.interpolant(
            time_distribution=time_distribution_seq, prior_distribution=prior_seq, device=device
        )
        interpolant_struc = self.interpolant(
            time_distribution=time_distribution_struc, prior_distribution=prior_struc, device=device
        )
        inference_schedule = self.inference_schedule(nsteps=1000)
        self.interpolant_seq = interpolant_seq
        self.interpolant_struc = interpolant_struc
        # 3Di track interpolant (masked prior over 20 states + mask). Built only when the track is enabled.
        if use_3di_track:
            if use_masked_prior:
                prior_tri = DiscreteMaskedPrior(
                    num_classes=self.num_tri_classes, mask_dim=self.mask_index_tri, inclusive=True
                )
            else:
                prior_tri = self.prior_distribution_seq(num_classes=self.num_tri_classes)
            self.interpolant_tri = self.interpolant(
                time_distribution=self.time_distribution_seq(), prior_distribution=prior_tri, device=device
            )
        else:
            self.interpolant_tri = None
        self.inference_schedule = inference_schedule

        logger.info(f"Using prior distribution seq: {self.prior_distribution_seq}")
        logger.info(f"Using prior distribution struc: {self.prior_distribution_struc}")
        logger.info(f"Using time distribution seq: {self.time_distribution_seq}")
        logger.info(f"Using time distribution struc: {self.time_distribution_struc}")
        logger.info(f"Using interpolant: {self.interpolant}")
        logger.info(f"Using training inference schedule: {self.inference_schedule}")

        self.mask_index_struc_tokens = self.quantizer.n_tokens
        self.padding_index_struc_tokens = self.quantizer.n_tokens + 1
        self.num_struc_classes = self.quantizer.n_tokens + 2

        # Template-token conditioning: "no template" index = structure_token_vocab_size
        # (the extra embedding row); templated residues use clean tokens 0..n_tokens-1.
        self.no_template_idx = self.num_struc_classes
        self.encoder = LeFlurSequenceStructureEncoderModule(
            auxiliary_tasks=auxiliary_tasks,
            sequence_token_vocab_size=self.vocab_size,
            structure_token_vocab_size=self.num_struc_classes,
            sequence_token_pad_token_id=self.pad_token_id,
            structure_token_pad_token_id=self.padding_index_struc_tokens,
            model_ckpt=ckpt_path,
            use_template_conditioning=(template_percentage > 0),
            use_3di_track=use_3di_track,
            **encoder_kwargs or {},
        )

    def embed_sequences(
        self, sequences: Sequence[str] | str, modality: ModalityType | Modality = None, aggregate: bool = True
    ) -> Tensor:
        raise NotImplementedError("Embedding for sequence and structure encoder is not implemented")

    def embed_structures(
        self, structures: Sequence[str] | str, modality: ModalityType | Modality = None, aggregate: bool = True
    ) -> Tensor:
        raise NotImplementedError("Embedding for structure encoder is not implemented")

    def embed_sequences_and_structures(
        self,
        sequences: Sequence[str] | str,
        structures: Sequence[str] | str,
        modality: ModalityType | Modality = None,
        aggregate: bool = True,
    ) -> Tensor:
        raise NotImplementedError("Embedding for sequence and structure encoder is not implemented")

    def decode_structure(self, unmasked_x: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
        """Decode the model output."""
        decoder_name = "vit_decoder"
        decoded_x = {}
        struc_tokens = unmasked_x["structure_logits"][..., : self.quantizer.n_tokens]
        temp = 0.1
        struc_tokens_ = torch.softmax(struc_tokens / temp, dim=-1)
        decoded_x[decoder_name] = self.decoder_factory.decoders[decoder_name](struc_tokens_, mask)

        return decoded_x

    @torch.no_grad()
    def _build_pair_bias_inputs(
        self,
        structure_tokens: Tensor,
        mask: Tensor,
        residue_index: Tensor,
        chain_ids: Tensor | None,
        conditioning_tensor: Tensor | None,
    ) -> dict[str, Tensor]:
        """Build the raw per-pair features for pair-bias attention from the CURRENT (noised) structure
        tokens. DETACHED (no grad through the ViT decoder) — geometry is a fixed input; only the
        per-layer to_bias/pair_norm learn. Masking follows the template-track rule (valid = mask &
        token<n_tokens). Returns bin ids / relpos ids / chain-diff / hotspot / pair_valid, all (B,L,L)."""
        from lobster.model.latent_generator.utils._kinematics import get_Cb

        B, L = structure_tokens.shape
        device = structure_tokens.device
        n_tok = self.quantizer.n_tokens

        # Normalize aux tensors to (B, L): training passes (B,L); generation may pass 1-D (L,) or (1,L).
        def _to_bl(t):
            if t is None:
                return None
            if t.dim() == 1:
                t = t.unsqueeze(0)
            if t.shape[0] == 1 and B > 1:
                t = t.expand(B, -1)
            return t

        mask = _to_bl(mask)
        residue_index = _to_bl(residue_index)
        chain_ids = _to_bl(chain_ids)
        valid = mask.bool() & (structure_tokens < n_tok)  # 1=valid (template-track rule)
        # One-hot the current codebook tokens (zero the mask/pad rows), decode to coords with the valid
        # mask so masked positions are ignored (mirrors encode_structure(coords, mask_c, ...)).
        onehot = torch.nn.functional.one_hot(structure_tokens.clamp(0, n_tok - 1), n_tok).to(self.dtype)
        onehot = onehot * valid.unsqueeze(-1).to(self.dtype)
        coords = self.decoder_factory.decoders["vit_decoder"](onehot, valid.to(self.dtype))  # (B,L,3,3)
        cb = torch.nan_to_num(get_Cb(coords[:, :, :3, :]))  # (B,L,3)
        dist2 = ((cb[:, :, None, :] - cb[:, None, :, :]) ** 2).sum(-1)  # (B,L,L)
        boundaries = (
            torch.linspace(self.distogram_min_bin, self.distogram_max_bin, self.distogram_num_bins - 1, device=device)
            ** 2
        )
        bin_ids = torch.sum(dist2[..., None] > boundaries, dim=-1)  # (B,L,L) in [0, num_bins-1]
        pair_valid = valid[:, None, :] & valid[:, :, None]  # (B,L,L)
        # relative sequence separation, one-hot, clamped to ±64 -> 129 bins
        k = 64
        ridx = residue_index.long()
        sep = ridx[:, :, None] - ridx[:, None, :]  # (B,L,L)
        relpos_ids = (sep.clamp(-k, k) + k).long()
        # chain-diff (1 if different chain)
        if chain_ids is not None:
            chain_diff = (chain_ids[:, :, None] != chain_ids[:, None, :]).float()
        else:
            chain_diff = torch.zeros(B, L, L, device=device)
        # hotspot: 1 if either residue is an epitope/hotspot (from the conditioning channel)
        if conditioning_tensor is not None:
            ct = conditioning_tensor
            if ct.dim() == 3:
                ct = ct[..., 0]  # (B,L,1) -> (B,L)
            ct = _to_bl(ct)
            epi = ct > 0  # (B,L)
            hotspot = (epi[:, :, None] | epi[:, None, :]).float()
        else:
            hotspot = torch.zeros(B, L, L, device=device)
        return {
            "pair_bin_ids": bin_ids,
            "pair_relpos_ids": relpos_ids,
            "pair_chain_diff": chain_diff,
            "pair_hotspot": hotspot,
            "pair_valid": pair_valid,
        }

    def encode_structure(self, x_gt: Tensor, mask: Tensor, residue_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode the model input."""
        x_emb = self.strucure_encoder(x_gt, mask, residue_index=residue_index)
        x_quant, x_quant_emb, mask = self.quantizer.quantize(x_emb, mask=mask, batch_size=x_gt.shape[0])

        return x_quant, x_quant_emb, mask

    def apply_interpolant_loss(
        self,
        split: str,
        x_gt: dict[str, Tensor],
        unmasked_x: dict[str, Tensor],
        mask: Tensor,
        total_loss: Tensor,
        loss_dict: dict[str, Tensor],
        timesteps: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the interpolant loss to the model."""
        loss_seq = self.interpolant_seq.loss(
            unmasked_x["sequence_logits"], x_gt["sequence_tokens"], timesteps["sequence_tokens"]
        ).mean()
        loss_struc = self.interpolant_struc.loss(
            unmasked_x["structure_logits"], x_gt["structure_tokens"], timesteps["structure_tokens"]
        ).mean()
        loss = loss_seq + loss_struc
        total_loss += loss
        loss_dict[f"{split}_interpolant_seq"] = loss_seq
        loss_dict[f"{split}_interpolant_struc"] = loss_struc
        loss_dict[f"{split}_timesteps_seq"] = timesteps["sequence_tokens"].mean()
        loss_dict[f"{split}_timesteps_struc"] = timesteps["structure_tokens"].mean()
        # 3Di track loss (weighted). Present only when the track is enabled and GT/logits exist.
        if self.use_3di_track and "tri_logits" in unmasked_x and "tri_tokens" in x_gt:
            loss_tri = self.interpolant_tri.loss(
                unmasked_x["tri_logits"], x_gt["tri_tokens"], timesteps["tri_tokens"]
            ).mean()
            total_loss += self.tri_loss_weight * loss_tri
            loss_dict[f"{split}_interpolant_tri"] = loss_tri
        return total_loss, loss_dict

    def apply_distogram_loss(
        self,
        split: str,
        batch: dict[str, Tensor],
        unmasked_x: dict[str, Tensor],
        mask: Tensor,
        chain_ids: Tensor | None,
        total_loss: Tensor,
        loss_dict: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """AF3-style distogram auxiliary loss (binned Cb-Cb distance CE).

        Targets are built from the GT backbone coords (``batch["input"][0]``,
        atom order N/CA/C -> pseudo-Cb). The prediction comes from the encoder's
        distogram head, which reads the (noised) hidden state, so this pushes the
        representation to encode explicit geometry. The full L x L map is supervised
        (intra + inter chain); inter-chain pairs (different, non-pad chain ids) are
        up-weighted by ``distogram_inter_chain_weight`` because docking is the sparse
        signal we care about. Intra/inter mean CE are logged for monitoring.
        """
        from lobster.model.latent_generator.utils._kinematics import get_Cb

        logits = unmasked_x["distogram_logits"]  # (B, L, L, num_bins)
        coords = batch["input"][0]  # (B, L, >=3, 3), atom order N, CA, C
        pb = get_Cb(coords[:, :, :3, :])  # (B, L, 3) pseudo-Cb
        pb = torch.nan_to_num(pb, nan=0.0, posinf=0.0, neginf=0.0)

        no_bins = self.distogram_num_bins
        boundaries = (
            torch.linspace(self.distogram_min_bin, self.distogram_max_bin, no_bins - 1, device=logits.device) ** 2
        )
        dists2 = ((pb[:, :, None, :] - pb[:, None, :, :]) ** 2).sum(-1)  # (B, L, L)
        true_bins = torch.sum(dists2[..., None] > boundaries, dim=-1)  # (B, L, L) in [0, no_bins-1]

        logp = torch.log_softmax(logits.float(), dim=-1)
        ce = -logp.gather(-1, true_bins.unsqueeze(-1)).squeeze(-1)  # (B, L, L)

        m = mask.bool()
        pair_mask = (m[:, :, None] & m[:, None, :]).float()  # (B, L, L)
        weight = torch.ones_like(pair_mask)
        inter = None
        if chain_ids is not None:
            ci = chain_ids.long()
            inter = (ci[:, :, None] != ci[:, None, :]) & (ci[:, :, None] > 0) & (ci[:, None, :] > 0)
            weight = torch.where(inter, weight * self.distogram_inter_chain_weight, weight)
        wm = pair_mask * weight
        loss = (ce * wm).sum() / wm.sum().clamp_min(1.0)

        total_loss = total_loss + self.distogram_loss_weight * loss
        loss_dict[f"{split}_distogram"] = loss
        with torch.no_grad():
            if inter is not None:
                inter_mask = pair_mask * inter.float()
                intra_mask = pair_mask * (~inter).float()
                if inter_mask.sum() > 0:
                    loss_dict[f"{split}_distogram_inter"] = (ce * inter_mask).sum() / inter_mask.sum().clamp_min(1.0)
            else:
                intra_mask = pair_mask
            loss_dict[f"{split}_distogram_intra"] = (ce * intra_mask).sum() / intra_mask.sum().clamp_min(1.0)
        return total_loss, loss_dict

    def apply_structure_decoder_loss(
        self,
        split: str,
        decoder_gt: dict[str, Tensor],
        decoded_x: dict[str, Tensor],
        mask: Tensor,
        total_loss: Tensor,
        loss_dict: dict[str, Tensor],
        just_loss: bool = False,
        keep_batch_dim: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the structure decoder loss to the model."""
        decoder_name = "vit_decoder"
        loss2apply = self.decoder_factory.get_loss(decoder_name)

        for loss2apply_ in loss2apply:
            loss = self.loss_factory(
                loss2apply_, decoder_gt, decoded_x[decoder_name], mask, keep_batch_dim=keep_batch_dim
            )
            if just_loss:
                return loss
            # apply loss weighting from weight_dict in loss_factory; setting to 0 b/c will need different way to set weights for different losses
            # total_loss += self.loss_factory.weight_dict[loss2apply_] * loss
            total_loss += 0 * loss
            loss_dict[f"{split}_{loss2apply_}"] = loss

        return total_loss, loss_dict

    # Names of the per-design scalar conditioning signals; batch key = f"scalar_cond__{name}".
    _SCALAR_COND_NAMES = ("rg_ratio", "iface_frac", "iface_helix", "iface_sheet", "iface_coil", "frac_arom")

    def build_scalar_cond_bins(self, batch: dict[str, Tensor], training: bool) -> dict | None:
        """Pull per-design scalar conditioning bins from the batch and apply per-signal CFG dropout.

        Returns {name: (B,L) Long bin ids} for signals present in the batch, or None when scalar
        conditioning is disabled / no signals present. During training each signal is independently set to
        NULL (bin 0) per-example with its dropout prob, so the model learns to run with any subset.
        """
        if not getattr(self.encoder, "use_scalar_conditioning", False):
            return None
        dev = batch["sequence"].device
        B = batch["sequence"].shape[0]
        dd = self._scalar_cond_dropout
        # Two-level schedule: with prob `_all`, drop EVERY signal for an example (clean unconditioned mode
        # -> preserves the strong base generation); otherwise drop each signal independently at its own rate.
        p_all = dd.get("_all", 0.0) if isinstance(dd, dict) else 0.0
        all_drop = (
            (torch.rand(B, device=dev) < p_all)
            if (training and p_all > 0)
            else torch.zeros(B, dtype=torch.bool, device=dev)
        )
        bins: dict[str, Tensor] = {}
        for name in self._SCALAR_COND_NAMES:
            key = f"scalar_cond__{name}"
            if key not in batch or batch[key] is None:
                continue
            t = batch[key].to(dev).long()
            if training:
                p = dd.get(name, 0.2) if isinstance(dd, dict) else float(dd)
                drop = all_drop | (torch.rand(B, device=dev) < p) if p > 0 else all_drop
                if drop.any():
                    t = t.clone()
                    t[drop] = 0  # NULL the whole example for this signal
            bins[name] = t
        return bins or None

    def get_gen_gt_and_conditioning_tensor(
        self, batch: dict[str, Tensor], cond_percentage: float | None = None
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, bool]:
        """Get the conditioning tensor for the model."""
        if cond_percentage is None:
            cond_percentage = 0.0
        if cond_percentage > 0.0 and "epitope_tensor" in batch:
            conditioning = True
        else:
            conditioning = False

        B = batch["sequence"].shape[0]
        device = batch["sequence"].device
        mask = batch["mask"]
        residue_index = batch["indices"]
        seq_gt = batch["sequence"]
        x_quant, x_quant_emb, mask = self.encode_structure(*batch["input"])
        x_1_struc_tokens_argmax = torch.argmax(x_quant, dim=-1)
        x_1_struc_tokens_argmax[~mask.bool()] = self.padding_index_struc_tokens

        x_gt = {"structure_tokens": x_1_struc_tokens_argmax, "sequence_tokens": seq_gt}

        # 3Di GT tokens (Foldseek states 0..19). Pad positions -> padding_index_tri (masked out of the loss
        # via the interpolant's pad handling / our mask). Only built when the track is enabled and the
        # transform emitted 3di_states.
        if self.use_3di_track and "3di_states" in batch and batch["3di_states"] is not None:
            tri_gt = batch["3di_states"].to(device).long().clone()
            L3 = tri_gt.shape[1]
            m3 = mask.bool()[:, :L3]
            tri_gt = tri_gt.clamp(0, self.mask_index_tri - 1)  # guard any stray values into [0,19]
            tri_gt[~m3] = self.padding_index_tri
            x_gt["tri_tokens"] = tri_gt

        # Generate a random mask for each batch index
        conditioning_mask = torch.rand(B, device=device) < cond_percentage
        epitope_cond = torch.full((B, x_quant.shape[1], 1), 0, device=device, requires_grad=True, dtype=torch.float)

        # Apply conditioning logic for indices where conditioning_mask is True
        for i in range(B):
            if conditioning_mask[i]:
                if "epitope_tensor" in batch:
                    epitope_cond[i] = batch["epitope_tensor"][i, :, None]
                    # Mask 0 - 100% of non-zero indices in epitope tensor
                    epitope_mask = torch.rand_like(epitope_cond[i].float()) < torch.rand(1).item()
                    epitope_cond[i] = epitope_cond[i] * epitope_mask
                # Bidirectional hotspots: fold a FEW binder-side (paratope) interface residues into the SAME
                # channel, RARER + SPARSER than epitope, so the model can locate the binder side of the
                # interface at inference (for 3Di interaction-type guidance) without depending on it.
                if (
                    self.use_paratope_conditioning
                    and "paratope_tensor" in batch
                    and batch["paratope_tensor"] is not None
                    and torch.rand(1).item() < self.paratope_cond_percentage
                ):
                    para_idx = batch["paratope_tensor"][i].nonzero(as_tuple=True)[0]
                    if para_idx.numel() > 0:
                        k = int(torch.randint(1, self.paratope_max_residues + 1, (1,)).item())
                        sel = para_idx[torch.randperm(para_idx.numel(), device=para_idx.device)[:k]]
                        epitope_cond[i, sel, 0] = 1.0

        conditioning_tensor = epitope_cond
        return x_gt, conditioning_tensor, mask, residue_index, conditioning

    def build_template_tokens(self, batch: dict[str, Tensor], chain_ids: Tensor) -> Tensor:
        """Leak-free per-chain TEMPLATE structure tokens with random-crop augmentation.

        For each example, with prob ``template_percentage`` provide templates; within
        that, uniformly pick {both, one(random), none} chains. Each templated chain is
        encoded IN ISOLATION (partner masked out of the structure encoder, so no
        interface/partner leakage) from an INDEPENDENT random SE(3) copy of the coords
        (so the relative pose is NOT leaked -> model must still learn to dock).

        Data augmentation: every time a chain is templated it is randomly CROPPED --- a
        fraction ~U(0, 0.9) of its residues is dropped BEFORE encoding (mask-in-place),
        so the template covers only the kept residues and their tokens are computed from
        the partial structure. This matches how a cropped template behaves at inference
        (see scripts/_exp_template_crop.py) and teaches the model to use partial
        templates. Returns (B, L) long; no_template_idx where not templated / cropped.
        ``chain_ids`` = remapped per-residue chain ids (0=none/pad, 1/2=chains).
        """
        import random as _random

        coords, enc_mask, ridx = batch["input"]
        B, L = chain_ids.shape
        device = chain_ids.device
        template = torch.full((B, L), self.no_template_idx, dtype=torch.long, device=device)

        chain_vals = [int(c) for c in torch.unique(chain_ids).tolist() if c > 0]

        # 1) Per-example schedule + per-(example, chain) random-CROP keep-mask, decided
        #    BEFORE encoding so the crop is applied at the encoder input (template tokens
        #    are then computed from the partial structure). keep_by_chain[c][i] marks the
        #    residues of chain c in example i that are templated AND survive the crop.
        keep_by_chain: dict[int, Tensor] = {c: torch.zeros(B, L, dtype=torch.bool, device=device) for c in chain_vals}
        for i in range(B):
            ci = [c for c in chain_vals if bool((chain_ids[i] == c).any())]
            # Only template multi-chain (complex) examples — never single-chain/monomer.
            if len(ci) < 2 or torch.rand(1).item() >= self.template_percentage:
                continue
            mode = _random.choice(["both", "one", "none"])
            if mode == "none":
                continue
            chosen = ci if mode == "both" else [_random.choice(ci)]
            for c in chosen:
                cmask = (chain_ids[i] == c) & enc_mask[i].bool()
                idxs = cmask.nonzero(as_tuple=True)[0]
                n = int(idxs.numel())
                if n == 0:
                    continue
                # crop augmentation: drop a random 0-90% of this chain's residues
                crop_frac = torch.rand(1).item() * 0.9
                n_keep = max(1, int(round(n * (1.0 - crop_frac))))
                perm = torch.randperm(n, device=device)[:n_keep]
                keep_by_chain[c][i, idxs[perm]] = True

        # 2) Batched per-chain ISOLATED encoding on the CROPPED keep-mask, each chain
        #    from an independent random SE(3) frame. Assign the resulting tokens only to
        #    the kept residues; cropped / unkept residues stay at no_template_idx.
        for c in chain_vals:
            keep_c = keep_by_chain[c]
            if not keep_c.any():
                continue
            coords_c = apply_random_se3_batched(coords.clone())
            xq_c, _, _ = self.encode_structure(coords_c, keep_c.float(), ridx)
            tok_c = xq_c.argmax(dim=-1)  # (B, L)
            template[keep_c] = tok_c[keep_c]

        return template

    def get_timesteps(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Get the timesteps for the model."""
        timesteps_seq = self.interpolant_seq.sample_time(batch["sequence"].shape[0])
        timesteps_struc = self.interpolant_struc.sample_time(batch["sequence"].shape[0])
        timesteps = {"sequence_tokens": timesteps_seq, "structure_tokens": timesteps_struc}
        if self.use_3di_track:
            timesteps["tri_tokens"] = self.interpolant_tri.sample_time(batch["sequence"].shape[0])
        return timesteps

    def interpolate_tokens(self, input_tokens: dict[str, Tensor], timesteps: dict[str, Tensor]) -> dict[str, Tensor]:
        """Interpolate the tokens for the model."""

        x_1_seq = input_tokens["sequence_tokens"]
        x_1_struc = input_tokens["structure_tokens"]
        x_0_seq = self.interpolant_seq.sample_prior(x_1_seq.shape)
        x_0_struc = self.interpolant_struc.sample_prior(x_1_struc.shape)
        timesteps_seq = timesteps["sequence_tokens"]
        timesteps_struc = timesteps["structure_tokens"]
        if self.inverse_folding:
            timesteps_struc = torch.ones_like(timesteps_struc)
        x_t_seq = self.interpolant_seq.interpolate(x_1_seq, timesteps_seq, x_0_seq)
        x_t_struc = self.interpolant_struc.interpolate(x_1_struc, timesteps_struc, x_0_struc)

        x_t = {"sequence_tokens": x_t_seq, "structure_tokens": x_t_struc}
        # 3Di track: noise from the masked prior like the sequence track.
        if self.use_3di_track and "tri_tokens" in input_tokens:
            x_1_tri = input_tokens["tri_tokens"]
            x_0_tri = self.interpolant_tri.sample_prior(x_1_tri.shape)
            x_t["tri_tokens"] = self.interpolant_tri.interpolate(x_1_tri, timesteps["tri_tokens"], x_0_tri)
        return x_t

    def forward(
        self,
        x_t: dict[str, Tensor],
        mask: Tensor,
        residue_index: Tensor,
        conditioning_tensor: Tensor,
        timesteps: dict[str, Tensor] | None = None,
        chain_ids: Tensor | None = None,
        template_structure_tokens: Tensor | None = None,
        scalar_cond_bins: dict | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass of the model, for inference."""
        if timesteps is not None:
            timesteps = timesteps.copy()
            # expand to be same length as x_t e.g from B to B,L
            B, L = x_t["sequence_tokens"].shape
            timesteps["sequence_tokens"] = timesteps["sequence_tokens"][:, None].expand(-1, L)[:, :, None]
            timesteps["structure_tokens"] = timesteps["structure_tokens"][:, None].expand(-1, L)[:, :, None]

        # Pair-bias attention: build the geometry-grounded per-pair features from the current structure
        # tokens (detached decode) and hand them to the encoder, which assembles + projects them.
        pair_kwargs: dict[str, Tensor] = {}
        if getattr(self.encoder, "use_pair_bias_attention", False):
            pair_kwargs = self._build_pair_bias_inputs(
                x_t["structure_tokens"], mask, residue_index, chain_ids, conditioning_tensor
            )

        unmasked_x = self.encoder(
            sequence_input_ids=x_t["sequence_tokens"],
            structure_input_ids=x_t["structure_tokens"],
            position_ids=residue_index,
            attention_mask=mask,
            conditioning_tensor=conditioning_tensor,
            chain_ids=chain_ids,
            timesteps=timesteps,
            template_structure_tokens=template_structure_tokens,
            scalar_cond_bins=scalar_cond_bins,
            tri_input_ids=x_t.get("tri_tokens"),
            return_auxiliary_tasks=False,
            **pair_kwargs,
        )

        return unmasked_x

    @torch.no_grad()
    def score_pll(
        self,
        sequence_tokens: Tensor,
        structure_tokens: Tensor,
        mask: Tensor,
        residue_index: Tensor | None = None,
        *,
        K: int = 32,
        eps: float = 0.02,
        seed: int = 0,
        variants: tuple[str, ...] | None = None,
    ) -> dict[str, Tensor]:
        """Compute pseudo-NLL ranking scores for a batch of (seq, struc) pairs.

        Centralized PLL scoring for the protein-only LeFlur checkpoint. See
        :mod:`lobster.model.leflur._pll_scoring` for the full algorithmic
        description.

        Parameters
        ----------
        sequence_tokens : Tensor ``(B, L)``
        structure_tokens : Tensor ``(B, L)``
        mask : Tensor ``(B, L)`` of validity (1) vs padding (0)
        residue_index : Tensor ``(B, L)`` or ``None``
            When ``None``, ``torch.arange(L)`` is broadcast over the batch.
        K : Monte-Carlo draws per modality (default 32).
        eps : Stratified-t endpoint margin (default 0.02).
        seed : Base seed; sample ``b`` uses ``seed + b`` so the per-sample
            scoring is deterministic and order-independent.
        variants : Subset of
            :data:`lobster.model.leflur._pll_scoring.PROTEIN_VARIANTS`.
            ``None`` returns the full default tuple.

        Returns
        -------
        dict mapping each requested variant name (and the diagnostic
        ``seq_score_arllh`` / ``struc_score_arllh`` weighted estimators when
        ``seq`` / ``struc`` are requested) to a ``(B,)`` float tensor.
        Lower NLL = higher likelihood; rank by ``argmin``.
        """
        from ._pll_scoring import PROTEIN_VARIANTS, score_protein_pll

        was_training = self.training
        self.eval()
        try:
            return score_protein_pll(
                self,
                sequence_tokens=sequence_tokens,
                structure_tokens=structure_tokens,
                mask=mask,
                residue_index=residue_index,
                K=K,
                eps=eps,
                seed=seed,
                variants=variants if variants is not None else PROTEIN_VARIANTS,
            )
        finally:
            if was_training:
                self.train()

    def step(
        self, batch: dict[str, Tensor], batch_idx: int, split: Literal["train", "val"] = "train"
    ) -> dict[str, Tensor]:
        """Single training/val/test step of the model."""
        # set device
        device = batch["sequence"].device
        self.interpolant_seq.device = device
        self.interpolant_struc.device = device
        if self.use_3di_track:
            self.interpolant_tri.device = device

        # set losses
        total_loss = 0.0
        loss_dict = {}

        # prep the input
        with torch.no_grad():
            x_gt, conditioning_tensor, mask, residue_index, conditioning = self.get_gen_gt_and_conditioning_tensor(
                batch, cond_percentage=self.cond_percentage
            )

        timesteps = self.get_timesteps(batch)
        x_t = self.interpolate_tokens(x_gt, timesteps)

        # Per-residue chain-id signal (Commit C). Pulled directly from the
        # batch -- the upstream `RemapChainIdsForEmbedding` transform writes
        # this key only when the transform yaml requests it AND the data
        # actually has chain info. When absent the encoder's chain_embedding
        # short-circuits to a zero contribution (padding_idx=0 + None guard).
        chain_ids = batch.get("chain_ids_for_embedding")

        # Per-residue leak-free TEMPLATE structure tokens (built only when enabled and
        # chain info is present; monomers have no chain_ids -> no template).
        template_structure_tokens = None
        if self.template_percentage > 0 and chain_ids is not None:
            with torch.no_grad():
                template_structure_tokens = self.build_template_tokens(batch, chain_ids)

        # Per-design scalar (categorical) conditioning bins with per-signal CFG dropout (train only).
        scalar_cond_bins = self.build_scalar_cond_bins(batch, training=self.training)

        # gen tokens
        unmasked_x = self.forward(
            x_t,
            mask,
            residue_index,
            conditioning_tensor,
            timesteps=timesteps,
            chain_ids=chain_ids,
            template_structure_tokens=template_structure_tokens,
            scalar_cond_bins=scalar_cond_bins,
        )

        total_loss, loss_dict = self.apply_interpolant_loss(
            split, x_gt, unmasked_x, mask, total_loss, loss_dict, timesteps
        )

        # Auxiliary distogram loss (no-op unless distogram_loss_weight>0 AND the
        # encoder was built with a distogram head, i.e. "distogram_logits" is present).
        if self.distogram_loss_weight > 0 and "distogram_logits" in unmasked_x:
            total_loss, loss_dict = self.apply_distogram_loss(
                split, batch, unmasked_x, mask, chain_ids, total_loss, loss_dict
            )

        # Decode the tokens if needed
        if self.decode_tokens_during_training:
            # decode the tokens
            decoder_gt = batch
            decoded_x = self.decode_structure(unmasked_x, mask)
            total_loss, loss_dict = self.apply_structure_decoder_loss(
                split, decoder_gt, decoded_x, mask, total_loss, loss_dict
            )
        else:
            decoder_gt = None
            decoded_x = None

        # With multiple val DataLoaders Lightning's _Metadata equality
        # includes ``dataloader_idx``, so logging the same key from
        # different dataloaders with ``add_dataloader_idx=False`` raises
        # ``MisconfigurationException: ... twice in validation_step with
        # different arguments``. Letting Lightning auto-suffix to
        # ``val_loss/dataloader_idx_N`` avoids that; the cleaner semantic
        # names (``val_loss_cameo``, ``val_loss_pinder_val``) are emitted
        # explicitly in ``validation_step`` from a fresh log call.
        self.log_dict({f"{split}_loss": total_loss, **loss_dict}, batch_size=x_gt["sequence_tokens"].shape[0])

        return {
            "loss": total_loss,
            "x_gt": x_gt,
            "unmasked_x": unmasked_x,
            "decoder_gt": decoder_gt,
            "decoded_x": decoded_x,
            "conditioning": conditioning,
            f"{split}_timesteps_seq": timesteps["sequence_tokens"],
            f"{split}_timesteps_struc": timesteps["structure_tokens"],
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        out = self.step(batch, batch_idx, "val")
        # Per-source breakdown: when the datamodule splits val into
        # multiple DataLoaders (e.g. CAMEO monomers + Pinder dimers),
        # emit a clean per-source key so ModelCheckpoint can monitor a
        # single dataset (``val_loss_cameo``, ``val_loss_pinder_val``).
        # ``add_dataloader_idx=False`` is safe here because each
        # dataloader logs a DIFFERENT key (unique source name), so
        # Lightning's per-key metadata check doesn't collide.
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        names = getattr(dm, "_val_source_names", None) if dm is not None else None
        if names and 0 <= dataloader_idx < len(names):
            self.log(
                f"val_loss_{names[dataloader_idx]}",
                out["loss"],
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch["sequence"].shape[0],
            )
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        scheduler = transformers.get_scheduler(
            self.scheduler,
            optimizer,
            num_training_steps=self.scheduler_kwargs.pop("num_training_steps", None),
            num_warmup_steps=self.scheduler_kwargs.pop("num_warmup_steps", None),
            scheduler_specific_kwargs=self.scheduler_kwargs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def generate_sample(
        self,
        length,
        num_samples,
        inference_schedule_seq: Callable[..., LinearInferenceSchedule] = LogInferenceSchedule,
        inference_schedule_struc: Callable[..., LinearInferenceSchedule] = LinearInferenceSchedule,
        nsteps: int = 200,
        stochasticity_seq: int = 20,
        stochasticity_struc: int = 20,
        temperature_seq: float = 0.5,
        temperature_struc: float = 1.0,
        inverse_folding: bool = False,
        forward_folding: bool = False,
        inpainting: bool = False,
        input_structure_coords: Tensor = None,
        input_sequence_tokens: Tensor = None,
        input_mask: Tensor = None,
        input_indices: Tensor = None,
        inpainting_mask_sequence: Tensor = None,
        inpainting_mask_structure: Tensor = None,
        asynchronous_sampling: bool = False,
        sequence_anchor_tokens: Tensor = None,
        sequence_anchor_mask: Tensor = None,
        sequence_logit_bias: Tensor | None = None,
        sequence_logit_bias_steps: int = 10,
        # Per-SEQUENCE diversity (frequency) penalty: at each step subtract
        # `sequence_diversity_penalty * (running per-design frequency of each AA)` from the sequence
        # logits, so whichever residue is currently over-represented in a given design is suppressed.
        # Adaptive + residue-agnostic -> fights single-AA collapse WITHIN each sequence (unlike the
        # static composition bias which only matches the aggregate marginal). 0 = off.
        sequence_diversity_penalty: float = 0.0,
        # Per-residue chain-id signal for the encoder's chain_embedding
        # layer (Commit C). Shape (num_samples, length), long. None or
        # all-zero disables the chain channel (padding_idx=0 -> embedding
        # contributes 0). Required for any meaningful dimer-aware inference
        # on a model trained with `max_num_chains > 0`.
        chain_ids: Tensor | None = None,
        # Per-residue epitope/conditioning input for the encoder's
        # `conditioning_embedding` layer. Shape (num_samples, length, 1),
        # float in {0, 1}. None falls back to the all-zero default (no
        # conditioning, matches unconditional leflur-base behaviour).
        conditioning_tensor_override: Tensor | None = None,
        # Per-residue TEMPLATE structure tokens (leak-free per-chain), (num_samples, length)
        # long, no_template_idx where absent. Passed to the encoder's template embedding.
        template_structure_tokens: Tensor | None = None,
        # Per-design scalar conditioning bins {name -> (num_samples, length) Long, 0=NULL}. Applied at every
        # denoising step (both the conditional and the epitope-CFG uncond forward). None = off.
        scalar_cond_bins: dict | None = None,
        # 3Di track sampling. By DEFAULT the 3Di track uses the SAME schedule as the LG structure track
        # (reuses struc per-step t/dt) and, when temperature_tri/stochasticity_tri are None, the SAME
        # temperature/stochasticity as the struc track — so callbacks need no 3Di-specific config (to be
        # ablated post-training). `input_3di_tokens` (num_samples,length) Long supplies 3Di:
        #   - with `inpainting_mask_3di` (1=generate, 0=hold): partial 3Di spec (interaction-type guidance),
        #   - with `pin_3di_clean=True`: hold the WHOLE supplied 3Di clean (condition-on-3Di mode).
        # `tri_logit_bias` (like sequence_logit_bias) nudges the 3Di logits for the first N steps.
        temperature_tri: float | None = None,
        stochasticity_tri: int | None = None,
        # Decode the 3Di track FASTER than the LG structure track: warp 3Di time t_tri = min(t_struc*accel,
        # 0.9995) and hold it clean once resolved, so the interpretable 3Di fold commits early and scaffolds
        # the LG denoising. accel=1.0 (default) = lockstep with struc (bit-exact baseline). accel=2 -> 3Di
        # clean by ~50% of steps; accel=4 -> ~25%. Tests the "3Di leads structure prediction" hypothesis.
        tri_time_accel: float = 1.0,
        # INDEPENDENT 3Di inference schedule (its OWN shape: Linear/Log/Power), fully decoupled from the LG
        # (struc) schedule. None = 3Di reuses the struc schedule (then tri_time_accel warps it). When set,
        # the 3Di track is denoised on this schedule's own t/dt (a genuinely different schedule, not a
        # time-warp of struc). Callable(nsteps=...) like inference_schedule_struc.
        inference_schedule_tri: Callable | None = None,
        input_3di_tokens: Tensor | None = None,
        inpainting_mask_3di: Tensor | None = None,
        pin_3di_clean: bool = False,
        tri_logit_bias: Tensor | None = None,
        # 3Di diversity penalty (analog of sequence_diversity_penalty): subtract
        # `tri_diversity_penalty * (running per-design 3Di-token frequency)` from the 3Di logits each step,
        # suppressing over-used structural states. Motivated by 3Di degeneracy predicting binder failure
        # (>0.5 max-token designs pass ~2% vs ~9% for diverse). Applied over generated (inpainting) positions.
        tri_diversity_penalty: float = 0.0,
        # Inpainting only: if True, exclude the to-be-generated (inpainting_mask==1)
        # region from structure encoding, so the placeholder geometry there does not
        # perturb the fixed (target) structure tokens via the encoder's attention. The
        # generated region is noised from the prior regardless; this just keeps the
        # target tokens clean. The full mask is still used for the sampling loop.
        encode_target_only: bool = False,
        # Classifier-free guidance on the epitope/conditioning channel. >1.0 runs a second
        # (unconditional: conditioning_tensor=0) forward per step and extrapolates the logits:
        # guided = uncond + cfg_weight*(cond - uncond). 1.0 = off. Needs a cond-dropout-trained
        # ckpt (cond_percentage>0) for the uncond branch to be in-distribution.
        cfg_weight: float = 1.0,
        # Optional per-timestep hook for analysis (e.g. distogram-vs-decoded-structure consistency).
        # Called as step_callback(step_idx, t_struc, unmasked_x, mask) right after each forward pass,
        # before the interpolant step. `unmasked_x` holds structure_logits / sequence_logits and, when
        # the encoder has the distogram head on, distogram_logits. No-op when None (default) — zero
        # effect on generation.
        step_callback: Callable[..., None] | None = None,
        # GRPO rollout capture (no-op unless a dict is supplied). When given, it is populated with two keys:
        #   "static" -> the conditioning shared across steps (mask, residue_index, conditioning_tensor,
        #               chain_ids, template_structure_tokens, scalar_cond_bins, cfg_weight, the bias/diversity
        #               config, and the effective generated-position + diversity masks per track), and
        #   "steps"  -> one record per denoising step, each holding the pre-step state (`xt` dict), the
        #               per-track sampled next state (`x_next` = the raw multinomial output, before any
        #               inpainting re-pin), the padded time/step size, and temperature/stochasticity.
        # Together these are exactly what `logprob_over_trajectory` needs to reconstruct the biased per-step
        # logits (one forward + optional CFG + bias/diversity) and recompute differentiable transition
        # log-probs. Generation is bit-identical when `trajectory_store is None`.
        trajectory_store: dict | None = None,
    ):
        """Generate with model, with option to return full unmasking trajectory and likelihood."""
        device = next(self.parameters()).device
        self.interpolant_seq.device = device
        self.interpolant_struc.device = device
        if self.use_3di_track:
            self.interpolant_tri.device = device
        xt_seq = self.interpolant_seq.sample_prior((num_samples, length))
        xt_struc = self.interpolant_struc.sample_prior((num_samples, length))
        xt = {"sequence_tokens": xt_seq, "structure_tokens": xt_struc}
        # 3Di track init (masked prior). Reuses the struc inference schedule for per-step t/dt.
        xt_tri = xt_tri_input = None
        if self.use_3di_track:
            xt_tri = self.interpolant_tri.sample_prior((num_samples, length))
            if input_3di_tokens is not None:
                xt_tri_input = input_3di_tokens.to(device).long()
                if pin_3di_clean:
                    # Hold the entire supplied 3Di clean (condition-on-3Di).
                    xt_tri = xt_tri_input.clone()
                elif inpainting_mask_3di is not None:
                    # Keep supplied 3Di where mask=0, noise from prior where mask=1 (partial spec).
                    xt_tri = torch.where(inpainting_mask_3di.bool(), xt_tri, xt_tri_input)
                else:
                    xt_tri = xt_tri_input.clone()
            xt["tri_tokens"] = xt_tri
        if inference_schedule_seq is None:
            inference_schedule_seq = self.inference_schedule
        else:
            inference_schedule_seq = inference_schedule_seq(nsteps=nsteps)
        if inference_schedule_struc is None:
            inference_schedule_struc = self.inference_schedule
        else:
            inference_schedule_struc = inference_schedule_struc(nsteps=nsteps)
        logger.info(f"Using generation inference schedule seq: {inference_schedule_seq}")
        logger.info(f"Using generation inference schedule struc: {inference_schedule_struc}")
        ts_seq = inference_schedule_seq.generate_schedule(device=device)
        ts_struc = inference_schedule_struc.generate_schedule(device=device)
        dts_seq = inference_schedule_seq.discretize(device=device)
        dts_struc = inference_schedule_struc.discretize(device=device)
        # INDEPENDENT 3Di schedule. If inference_schedule_tri given, the 3Di track runs on its OWN schedule
        # shape (Linear/Log/Power) — genuinely decoupled from struc, not a time-warp. Else reuse struc's
        # schedule (and tri_time_accel warps it). Same nsteps so the sampling loop stays aligned.
        _sched_tri_obj = None
        if self.use_3di_track and inference_schedule_tri is not None:
            _sched_tri_obj = inference_schedule_tri(nsteps=nsteps)
            ts_tri = _sched_tri_obj.generate_schedule(device=device)
            dts_tri = _sched_tri_obj.discretize(device=device)
            logger.info(f"Using INDEPENDENT 3Di inference schedule: {_sched_tri_obj}")
        else:
            ts_tri, dts_tri = ts_struc, dts_struc
        mask = torch.ones((num_samples, length), device=device)
        residue_index = torch.arange(length, device=device)
        conditioning_tensor = torch.zeros((num_samples, length, 1), device=device)
        # Caller-provided overrides (Commit C wiring + epitope eval). The
        # all-zero defaults above match the historical unconditional
        # generation path; overrides activate the trained chain_embedding
        # and conditioning_embedding signals at inference time.
        if conditioning_tensor_override is not None:
            conditioning_tensor = conditioning_tensor_override.to(device)
        if chain_ids is not None:
            chain_ids = chain_ids.to(device)
        if inverse_folding:
            if input_structure_coords is not None:
                x_quant, x_quant_emb, mask = self.encode_structure(
                    x_gt=input_structure_coords, mask=input_mask, residue_index=input_indices
                )
                xt_struc = x_quant.argmax(dim=-1)
                xt["structure_tokens"] = xt_struc
                ts_struc = torch.full_like(ts_struc, 0.9950)
            else:
                raise ValueError("Structure path is required for inverse folding")
        elif forward_folding:
            if input_sequence_tokens is not None:
                xt_seq = input_sequence_tokens
                xt["sequence_tokens"] = xt_seq
                ts_seq = torch.full_like(ts_seq, 0.9950)
            else:
                raise ValueError("Sequence tokens are required for forward folding")
        elif inpainting:
            # For inpainting, we need both input structure and sequence, plus masks
            if input_structure_coords is None:
                raise ValueError("Structure coordinates are required for inpainting")
            if input_sequence_tokens is None:
                raise ValueError("Sequence tokens are required for inpainting")

            # Encode the input structure. When encode_target_only, mask out the
            # to-be-generated region so its placeholder geometry is not seen by the
            # structure encoder (keeps the target/fixed tokens clean); restore the full
            # mask afterwards for the sampling loop.
            struct_encode_mask = input_mask
            if encode_target_only and inpainting_mask_structure is not None:
                struct_encode_mask = input_mask.clone()
                struct_encode_mask[inpainting_mask_structure.bool()] = 0
            x_quant, x_quant_emb, mask = self.encode_structure(
                x_gt=input_structure_coords, mask=struct_encode_mask, residue_index=input_indices
            )
            if encode_target_only:
                mask = input_mask
            xt_struc_input = x_quant.argmax(dim=-1)
            xt_seq_input = input_sequence_tokens

            # Initialize with input values
            xt_seq = xt_seq_input.clone()
            xt_struc = xt_struc_input.clone()

            # Apply masks: keep input where mask=0, randomize where mask=1
            if inpainting_mask_sequence is not None:
                # Generate random tokens for masked positions
                random_seq = self.interpolant_seq.sample_prior((num_samples, length))
                # Keep original where mask=0, use random where mask=1
                xt_seq = torch.where(inpainting_mask_sequence.bool(), random_seq, xt_seq_input)

            if inpainting_mask_structure is not None:
                # Generate random tokens for masked positions
                random_struc = self.interpolant_struc.sample_prior((num_samples, length))
                # Keep original where mask=0, use random where mask=1
                xt_struc = torch.where(inpainting_mask_structure.bool(), random_struc, xt_struc_input)

            xt = {"sequence_tokens": xt_seq, "structure_tokens": xt_struc}

        # GRPO rollout capture — effective per-track "generated position" masks (actions live only here).
        # Computed once; bit-identical no-op when trajectory_store is None.
        _gm_seq = _gm_struc = _gm_tri = None
        if trajectory_store is not None:
            # Inline behaviour-policy log-prob capture (GRPO): compute each step's log-prob from the
            # exact biased logits the sampler drew from, so `old_lp` is faithful by construction and
            # needs no post-hoc recompute. Same differentiable kernel used by `logprob_over_trajectory`.
            from lobster.rl_training._dfm_logprob import dfm_step_logprob as _dfm_step_logprob

            _ones = torch.ones((num_samples, length), dtype=torch.bool, device=device)
            # Action masks (where a transition is actually sampled): inpaint AND anchor (seq only).
            _gm_seq = inpainting_mask_sequence.bool() if inpainting_mask_sequence is not None else _ones.clone()
            if sequence_anchor_mask is not None:
                _gm_seq = _gm_seq & sequence_anchor_mask.bool()
            _gm_struc = inpainting_mask_structure.bool() if inpainting_mask_structure is not None else _ones.clone()
            if self.use_3di_track:
                if pin_3di_clean:
                    _gm_tri = torch.zeros((num_samples, length), dtype=torch.bool, device=device)
                elif inpainting_mask_3di is not None:
                    _gm_tri = inpainting_mask_3di.bool()
                else:
                    _gm_tri = _ones.clone()
            # Diversity-penalty frequency masks: the RAW inpainting mask (NO anchor AND) — this is exactly
            # what the sampler uses to measure per-design AA/3Di frequency, and it differs from the action
            # gen_mask above (which ANDs the anchor). Recompute must mirror the sampler's biased logits.
            _div_seq = inpainting_mask_sequence.bool() if inpainting_mask_sequence is not None else _ones.clone()
            _div_tri = None
            if self.use_3di_track:
                _div_tri = inpainting_mask_3di.bool() if inpainting_mask_3di is not None else _ones.clone()
            # Static conditioning shared across every step — everything `logprob_over_trajectory` needs to
            # re-run `forward` (+ CFG) and reproduce the biased logits deterministically. Kept on-device and
            # detached (small: (B,L)-ish); the per-step xt/x_next are the memory-heavy parts and go to CPU.
            trajectory_store["static"] = {
                "mask": mask.detach(),
                "residue_index": residue_index.detach(),
                "conditioning_tensor": conditioning_tensor.detach(),
                "chain_ids": None if chain_ids is None else chain_ids.detach(),
                "template_structure_tokens": (
                    None if template_structure_tokens is None else template_structure_tokens.detach()
                ),
                "scalar_cond_bins": scalar_cond_bins,
                "cfg_weight": float(cfg_weight),
                "use_3di_track": bool(self.use_3di_track),
                # Bias / diversity config (replayed per step to reconstruct the exact biased logits).
                "sequence_logit_bias": None if sequence_logit_bias is None else sequence_logit_bias.detach(),
                "sequence_logit_bias_steps": int(sequence_logit_bias_steps),
                "sequence_diversity_penalty": float(sequence_diversity_penalty or 0.0),
                "tri_logit_bias": None if tri_logit_bias is None else tri_logit_bias.detach(),
                "tri_diversity_penalty": float(tri_diversity_penalty or 0.0),
                # Masks: actions vs diversity-frequency (deliberately different — see above).
                "gen_mask_seq": _gm_seq.detach(),
                "gen_mask_struc": _gm_struc.detach(),
                "gen_mask_tri": None if _gm_tri is None else _gm_tri.detach(),
                "div_mask_seq": _div_seq.detach(),
                "div_mask_tri": None if _div_tri is None else _div_tri.detach(),
                # Track-level mask indices for the absorbing prior (for the recompute kernel).
                "mask_index_seq": int(self.mask_token_id),
                "mask_index_struc": int(self.mask_index_struc_tokens),
                "mask_index_tri": int(self.mask_index_tri),
            }
            trajectory_store["steps"] = []

        for step_idx, (dt_seq, dt_struc, dt_tri, t_seq, t_struc, t_tri_raw) in enumerate(
            tqdm(zip(dts_seq, dts_struc, dts_tri, ts_seq, ts_struc, ts_tri), desc="Generating samples")
        ):
            t_seq = inference_schedule_seq.pad_time(num_samples, t_seq, device)
            t_struc = inference_schedule_struc.pad_time(num_samples, t_struc, device)
            # 3Di time on its OWN schedule (independent shape) or struc's if none; pad with whichever built it.
            _pad = (_sched_tri_obj or inference_schedule_struc).pad_time
            t_tri_sched = _pad(num_samples, t_tri_raw, device)
            timesteps = {"sequence_tokens": t_seq, "structure_tokens": t_struc}

            # GRPO capture: open one record per step, holding the state that feeds `forward` this step.
            _traj_rec = None
            if trajectory_store is not None:
                _traj_rec = {
                    "step_idx": step_idx,
                    "xt": {k: v.detach().to("cpu", torch.int32) for k, v in xt.items()},
                    "t_seq": t_seq.detach().cpu(),
                    "t_struc": t_struc.detach().cpu(),
                    "tracks": {},
                }
                trajectory_store["steps"].append(_traj_rec)

            unmasked_x = self.forward(
                xt,
                mask,
                residue_index,
                conditioning_tensor,
                timesteps=timesteps,
                chain_ids=chain_ids,
                template_structure_tokens=template_structure_tokens,
                scalar_cond_bins=scalar_cond_bins,
            )
            # Classifier-free guidance: extrapolate logits away from the unconditional (no-epitope)
            # prediction to amplify the hotspot signal (which steers the interface, esp. L36).
            if cfg_weight != 1.0:
                uncond_x = self.forward(
                    xt,
                    mask,
                    residue_index,
                    torch.zeros_like(conditioning_tensor),
                    timesteps=timesteps,
                    chain_ids=chain_ids,
                    template_structure_tokens=template_structure_tokens,
                    scalar_cond_bins=scalar_cond_bins,
                )
                for _k in ("sequence_logits", "structure_logits"):
                    unmasked_x[_k] = uncond_x[_k] + cfg_weight * (unmasked_x[_k] - uncond_x[_k])
            # Analysis hook (no-op unless a callback is supplied): lets callers capture per-timestep
            # distogram logits + decode the current structure without altering the sampling path.
            if step_callback is not None:
                step_callback(step_idx, t_struc, unmasked_x, mask)
            unmasked_sequence_tokens = unmasked_x["sequence_logits"]
            if sequence_logit_bias is not None and step_idx < sequence_logit_bias_steps:
                unmasked_sequence_tokens = unmasked_sequence_tokens + sequence_logit_bias
            if sequence_diversity_penalty and sequence_diversity_penalty > 0 and step_idx < sequence_logit_bias_steps:
                # Adaptive per-design frequency penalty: subtract alpha * (current AA frequency) so the
                # residue a given design is over-using gets suppressed. Frequency measured from the
                # current x0-hat prediction over the positions being generated (inpainting mask).
                pred = unmasked_sequence_tokens.argmax(dim=-1)  # (B, L)
                B, L, V = unmasked_sequence_tokens.shape
                genmask = (
                    inpainting_mask_sequence.bool()
                    if inpainting_mask_sequence is not None
                    else torch.ones(B, L, dtype=torch.bool, device=unmasked_sequence_tokens.device)
                )
                onehot = torch.nn.functional.one_hot(pred, V).float() * genmask.unsqueeze(-1).float()
                freq = onehot.sum(dim=1) / genmask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, V)
                unmasked_sequence_tokens = unmasked_sequence_tokens - sequence_diversity_penalty * freq.unsqueeze(1)
            xt_seq_new = self.interpolant_seq.step(
                unmasked_sequence_tokens,
                t_seq,
                xt_seq,
                dt_seq,
                stochasticity=stochasticity_seq,
                temperature=temperature_seq,
            )
            if _traj_rec is not None:
                _traj_rec["tracks"]["sequence_tokens"] = {
                    "xt": xt_seq.detach().to("cpu", torch.int32),
                    "x_next": xt_seq_new.detach().to("cpu", torch.int32),
                    "t": t_seq.detach().cpu(),
                    "dt": dt_seq.detach().cpu() if torch.is_tensor(dt_seq) else float(dt_seq),
                    "temperature": float(temperature_seq),
                    "stochasticity": float(stochasticity_seq),
                    "gen_mask": _gm_seq.detach().cpu(),
                    # Behaviour-policy log-prob from the SAME biased logits the sampler drew from.
                    "logprob": _dfm_step_logprob(
                        unmasked_sequence_tokens,
                        t_seq,
                        dt_seq,
                        xt_seq,
                        xt_seq_new,
                        _gm_seq,
                        mask_index=self.mask_token_id,
                        temperature=temperature_seq,
                        stochasticity=stochasticity_seq,
                    )
                    .detach()
                    .cpu(),
                }
            # if asynchronous_sampling: # onestep structure prediction
            #    unmasked_x = self.forward({"sequence_tokens": unmasked_sequence_tokens.argmax(dim=-1), "structure_tokens": self.interpolant_struc.sample_prior((num_samples, length))}, mask, residue_index, conditioning_tensor, timesteps={"sequence_tokens": torch.full_like(t_seq, 0.9950), "structure_tokens": torch.full_like(t_seq, 0.0000)})

            unmasked_structure_tokens = unmasked_x["structure_logits"]
            xt_struc_new = self.interpolant_struc.step(
                unmasked_structure_tokens,
                t_struc,
                xt_struc,
                dt_struc,
                stochasticity=stochasticity_struc,
                temperature=temperature_struc,
            )
            if _traj_rec is not None:
                _traj_rec["tracks"]["structure_tokens"] = {
                    "xt": xt_struc.detach().to("cpu", torch.int32),
                    "x_next": xt_struc_new.detach().to("cpu", torch.int32),
                    "t": t_struc.detach().cpu(),
                    "dt": dt_struc.detach().cpu() if torch.is_tensor(dt_struc) else float(dt_struc),
                    "temperature": float(temperature_struc),
                    "stochasticity": float(stochasticity_struc),
                    "gen_mask": _gm_struc.detach().cpu(),
                    "logprob": _dfm_step_logprob(
                        unmasked_structure_tokens,
                        t_struc,
                        dt_struc,
                        xt_struc,
                        xt_struc_new,
                        _gm_struc,
                        mask_index=self.mask_index_struc_tokens,
                        temperature=temperature_struc,
                        stochasticity=stochasticity_struc,
                    )
                    .detach()
                    .cpu(),
                }

            # For inpainting, keep unmasked positions fixed
            if inpainting:
                if inpainting_mask_sequence is not None:
                    # Only update masked positions, keep unmasked positions from input
                    xt_seq = torch.where(inpainting_mask_sequence.bool(), xt_seq_new, xt_seq_input)
                else:
                    xt_seq = xt_seq_new

                if inpainting_mask_structure is not None:
                    # Only update masked positions, keep unmasked positions from input
                    xt_struc = torch.where(inpainting_mask_structure.bool(), xt_struc_new, xt_struc_input)
                else:
                    xt_struc = xt_struc_new
            else:
                xt_seq = xt_seq_new
                xt_struc = xt_struc_new

            # Apply sequence anchors: keep anchored positions fixed (mask=0), update free positions (mask=1)
            if sequence_anchor_tokens is not None and sequence_anchor_mask is not None:
                xt_seq = torch.where(sequence_anchor_mask.bool(), xt_seq, sequence_anchor_tokens)

            # 3Di track step. By default shares the struc schedule (accel=1). With tri_time_accel>1 the 3Di
            # time is warped ahead so it decodes faster and commits early (then holds clean). Optional logit
            # bias, then hold-fixed for pinned/inpainted 3Di.
            if self.use_3di_track:
                unmasked_tri = unmasked_x["tri_logits"]
                if tri_logit_bias is not None and step_idx < sequence_logit_bias_steps:
                    unmasked_tri = unmasked_tri + tri_logit_bias
                if tri_diversity_penalty and tri_diversity_penalty > 0:
                    # Adaptive per-design 3Di-frequency penalty: subtract alpha * (current 3Di-token
                    # frequency over generated positions) so an over-used structural state is suppressed.
                    _pred = unmasked_tri.argmax(dim=-1)  # (B, L)
                    _B, _L, _V = unmasked_tri.shape
                    _gm = (
                        inpainting_mask_3di.bool()
                        if inpainting_mask_3di is not None
                        else torch.ones(_B, _L, dtype=torch.bool, device=unmasked_tri.device)
                    )
                    _oh = torch.nn.functional.one_hot(_pred, _V).float() * _gm.unsqueeze(-1).float()
                    _freq = _oh.sum(dim=1) / _gm.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, V)
                    unmasked_tri = unmasked_tri - tri_diversity_penalty * _freq.unsqueeze(1)
                # Base 3Di time = its INDEPENDENT schedule (t_tri_sched/dt_tri) — or struc's if no
                # inference_schedule_tri (then ts_tri==ts_struc). tri_time_accel optionally warps it further.
                _base_t, _base_dt = t_tri_sched, dt_tri
                _accel = max(1.0, float(tri_time_accel))
                if _accel == 1.0:
                    _t_tri, _dt_tri, _do_step = _base_t, _base_dt, True  # pure schedule (baseline if struc-shared)
                else:
                    _t_tri = torch.clamp(_base_t * _accel, max=0.9995)
                    _dt_tri = _base_dt * _accel
                    _do_step = float(_t_tri.reshape(-1)[0]) < 0.999  # once 3Di is clean, hold it fixed
                if _do_step:
                    _stoch_tri = stochasticity_tri if stochasticity_tri is not None else stochasticity_struc
                    _temp_tri = temperature_tri if temperature_tri is not None else temperature_struc
                    xt_tri_new = self.interpolant_tri.step(
                        unmasked_tri,
                        _t_tri,
                        xt_tri,
                        _dt_tri,
                        # Default to the LG (struc) track's knobs unless explicitly overridden.
                        stochasticity=_stoch_tri,
                        temperature=_temp_tri,
                    )
                    if _traj_rec is not None:
                        _traj_rec["tracks"]["tri_tokens"] = {
                            "xt": xt_tri.detach().to("cpu", torch.int32),
                            "x_next": xt_tri_new.detach().to("cpu", torch.int32),
                            "t": _t_tri.detach().cpu(),
                            "dt": _dt_tri.detach().cpu() if torch.is_tensor(_dt_tri) else float(_dt_tri),
                            "temperature": float(_temp_tri),
                            "stochasticity": float(_stoch_tri),
                            "gen_mask": _gm_tri.detach().cpu(),
                            "logprob": _dfm_step_logprob(
                                unmasked_tri,
                                _t_tri,
                                _dt_tri,
                                xt_tri,
                                xt_tri_new,
                                _gm_tri,
                                mask_index=self.mask_index_tri,
                                temperature=_temp_tri,
                                stochasticity=_stoch_tri,
                            )
                            .detach()
                            .cpu(),
                        }
                else:
                    xt_tri_new = xt_tri  # 3Di already resolved -> keep clean while LG finishes
                if pin_3di_clean and xt_tri_input is not None:
                    xt_tri = xt_tri_input
                elif inpainting_mask_3di is not None and xt_tri_input is not None:
                    xt_tri = torch.where(inpainting_mask_3di.bool(), xt_tri_new, xt_tri_input)
                else:
                    xt_tri = xt_tri_new

            xt = {"sequence_tokens": xt_seq, "structure_tokens": xt_struc}
            if self.use_3di_track:
                xt["tri_tokens"] = xt_tri

        # GRPO: record the sampled endpoint (the discrete state actually produced, post inpaint-repin and
        # anchor pinning). The reward must score THIS sequence — not the final logits argmax — so that the
        # rewarded object is exactly the action whose log-prob the policy ratio evaluates.
        if trajectory_store is not None:
            trajectory_store["final_xt"] = {k: v.detach().to("cpu", torch.int32) for k, v in xt.items()}

        return unmasked_x

    # ------------------------------------------------------------------ #
    # GRPO trajectory recompute (differentiable log-prob / KL)            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def rollout_with_logprobs(self, **generate_kwargs) -> dict:
        """Sample a group of designs while capturing everything needed to recompute log-probs.

        Thin wrapper around :meth:`generate_sample`: it allocates a fresh
        ``trajectory_store`` and forwards all keyword arguments unchanged, so the
        returned rollout is drawn from the exact production sampler (same CFG, bias,
        diversity, schedules). The returned dict carries ``"static"`` (shared
        conditioning), ``"steps"`` (per-step captures), and ``"final_xt"`` (the
        sampled endpoint to be scored by the reward). Feed it directly to
        :meth:`logprob_over_trajectory` / :meth:`kl_over_trajectory`.

        Parameters
        ----------
        **generate_kwargs
            Any keyword arguments accepted by :meth:`generate_sample` (e.g.
            ``num_samples``, the inpainting masks/tokens, ``cfg_weight``, the
            bias/diversity/stochasticity knobs). ``trajectory_store`` must not be
            passed — it is supplied internally.

        Returns
        -------
        dict
            The populated trajectory store.

        Raises
        ------
        ValueError
            If ``trajectory_store`` is passed in ``generate_kwargs``.
        """
        if "trajectory_store" in generate_kwargs:
            raise ValueError("rollout_with_logprobs supplies trajectory_store internally; do not pass it.")
        trajectory: dict = {}
        self.generate_sample(**generate_kwargs, trajectory_store=trajectory)
        return trajectory

    def decode_endpoint_aa(self, trajectory: dict) -> Tensor:
        """Standard-AA token ids ``(B, L)`` for a rollout's sampled endpoint sequence.

        Decodes ``trajectory["final_xt"]["sequence_tokens"]`` (the sampled sequence,
        not the final logits argmax) from the 33-token vocabulary to the standard
        23-token representation (0-19 = amino acids, 20 = X, 21/22 = gaps). Callers
        slice the binder positions and map to letters before scoring.

        Parameters
        ----------
        trajectory : dict
            A rollout store carrying ``"final_xt"`` (from :meth:`rollout_with_logprobs`).

        Returns
        -------
        Tensor
            Standard-AA token ids of shape ``(B, L)``.
        """
        from lobster.model.latent_generator.utils.residue_constants import (
            convert_lobster_aa_tokenization_to_standard_aa,
        )

        seq_tokens = trajectory["final_xt"]["sequence_tokens"].long()  # (B, L) sampled ids
        # The converter requires a 33-wide last dim (it argmaxes), so one-hot the sampled tokens:
        # argmax of the one-hot recovers exactly the sampled token, decoding the SAMPLED sequence.
        onehot = torch.nn.functional.one_hot(seq_tokens, self.vocab_size).float()
        return convert_lobster_aa_tokenization_to_standard_aa(onehot)

    def _recompute_biased_step_logits(
        self,
        xt: dict[str, Tensor],
        t_seq: Tensor,
        t_struc: Tensor,
        step_idx: int,
        static: dict,
    ) -> dict[str, Tensor | None]:
        """Reproduce the exact per-track biased logits a ``generate_sample`` step used.

        Re-runs ``forward`` (plus the optional classifier-free-guidance second
        forward on seq+struc only) and re-applies the sequence / 3Di logit-bias and
        adaptive diversity penalties, mirroring the sampler bit-for-bit so the
        resulting logits — evaluated at the sampled transition — reproduce the exact
        action distribution used during generation. The returned logits are
        differentiable in the encoder parameters (no ``torch.no_grad`` here).

        Parameters
        ----------
        xt : dict[str, Tensor]
            The pre-step state fed to ``forward`` this step (``sequence_tokens``,
            ``structure_tokens``, and — from step 1 onward — ``tri_tokens``), on the
            model device and integer-typed.
        t_seq, t_struc : Tensor
            The padded ``(B,)`` sequence and structure times for this step (the only
            timesteps ``forward`` consumes).
        step_idx : int
            The denoising step index, used to gate the bias/diversity schedules
            exactly as the sampler does.
        static : dict
            The ``"static"`` conditioning dict populated by ``generate_sample``.

        Returns
        -------
        dict[str, Tensor | None]
            Biased logits keyed ``sequence_tokens`` / ``structure_tokens`` /
            ``tri_tokens`` (the last is ``None`` when the 3Di track is inactive).
        """
        timesteps = {"sequence_tokens": t_seq, "structure_tokens": t_struc}
        out = self.forward(
            xt,
            static["mask"],
            static["residue_index"],
            static["conditioning_tensor"],
            timesteps=timesteps,
            chain_ids=static["chain_ids"],
            template_structure_tokens=static["template_structure_tokens"],
            scalar_cond_bins=static["scalar_cond_bins"],
        )
        # Classifier-free guidance touches seq+struc only (NOT tri) — mirror the sampler.
        cfg_weight = static["cfg_weight"]
        if cfg_weight != 1.0:
            uncond = self.forward(
                xt,
                static["mask"],
                static["residue_index"],
                torch.zeros_like(static["conditioning_tensor"]),
                timesteps=timesteps,
                chain_ids=static["chain_ids"],
                template_structure_tokens=static["template_structure_tokens"],
                scalar_cond_bins=static["scalar_cond_bins"],
            )
            for _k in ("sequence_logits", "structure_logits"):
                out[_k] = uncond[_k] + cfg_weight * (out[_k] - uncond[_k])

        # Sequence bias then adaptive diversity penalty (both gated by step < bias_steps).
        seq_logits = out["sequence_logits"]
        bias_steps = static["sequence_logit_bias_steps"]
        if static["sequence_logit_bias"] is not None and step_idx < bias_steps:
            seq_logits = seq_logits + static["sequence_logit_bias"]
        if static["sequence_diversity_penalty"] > 0 and step_idx < bias_steps:
            pred = seq_logits.argmax(dim=-1)  # (B, L) — argmax on the biased logits, as in the sampler
            _, _, V = seq_logits.shape
            gm = static["div_mask_seq"].to(seq_logits.device)
            onehot = torch.nn.functional.one_hot(pred, V).to(seq_logits.dtype) * gm.unsqueeze(-1).to(seq_logits.dtype)
            freq = onehot.sum(dim=1) / gm.to(seq_logits.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, V)
            seq_logits = seq_logits - static["sequence_diversity_penalty"] * freq.unsqueeze(1)

        struc_logits = out["structure_logits"]

        tri_logits = None
        if static["use_3di_track"] and out.get("tri_logits") is not None:
            tri_logits = out["tri_logits"]
            if static["tri_logit_bias"] is not None and step_idx < bias_steps:
                tri_logits = tri_logits + static["tri_logit_bias"]
            # 3Di diversity penalty has NO step gate (matches the sampler).
            if static["tri_diversity_penalty"] > 0:
                _pred = tri_logits.argmax(dim=-1)
                _, _, _V = tri_logits.shape
                _gm = static["div_mask_tri"].to(tri_logits.device)
                _oh = torch.nn.functional.one_hot(_pred, _V).to(tri_logits.dtype) * _gm.unsqueeze(-1).to(
                    tri_logits.dtype
                )
                _freq = _oh.sum(dim=1) / _gm.to(tri_logits.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
                tri_logits = tri_logits - static["tri_diversity_penalty"] * _freq.unsqueeze(1)

        return {"sequence_tokens": seq_logits, "structure_tokens": struc_logits, "tri_tokens": tri_logits}

    def _iter_traj_steps(self, trajectory: dict, step_indices: Sequence[int] | None):
        """Yield ``(record, xt_on_device, t_seq, t_struc)`` for each requested step."""
        static = trajectory["static"]
        device = static["mask"].device
        steps = trajectory["steps"]
        if step_indices is not None:
            steps = [steps[i] for i in step_indices]
        for rec in steps:
            xt_dev = {k: v.to(device=device, dtype=torch.long) for k, v in rec["xt"].items()}
            yield rec, xt_dev, rec["t_seq"].to(device), rec["t_struc"].to(device)

    def logprob_over_trajectory(
        self,
        trajectory: dict,
        tracks: Sequence[str] = ("sequence_tokens", "structure_tokens", "tri_tokens"),
        step_indices: Sequence[int] | None = None,
    ) -> Tensor:
        """Differentiable summed transition log-prob of a captured rollout under this policy.

        For every requested step, reproduces the biased per-track logits
        (:meth:`_recompute_biased_step_logits`) and accumulates
        :func:`lobster.rl_training._dfm_logprob.dfm_step_logprob` over the requested
        tracks at that step's sampled transition. The returned ``(B,)`` tensor is
        differentiable in the encoder parameters — this is the GRPO policy log-prob.

        Parameters
        ----------
        trajectory : dict
            The ``{"static": ..., "steps": [...]}`` structure populated by
            ``generate_sample`` when a ``trajectory_store`` is supplied.
        tracks : Sequence[str], optional
            Which tracks to include in the ratio. Defaults to all three; pass e.g.
            ``("sequence_tokens",)`` for a seq-only ablation.
        step_indices : Sequence[int] | None, optional
            Optional subset of step positions (diffu-GRPO random step-subsampling).
            ``None`` uses every captured step.

        Returns
        -------
        Tensor
            Per-design summed log-prob of shape ``(B,)``.
        """
        from lobster.rl_training._dfm_logprob import dfm_step_logprob

        static = trajectory["static"]
        device = static["mask"].device
        mask_index = {
            "sequence_tokens": static["mask_index_seq"],
            "structure_tokens": static["mask_index_struc"],
            "tri_tokens": static["mask_index_tri"],
        }
        gen_mask = {
            "sequence_tokens": static["gen_mask_seq"],
            "structure_tokens": static["gen_mask_struc"],
            "tri_tokens": static["gen_mask_tri"],
        }
        total: Tensor | None = None
        for rec, xt_dev, t_seq, t_struc in self._iter_traj_steps(trajectory, step_indices):
            biased = self._recompute_biased_step_logits(xt_dev, t_seq, t_struc, rec["step_idx"], static)
            for track in tracks:
                tr = rec["tracks"].get(track)
                if tr is None or biased.get(track) is None or gen_mask.get(track) is None:
                    continue  # e.g. tri track held clean this step, or track not generated
                lp = dfm_step_logprob(
                    biased[track],
                    tr["t"].to(device),
                    tr["dt"].to(device) if torch.is_tensor(tr["dt"]) else tr["dt"],
                    tr["xt"].to(device=device, dtype=torch.long),
                    tr["x_next"].to(device=device, dtype=torch.long),
                    gen_mask[track].to(device),
                    mask_index=mask_index[track],
                    temperature=tr["temperature"],
                    stochasticity=tr["stochasticity"],
                )
                total = lp if total is None else total + lp
        if total is None:
            return torch.zeros(static["mask"].shape[0], device=device)
        return total

    def captured_logprob_per_step(
        self,
        trajectory: dict,
        tracks: Sequence[str] = ("sequence_tokens", "structure_tokens", "tri_tokens"),
    ) -> Tensor:
        """Per-step behaviour-policy log-prob captured inline during ``generate_sample``.

        Reads the log-probabilities the sampler actually drew from — stored per step
        per track when a ``trajectory_store`` was supplied — instead of recomputing
        them with a second forward pass (:meth:`logprob_over_trajectory` with
        ``step_indices=[i]``). This is the *faithful* behaviour-policy log-prob for
        GRPO: it uses the exact biased logits the sampler used, so no
        sampler/recompute mirror can silently drift, and it costs zero extra
        forwards. With inline ``old_lp``, the first inner PPO iteration's importance
        ratio should be ``~1`` — a live consistency check on the recompute path used
        for ``new_lp``.

        Parameters
        ----------
        trajectory : dict
            Rollout store from :meth:`rollout_with_logprobs`. Each step's per-track
            dict must carry a ``"logprob"`` entry (present for rollouts captured with
            inline log-prob support).
        tracks : Sequence[str], optional
            Tracks summed into each step's log-prob. Defaults to all three; pass
            ``("sequence_tokens",)`` for the seq-only ablation.

        Returns
        -------
        Tensor
            Behaviour-policy log-prob of shape ``(n_steps, B)`` on the model device,
            summed over the requested tracks (a track held clean at a step
            contributes zero there).

        Raises
        ------
        KeyError
            If a requested step/track lacks an inline ``"logprob"`` — i.e. the
            rollout predates inline capture; recompute ``old_lp`` instead.
        """
        static = trajectory["static"]
        device = static["mask"].device
        batch = static["mask"].shape[0]
        rows: list[Tensor] = []
        for rec in trajectory["steps"]:
            acc = torch.zeros(batch, device=device)
            for track in tracks:
                tr = rec["tracks"].get(track)
                if tr is None:
                    continue  # track held clean this step (e.g. 3Di already resolved)
                if "logprob" not in tr:
                    raise KeyError(
                        f"step {rec['step_idx']} track {track!r} has no inline 'logprob'; "
                        "rollout predates inline capture — recompute old_lp instead"
                    )
                acc = acc + tr["logprob"].to(device)
            rows.append(acc)
        return torch.stack(rows)  # (n_steps, B)

    def kl_over_trajectory(
        self,
        trajectory: dict,
        ref_module: "LeFlurSequenceStructureEncoderLightningModule",
        tracks: Sequence[str] = ("sequence_tokens", "structure_tokens", "tri_tokens"),
        step_indices: Sequence[int] | None = None,
    ) -> Tensor:
        """Differentiable summed categorical KL ``KL(pi_self || pi_ref)`` over a rollout.

        For each requested step and track, materializes both policies' step
        distributions (this module with grad, ``ref_module`` under ``no_grad``) via
        the same biased-logit reconstruction, then accumulates the closed-form
        categorical KL (:func:`lobster.rl_training._dfm_logprob.dfm_step_kl`) over
        the generated positions. Used as the GRPO KL-to-reference regularizer.

        Parameters
        ----------
        trajectory : dict
            Captured rollout (see :meth:`logprob_over_trajectory`).
        ref_module : LeFlurSequenceStructureEncoderLightningModule
            The frozen reference policy.
        tracks : Sequence[str], optional
            Tracks to include. Defaults to all three.
        step_indices : Sequence[int] | None, optional
            Optional step subset. ``None`` uses every captured step.

        Returns
        -------
        Tensor
            Per-design summed KL of shape ``(B,)``, differentiable in this module.
        """
        from lobster.rl_training._dfm_logprob import dfm_step_kl, dfm_step_prob

        static = trajectory["static"]
        device = static["mask"].device
        mask_index = {
            "sequence_tokens": static["mask_index_seq"],
            "structure_tokens": static["mask_index_struc"],
            "tri_tokens": static["mask_index_tri"],
        }
        gen_mask = {
            "sequence_tokens": static["gen_mask_seq"],
            "structure_tokens": static["gen_mask_struc"],
            "tri_tokens": static["gen_mask_tri"],
        }
        total: Tensor | None = None
        for rec, xt_dev, t_seq, t_struc in self._iter_traj_steps(trajectory, step_indices):
            biased = self._recompute_biased_step_logits(xt_dev, t_seq, t_struc, rec["step_idx"], static)
            with torch.no_grad():
                biased_ref = ref_module._recompute_biased_step_logits(xt_dev, t_seq, t_struc, rec["step_idx"], static)
            for track in tracks:
                tr = rec["tracks"].get(track)
                if tr is None or biased.get(track) is None or gen_mask.get(track) is None:
                    continue
                kwargs = dict(
                    t=tr["t"].to(device),
                    dt=tr["dt"].to(device) if torch.is_tensor(tr["dt"]) else tr["dt"],
                    xt=tr["xt"].to(device=device, dtype=torch.long),
                    mask_index=mask_index[track],
                    temperature=tr["temperature"],
                    stochasticity=tr["stochasticity"],
                )
                sp_theta = dfm_step_prob(biased[track], **kwargs)
                with torch.no_grad():
                    sp_ref = dfm_step_prob(biased_ref[track], **kwargs)
                kl = dfm_step_kl(sp_theta, sp_ref, gen_mask[track].to(device))
                total = kl if total is None else total + kl
        if total is None:
            return torch.zeros(static["mask"].shape[0], device=device)
        return total
