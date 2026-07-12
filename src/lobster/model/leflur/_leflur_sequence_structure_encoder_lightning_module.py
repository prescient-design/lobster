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
        # Fresh-finetune-from-pretrained knobs. These are NOT used inside
        # __init__ — they're consumed by `cmdline/train.py` AFTER model
        # instantiation. Declared here purely so Hydra can pass them
        # through `model.pretrained_ckpt=...` without raising on an
        # unexpected kwarg, and so they get saved into the checkpoint's
        # `hyper_parameters` block for reproducibility.
        pretrained_ckpt: str | None = None,
        zero_init_conditioning_on_load: bool = False,
        zero_init_chain_embedding_on_load: bool = False,
    ):
        self.save_hyperparameters()
        self.cond_percentage = cond_percentage
        self.template_percentage = template_percentage

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
    ) -> dict[str, Tensor]:
        """Forward pass of the model, for inference."""
        if timesteps is not None:
            timesteps = timesteps.copy()
            # expand to be same length as x_t e.g from B to B,L
            B, L = x_t["sequence_tokens"].shape
            timesteps["sequence_tokens"] = timesteps["sequence_tokens"][:, None].expand(-1, L)[:, :, None]
            timesteps["structure_tokens"] = timesteps["structure_tokens"][:, None].expand(-1, L)[:, :, None]

        unmasked_x = self.encoder(
            sequence_input_ids=x_t["sequence_tokens"],
            structure_input_ids=x_t["structure_tokens"],
            position_ids=residue_index,
            attention_mask=mask,
            conditioning_tensor=conditioning_tensor,
            chain_ids=chain_ids,
            timesteps=timesteps,
            template_structure_tokens=template_structure_tokens,
            return_auxiliary_tasks=False,
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

        # gen tokens
        unmasked_x = self.forward(
            x_t,
            mask,
            residue_index,
            conditioning_tensor,
            timesteps=timesteps,
            chain_ids=chain_ids,
            template_structure_tokens=template_structure_tokens,
        )

        total_loss, loss_dict = self.apply_interpolant_loss(
            split, x_gt, unmasked_x, mask, total_loss, loss_dict, timesteps
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
    ):
        """Generate with model, with option to return full unmasking trajectory and likelihood."""
        device = next(self.parameters()).device
        xt_seq = self.interpolant_seq.sample_prior((num_samples, length))
        xt_struc = self.interpolant_struc.sample_prior((num_samples, length))
        xt = {"sequence_tokens": xt_seq, "structure_tokens": xt_struc}
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

        for step_idx, (dt_seq, dt_struc, t_seq, t_struc) in enumerate(
            tqdm(zip(dts_seq, dts_struc, ts_seq, ts_struc), desc="Generating samples")
        ):
            t_seq = inference_schedule_seq.pad_time(num_samples, t_seq, device)
            t_struc = inference_schedule_struc.pad_time(num_samples, t_struc, device)
            timesteps = {"sequence_tokens": t_seq, "structure_tokens": t_struc}

            unmasked_x = self.forward(
                xt,
                mask,
                residue_index,
                conditioning_tensor,
                timesteps=timesteps,
                chain_ids=chain_ids,
                template_structure_tokens=template_structure_tokens,
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
                )
                for _k in ("sequence_logits", "structure_logits"):
                    unmasked_x[_k] = uncond_x[_k] + cfg_weight * (unmasked_x[_k] - uncond_x[_k])
            unmasked_sequence_tokens = unmasked_x["sequence_logits"]
            if sequence_logit_bias is not None and step_idx < sequence_logit_bias_steps:
                unmasked_sequence_tokens = unmasked_sequence_tokens + sequence_logit_bias
            xt_seq_new = self.interpolant_seq.step(
                unmasked_sequence_tokens,
                t_seq,
                xt_seq,
                dt_seq,
                stochasticity=stochasticity_seq,
                temperature=temperature_seq,
            )
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

            xt = {"sequence_tokens": xt_seq, "structure_tokens": xt_struc}

        return unmasked_x
