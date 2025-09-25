import logging
from collections.abc import Sequence
from typing import Literal, Callable

import torch
import torch.nn as nn
import transformers
from lightning import LightningModule
from torch import Tensor

from lobster.constants import Modality, ModalityType
from lobster.model.neobert import mask_tokens
from lobster.model.utils import _detect_modality
from lobster.tokenization import UMETokenizerTransform

from ._gen_ume_sequence_structure_encoder import AuxiliaryTask, UMESequenceStructureEncoderModule

#latent generator code:
from lobster.model.latent_generator.cmdline import LatentEncoderDecoder
from lobster.model.latent_generator.cmdline import methods as latent_generator_methods

#bionemo interpolant code:
from bionemo.moco.distributions.prior import DiscreteUniformPrior, DiscreteMaskedPrior
from bionemo.moco.distributions.time import UniformTimeDistribution
from bionemo.moco.interpolants import DiscreteFlowMatcher
from bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule, PowerInferenceSchedule, LogInferenceSchedule


logger = logging.getLogger(__name__)


class UMESequenceStructureEncoderLightningModule(LightningModule):
    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        special_token_ids: list[int],
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        seed: int = 0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        scheduler: str = "constant",
        scheduler_kwargs: dict | None = None,
        encoder_kwargs: dict | None = None,
        ckpt_path: str | None = None,
        #LatentGenerator params
        decode_tokens_during_training: bool = True,
        latent_generator_model_name: str = "LG 20A seq 3di c6d Aux",
        #generation params
        prior_distribution_seq: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        prior_distribution_struc: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        time_distribution_seq: Callable[..., UniformTimeDistribution] = UniformTimeDistribution,
        time_distribution_struc: Callable[..., UniformTimeDistribution] = UniformTimeDistribution,
        interpolant: Callable[..., DiscreteFlowMatcher] = DiscreteFlowMatcher,
        inference_schedule: Callable[..., LinearInferenceSchedule] = LinearInferenceSchedule,
        use_masked_prior: bool = True,
        num_warmup_steps: int = 20_000,
        num_training_steps: int = 100_000,
    ):
        self.save_hyperparameters()

        super().__init__()

        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = special_token_ids
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


        #LatentGenerator params
        #self.structure_encoder = structure_encoder
        #self.quantizer = quantizer
        #self.demasker = demasker
        #self.decoder_factory = decoder_factory
        #self.freeze_tokenizer = freeze_tokenizer
        #self.freeze_decoder = freeze_decoder
        self.decode_tokens_during_training = decode_tokens_during_training
        self.structure_latent_encoder_decoder = LatentEncoderDecoder()
        self.structure_latent_encoder_decoder.load_model(
            latent_generator_methods[latent_generator_model_name].model_config.checkpoint, 
            latent_generator_methods[latent_generator_model_name].model_config.config_path, 
            latent_generator_methods[latent_generator_model_name].model_config.config_name, 
            overrides=latent_generator_methods[latent_generator_model_name].model_config.overrides)
        self.quantizer = self.structure_latent_encoder_decoder.model.quantizer
        self.strucure_encoder = self.structure_latent_encoder_decoder.model.encoder
        self.decoder_factory = self.structure_latent_encoder_decoder.model.decoder_factory
        self.loss_factory = self.structure_latent_encoder_decoder.model.loss_factory

        #generation params
        self.prior_distribution_seq = prior_distribution_seq
        self.prior_distribution_struc = prior_distribution_struc
        if use_masked_prior:
            self.prior_distribution_seq = DiscreteMaskedPrior(num_classes=self.vocab_size,  mask_dim=self.mask_token_id, inclusive=True)
            self.prior_distribution_struc = DiscreteMaskedPrior(num_classes=self.quantizer.n_tokens+2, mask_dim=self.quantizer.n_tokens, inclusive=True)
            prior_seq = self.prior_distribution_seq
            prior_struc = self.prior_distribution_struc
            
        else:
            prior_seq = self.prior_distribution_seq(num_classes=self.vocab_size)
            prior_struc = self.prior_distribution_struc(num_classes=self.quantizer.n_tokens+2)
        self.time_distribution_seq = time_distribution_seq
        self.time_distribution_struc = time_distribution_struc
        self.interpolant = interpolant
        self.inference_schedule = inference_schedule
        time_distribution_seq = self.time_distribution_seq()
        time_distribution_struc = self.time_distribution_struc()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        interpolant_seq = self.interpolant(time_distribution=time_distribution_seq, prior_distribution=prior_seq, device=device)
        interpolant_struc = self.interpolant(time_distribution=time_distribution_struc, prior_distribution=prior_struc, device=device)
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

        self.encoder = UMESequenceStructureEncoderModule(
            auxiliary_tasks=auxiliary_tasks,
            pad_token_id=self.pad_token_id,
            model_ckpt= ckpt_path,
            **encoder_kwargs or {},
        )

    def embed_sequences(
        self, sequences: Sequence[str] | str, modality: ModalityType | Modality = None, aggregate: bool = True
    ) -> Tensor:
        raise NotImplementedError("Embedding for sequence and structure encoder is not implemented")

    def decode_structure(self, unmasked_x: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
        """Decode the model output."""
        decoder_name = "vit_decoder"

        decoded_x = {}
        struc_tokens=unmasked_x["structure_logits"][...,:self.quantizer.n_tokens]
        #softmax with temp
        temp = 0.1
        struc_tokens_ = torch.softmax(struc_tokens / temp, dim=-1)

        decoded_x[decoder_name] = self.decoder_factory.decoders[decoder_name](
            struc_tokens_, mask
        )
        
        return decoded_x

    def encode_structure(self, x_gt: Tensor, mask: Tensor, residue_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode the model input."""
        x_emb = self.strucure_encoder(x_gt, mask, residue_index=residue_index)
        x_quant, x_quant_emb, mask = self.quantizer.quantize(x_emb, mask=mask, batch_size=x_gt.shape[0])
        
        return x_quant, x_quant_emb, mask

    def apply_interpolant_loss(self, split: str, x_gt: dict[str, Tensor], unmasked_x: dict[str, Tensor], mask: Tensor, total_loss: Tensor, loss_dict: dict[str, Tensor], timesteps: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        loss_seq = self.interpolant_seq.loss(unmasked_x['sequence_logits'], x_gt["sequence_tokens"], timesteps["sequence_tokens"]).mean()
        loss_struc = self.interpolant_struc.loss(unmasked_x['structure_logits'], x_gt["structure_tokens"], timesteps["structure_tokens"]).mean()
        loss = loss_seq + loss_struc
        total_loss += loss
        loss_dict[f"{split}_interpolant_seq"] = loss_seq
        loss_dict[f"{split}_interpolant_struc"] = loss_struc
        loss_dict[f"{split}_timesteps_seq"] = timesteps["sequence_tokens"].mean()
        loss_dict[f"{split}_timesteps_struc"] = timesteps["structure_tokens"].mean()
        return total_loss, loss_dict

    def apply_structure_decoder_loss(self, split: str, decoder_gt: dict[str, Tensor], decoded_x: dict[str, Tensor], mask: Tensor, total_loss: Tensor, loss_dict: dict[str, Tensor], just_loss: bool = False, keep_batch_dim: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        decoder_name = "vit_decoder"
        loss2apply = self.decoder_factory.get_loss(decoder_name)

        for loss2apply_ in loss2apply:
            loss = self.loss_factory(loss2apply_, decoder_gt, decoded_x[decoder_name], mask, keep_batch_dim=keep_batch_dim)
            if just_loss:
                return loss
            # apply loss weighting from weight_dict in loss_factory; setting to 0 b/c will need different way to set weights for different losses
            #total_loss += self.loss_factory.weight_dict[loss2apply_] * loss
            total_loss += 0 * loss
            loss_dict[f"{split}_{loss2apply_}"] = loss

        return total_loss, loss_dict

    def compute_mlm_loss(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("MLM loss is not implemented for the model.")

    def compute_auxiliary_tasks_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError("Auxiliary tasks are not implemented for the model.")

    def get_gen_gt_and_conditioning_tensor(self, batch: dict[str, Tensor], cond_percentage: float | None = None) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, bool]:
        """Get the conditioning tensor for the model."""
        if cond_percentage is None:
            cond_percentage = 0.0
        if cond_percentage > 0.0 and "epitope_tensor" in batch:
            conditioning = True
        else:
            conditioning = False

        B = batch["sequence"].shape[0]
        L = batch["sequence"].shape[1]
        device = batch["sequence"].device
        mask = batch["mask"]
        residue_index = batch["indices"]
        seq_gt = batch["sequence"]
        x_quant, x_quant_emb, mask = self.encode_structure(*batch["input"])
        x_1_struc_tokens_argmax = torch.argmax(x_quant, dim=-1)
        x_1_struc_tokens_argmax[~mask.bool()] = self.padding_index_struc_tokens

        x_gt =  {"structure_tokens": x_1_struc_tokens_argmax, "sequence_tokens": seq_gt}

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

    def get_timesteps(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Get the timesteps for the model."""
        timesteps_seq = self.interpolant_seq.sample_time(batch["sequence"].shape[0])
        timesteps_struc = self.interpolant_struc.sample_time(batch["sequence"].shape[0])
        timesteps = {"sequence_tokens": timesteps_seq, "structure_tokens": timesteps_struc}
        return timesteps

    def interpolate_tokens(self, input_tokens: dict[str, Tensor], timesteps: dict[str, Tensor]) -> dict[str, Tensor]:
        """Interpolate the tokens for the model."""
        #sequence tokens
        x_1_seq = input_tokens["sequence_tokens"]
        x_1_struc = input_tokens["structure_tokens"]
        x_0_seq = self.interpolant_seq.sample_prior(x_1_seq.shape)
        x_0_struc = self.interpolant_struc.sample_prior(x_1_struc.shape)
        timesteps_seq = timesteps["sequence_tokens"]
        timesteps_struc = timesteps["structure_tokens"]
        x_t_seq = self.interpolant_seq.interpolate(x_1_seq, timesteps_seq, x_0_seq)
        x_t_struc = self.interpolant_struc.interpolate(x_1_struc, timesteps_struc, x_0_struc)

        x_t = {"sequence_tokens": x_t_seq, "structure_tokens": x_t_struc}
        return x_t

    def forward(self, x_t: dict[str, Tensor], mask: Tensor, residue_index: Tensor, conditioning_tensor: Tensor, timesteps: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Forward pass of the model, for inference."""
        if timesteps is not None:
            timesteps = timesteps.copy()
            #expand to be same length as x_t e.g from B to B,L
            B, L = x_t["sequence_tokens"].shape
            timesteps["sequence_tokens"] = timesteps["sequence_tokens"][:,None].expand(-1, L)[:,:,None]
            timesteps["structure_tokens"] = timesteps["structure_tokens"][:,None].expand(-1, L)[:,:,None]
        
        unmasked_x = self.encoder(
            sequence_input_ids=x_t["sequence_tokens"], structure_input_ids=x_t["structure_tokens"], position_ids=residue_index, attention_mask=mask, conditioning_tensor=conditioning_tensor, timesteps=timesteps, return_auxiliary_tasks=False
        )

        return unmasked_x

    def step(self, batch: dict[str, Tensor], batch_idx: int, split: Literal["train", "val"] = "train") -> dict[str, Tensor]:
        """Single training/val/test step of the model."""

        #set losses
        total_loss = 0.0
        loss_dict = {}

        #prep the input
        with torch.no_grad():
            x_gt, conditioning_tensor, mask, residue_index, conditioning = self.get_gen_gt_and_conditioning_tensor(batch)

        timesteps = self.get_timesteps(batch)
        x_t = self.interpolate_tokens(x_gt, timesteps)

        #gen tokens
        unmasked_x = self.forward(x_t, mask, residue_index, conditioning_tensor, timesteps=timesteps)

        total_loss, loss_dict = self.apply_interpolant_loss(split, x_gt, unmasked_x, mask, total_loss, loss_dict, timesteps)

        #Decode the tokens if needed
        if self.decode_tokens_during_training:
            #decode the tokens
            decoder_gt = batch
            decoded_x = self.decode_structure(unmasked_x, mask)
            total_loss, loss_dict = self.apply_structure_decoder_loss(split, decoder_gt, decoded_x, mask, total_loss, loss_dict)
        else:
            decoder_gt = None
            decoded_x = None


        self.log_dict({f"{split}_loss": total_loss, **loss_dict}, batch_size=x_gt["sequence_tokens"].shape[0])


        return {"loss": total_loss, "x_gt": x_gt, "unmasked_x": unmasked_x, "decoder_gt": decoder_gt, "decoded_x": decoded_x, "conditioning": conditioning, f"{split}_timesteps_seq": timesteps["sequence_tokens"], f"{split}_timesteps_struc": timesteps["structure_tokens"]}

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, "val")

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
    
    def generate_sample(self, length, num_samples, inference_schedule_seq: Callable[..., LinearInferenceSchedule] = LinearInferenceSchedule, inference_schedule_struc: Callable[..., LinearInferenceSchedule] = LinearInferenceSchedule, nsteps:int=100, stochasticity_seq: int=0, stochasticity_struc: int=0, temperature_seq: float=1.0, temperature_struc: float=1.0):
        """Generate with model, with option to return full unmasking trajectory and likelihood.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        xt_seq = self.interpolant_seq.sample_prior((num_samples, length))
        xt_struc = self.interpolant_struc.sample_prior((num_samples, length))
        xt_seq_oh = torch.nn.functional.one_hot(xt_seq.long(), num_classes=residue_constants.PEPTIDE_ALPHABET.index("-")+1).float()
        xt_struc_oh = torch.nn.functional.one_hot(xt_struc.long(), num_classes=self.num_struc_classes).float()
        xt = {"sequence_tokens_oh": xt_seq_oh, "structure_tokens_oh": xt_struc_oh}
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

        for dt_seq, dt_struc, t_seq, t_struc in tqdm(zip(dts_seq, dts_struc, ts_seq, ts_struc), desc="Generating samples"):
            t_seq = inference_schedule_seq.pad_time(num_samples, t_seq, device)
            t_struc = inference_schedule_struc.pad_time(num_samples, t_struc, device)
            timesteps = {"sequence_tokens": t_seq, "structure_tokens": t_struc}
            unmasked_x = self.forward(xt, mask, residue_index, conditioning_tensor, timesteps=timesteps)
            unmasked_sequence_tokens = unmasked_x["mb_demasker"]["sequence_tokens"]
            unmasked_structure_tokens = unmasked_x["mb_demasker"]["structure_tokens"]
            xt_seq = self.interpolant_seq.step(unmasked_sequence_tokens, t_seq, xt_seq, dt_seq, stochasticity=stochasticity_seq, temperature=temperature_seq)
            xt_struc = self.interpolant_struc.step(unmasked_structure_tokens, t_struc, xt_struc, dt_struc, stochasticity=stochasticity_struc, temperature=temperature_struc)
            xt_seq_oh = torch.nn.functional.one_hot(xt_seq.long(), num_classes=residue_constants.PEPTIDE_ALPHABET.index("-")+1).float()
            xt_struc_oh = torch.nn.functional.one_hot(xt_struc.long(), num_classes=self.num_struc_classes).float()
            xt = {"sequence_tokens_oh": xt_seq_oh, "structure_tokens_oh": xt_struc_oh}

        return unmasked_x
