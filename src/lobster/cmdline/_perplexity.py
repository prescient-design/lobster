import importlib.resources
import math

import esm
import hydra
import lightning.pytorch as pl
import pandas as pd
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from lobster.model._utils import model_typer
from lobster.tokenization import PmlmConceptTokenizerTransform

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, dataframe, keyword):
        self.dataframe = dataframe
        self.seq = dataframe[keyword].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.seq[idx]


@hydra.main(version_base=None, config_path="../hydra_config", config_name="perplexity")
def perplexity(cfg: DictConfig) -> bool:
    WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.experiment,
        log_model="all",
    )

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    dataset_name = cfg.dataset_name
    print(f"Experiment:\ndata: {dataset_name}\nModel: {cfg.model.model_type} {cfg.model.model_size}")

    path = importlib.resources.files("lobster") / "assets" / "pmlm_tokenizer"

    transform_fn = PmlmConceptTokenizerTransform(
        path,
        padding="max_length",
        truncation=True,
        max_length=512,
        normalize=False,
    )
    transform_fn_normalized = PmlmConceptTokenizerTransform(
        path,
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    if cfg.model.model_type == "esm":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        model_helper = (
            model_typer[cfg.model_helper.model_type].load_from_checkpoint(cfg.model_helper.checkpoint).to(device)
        )
    else:
        model = model_typer[cfg.model.model_type].load_from_checkpoint(cfg.model.checkpoint).to(device)

    model.to(device).eval()
    df = pd.read_csv(cfg.path)
    dataset = CustomDataset(df, cfg.data.key)

    predict_dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    total_loss = 0
    for i, sequences in enumerate(predict_dataloader):
        if i % 10 == 0:
            print(f"{i} of {len(predict_dataloader)}")

        original_concepts_seq = torch.concat(
            [toks["all_concepts"].unsqueeze(0).to(device) for toks in transform_fn_normalized(sequences)]
        )
        input_ids = torch.concat([toks["input_ids"].to(device) for toks in transform_fn(sequences)])
        attention_mask = torch.concat([toks["attention_mask"].to(device) for toks in transform_fn(sequences)])
        labels = input_ids.clone()

        if cfg.model.model_type == "esm":
            masked_toks = model_helper._mask_inputs(input_ids, p_mask=cfg.mask_percentage)
            labels[masked_toks != model_helper.tokenizer.mask_token_id] = -100  # Only calculate loss on masked tokens
        else:
            masked_toks = model._mask_inputs(input_ids, p_mask=cfg.mask_percentage)
            labels[masked_toks != model.tokenizer.mask_token_id] = -100  # Only calculate loss on masked tokens

        if cfg.model.conditional_model:
            if model.config.conditioning_type in ["pre_encoder", "pre_encoder_with_classifier"]:
                concepts_to_emb = torch.zeros(original_concepts_seq.shape[0], model.config.concept_input_size[0]).to(
                    device
                )
                outputs = model.model(
                    input_ids=masked_toks,
                    concepts=original_concepts_seq,
                    concepts_to_emb=[concepts_to_emb],
                    attention_mask=attention_mask,
                    inference=True,
                    labels=labels,
                )
            else:
                outputs = model.model(
                    input_ids=masked_toks,
                    concepts=original_concepts_seq,
                    attention_mask=attention_mask,
                    inference=True,
                    labels=labels,
                )
            loss = outputs["loss"].detach()
        else:
            if cfg.model.model_type == "esm":
                logits = model(masked_toks)["logits"].detach()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, model_helper.config.vocab_size + 1), labels.view(-1)).detach()
            else:
                loss = model.model(input_ids=masked_toks, attention_mask=attention_mask, labels=labels)["loss"].detach()

        total_loss += loss.item()
        ppl = math.exp(loss.item())
        print(ppl)

    average_loss = total_loss / len(predict_dataloader)
    ppl = math.exp(average_loss)
    print("perplexity:", ppl)
    wandb.log({"perplexity": ppl})
    wandb.finish()
