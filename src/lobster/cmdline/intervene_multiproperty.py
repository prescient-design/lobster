import importlib.resources
import os

import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from lobster.concepts._utils import supported_biopython_concepts
from lobster.model._utils import model_typer
from lobster.tokenization import PmlmConceptTokenizerTransform


def create_mask(tensor, num_features=None, p_mask=0.2, intervention_type="positive"):
    """Create a mask for features based on the intervention type."""
    masks = []
    for i in range(len(tensor)):
        num_features_to_zero = int(num_features[i] * p_mask)
        if intervention_type == "positive":
            sorted_tensor, sorted_inds = torch.sort(tensor[i, 1 : num_features[i] + 1], dim=0, descending=False)
        elif intervention_type == "negative":
            sorted_tensor, sorted_inds = torch.sort(tensor[i, 1 : num_features[i] + 1], dim=0, descending=True)
        sorted_inds = sorted_inds + 1
        mask = torch.zeros_like(tensor[i], dtype=torch.bool)
        mask[sorted_inds[:num_features_to_zero]] = True
        masks.append(mask)
    return torch.stack(masks, dim=0)


device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, dataframe, keyword):
        self.dataframe = dataframe
        self.seq = dataframe[keyword].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.seq[idx]


@hydra.main(version_base=None, config_path="../hydra_config", config_name="intervene")
def intervene_multiproperty(cfg: DictConfig) -> bool:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    dataset_name = cfg.dataset_name
    print(
        f"Experiment:\ndata: {dataset_name}\nModel: {cfg.model.model_type} {cfg.model.model_size}\n"
        f"Intervention: {cfg.intervention.intervention_type} percentage: {cfg.intervention.mask_percentage} "
        f"iteration: {cfg.intervention.iteration} use attribution: {cfg.intervention.use_feature_att}"
    )

    path = importlib.resources.files("lobster") / "assets" / "pmlm_tokenizer"
    aa_toks = list("ARNDCEQGHILKMFPSTWYV")

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

    model = model_typer[cfg.model.model_type].load_from_checkpoint(cfg.model.checkpoint).to(device)
    model.eval()
    concepts = supported_biopython_concepts if cfg.concepts.concept_name == "all" else [cfg.concepts.concept_name]
    concept_folders = [f for f in os.listdir(cfg.data.path) if os.path.isdir(os.path.join(cfg.data.path, f))]
    columns = ["input_data", "masked_predicted_data", "masked_predicted_all_at_once"]

    for folder in concept_folders:
        data_folder_dir = f"{cfg.paths.root_dir}data_{dataset_name}_{cfg.model.model_type}"
        if cfg.intervention.use_feature_att:
            data_folder_dir += "withAtt"
        data_folder_dir += f"_{cfg.model.model_size}"
        if cfg.intervention.intervene:
            data_folder_dir += f"_{cfg.intervention.intervention_type}_maskingPercent{cfg.intervention.mask_percentage}_iter{cfg.intervention.iteration}"
        data_folder_dir += f"/Multi_Concepts/{folder}"
        os.makedirs(data_folder_dir, exist_ok=True)

        csv_path = (
            cfg.data.path
            + folder
            + ("/lowest.csv" if cfg.intervention.intervention_type == "positive" else "/highest.csv")
        )
        df = pd.read_csv(csv_path)
        dataset = CustomDataset(df, cfg.data.key)

        predict_dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
        )

        concepts = df.columns[1:].tolist()
        concepts_indices = [idx for idx, value in enumerate(supported_biopython_concepts) if value in concepts]

        if cfg.model.conditional_model:
            print(f"Let's intervene on {model.config.conditioning_type}....")

        index__ = 0
        all_data = None

        for i, sequences in enumerate(predict_dataloader):
            original_concepts_seq = torch.concat(
                [toks["all_concepts"].unsqueeze(0).to(device) for toks in transform_fn_normalized(sequences)]
            )
            input_ids = torch.concat([toks["input_ids"].to(device) for toks in transform_fn(sequences)])
            attention_mask = torch.concat([toks["attention_mask"].to(device) for toks in transform_fn(sequences)])

            for iter_ in range(cfg.intervention.iteration):
                if cfg.intervention.intervene_multiproperty_type == "sequential":
                    for i_c, c in enumerate(concepts_indices):
                        masked_toks = model._mask_inputs(input_ids, p_mask=cfg.intervention.mask_percentage)

                        if cfg.model.conditional_model:
                            if model.config.conditioning_type in ("pre_encoder", "pre_encoder_with_classifier"):
                                logits_masked = model.model(
                                    input_ids=masked_toks,
                                    concepts=original_concepts_seq,
                                    concepts_to_emb=None,
                                    inference=True,
                                    attention_mask=attention_mask,
                                )["logits"]
                            else:
                                logits_masked = model.model(
                                    input_ids=masked_toks, inference=True, attention_mask=attention_mask
                                )["logits"]
                        else:
                            logits_masked = model.model(input_ids=masked_toks, attention_mask=attention_mask)["logits"]

                        logits_masked = logits_masked.detach()
                        pred_masked_seqs = []

                        for j, logit in enumerate(logits_masked):
                            pred_tok = logit.argmax(dim=-1)
                            mask = masked_toks[j].eq(model.tokenizer.mask_token_id).int()
                            pred_masked_token = (input_ids[j] * (1 - mask)) + (pred_tok * mask)
                            pred_masked_seq = model.tokenizer.decode(pred_masked_token).replace(" ", "")
                            pred_masked_seq = "".join([t for t in pred_masked_seq if t in aa_toks])
                            pred_masked_seqs.append(pred_masked_seq)

                        pred_seq = []
                        num_features = torch.sum(
                            (input_ids != model.tokenizer.cls_token_id)
                            * (input_ids != model.tokenizer.pad_token_id)
                            * (input_ids != model.tokenizer.eos_token_id),
                            dim=1,
                        )

                        if (
                            cfg.intervention.use_feature_att
                            and cfg.model.conditional_model
                            and model.config.conditioning_type in ("cbm", "pre_encoder_with_classifier")
                        ):
                            mask_emb = model.model.LMBase.embeddings.word_embeddings.weight[-1].detach()
                            forward_output = model.model(
                                input_ids=input_ids,
                                concepts=original_concepts_seq,
                                inference=True,
                                attention_mask=attention_mask,
                                requires_grad=True,
                            )
                            concept_ = forward_output["concepts"][:, c]
                            input_ = forward_output["input_emb"]
                            attribution_orig = torch.autograd.grad(torch.unbind(concept_), input_, allow_unused=True)[0]
                            attribution = torch.sum(attribution_orig * (input_ - mask_emb), dim=2)
                            mask = create_mask(
                                attribution,
                                num_features=num_features,
                                p_mask=cfg.intervention.mask_percentage,
                                intervention_type=cfg.intervention.intervention_type,
                            )

                            masked_toks = model._mask_inputs(input_ids, mask_arr=mask)

                        if model.config.conditioning_type in ("pre_encoder", "pre_encoder_with_classifier"):
                            original_concepts = original_concepts_seq
                        else:
                            if model._n_concepts > original_concepts_seq.shape[1]:
                                original_concepts_ds = torch.zeros(
                                    (original_concepts_seq.shape[0], model._n_concepts - original_concepts_seq.shape[1])
                                ).to(device)
                                original_concepts = torch.concat((original_concepts_seq, original_concepts_ds), dim=1)
                            else:
                                original_concepts = original_concepts_seq

                        concept_mask = torch.zeros_like(original_concepts)
                        concept_mask[:, concepts_indices] = 1

                        new_concepts_value = original_concepts.clone()
                        if cfg.intervention.intervention_type == "positive":
                            new_concepts_value[:, c] = torch.ones_like(new_concepts_value[:, c])
                        else:
                            new_concepts_value[:, c] = torch.zeros_like(new_concepts_value[:, c])
                        new_concepts_value = new_concepts_value.clamp(max=1).clamp(min=0)

                        if model.config.conditioning_type in ("pre_encoder", "pre_encoder_with_classifier"):
                            logits_masked = model.model(
                                input_ids=masked_toks,
                                concepts=new_concepts_value,
                                inference=True,
                                attention_mask=attention_mask,
                            )["logits"]
                        else:
                            intervene_value = (concept_mask, new_concepts_value)
                            logits_masked = model.model(
                                input_ids=masked_toks,
                                inference=True,
                                intervene=intervene_value,
                                attention_mask=attention_mask,
                            )["logits"]

                        logits_masked = logits_masked.detach()
                        pred_masked_tokens = []

                        for j, logit in enumerate(logits_masked):
                            pred_tok = logit.argmax(dim=-1)
                            mask = masked_toks[j].eq(model.tokenizer.mask_token_id).int()
                            pred_masked_token = (input_ids[j] * (1 - mask)) + (pred_tok * mask)
                            pred_masked_token = model.tokenizer.decode(pred_masked_token).replace(" ", "")
                            pred_masked_token = "".join([t for t in pred_masked_token if t in aa_toks])
                            pred_masked_tokens.append(pred_masked_token)

                        pred_seq = pred_masked_tokens
                        if iter_ < (cfg.intervention.iteration - 1) or i_c < len(concepts_indices) - 1:
                            original_concepts_seq = torch.concat(
                                [
                                    toks["all_concepts"].unsqueeze(0).to(device)
                                    for toks in transform_fn_normalized(pred_masked_tokens)
                                ]
                            )
                            input_ids = torch.concat(
                                [toks["input_ids"].to(device) for toks in transform_fn(pred_masked_tokens)]
                            )
                            attention_mask = torch.concat(
                                [toks["attention_mask"].to(device) for toks in transform_fn(pred_masked_tokens)]
                            )
                            masked_toks = model._mask_inputs(input_ids, p_mask=cfg.intervention.mask_percentage)
                else:
                    masked_toks = model._mask_inputs(input_ids, p_mask=cfg.intervention.mask_percentage)
                    if cfg.model.conditional_model:
                        if model.config.conditioning_type in ("pre_encoder", "pre_encoder_with_classifier"):
                            logits_masked = model.model(
                                input_ids=masked_toks,
                                concepts=original_concepts_seq,
                                concepts_to_emb=None,
                                inference=True,
                                attention_mask=attention_mask,
                            )["logits"]
                        else:
                            logits_masked = model.model(
                                input_ids=masked_toks, inference=True, attention_mask=attention_mask
                            )["logits"]
                    else:
                        logits_masked = model.model(input_ids=masked_toks, attention_mask=attention_mask)["logits"]

                    logits_masked = logits_masked.detach()
                    pred_masked_seqs = []

                    for j, logit in enumerate(logits_masked):
                        pred_tok = logit.argmax(dim=-1)
                        mask = masked_toks[j].eq(model.tokenizer.mask_token_id).int()
                        pred_masked_token = (input_ids[j] * (1 - mask)) + (pred_tok * mask)
                        pred_masked_seq = model.tokenizer.decode(pred_masked_token).replace(" ", "")
                        pred_masked_seq = "".join([t for t in pred_masked_seq if t in aa_toks])
                        pred_masked_seqs.append(pred_masked_seq)

                    pred_seq = []
                    num_features = torch.sum(
                        (input_ids != model.tokenizer.cls_token_id)
                        * (input_ids != model.tokenizer.pad_token_id)
                        * (input_ids != model.tokenizer.eos_token_id),
                        dim=1,
                    )

                    if (
                        cfg.intervention.use_feature_att
                        and cfg.model.conditional_model
                        and model.config.conditioning_type in ("cbm", "pre_encoder_with_classifier")
                    ):
                        mask_emb = model.model.LMBase.embeddings.word_embeddings.weight[-1].detach()
                        forward_output = model.model(
                            input_ids=input_ids,
                            concepts=original_concepts_seq,
                            inference=True,
                            attention_mask=attention_mask,
                            requires_grad=True,
                        )
                        concept_ = sum([forward_output["concepts"][:, c] for c in concepts_indices])
                        input_ = forward_output["input_emb"]
                        attribution_orig = torch.autograd.grad(torch.unbind(concept_), input_, allow_unused=True)[0]
                        attribution = torch.sum(attribution_orig * (input_ - mask_emb), dim=2)
                        mask = create_mask(
                            attribution,
                            num_features=num_features,
                            p_mask=cfg.intervention.mask_percentage,
                            intervention_type=cfg.intervention.intervention_type,
                        )
                        masked_toks = model._mask_inputs(input_ids, mask_arr=mask)

                    if model.config.conditioning_type in ("pre_encoder", "pre_encoder_with_classifier"):
                        original_concepts = original_concepts_seq
                    else:
                        if model._n_concepts > original_concepts_seq.shape[1]:
                            original_concepts_ds = torch.zeros(
                                (original_concepts_seq.shape[0], model._n_concepts - original_concepts_seq.shape[1])
                            ).to(device)
                            original_concepts = torch.concat((original_concepts_seq, original_concepts_ds), dim=1)
                        else:
                            original_concepts = original_concepts_seq

                    concept_mask = torch.zeros_like(original_concepts)
                    concept_mask[:, concepts_indices] = 1

                    new_concepts_value = original_concepts.clone()
                    if cfg.intervention.intervention_type == "positive":
                        for c in concepts_indices:
                            new_concepts_value[:, c] = torch.ones_like(new_concepts_value[:, c])
                    else:
                        for c in concepts_indices:
                            new_concepts_value[:, c] = torch.zeros_like(new_concepts_value[:, c])
                    new_concepts_value = new_concepts_value.clamp(max=1).clamp(min=0)

                    if model.config.conditioning_type in ("pre_encoder", "pre_encoder_with_classifier"):
                        logits_masked = model.model(
                            input_ids=masked_toks,
                            concepts=new_concepts_value,
                            inference=True,
                            attention_mask=attention_mask,
                        )["logits"]
                    else:
                        intervene_value = (concept_mask, new_concepts_value)
                        logits_masked = model.model(
                            input_ids=masked_toks,
                            inference=True,
                            intervene=intervene_value,
                            attention_mask=attention_mask,
                        )["logits"]

                    logits_masked = logits_masked.detach()
                    pred_masked_tokens = []

                    for j, logit in enumerate(logits_masked):
                        pred_tok = logit.argmax(dim=-1)
                        mask = masked_toks[j].eq(model.tokenizer.mask_token_id).int()
                        pred_masked_token = (input_ids[j] * (1 - mask)) + (pred_tok * mask)
                        pred_masked_token = model.tokenizer.decode(pred_masked_token).replace(" ", "")
                        pred_masked_token = "".join([t for t in pred_masked_token if t in aa_toks])
                        pred_masked_tokens.append(pred_masked_token)

                    pred_seq = pred_masked_tokens

                    if iter_ < (cfg.intervention.iteration - 1):
                        original_concepts_seq = torch.concat(
                            [
                                toks["all_concepts"].unsqueeze(0).to(device)
                                for toks in transform_fn_normalized(pred_masked_tokens)
                            ]
                        )
                        input_ids = torch.concat(
                            [toks["input_ids"].to(device) for toks in transform_fn(pred_masked_tokens)]
                        )
                        attention_mask = torch.concat(
                            [toks["attention_mask"].to(device) for toks in transform_fn(pred_masked_tokens)]
                        )
                        masked_toks = model._mask_inputs(input_ids, p_mask=cfg.intervention.mask_percentage)

            if i == 0:
                data = [sequences, pred_masked_seqs, pred_seq]
                all_data = np.array(data)
            else:
                data = [sequences, pred_masked_seqs, pred_seq]
                all_data = np.concatenate((all_data, data), axis=1) if all_data is not None else np.array(data)

            if all_data.shape[1] > 1000:
                print(f"{i + 1} of {len(predict_dataloader)} Saving")
                df = pd.DataFrame(all_data.transpose(), columns=columns)
                df.to_csv(f"{data_folder_dir}/{index__ + 1}_indexed.csv", index=False)
                index__ += 1
                all_data = None

        if all_data is not None:
            print(f"{i + 1} of {len(predict_dataloader)} Saving")
            df = pd.DataFrame(all_data.transpose(), columns=columns)
            df.to_csv(f"{data_folder_dir}/{index__ + 1}_indexed.csv", index=False)
