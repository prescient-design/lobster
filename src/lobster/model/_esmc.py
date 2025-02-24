from typing import Iterable, Optional, List, Dict
import copy

import lightning.pytorch as pl
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM

class LobsterESMC(pl.LightningModule):
    def __init__(
        self,
        use_bfloat16: bool = False,
        max_length: int = 512,
    ):
        """
        Prescient Protein Masked Language Model.

        Parameters
        ----------
        use_bfloat: whether to load model weights as bfloat16 instead of float32
        max_length: max sequence length the model will see

        """
        super().__init__()
        self._max_length = max_length

        if not use_bfloat16:
            self.model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.tokenizer = self.model.tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model for ONNX compiling.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input tensor.
        attention_mask: torch.Tensor
            The attention mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.

        """
        preds = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = preds["hidden_states"]  # hidden reps (B, L, H)

        hidden_states = torch.stack(hidden_states, dim=1)  # (B, num layers, L, H)

        return hidden_states


    def predict_step(self, batch, batch_idx) -> pd.DataFrame:
        # batch, _targets = batch  # targets are the FASTA IDs
        toks = batch["input_ids"].squeeze()
        toks = toks.to(self.device)
        attention_mask = batch["attention_mask"].squeeze().to(self.device)
        with torch.inference_mode():
            preds = self.model(input_ids=toks, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = preds["hidden_state"][-1]  # last layer hidden reps (B, L, H)

        # mean pool over AAs
        df = pd.DataFrame(
            hidden_states.mean(dim=1).cpu(),
            columns=[f"embedding_{idx}" for idx in range(hidden_states.shape[-1])],
        )

        return df
    

    def embed_dataset(self, sequences: List[str], batch_size: int = 32, residue_wise: bool = True, num_workers: int = 0) -> Dict[str, torch.Tensor]:
        embeddings = self.model.embed_dataset(
            sequences=sequences, # list of protein strings
            batch_size=batch_size, # embedding batch size
            max_len=self._max_length, # truncate to max_len
            full_embeddings=residue_wise, # return residue-wise embeddings
            full_precision=False, # store as float32
            pooling_type='mean', # use mean pooling if protein-wise embeddings
            num_workers=0, # data loading num workers
            sql=False, # return dictionary of sequences and embeddings
        )
        return embeddings

    def naturalness(self, sequences: Iterable[str]) -> torch.Tensor:
        out = [
            self._naturalness_single_sequence(
                seq,
                batch_size=32,
            )
            for seq in sequences
        ]

        return torch.tensor(out)

    def _naturalness_single_sequence(
        self,
        sequence: str,
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> tuple[float, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        N = len(sequence)

        # Tokenize full sequence
        encoded_seq = self.tokenizer.encode(sequence)
        ref_seq_indices = torch.tensor(encoded_seq) - 4
        # -4 to align since first tokens are special, BERT-related.

        # Generate full sequence variants with aa's masked one by one, tokenize
        seqs_masked = [copy.deepcopy(encoded_seq) for x in range(N)]
        for i in range(N):
            seqs_masked[i][i+1] = self.tokenizer.mask_token_id
        seqs_mask_encoded = torch.tensor(seqs_masked, device=self.device)

        if N < batch_size:
            batch_size_ = N
        else:
            batch_size_ = batch_size
        with torch.inference_mode():
            logits = torch.vstack(
                [
                    self.model(input_ids=toks.to(self.device))["logits"]
                    for toks in torch.tensor_split(seqs_mask_encoded, N // batch_size_)
                ]
            )

        # raw_log_probs [N, 20]: log probability for each WT amino acid
        raw_log_probs = torch.nn.functional.log_softmax(logits[:, :, 4:24], dim=-1)[
            torch.arange(N), torch.arange(N), :
        ]
        # sum of log probabilities that the model assigns to the true amino acid in each masked position
        sum_log_probs = raw_log_probs[torch.arange(N), ref_seq_indices[1:-1]].sum()  # chop off bos/eos

        naturalness_score = (1.0 / torch.exp(-sum_log_probs / N)).item()

        if return_probs:
            return naturalness_score, (raw_log_probs, ref_seq_indices[1:-1].detach())
        else:
            return naturalness_score

    def latent_embeddings_to_sequences(self, x: torch.Tensor) -> List[str]:
        """x: (B, L, H) size tensor of hidden states"""
        with torch.inference_mode():
            logits = self.model.sequence_head(x)
        tokens = [self.tokenizer.decode(logit.argmax(dim=-1)) for logit in logits]
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        tokens = [t.replace(" ", "") for t in tokens]
        tokens = ["".join([t for t in seq if t in aa_toks]) for seq in tokens]
        return tokens

    def sequences_to_latents(self, sequences: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(sequences, padding="max_length", truncation=True, max_length=self._max_length, return_tensors='pt')
        with torch.inference_mode():
            hidden_states = self.model(input_ids=tokens["input_ids"].to(self.device), output_hidden_states=True)["hidden_states"]
        return hidden_states

    def _perturb_seq(self, sequences: List[str], sigma: float = 5.0) -> List[str]:
        h = self.sequences_to_latents(sequences)
        h_perturbed = h + torch.randn(h.shape) * sigma * h.var()
        sequences = self.latent_embeddings_to_sequences(h_perturbed)

        return sequences

