import pytorch_lightning as L
import torch

from lobster._ensure_package import ensure_package

# Ensure CaLM package is available
ensure_package("calm", group="calm")

from calm.pretrained import CaLM as PretrainedCaLM


class CaLM(L.LightningModule):
    """
    Minimal LightningModule wrapper for the CaLM pretrained model.

    Only supports inference via `embed_sequences`.

    Parameters
    ----------
    device : str, optional
        Device for returned tensors. Default is "cpu".
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.device_type = device

        # Initialize pretrained CaLM model (kept on its default device)
        self.calm = PretrainedCaLM()
        # Put underlying torch model in eval mode
        if hasattr(self.calm, "model") and hasattr(self.calm.model, "eval"):
            self.calm.model.eval()

    def _normalize_to_codon_sequences(self, sequences: list[str]) -> list[str]:
        """
        Normalize input sequences to RNA codon strings that CaLM expects.

        - If input looks like nucleotides (A/C/G/T/U), convert T->U and trim
          any trailing incomplete codon chunk.
        - If input looks like amino acids, back-translate each residue to a
          canonical codon (most common option) to form an RNA sequence.
        """
        # Simple heuristic alphabets
        nuc_letters = set("ACGTU")
        set("ACDEFGHIKLMNPQRSTVWY")

        # Canonical back-translation map (RNA, using common codons)
        aa_to_codon = {
            "A": "GCU",
            "C": "UGC",
            "D": "GAU",
            "E": "GAA",
            "F": "UUU",
            "G": "GGU",
            "H": "CAU",
            "I": "AUU",
            "K": "AAA",
            "L": "CUU",
            "M": "AUG",
            "N": "AAU",
            "P": "CCU",
            "Q": "CAA",
            "R": "CGU",
            "S": "UCU",
            "T": "ACU",
            "V": "GUU",
            "W": "UGG",
            "Y": "UAU",
        }

        normalized: list[str] = []
        for seq in sequences:
            if not isinstance(seq, str):
                seq = str(seq)
            seq = seq.strip().upper().replace(" ", "")

            # Decide modality by character set
            unique_chars = set(seq)
            is_nucleotide = unique_chars.issubset(nuc_letters) and len(unique_chars) > 0

            if is_nucleotide:
                # Convert DNA to RNA
                rna = seq.replace("T", "U")
                # Trim trailing incomplete codon
                trim_len = len(rna) - (len(rna) % 3)
                rna = rna[:trim_len]
                normalized.append(rna)
            else:
                # Treat as amino acid sequence and back-translate to RNA
                # Replace any non-standard AA with a reasonable placeholder (glycine)
                codons = []
                for aa in seq:
                    codons.append(aa_to_codon.get(aa, "GGU"))
                normalized.append("".join(codons))

        return normalized

    def embed_sequences(self, sequences: str | list[str], **kwargs) -> torch.Tensor:
        """
        Embed cDNA sequences using CaLM and return averaged embeddings.

        Parameters
        ----------
        sequences : str | list[str]
            One sequence or list of sequences to embed.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, embedding_dim).
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Normalize to RNA codon strings in multiples of 3
        normalized_sequences = self._normalize_to_codon_sequences(sequences)

        with torch.no_grad():
            # Process sequences individually to avoid OOM with very long sequences
            # This is less efficient but more memory-safe
            embeddings_list = []

            for seq in normalized_sequences:
                # Check if sequence is extremely long and might cause OOM
                if len(seq) > 50000:  # ~50k characters is very long
                    # For extremely long sequences, truncate to manageable size
                    # Take from the middle to avoid losing important regions
                    start_idx = len(seq) // 4
                    end_idx = start_idx + 30000  # Take 30k characters from middle
                    seq = seq[start_idx:end_idx]
                    print(
                        f"Warning: Truncated very long sequence from {len(normalized_sequences[0])} to {len(seq)} characters"
                    )

                # Process single sequence
                single_embedding = self.calm.embed_sequences([seq])
                embeddings_list.append(single_embedding)

            # Concatenate all embeddings
            embeddings = torch.cat(embeddings_list, dim=0)

        # Ensure returned tensor is on the requested device
        return embeddings.to(self.device_type)
