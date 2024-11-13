# Lobster Model Inference

import torch
from lobster.model import LobsterCBMPMLM, LobsterPMLM

# Define the test protein sequence
test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"

# Determine the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the LobsterPMLM model
lobster = LobsterPMLM("asalam91/lobster_24M").to(device)
lobster.eval()

# Get MLM representation
mlm_representation = lobster.sequences_to_latents([test_protein])[-1]
cls_token_mlm_representation = mlm_representation[:, 0, :]
pooled_mlm_representation = torch.mean(mlm_representation, dim=1)

# Load the LobsterCBMPMLM model
cb_lobster = LobsterCBMPMLM("asalam91/cb_lobster_24M").to(device)
cb_lobster.eval()

# Get CB MLM representation
cb_mlm_representation = cb_lobster.sequences_to_latents([test_protein])[-1]
cls_token_cb_mlm_representation = cb_mlm_representation[:, 0, :]
pooled_cb_mlm_representation = torch.mean(cb_mlm_representation, dim=1)

# Get protein concepts
test_protein_concepts = cb_lobster.sequences_to_concepts([test_protein])[-1]
test_protein_concepts_emb = cb_lobster.sequences_to_concepts_emb([test_protein])[-1][
    0
]  # All of the known concepts are the same for all tokens...
test_protein_concepts_unknown_emb = cb_lobster.sequences_to_concepts_emb([test_protein])[-1]

# Print results
print("CLS token MLM representation:", cls_token_mlm_representation.shape)
print("Pooled MLM representation:", pooled_mlm_representation.shape)
print("CLS token CB MLM representation:", cls_token_cb_mlm_representation.shape)
print("Pooled CB MLM representation:", pooled_cb_mlm_representation.shape)
print("Test protein concepts:", test_protein_concepts.shape)
print("Test protein concepts embedding:", test_protein_concepts_emb.shape)
print("Test protein unknown concepts embedding:", test_protein_concepts_unknown_emb.shape)
