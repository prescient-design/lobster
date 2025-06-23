from lobster.model import UME

ume = UME().to("cuda")
ume.eval()

# Example protein sequences
protein_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Sample protein fragment
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",  # Hemoglobin beta chain
]

# Get embeddings for protein sequences
protein_embeddings = ume.embed_sequences(protein_sequences, modality="amino_acid")

print(f"Embeddings stats: mean={protein_embeddings.mean().item():.6f}, std={protein_embeddings.std().item():.6f}")
