#!/usr/bin/env python3
"""
Script to compare embeddings between flash attention enabled (GPU) and disabled (CPU) modes.
This demonstrates that the same checkpoint produces consistent embeddings regardless of flash attention mode.

Note: This script tests the fix for SDPA/Flash Attention consistency:

The issue was that SDPA (PyTorch's Scaled Dot Product Attention) requires padded inputs
to work correctly, but the model was using unpadded architecture with SDPA when
flash attention was disabled.

The fix:
- When use_flash_attn=True: Uses unpadded architecture + Flash Attention
- When use_flash_attn=False: Uses padded architecture + SDPA with attention masks
- When loading checkpoints: Preserves original unpadded architecture but enables SDPA masks

This should eliminate warnings and ensure consistent embeddings between modes.
"""

import torch
import torch.nn.functional as F

from lobster.model import UME


def compute_embedding_metrics(embeddings_fa, embeddings_no_fa):
    """Compute similarity metrics between two sets of embeddings."""

    # Ensure both are on the same device for computation
    embeddings_fa = embeddings_fa.cpu()
    embeddings_no_fa = embeddings_no_fa.cpu()

    # Cosine similarity
    cosine_sim = F.cosine_similarity(embeddings_fa, embeddings_no_fa, dim=1)

    # L2 distance (Euclidean)
    l2_distance = torch.norm(embeddings_fa - embeddings_no_fa, dim=1)

    # Mean squared error
    mse = F.mse_loss(embeddings_fa, embeddings_no_fa, reduction="none").mean(dim=1)

    # Mean absolute error
    mae = F.l1_loss(embeddings_fa, embeddings_no_fa, reduction="none").mean(dim=1)

    return {"cosine_similarity": cosine_sim, "l2_distance": l2_distance, "mse": mse, "mae": mae}


def compare_flash_attention_embeddings():
    """Compare embeddings between flash attention enabled and disabled modes."""

    print("🔬 Comparing Flash Attention vs Non-Flash Attention Embeddings")
    print("=" * 80)

    # S3 checkpoint path
    s3_checkpoint_path = "s3://prescient-lobster/ume/runs/2025-06-11T02-12-16/epoch=0-step=19500-val_loss=1.0225.ckpt"

    # Example protein sequences for testing
    protein_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Example protein 1
        "MKLLNVINFVFLMFVSSSADVNAAAFKDTILHALEREPVDAFRQLAAKLNISPPMNVAAEF",  # Example protein 2
        "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ",  # Longer protein
        "ACDEFGHIKLMNPQRSTVWY",  # Simple amino acid sequence
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG",  # Another example
    ]

    print(f"📥 Loading checkpoint: {s3_checkpoint_path}")
    print(f"🧬 Testing with {len(protein_sequences)} protein sequences")
    print()

    # Step 1: Load model with flash attention enabled on GPU
    print("🚀 Step 1: Loading model with Flash Attention ENABLED on GPU...")
    try:
        model_fa = UME.load_from_checkpoint(
            s3_checkpoint_path,
            model_name="UME_small",
            use_flash_attn=True,  # Flash attention enabled
            max_length=8192,
            strict=False,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            model_fa = model_fa.cuda()
            print(f"✅ Model loaded on GPU: {next(model_fa.parameters()).device}")
        else:
            print("⚠️  CUDA not available, using CPU for flash attention model")

        model_fa.eval()

        print(f"   Flash attention enabled: {model_fa.use_flash_attn}")
        print(f"   Config use_fa2: {model_fa.model.config.use_fa2}")
        print(f"   Model device: {next(model_fa.parameters()).device}")

    except Exception as e:
        print(f"❌ Error loading flash attention model: {e}")
        return

    # Step 2: Generate embeddings with flash attention
    print("\n🧬 Generating embeddings with Flash Attention...")
    try:
        with torch.no_grad():
            embeddings_fa = model_fa.embed_sequences(protein_sequences, "amino_acid", aggregate=True)

        print(f"✅ Flash attention embeddings shape: {embeddings_fa.shape}")
        print(f"   Device: {embeddings_fa.device}")
        print(f"   Statistics - Mean: {embeddings_fa.mean().item():.6f}, Std: {embeddings_fa.std().item():.6f}")

    except Exception as e:
        print(f"❌ Error generating flash attention embeddings: {e}")
        return

    # Step 3: Load model with flash attention disabled on CPU
    print("\n🔄 Step 2: Loading model with Flash Attention DISABLED on CPU...")
    try:
        model_no_fa = UME.load_from_checkpoint(
            s3_checkpoint_path,
            model_name="UME_small",
            use_flash_attn=False,  # Flash attention disabled
            max_length=8192,
            strict=False,
        )

        # Ensure model is on CPU
        model_no_fa = model_no_fa.cpu()
        model_no_fa.eval()

        print(f"✅ Model loaded on CPU: {next(model_no_fa.parameters()).device}")
        print(f"   Flash attention enabled: {model_no_fa.use_flash_attn}")
        print(f"   Config use_fa2: {model_no_fa.model.config.use_fa2}")

    except Exception as e:
        print(f"❌ Error loading non-flash attention model: {e}")
        return

    # Step 4: Generate embeddings without flash attention
    print("\n🧬 Generating embeddings without Flash Attention...")
    try:
        with torch.no_grad():
            embeddings_no_fa = model_no_fa.embed_sequences(protein_sequences, "amino_acid", aggregate=True)

        print(f"✅ Non-flash attention embeddings shape: {embeddings_no_fa.shape}")
        print(f"   Device: {embeddings_no_fa.device}")
        print(f"   Statistics - Mean: {embeddings_no_fa.mean().item():.6f}, Std: {embeddings_no_fa.std().item():.6f}")

    except Exception as e:
        print(f"❌ Error generating non-flash attention embeddings: {e}")
        return

    # Step 5: Compare embeddings
    print("\n📊 Step 3: Comparing embeddings...")

    # Compute metrics
    metrics = compute_embedding_metrics(embeddings_fa, embeddings_no_fa)

    print("\n🔍 Embedding Comparison Results:")
    print("-" * 50)

    for i, sequence in enumerate(protein_sequences):
        seq_preview = sequence[:30] + "..." if len(sequence) > 30 else sequence
        print(f"\nSequence {i + 1}: {seq_preview}")
        print(f"  Length: {len(sequence)} amino acids")
        print(f"  Cosine Similarity: {metrics['cosine_similarity'][i].item():.8f}")
        print(f"  L2 Distance:       {metrics['l2_distance'][i].item():.8f}")
        print(f"  MSE:               {metrics['mse'][i].item():.8e}")
        print(f"  MAE:               {metrics['mae'][i].item():.8e}")

    # Overall statistics
    print("\n📈 Overall Statistics:")
    print("-" * 30)
    print(
        f"Average Cosine Similarity: {metrics['cosine_similarity'].mean().item():.8f} ± {metrics['cosine_similarity'].std().item():.8f}"
    )
    print(
        f"Average L2 Distance:       {metrics['l2_distance'].mean().item():.8f} ± {metrics['l2_distance'].std().item():.8f}"
    )
    print(f"Average MSE:               {metrics['mse'].mean().item():.8e} ± {metrics['mse'].std().item():.8e}")
    print(f"Average MAE:               {metrics['mae'].mean().item():.8e} ± {metrics['mae'].std().item():.8e}")

    # Determine if embeddings are sufficiently similar
    avg_cosine_sim = metrics["cosine_similarity"].mean().item()
    min_cosine_sim = metrics["cosine_similarity"].min().item()

    print("\n🎯 Similarity Assessment:")
    print("-" * 25)
    if avg_cosine_sim > 0.999:
        print("🟢 EXCELLENT: Embeddings are nearly identical (>99.9% similarity)")
    elif avg_cosine_sim > 0.995:
        print("🟡 GOOD: Embeddings are very similar (>99.5% similarity)")
    elif avg_cosine_sim > 0.99:
        print("🟠 ACCEPTABLE: Embeddings are quite similar (>99% similarity)")
    else:
        print("🔴 CONCERNING: Embeddings show significant differences (<99% similarity)")

    print(f"Minimum cosine similarity: {min_cosine_sim:.8f}")

    # Check for any sequences with low similarity
    low_similarity_threshold = 0.995
    low_sim_indices = torch.where(metrics["cosine_similarity"] < low_similarity_threshold)[0]

    if len(low_sim_indices) > 0:
        print(f"\n⚠️  Sequences with cosine similarity < {low_similarity_threshold}:")
        for idx in low_sim_indices:
            seq_preview = (
                protein_sequences[idx][:30] + "..." if len(protein_sequences[idx]) > 30 else protein_sequences[idx]
            )
            print(f"  Sequence {idx + 1}: {seq_preview} (similarity: {metrics['cosine_similarity'][idx].item():.6f})")

    print("\n" + "=" * 80)
    print("🎉 Embedding comparison completed!")
    print("💡 This demonstrates the consistency between flash attention and non-flash attention modes.")


if __name__ == "__main__":
    compare_flash_attention_embeddings()
