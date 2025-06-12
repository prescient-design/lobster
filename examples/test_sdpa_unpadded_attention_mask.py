#!/usr/bin/env python3
"""
Test SDPA with unpadded architecture and attention mask propagation in Ume.

This test initializes a Ume mini model with flash attention disabled (SDPA fallback),
runs embed_sequences on short amino acid sequences, and prints the attention mask
used in the batch to verify correct creation and propagation.
"""

import torch

from lobster.model import Ume


def test_sdpa_unpadded_attention_mask():
    print("\n=== SDPA Unpadded Attention Mask Test ===\n")

    # Initialize Ume mini model with flash attention disabled (SDPA fallback)
    # model = Ume(
    #     model_name="UME_mini",
    #     use_flash_attn=False,  # Force SDPA
    #     max_length=32,        # Short for test
    # )
    s3_checkpoint_path = "s3://prescient-lobster/ume/runs/2025-06-11T02-12-16/epoch=0-step=19500-val_loss=1.0225.ckpt"
    model = Ume.load_from_checkpoint(
        s3_checkpoint_path,
        model_name="UME_small",
        use_flash_attn=False,
        max_length=32,
        strict=False,
    )

    model = model.cpu()
    model.eval()
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model use_flash_attn: {model.use_flash_attn}")
    print(f"Model config.use_fa2: {model.model.config.use_fa2}")
    print(f"Model config.padding: {model.model.config.padding}")
    print(f"Model config.use_sdpa_attn_mask: {getattr(model.model.config, 'use_sdpa_attn_mask', None)}")

    # Example short amino acid sequences
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTVRQERLKSIVRILERSK",
        "GAVLIPFYWSTCMNQDEKRH",
    ]
    print(f"Input sequences: {sequences}")

    # Patch the attention layer's forward method to print the attention mask
    # Get the first attention layer from the first encoder block
    attention_layer = model.model.model.encoder.layers[0].attn
    orig_forward = attention_layer.forward

    def patched_forward(self, hidden_states, cu_seqlens, max_seqlen, indices, attn_mask=None):
        print("\n[DEBUG] Attention mask in forward pass:")
        print(attn_mask)
        return orig_forward(hidden_states, cu_seqlens, max_seqlen, indices, attn_mask)

    attention_layer.forward = patched_forward.__get__(attention_layer)

    # Run embed_sequences (should trigger the debug print)
    with torch.no_grad():
        embeddings = model.embed_sequences(sequences, "amino_acid", aggregate=True)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings stats: mean={embeddings.mean().item():.6f}, std={embeddings.std().item():.6f}")
    print("\n=== End of Test ===\n")


if __name__ == "__main__":
    test_sdpa_unpadded_attention_mask()
