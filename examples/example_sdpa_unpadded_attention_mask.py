#!/usr/bin/env python3
"""
Test SDPA with unpadded architecture and attention mask propagation in UME.

This test initializes a UME mini model with flash attention disabled (SDPA fallback),
runs embed_sequences on short amino acid sequences, and prints the attention mask
used in the batch to verify correct creation and propagation.
"""

import torch
import torch.nn.functional as F

from lobster.model import UME


def test_sdpa_unpadded_attention_mask():
    print("\n=== SDPA Unpadded Attention Mask Test ===\n")

    # s3_checkpoint_path = "s3://prescient-lobster/ume/runs/2025-06-11T02-12-16/epoch=0-step=19500-val_loss=1.0225.ckpt"
    s3_checkpoint_path = "s3://prescient-lobster/ume/runs/2025-06-17T13-45-59/epoch=0-step=2500-val_loss=0.8203.ckpt"  # with Rope embedding

    # Test both CPU and GPU configurations
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")

    for device in devices:
        print(f"\n=== Testing on {device.upper()} ===")

# Removed commented-out block for cleaner and more maintainable code.
        model = UME.load_from_checkpoint(s3_checkpoint_path, device=device)

        model = model.to(device)
        model.eval()
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model use_flash_attn: {model.use_flash_attn}")
        print(f"Model config.use_fa2: {model.model.config.use_fa2}")
        print(f"Model config.padding: {model.model.config.padding}")
        print(f"Model config.use_sdpa_attn_mask: {getattr(model.model.config, 'use_sdpa_attn_mask', None)}")

        # Print attention layer configuration
        print("\n[DEBUG] Attention layer configuration:")
        print(f"Attention layer type: {model.model.config.attention_layer}")
        print(f"Initial attention layer type: {getattr(model.model.config, 'initial_attention_layer', None)}")
        print(f"Number of initial layers: {model.model.config.num_initial_layers}")
        print(f"Rotary embedding dimension: {model.model.config.rotary_emb_dim}")
        print(f"Rotary embedding base: {model.model.config.rotary_emb_base}")
        print(f"Rotary embedding interleaved: {model.model.config.rotary_emb_interleaved}")

        # Example short amino acid sequences
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTVRQERLKSIVRILERSK",
            "GAVLIPFYWSTCMNQDEKRH",
        ]
        print(f"Input sequences: {sequences}")

        # Get the first attention layer from the first encoder block
        attention_layer = model.model.model.encoder.layers[0].attn
        orig_forward = attention_layer.forward

        # Create a patched forward function based on the architecture
        if model.model.config.padding == "unpadded":

            def patched_forward(
                self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                max_seqlen: int,
                indices: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                print("\n[DEBUG] Attention mask in forward pass:")
                print(attn_mask)

                # Print attention layer type
                print(f"\n[DEBUG] Attention layer type: {type(self).__name__}")

                # Print rotary embedding configuration if it exists
                if hasattr(self, "rotary_emb"):
                    print("\n[DEBUG] Rotary embedding configuration:")
                    print(f"Type: {type(self.rotary_emb).__name__}")
                    print(f"Dimension: {self.rotary_emb.dim}")
                    print(f"Base: {self.rotary_emb.base}")
                    print(f"Interleaved: {self.rotary_emb.interleaved}")

                    # Get the cos and sin values
                    if hasattr(self.rotary_emb, "_cos_cached") and self.rotary_emb._cos_cached is not None:
                        print("\n[DEBUG] Rotary embedding cached values:")
                        print(f"Cos shape: {self.rotary_emb._cos_cached.shape}")
                        print(f"Cos mean: {self.rotary_emb._cos_cached.mean().item():.6f}")
                        print(f"Cos std: {self.rotary_emb._cos_cached.std().item():.6f}")
                        print(f"Cos first few values: {self.rotary_emb._cos_cached[0, 0, :5, 0]}")

                        print(f"\nSin shape: {self.rotary_emb._sin_cached.shape}")
                        print(f"Sin mean: {self.rotary_emb._sin_cached.mean().item():.6f}")
                        print(f"Sin std: {self.rotary_emb._sin_cached.std().item():.6f}")
                        print(f"Sin first few values: {self.rotary_emb._sin_cached[0, 0, :5, 0]}")

                return orig_forward(hidden_states, cu_seqlens, max_seqlen, indices, attn_mask)
        else:

            def patched_forward(
                self,
                hidden_states: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                print("\n[DEBUG] Attention mask in forward pass:")
                print(attn_mask)

                # Print attention layer type
                print(f"\n[DEBUG] Attention layer type: {type(self).__name__}")

                # Print rotary embedding configuration if it exists
                if hasattr(self, "rotary_emb"):
                    print("\n[DEBUG] Rotary embedding configuration:")
                    print(f"Type: {type(self.rotary_emb).__name__}")
                    print(f"Dimension: {self.rotary_emb.dim}")
                    print(f"Base: {self.rotary_emb.base}")
                    print(f"Interleaved: {self.rotary_emb.interleaved}")

                    # Get the cos and sin values
                    if hasattr(self.rotary_emb, "_cos_cached") and self.rotary_emb._cos_cached is not None:
                        print("\n[DEBUG] Rotary embedding cached values:")
                        print(f"Cos shape: {self.rotary_emb._cos_cached.shape}")
                        print(f"Cos mean: {self.rotary_emb._cos_cached.mean().item():.6f}")
                        print(f"Cos std: {self.rotary_emb._cos_cached.std().item():.6f}")
                        print(f"Cos first few values: {self.rotary_emb._cos_cached[0, 0, :5, 0]}")

                        print(f"\nSin shape: {self.rotary_emb._sin_cached.shape}")
                        print(f"Sin mean: {self.rotary_emb._sin_cached.mean().item():.6f}")
                        print(f"Sin std: {self.rotary_emb._sin_cached.std().item():.6f}")
                        print(f"Sin first few values: {self.rotary_emb._sin_cached[0, 0, :5, 0]}")

                return orig_forward(hidden_states, attn_mask)

        attention_layer.forward = patched_forward.__get__(attention_layer)

        # Run embed_sequences (should trigger the debug print)
        with torch.no_grad():
            embeddings = model.embed_sequences(sequences, "amino_acid", aggregate=True)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embeddings stats: mean={embeddings.mean().item():.6f}, std={embeddings.std().item():.6f}")

        # Store embeddings for comparison
        if device == "cpu":
            cpu_embeddings = embeddings
        else:
            gpu_embeddings = embeddings

    # Compare embeddings if both CPU and GPU were tested
    if len(devices) > 1:
        print("\n=== Comparing CPU vs GPU Embeddings ===")
        # Absolute differences
        diff = (cpu_embeddings - gpu_embeddings.cpu()).abs()
        print(f"Max difference: {diff.max().item():.6f}")
        print(f"Mean difference: {diff.mean().item():.6f}")
        print(f"Std of differences: {diff.std().item():.6f}")

        # Cosine similarity
        cpu_embeddings_norm = F.normalize(cpu_embeddings, p=2, dim=-1)
        gpu_embeddings_norm = F.normalize(gpu_embeddings.cpu(), p=2, dim=-1)
        cosine_sim = (cpu_embeddings_norm * gpu_embeddings_norm).sum(dim=-1)
        print("\nCosine similarity between CPU and GPU embeddings:")
        print(f"Mean cosine similarity: {cosine_sim.mean().item():.6f}")
        print(f"Min cosine similarity: {cosine_sim.min().item():.6f}")
        print(f"Max cosine similarity: {cosine_sim.max().item():.6f}")

        # Print per-sequence cosine similarities
        print("\nPer-sequence cosine similarities:")
        for i, seq in enumerate(sequences):
            print(f"Sequence {i + 1} ('{seq}'): {cosine_sim[i].item():.6f}")

    print("\n=== End of Test ===\n")


if __name__ == "__main__":
    test_sdpa_unpadded_attention_mask()
