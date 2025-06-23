#!/usr/bin/env python3
"""
Test script for the Lobster MCP Server

This script tests the basic functionality of the MCP server without
requiring the full MCP protocol setup.
"""

import asyncio
import os
import sys

# Add the parent directory to path to import lobster
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lobster_inference_server import LobsterInferenceServer


async def test_server():
    """Test the basic functionality of the Lobster MCP server"""

    print("🦞 Testing Lobster MCP Server...")
    server = LobsterInferenceServer()

    # Test protein sequence
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

    print(f"\n📋 Device: {server.device}")
    print(f"📋 Test sequence: {test_sequence[:50]}...")

    try:
        # Test 1: Get sequence representations with masked LM
        print("\n🧪 Test 1: Getting sequence representations (Masked LM)...")
        result = await server.get_sequence_representations(
            sequences=[test_sequence], model_name="lobster_24M", model_type="masked_lm", representation_type="pooled"
        )
        print(f"✅ Got embeddings with shape: {len(result['embeddings'][0])} dimensions")

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    try:
        # Test 2: Get supported concepts
        print("\n🧪 Test 2: Getting supported concepts...")
        result = await server.get_supported_concepts(model_name="cb_lobster_24M")
        concepts = result["supported_concepts"]
        print(f"✅ Found {len(concepts) if concepts else 0} supported concepts")
        if concepts and len(concepts) > 0:
            print(f"📋 First few concepts: {concepts[:5] if isinstance(concepts, list) else str(concepts)[:100]}")

        # Test 3: Get concept predictions
        print("\n🧪 Test 3: Getting concept predictions...")
        result = await server.get_sequence_concepts(sequences=[test_sequence], model_name="cb_lobster_24M")
        print(f"✅ Got concept predictions with {result['num_concepts']} concepts")

        # Test 4: Concept intervention (if concepts are available)
        if concepts and isinstance(concepts, list) and len(concepts) > 0:
            # Use first available concept for testing
            test_concept = concepts[0] if isinstance(concepts[0], str) else "gravy"
            print(f"\n🧪 Test 4: Performing concept intervention on '{test_concept}'...")
            result = await server.intervene_on_sequence(
                sequence=test_sequence,
                concept=test_concept,
                model_name="cb_lobster_24M",
                edits=3,
                intervention_type="negative",
            )
            print("✅ Intervention successful!")
            print(f"📋 Original length: {len(result['original_sequence'])}")
            print(f"📋 Modified length: {len(result['modified_sequence'])}")
            if result["edit_distance"]:
                print(f"📋 Edit distance: {result['edit_distance']}")

    except Exception as e:
        print(f"❌ Concept-related tests failed: {e}")

    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    asyncio.run(test_server())
