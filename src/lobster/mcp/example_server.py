#!/usr/bin/env python3
"""
Test script for the Lobster MCP Server (FastMCP version)

This script tests the basic functionality of the MCP server components
using the new FastMCP approach with native type hints instead of Pydantic models.
"""

from lobster.mcp.tools.concepts import get_sequence_concepts, get_supported_concepts
from lobster.mcp.tools.interventions import intervene_on_sequence
from lobster.mcp.tools.representations import get_sequence_representations
from lobster.mcp.tools.tool_utils import list_available_models


def test_server():
    """Test the basic functionality of the Lobster MCP server components using FastMCP approach"""

    print("🦞 Testing Lobster MCP Server Components (FastMCP version)...")

    # Get device info from list_available_models
    device_info = list_available_models()

    # Test protein sequence
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

    print(f"\n📋 Device: {device_info['device']}")
    print(f"📋 Test sequence: {test_sequence[:50]}...")

    try:
        # Test 1: Get sequence representations with masked LM
        print("\n🧪 Test 1: Getting sequence representations (FastMCP version)...")
        result = get_sequence_representations(
            model_name="lobster_24M", sequences=[test_sequence], model_type="masked_lm", representation_type="pooled"
        )
        print(f"✅ Got embeddings with shape: {len(result.embeddings[0])} dimensions")
        print(f"📋 Model used: {result.model_used}")

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    try:
        # Test 2: Get supported concepts
        print("\n🧪 Test 2: Getting supported concepts (FastMCP version)...")
        result = get_supported_concepts("cb_lobster_24M")
        concepts = result["concepts"]
        print(f"✅ Found {len(concepts) if concepts else 0} supported concepts")
        if concepts and len(concepts) > 0:
            print(f"📋 First few concepts: {concepts[:5] if isinstance(concepts, list) else str(concepts)[:100]}")

        # Test 3: Get concept predictions
        print("\n🧪 Test 3: Getting concept predictions (FastMCP version)...")
        result = get_sequence_concepts(model_name="cb_lobster_24M", sequences=[test_sequence])
        print(f"✅ Got concept predictions with {result['num_concepts']} concepts")

        # Test 4: Concept intervention (if concepts are available)
        if concepts and isinstance(concepts, list) and len(concepts) > 0:
            # Use first available concept for testing
            test_concept = concepts[0] if isinstance(concepts[0], str) else "gravy"
            print(f"\n🧪 Test 4: Performing concept intervention on '{test_concept}' (FastMCP version)...")

            result = intervene_on_sequence(
                model_name="cb_lobster_24M",
                sequence=test_sequence,
                concept=test_concept,
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

    print("\n🎉 FastMCP Testing complete!")


if __name__ == "__main__":
    test_server()
