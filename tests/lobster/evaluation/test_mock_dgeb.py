#!/usr/bin/env python3
"""Test script for mock DGEB evaluation."""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lobster.evaluation.dgeb_mock_runner import run_mock_evaluation, generate_mock_report


def test_mock_evaluation(tmp_path):
    """Test the mock DGEB evaluation with a small subset of tasks."""

    print("Testing mock DGEB evaluation...")

    # Run evaluation on a small subset of protein tasks
    results = run_mock_evaluation(
        model_name="test-mock-ume",
        modality="protein",
        tasks=["ec_classification"],  # Just one task for testing
        output_dir=tmp_path / "test_mock_results",
        batch_size=4,
        max_seq_length=512,
        l2_norm=True,
        pool_type="mean",
        seed=42,
        embed_dim=256,  # Smaller dimension for testing
        num_layers=6,  # Fewer layers for testing
    )

    print("Evaluation completed successfully!")
    print(f"Model name: {results['model_name']}")
    print(f"Modality: {results['modality']}")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Tasks run: {results['tasks_run']}")

    # Check that we got results
    assert len(results["results"]) > 0, "No results returned"

    # Check that the model metadata indicates it's a mock
    assert results["model_metadata"]["is_mock"], "Model should be marked as mock"
    assert results["model_metadata"]["num_params"] == 0, "Mock model should have 0 parameters"

    # Generate a report
    report_dir = tmp_path / "test_mock_results" / "test_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    generate_mock_report(results, report_dir)

    print(f"Report generated at: {report_dir / 'mock_evaluation_report.md'}")
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mock_evaluation()
