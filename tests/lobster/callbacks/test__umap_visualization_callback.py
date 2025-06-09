import os

import pytest

try:
    from lobster.callbacks import UmapVisualizationCallback

    UMAP_INSTALLED = True
except ImportError:
    UMAP_INSTALLED = False


@pytest.mark.skipif(not UMAP_INSTALLED, reason="umap-learn not installed")
def test_umap_visualization_callback(tmp_path, dummy_model, dummy_datamodule):
    """Test that the UmapVisualizationCallback can generate visualizations without a trainer."""
    output_dir = tmp_path / "umap_test"
    os.makedirs(output_dir, exist_ok=True)

    dataloader = dummy_datamodule.val_dataloader()

    umap_callback = UmapVisualizationCallback(
        output_dir=output_dir,
        max_samples=100,
        group_by="dataset",
        group_colors={"group1": "blue", "group2": "red"},
    )

    output_file = output_dir / "test_umap.png"
    result_path = umap_callback.evaluate(dummy_model, dataloader, output_file=output_file)

    assert result_path.exists()


@pytest.mark.skipif(not UMAP_INSTALLED, reason="umap-learn not installed")
def test_umap_without_grouping(tmp_path, dummy_model, dummy_datamodule):
    """Test that the UmapVisualizationCallback works without grouping."""
    output_dir = tmp_path / "umap_test"
    os.makedirs(output_dir, exist_ok=True)

    dataloader = dummy_datamodule.val_dataloader()

    umap_callback = UmapVisualizationCallback(
        output_dir=output_dir,
        max_samples=100,
        group_by=None,  # Disable grouping
    )

    output_file = output_dir / "no_grouping_umap.png"
    result_path = umap_callback.evaluate(dummy_model, dataloader, output_file=output_file)

    assert result_path.exists()
