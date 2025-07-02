import os

import pytest

from lobster.callbacks import UmapVisualizationCallback

try:
    import umap  # noqa: F401

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap not installed")
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


@pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap not installed")
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


@pytest.mark.skipif(UMAP_AVAILABLE, reason="umap is installed")
def test_umap_not_installed_error(tmp_path, dummy_model, dummy_datamodule):
    """Test that ImportError is raised when umap is not installed."""
    output_dir = tmp_path / "umap_test"
    os.makedirs(output_dir, exist_ok=True)

    dataloader = dummy_datamodule.val_dataloader()

    umap_callback = UmapVisualizationCallback(
        output_dir=output_dir,
        max_samples=100,
        group_by=None,
    )

    output_file = output_dir / "test_umap.png"

    with pytest.raises(ImportError, match="UMAP is not installed"):
        umap_callback.evaluate(dummy_model, dataloader, output_file=output_file)
