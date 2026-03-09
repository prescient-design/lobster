from __future__ import annotations

from lobster.data._streaming_optimize import CollectionProgress, sequence_is_val, sort_files_by_size


class TestSequenceIsVal:
    def test_deterministic(self):
        results = [sequence_is_val("EVQLVESGG", 0.1, seed=42) for _ in range(100)]
        assert len(set(results)) == 1

    def test_different_seeds_differ(self):
        results = {sequence_is_val("EVQLVESGG", 0.5, seed=seed) for seed in range(50)}
        assert len(results) == 2

    def test_fraction_zero_always_train(self):
        sequences = [f"SEQ{index}" for index in range(1000)]
        assert not any(sequence_is_val(sequence, 0.0, seed=42) for sequence in sequences)

    def test_fraction_approximate(self):
        sequences = [f"SEQ{index}" for index in range(10_000)]
        n_val = sum(sequence_is_val(sequence, 0.1, seed=42) for sequence in sequences)
        assert 800 < n_val < 1200


class TestSortFilesBySize:
    def test_sorts_smallest_first(self, tmp_path):
        small = tmp_path / "small.txt"
        medium = tmp_path / "medium.txt"
        large = tmp_path / "large.txt"

        small.write_text("a")
        medium.write_text("a" * 100)
        large.write_text("a" * 10_000)

        files = [str(large), str(small), str(medium)]
        assert sort_files_by_size(files) == [str(small), str(medium), str(large)]

    def test_empty_list(self):
        assert sort_files_by_size([]) == []


class TestCollectionProgress:
    def test_fresh_start(self, tmp_path):
        progress = CollectionProgress(tmp_path / "progress")
        assert progress.done_files == set()
        assert progress.sequence_count == 0

    def test_record_and_resume(self, tmp_path):
        progress_dir = tmp_path / "progress"
        progress = CollectionProgress(progress_dir, split_name="train")

        progress.record_file("file1.csv", num_sequences=10)
        progress.record_file("file2.csv", num_sequences=5)

        progress2 = CollectionProgress(progress_dir, split_name="train")
        assert progress2.done_files == {"file1.csv", "file2.csv"}
        assert progress2.sequence_count == 15

    def test_filter_remaining(self, tmp_path):
        progress_dir = tmp_path / "progress"
        progress = CollectionProgress(progress_dir)
        progress.record_file("done.csv", num_sequences=1)

        assert progress.filter_remaining(["done.csv", "new.csv"]) == ["new.csv"]

    def test_clear(self, tmp_path):
        progress_dir = tmp_path / "progress"
        progress = CollectionProgress(progress_dir, split_name="train")
        progress.record_file("file.csv", num_sequences=1)
        progress.clear()

        assert not (progress_dir / "progress_train.json").exists()

    def test_independent_split_tracking(self, tmp_path):
        progress_dir = tmp_path / "progress"
        train_progress = CollectionProgress(progress_dir, split_name="train")
        val_progress = CollectionProgress(progress_dir, split_name="val")

        train_progress.record_file("file1.csv", num_sequences=10)

        assert "file1.csv" in train_progress.done_files
        assert "file1.csv" not in val_progress.done_files
