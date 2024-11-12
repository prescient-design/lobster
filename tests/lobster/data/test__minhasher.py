import os

import pytest
from lobster.data import LobsterMinHasher


@pytest.fixture
def fasta_file(scope="session"):
    return os.path.join(os.path.dirname(__file__), "../../../test_data/query.fasta")


@pytest.mark.skip(reason="Slow.")
class TestLobsterMinHasher:
    def test_lobster_minhasher(self, fasta_file: str, tmp_path):
        minhasher = LobsterMinHasher(32, k=3)
        minhasher.deduplicate_sequences(fasta_file, tmp_path / "deduplicated.fasta")
        assert (tmp_path / "deduplicated.fasta").exists()
        assert (tmp_path / "deduplicated.fasta").stat().st_size > 0
