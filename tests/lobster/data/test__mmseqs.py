import pytest
from lobster.data import MMSeqsRunner


@pytest.mark.skip(reason="Requires mmseqs.")
class TestMMSeqsRunner:
    def test_mmseqsrunner(self):
        runner = MMSeqsRunner()

        assert runner.mmseqs_cmd == "mmseqs"
