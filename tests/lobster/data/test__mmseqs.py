from lobster.data import MMSeqsRunner


class TestMMSeqsRunner:
    def test_mmseqsrunner(self):
        runner = MMSeqsRunner()

        assert runner.mmseqs_cmd == "mmseqs"
