import subprocess


class TestPplmCmdline:
    def test_train(self):
        output = subprocess.check_output("pplm_train -h", shell=True)

        output = output.splitlines()

        assert output[0] == b"_train is powered by Hydra."

        assert len(output) > 1

    def test_embed(self):
        output = subprocess.check_output("pplm_embed -h", shell=True)
        output = output.splitlines()

        assert output[0] == b"_embed is powered by Hydra."

        assert len(output) > 1
