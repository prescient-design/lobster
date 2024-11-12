import subprocess


class TestLobsterCmdline:
    def test_train(self):
        output = subprocess.check_output("lobster_train -h", shell=True)

        output = output.splitlines()

        assert output[0] == b"_train is powered by Hydra."

        assert len(output) > 1

    def test_embed(self):
        output = subprocess.check_output("lobster_embed -h", shell=True)
        output = output.splitlines()

        assert output[0] == b"_embed is powered by Hydra."

        assert len(output) > 1
