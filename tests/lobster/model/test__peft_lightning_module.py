from lobster.model import LobsterPEFT


class TestLobsterPMLM:
    def test_init(self):
        model = LobsterPEFT()
        num_train_params, num_total_params = model.model.get_nb_trainable_parameters()

        assert num_train_params < num_total_params
