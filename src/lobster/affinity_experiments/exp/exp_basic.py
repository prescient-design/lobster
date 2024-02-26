import os

import torch
import torch.nn as nn


class ExpBasic(object):
    """A basic experiment class that will be inherited by all other experiments."""

    def __init__(self, args):
        """Initializes a  ExpBasic instance.
        Args:
         args: parser arguments
        """

        self.args = args
        self.model_type = self.args.model
        self.model = self._build_model()
        # self.data = self._get_dataset()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_step = self.args.init_step
        self.path_for_init = self.args.path_for_init
        if torch.cuda.device_count() > 1 and self.args.use_multi_gpu:
            self.device = torch.device("cuda:0")
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_dataset(self):
        raise NotImplementedError

    def _get_data(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
