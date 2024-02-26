import importlib
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
from prescient.transforms import AutoTokenizerTransform
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

from lobster.tokenization import PmlmTokenizerTransform

# from prescient.datasets import AffinityDataset, GREDAffinityDataset
T = TypeVar("T")


class Affinity(Dataset):
    def __init__(
        self,
        args,
        flag: str = "train",
        use_target_transform_fn: bool = True,
        use_transform_fn: bool = True,
        max_length: int = 512,
        return_details: bool = False,
    ):

        self.args = args
        self._use_transform_fn = use_transform_fn
        self.use_target_transform_fn = use_target_transform_fn
        self._max_length = max_length
        self._flag = flag
        self._tokenizer_dir = "pmlm_tokenizer"

        if self._use_transform_fn:
            if "RLM" in self.args.model or "PLM" in self.args.model:
                # self._transform_fn = AutoTokenizerTransform(
                #                 "facebook/esm2_t6_8M_UR50D",
                #                 padding="max_length",
                #                 truncation=True,
                #                 max_length=self._max_length,
                #                 )

                path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
                self._transform_fn = PmlmTokenizerTransform(
                    path, padding="max_length", truncation=True, max_length=self._max_length
                )

            else:
                self._transform_fn = AutoTokenizerTransform(
                    "facebook/" + self.args.model,
                    padding="max_length",
                    truncation=True,
                    max_length=self._max_length,
                )

        else:
            self._transform_fn = None

        if self._use_transform_fn and not self.args.get("in_production", False):
            if self.args.task == "classification":
                if "s3" in self.args.val_data:
                    data = pd.read_csv(self.args.data_folder + args.val_data)
                else:
                    data = pd.read_csv(self.args.data_folder + args.val_data)
                self._target_transform_fn = OneHotEncoder().fit(data[[args.target_value]])

        else:
            self._target_transform_fn = None

        if flag == "train":
            print("Loading training data")
            if "s3" in self.args.train_data:
                self._data = pd.read_csv(self.args.train_data)
            else:
                self._data = pd.read_csv(self.args.data_folder + self.args.train_data)
        elif "val" in flag:
            print("Loading validation data")
            if "s3" in self.args.val_data:
                self._data = pd.read_csv(self.args.val_data)
            else:
                self._data = pd.read_csv(self.args.data_folder + self.args.val_data)
        else:
            print("Loading test data")
            if "s3" in self.args.test_data:
                if "parquet" in self.args.test_data:
                    self._data = pd.read_parquet(self.args.test_data, engine="fastparquet")
                else:
                    self._data = pd.read_csv(self.args.test_data)
            else:
                self._data = pd.read_csv(self.args.data_folder + self.args.test_data)

        self.antibody = [
            h + l for h, l in zip(self._data[self.args.vh_seq], self._data[self.args.vl_seq])
        ]
        self.antigen = self._data[self.args.target_seq].values
        if flag == "train" or flag == "val":
            if self._use_transform_fn:
                self.y = self._target_transform_fn.transform(
                    self._data[[self.args.target_value]]
                ).toarray()
            else:
                self.y = self._data[[self.args.target_value]].values
            self.y = torch.tensor(self.y.astype(np.float32))

    def __getitem__(self, idx):
        if self._transform_fn:
            antibody = self._transform_fn(self.antibody[idx])

            antigen = self._transform_fn(self.antigen[idx])

        else:
            antibody = self.antibody[idx]
            antigen = self.antigen[idx]
        if "pred" not in self._flag:
            return antibody, antigen, self.y[idx]
        else:
            return antibody, antigen

    def __len__(self):
        return len(self._data)
