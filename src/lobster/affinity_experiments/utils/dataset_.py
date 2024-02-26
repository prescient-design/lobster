from typing import TypeVar

import pandas as pd
from prescient.datasets import GREDAffinityDataset
from prescient.transforms import AutoTokenizerTransform
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

T = TypeVar("T")


class Affinity_PLM(Dataset):
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

        self.columns = [self.args.target_seq, self.args.vh_seq, self.args.vl_seq]
        self.target_columns = [self.args.target_value]

        if self._use_transform_fn:
            self._transform_fn = AutoTokenizerTransform(
                "facebook/esm2_t6_8M_UR50D",
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
            )
        else:
            self._transform_fn = None

        if self._use_transform_fn:

            self.data = pd.read_csv(
                "/homefs/home/ismaia11/affinity/manifold-sampler-pytorch/oracle/moe_affinity/data/"
                + args.test_data
            )
            self.target_transform_fn = OneHotEncoder().fit(self.data[[args.target_value]])

        else:
            self.target_transform_fn = None

        aff = GREDAffinityDataset(
            self.args.data_folder,
            columns=self.columns,
            target_columns=self.target_columns,
            download=True,
            transform_fn=self._transform_fn,
            target_transform_fn=self.target_transform_fn,
        )  # all training

        print(len(aff))


# columns: Optional[Sequence[str]] = None,
#         target_columns: Optional[Sequence[str]] = None,
#         train: bool = True,
#         download: bool = False,
#         transform_fn: Union[Callable, Transform, None] = None,
#         target_transform_fn: Union[Callable, Transform, None] = None,

# gred= GREDAffinityDataset(self.args.data_folder,

#  download=True, train=False) #this is just test (iid / currently NOT round aware.)


#     self.args=args
#     self._use_transform_fn = use_transform_fn
#     self._max_length=max_length
#     self.label_encoding=label_encoding
#     self.return_details=return_details
#     if self._use_transform_fn:
#         self._transform_fn = AutoTokenizerTransform(
#                         "facebook/esm2_t6_8M_UR50D",
#                         padding="max_length",
#                         truncation=True,
#                         max_length=self._max_length,
#                         )
#     else:
#         self._transform_fn=None


#     if(flag=="train"):
#         print("Loading training data")
#         self.data = pd.read_csv(self.args.data_folder+self.args.train_data)
#     elif("val" in flag):
#         print("Loading validation data")
#         self.data =  pd.read_csv(self.args.data_folder+self.args.val_data)
#     else:
#         print("Loading test data")
#         self.data = pd.read_csv(self.args.data_folder+self.args.test_data)


#     if(self.label_encoding):
#         self.y= self.label_encoding.transform(self.data[[self.args.target_value]]).toarray()
#     self.y = torch.tensor(self.y.astype(np.float32))


#     self.antibody  = [h + l for h, l in zip(self.data[self.args.vh_seq_aho], self.data[self.args.vl_seq_aho])]
#     self.antigen= self.data[self.args.target_seq].values

#     if(self.return_details):
#         self.data_source=  self.data["data_source"].values
#         self.data_source_long=self.data["data_source_long"].values

# def __getitem__(self, idx):
#     if self._transform_fn:
#         antibody = self._transform_fn(self.antibody[idx])
#         antibody=antibody["input_ids"]
#         antigen = self._transform_fn(self.antigen[idx])
#         antigen=antigen["input_ids"]


#     else:
#         antibody=self.antibody[idx]
#         antigen=self.antigen[idx]

#     if(self.return_details):
#         return antibody,antigen,self.y[idx],self.data_source[idx],self.data_source_long[idx]
#     return antibody,antigen,self.y[idx]
# def __len__(self):
#     return len(self.data)
