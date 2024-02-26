import pandas as pd
from sklearn.model_selection import train_test_split

columns_of_intrest = [
    "data_source",
    "data_source_long",
    "target",
    "affinity_antigen",
    "fv_heavy_aho",
    "fv_light_aho",
    "affinity_pkd",
    "affinity_is_binder_pkd_4",
    "fv_heavy",
    "fv_light",
]
save_path = "/data/bucket/ismaia11/affinity/data/"

######## Classification  Data#########
affinity_train = pd.read_csv(
    "s3://prescient-data-dev/raw/multi-property/affinity_is_binder_r8_dev_2023-03-10_train.csv"
)
affinity_train = affinity_train[columns_of_intrest]
affinity_train = affinity_train[affinity_train["data_source"] != "data-augmentation"]
print(1, affinity_train.shape)


######## r7 #########
columns_of_intrest = [
    "data_source",
    "target",
    "affinity_antigen",
    "fv_heavy_aho",
    "fv_light_aho",
    "affinity_pkd",
    "affinity_is_binder_pkd_4",
    "fv_heavy",
    "fv_light",
]

affinity_r7 = pd.read_csv(
    "s3://prescient-data-dev/sandbox/loukasa/affinity-data/lab/r7_cleaned.csv"
)
affinity_r7.dropna(subset=["affinity_antigen"], inplace=True)
affinity_r7 = affinity_r7[columns_of_intrest]
affinity_r7["data_source_long"] = "r7"


######## r8 #########
affinity_r8 = pd.read_csv(
    "s3://prescient-data-dev/sandbox/stantos5/data/expressors_r8_test_2023-07-11.csv"
)
affinity_r8 = affinity_r8[columns_of_intrest]
affinity_r8["data_source_long"] = "r8"

affinity_r8_train, affinity_r8_valid = train_test_split(affinity_r8, test_size=0.3)


# ########### merge data ###########
affinity_all = pd.concat(
    [affinity_train, affinity_r7, affinity_r8_train, affinity_r8_valid], axis=0
)


#### remove seeds since they are in the test dataset.
affinity_all = affinity_all[affinity_all["data_source_long"] != "seed"]


affinity_all.to_csv(save_path + "affinity_train_r10.csv", index=False)
affinity_r8_valid.to_csv(save_path + "affinity_val_r10.csv", index=False)


affinity_round = affinity_all[affinity_all["data_source"] == "genentech"]

affinity_round.to_csv(save_path + "affinity_round_train_r10.csv", index=False)
