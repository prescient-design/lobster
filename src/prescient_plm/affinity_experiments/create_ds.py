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


########### merge data ###########
affinity_all = pd.concat([affinity_train, affinity_r7], axis=0)


########### create a validation dataset using round only ###########


other_datasource = affinity_all[affinity_all["data_source"] != "genentech"]
round_data = affinity_all[affinity_all["data_source"] == "genentech"]

round_train, round_valid = train_test_split(round_data, test_size=0.1)


training_data = pd.concat([other_datasource, round_train], axis=0)
print(training_data.shape, save_path + "affinity_train.csv")
training_data.to_csv(save_path + "affinity_train.csv", index=False)


round_valid.to_csv(save_path + "affinity_valid_round_iid.csv", index=False)
print(round_valid.shape, save_path + "affinity_valid_round_iid.csv")
######## r8 #########
affinity_r8 = pd.read_csv(
    "s3://prescient-data-dev/sandbox/stantos5/data/expressors_r8_test_2023-07-11.csv"
)
affinity_r8 = affinity_r8[columns_of_intrest]
affinity_r8["data_source_long"] = "r8"

affinity_r8.to_csv(save_path + "affinity_test_r8.csv", index=False)
print(affinity_r8.shape, save_path + "affinity_test_r8.csv")
