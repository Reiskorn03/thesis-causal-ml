# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) Load ver7 `cand_kipa` & merged `tx_ki`
2) Stratified 70/30 train/test split on (T_365, Delta)
3) Apply the same split to `tx_ki` by PX_ID
4) Write out train/test subsets under relative paths
5) Copy meta_data_ver7.csv unchanged into the split directory
"""
import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = Path("Data_Pipeline")
CLEAN_IN = DATA_DIR / "clean_subsets_ver7"
SPLIT_OUT = DATA_DIR / "splits_ver7"

os.makedirs(SPLIT_OUT, exist_ok=True)

def main():
    cand = pd.read_parquet(CLEAN_IN / "cand_kipa_subset_ver7.parquet")
    tx = pd.read_parquet(CLEAN_IN / "tx_ki_subset_ver7.parquet")

    # build stratification key: T_365_Delta
    strat_key = cand["T_365"].astype(str) + "_" + cand["Delta"].astype(str)

    # stratified shuffle split (70% train, 30% test)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42
    )
    train_idx, test_idx = next(splitter.split(cand, strat_key))

    train_cand = cand.iloc[train_idx].reset_index(drop=True)
    test_cand = cand.iloc[test_idx] .reset_index(drop=True)

    # apply same PX_ID split to tx_ki
    train_ids = set(train_cand["PX_ID"])
    test_ids = set(test_cand["PX_ID"])

    train_tx = tx[tx["PX_ID"].isin(train_ids)].reset_index(drop=True)
    test_tx = tx[tx["PX_ID"].isin(test_ids)] .reset_index(drop=True)

    train_cand.to_parquet(SPLIT_OUT / "cand_kipa_train_ver7.parquet", index=False)
    test_cand .to_parquet(SPLIT_OUT / "cand_kipa_test_ver7.parquet",  index=False)

    train_tx.to_parquet(SPLIT_OUT / "tx_ki_train_ver7.parquet", index=False)
    test_tx .to_parquet(SPLIT_OUT / "tx_ki_test_ver7.parquet",  index=False)

    print("-> Written train/test splits.")

if __name__ == "__main__":
    main()
