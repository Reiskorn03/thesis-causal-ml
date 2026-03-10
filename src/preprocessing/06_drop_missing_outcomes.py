# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) Load ver4 clean subsets and meta data
2) Drop from cand_kipa any rows with missing or negative WL_TIME
3) Drop from tx_ki any rows with missing or negative TX_TIME or GRAFT_TIME
-> record those PX_IDs and also drop their matching cand_kipa and txf_ki rows
4) Write out cleaned ver5 subsets with no missing outcome variables
"""
import os
import shutil
from pathlib import Path
import pandas as pd

DATA_DIR = Path("Data_Pipeline")
CLEAN_IN = DATA_DIR / "clean_subsets_ver4"
CLEAN_OUT = DATA_DIR / "clean_subsets_ver5"
META_IN = DATA_DIR / "meta_data" / "meta_data_ver4.csv"
META_OUT = DATA_DIR / "meta_data" / "meta_data_ver5.csv"

def main():
    os.makedirs(CLEAN_OUT, exist_ok=True)

    cand = pd.read_parquet(CLEAN_IN / "cand_kipa_subset_ver4.parquet")
    tx = pd.read_parquet(CLEAN_IN / "tx_ki_subset_ver4.parquet")
    txf = pd.read_parquet(CLEAN_IN / "txf_ki_subset_ver4.parquet")

    # drop invalid WL_TIME
    dropped_wl_missing = cand.loc[cand["WL_TIME"].isna(), "PX_ID"]
    dropped_wl_negative = cand.loc[cand["WL_TIME"] < 0, "PX_ID"]
    cand = cand.loc[cand["WL_TIME"].notna() & (cand["WL_TIME"] >= 0)]

    # drop invalid TX/GRAFT times
    mask_missing_tx = tx["TX_TIME"].isna() | tx["GRAFT_TIME"].isna()
    mask_negative_tx = (tx["TX_TIME"] < 0) | (tx["GRAFT_TIME"] < 0)
    dropped_tx_missing = tx.loc[mask_missing_tx,"PX_ID"]
    dropped_tx_negative = tx.loc[mask_negative_tx,"PX_ID"]
    tx = tx.loc[~(mask_missing_tx | mask_negative_tx)]

    # combine all dropped‐TX IDs
    dropped_tx = pd.Index(dropped_tx_missing.tolist() + dropped_tx_negative.tolist()).unique()

    # remove those same PX_IDs from cand_kipa and txf_ki
    cand = cand.loc[~cand["PX_ID"].isin(dropped_tx)]
    txf = txf.loc[~txf["PX_ID"].isin(dropped_tx)]

    # save dropped PX_IDs
    pd.Series(dropped_wl_missing.unique(), name="dropped_wl_missing").to_csv(
        CLEAN_OUT / "dropped_wl_missing_pxids.csv", index=False)
    pd.Series(dropped_wl_negative.unique(), name="dropped_wl_negative").to_csv(
        CLEAN_OUT / "dropped_wl_negative_pxids.csv", index=False)
    pd.Series(dropped_tx, name="dropped_tx").to_csv(
        CLEAN_OUT / "dropped_tx_pxids.csv", index=False)
    
    cand.to_parquet(CLEAN_OUT / "cand_kipa_subset_ver5.parquet", index=False)
    tx.to_parquet(CLEAN_OUT / "tx_ki_subset_ver5.parquet", index=False)
    txf.to_parquet(CLEAN_OUT / "txf_ki_subset_ver5.parquet", index=False)

    shutil.copy(META_IN, META_OUT)
    print("-> Wrote cleaned ver5 subsets and copied meta unchanged.")

if __name__ == "__main__":
    main()

