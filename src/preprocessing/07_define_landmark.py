# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) Load ver5 clean subsets and meta
2) Compute each patients total_time & total_event
3) Apply 1-year landmark (L=365 days): keep only total_time ≥ L
4) Define T_365, post-landmark survival Y & event delta
5) Filter tx_ki & txf_ki to the same PX_IDs
6) Write out ver6 subsets 
"""
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("Data_Pipeline")
CLEAN_IN = DATA_DIR / "clean_subsets_ver5"
CLEAN_OUT = DATA_DIR / "clean_subsets_ver6"
META_IN = DATA_DIR / "meta_data" / "meta_data_ver5.csv"
META_OUT = DATA_DIR / "meta_data" / "meta_data_ver6.csv"

L = 365  # landmark horizon in days ->defined via EDA

def main():
    os.makedirs(CLEAN_OUT, exist_ok=True)

    cand = pd.read_parquet(CLEAN_IN/"cand_kipa_subset_ver5.parquet")
    tx = pd.read_parquet(CLEAN_IN/"tx_ki_subset_ver5.parquet")
    txf  = pd.read_parquet(CLEAN_IN/"txf_ki_subset_ver5.parquet")

    # merge transplant times/events into cand
    tx_small = tx[["PX_ID", "TX_TIME", "TX_EVENT"]]
    cand = cand.merge(tx_small, on="PX_ID", how="left")

    # compute total_time / total_event
    # for never-treated: just WL // for treated: WL + post-tx
    cand["total_time"] = np.where(
        cand["T"] == 1,
        cand["WL_TIME"] + cand["TX_TIME"],
        cand["WL_TIME"]
    )
    cand["total_event"] = np.where(
        cand["T"] == 1,
        cand["TX_EVENT"],
        cand["WL_EVENT"]
    )

    # apply landmark: keep only those with total_time >= L
    mask = cand["total_time"] >= L
    dropped = cand.loc[~mask, "PX_ID"].unique()
    cand = cand.loc[mask].copy()

    # define landmarked treatment & outcome
    # T_365 = 1 if transplanted by day L, else 0
    cand["T_365"] = np.where(
        (cand["T"] == 1) & (cand["WL_TIME"] <= L),
        1,
        0
    )

    # define landmarked treatment & outcome (three-way split)
    # 1. Early tx -> post-transplant follow-up
    mask_early = (cand["T"] == 1) & (cand["WL_TIME"] <= L)
    cand.loc[mask_early, "Y"] = cand["total_time"] - L
    cand.loc[mask_early, "Delta"] = cand["total_event"]

    # 2. Late tx -≥ censor at transplant (censoring indicator = 0)
    mask_late  = (cand["T"] == 1) & (cand["WL_TIME"] > L)
    cand.loc[mask_late, "Y"] = cand.loc[mask_late, "WL_TIME"] - L
    cand.loc[mask_late, "Delta"] = 0

    # 3. Never tx  -> use waiting-list follow-up
    mask_never = cand["T"] == 0
    cand.loc[mask_never, "Y"] = cand.loc[mask_never, "WL_TIME"] - L
    cand.loc[mask_never, "Delta"] = cand.loc[mask_never, "WL_EVENT"]

    # drop helper columns
    cand = cand.drop(columns=["total_time", "total_event", "TX_TIME", "TX_EVENT"])

    # synchronize tx & txf
    keep_ids = cand["PX_ID"]
    tx  = tx.loc[ tx["PX_ID"].isin(keep_ids) ].copy()
    txf = txf.loc[txf["PX_ID"].isin(keep_ids)].copy()

    cand.to_parquet(CLEAN_OUT/"cand_kipa_subset_ver6.parquet", index=False)
    tx.to_parquet(CLEAN_OUT/"tx_ki_subset_ver6.parquet", index=False)
    txf.to_parquet(CLEAN_OUT/"txf_ki_subset_ver6.parquet", index=False)

    # update meta
    meta = pd.read_csv(META_IN, dtype=str)

    # define new feature rows
    new_rows = [
        {"Variable":"T_365", "group":"c_flag_YNU", "subset":"cand_kipa", "coerce_done":"1"},
        {"Variable":"Y", "group":"a_continuous_numeric", "subset":"cand_kipa", "coerce_done":"1"},
        {"Variable":"Delta", "group":"c_flag_YNU", "subset":"cand_kipa", "coerce_done":"1"},
    ]
    new_df = pd.DataFrame(new_rows)

    for col in meta.columns:
        if col not in new_df.columns:
            new_df[col] = ""

    new_df = new_df[meta.columns]
    meta = pd.concat([meta, new_df], ignore_index=True)
    meta.to_csv(META_OUT, index=False)

    print("-> ver6 subsets written and meta updated with T_365, Y, Delta.")

if __name__ == "__main__":
    main()
