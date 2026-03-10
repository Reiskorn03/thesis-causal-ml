# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) Aggregate TXF_KI -> one row per PX_ID:
     - TFL_CREAT_MAX = max(TFL_CREAT)
     - TFL_CREAT_LAST = TFL_CREAT at most‐recent TFL_PX_STAT_DT
     - TFL_MALIG_AGG = 1 if any TFL_MALIG==1 else 0 (NaN if all missing)
     - TFL_HOSP_AGG = 1 if any TFL_HOSP==1  else 0 (NaN if all missing)
     - TFL_HOSP_NUM_MAX= max(TFL_HOSP_NUM)
     - TFL_PX_STAT_LAST= last non‐null TFL_PX_STAT (by TFL_PX_STAT_DT)
2) Merge those six into TX_KI
3) Copy CAND_KIPA unchanged -> ver4
4) Drop STATHIST_KIPA & TXF_KI entirely
5) Update meta_data -> drop stathist_kipa & txf_ki rows; append the six aggregates
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

DATA_DIR = Path("Data_Pipeline")
CLEAN_V6 = DATA_DIR / "clean_subsets_ver6"
CLEAN_V7 = DATA_DIR / "clean_subsets_ver7"
META_IN = DATA_DIR / "meta_data" / "meta_data_ver6.csv"
META_OUT = DATA_DIR / "meta_data" / "meta_data_ver7.csv"


def main():
    os.makedirs(CLEAN_V7, exist_ok=True)

    txf = pd.read_parquet(CLEAN_V6 / "txf_ki_subset_ver6.parquet")
    txf["TFL_PX_STAT_DT"] = pd.to_datetime(txf["TFL_PX_STAT_DT"], errors="coerce")

    grp = txf.groupby("PX_ID", sort=False)
    # continuous summaries
    creat_max = grp["TFL_CREAT"].max().rename("TFL_CREAT_MAX")
    hospnum_max  = grp["TFL_HOSP_NUM"].max().rename("TFL_HOSP_NUM_MAX")
    # last observed by date
    last_by_date = txf.sort_values("TFL_PX_STAT_DT").groupby("PX_ID", sort=False).last()
    creat_last = last_by_date["TFL_CREAT"].rename("TFL_CREAT_LAST")
    pxstat_last = last_by_date["TFL_PX_STAT"].rename("TFL_PX_STAT_LAST")

    # any-flag summaries
    def any_flag(s):
        non_null = s.dropna()
        if non_null.empty:
            return np.nan
        return int((non_null == 1).any())

    malig_agg = grp["TFL_MALIG"].apply(any_flag).rename("TFL_MALIG_AGG")
    hosp_agg = grp["TFL_HOSP"].apply(any_flag).rename("TFL_HOSP_AGG")

    # assemble aggregated dataframe
    agg = pd.concat([
        creat_max, creat_last,
        malig_agg, hosp_agg,
        hospnum_max,
        pxstat_last
    ], axis=1).reset_index()

    # merge aggregates into TX_KI
    tx = pd.read_parquet(CLEAN_V6 / "tx_ki_subset_ver6.parquet")
    tx = tx.merge(agg, on="PX_ID", how="left")
    tx.to_parquet(CLEAN_V7 / "tx_ki_subset_ver7.parquet", index=False)

    # copy cand_kipa unchanged!!!
    cand = pd.read_parquet(CLEAN_V6 / "cand_kipa_subset_ver6.parquet")
    cand.to_parquet(CLEAN_V7 / "cand_kipa_subset_ver7.parquet", index=False)

    # update meta dataset
    meta = pd.read_csv(META_IN, dtype=str)
    # remove existing stathist_kipa & txf_ki entries
    meta = meta[~meta["subset"].isin(["stathist_kipa", "txf_ki"])]

    # append new aggregate variable definitions
    new_vars = [
        ("TFL_CREAT_MAX", "a_continuous_numeric"),
        ("TFL_CREAT_LAST", "a_continuous_numeric"),
        ("TFL_MALIG_AGG", "c_flag_YNU"),
        ("TFL_HOSP_AGG", "c_flag_YNU"),
        ("TFL_HOSP_NUM_MAX", "a_continuous_numeric"),
        ("TFL_PX_STAT_LAST", "e_char_with_encoding"),
    ]
    new_rows = []
    for var, grp_name in new_vars:
        row = {col: '' for col in meta.columns}
        row.update({
            "Variable": var,
            "group": grp_name,
            "subset": "tx_ki",
            "coerce_done": "1",
        })
        new_rows.append(row)
    meta = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META_OUT, index=False)

    print("->ver7 subsets & meta ready (aggregated txf_ki into tx_ki, dropped stathist_kipa & txf_ki).")

if __name__ == "__main__":
    main()
