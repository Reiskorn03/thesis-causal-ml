# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) In cand_kipa: compute & append WL_EVENT, WL_TIME, T, then: drop date inputs
2) In tx_ki: compute & append TX_EVENT, TX_TIME, GRAFT_EVENT, GRAFT_TIME, then: drop date inputs
3) Copy stathist_kipa & txf_ki unchanged
4) Update meta: drop raw date rows, append entries for 7 new vars (including T)
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("Data_Pipeline")
CLEAN_IN = DATA_DIR / "clean_subsets_ver3"
CLEAN_OUT = DATA_DIR / "clean_subsets_ver4"
META_IN = DATA_DIR / "meta_data" / "meta_data_ver3.csv"
META_OUT = DATA_DIR / "meta_data" / "meta_data_ver4.csv"

def define_waitlist_and_treatment():
    cand = pd.read_parquet(CLEAN_IN/"cand_kipa_subset_ver3.parquet")
    status = pd.read_parquet(CLEAN_IN/"stathist_kipa_subset_ver3.parquet")
    tx = pd.read_parquet(CLEAN_IN/"tx_ki_subset_ver3.parquet")

    # coerce to datetime: double check
    for col in ("CAN_LISTING_DT","CAN_DEATH_DT","REC_TX_DT","CANHX_END_DT"):
        if col in cand: cand[col] = pd.to_datetime(cand[col], errors="coerce")
        if col in status: status[col] = pd.to_datetime(status[col], errors="coerce")

    # last follow‐up = max CANHX_END_DT per patient
    last_fu = (
        status
        .groupby("PX_ID")["CANHX_END_DT"]
        .max()
        .rename("LAST_FU_DT")
    )
    cand = cand.merge(last_fu, on="PX_ID", how="left")

    # WL_EVENT: died before transplant?
    tx_cutoff = cand["REC_TX_DT"].fillna(pd.Timestamp.max)
    died_before_tx = cand["CAN_DEATH_DT"].notna() & (cand["CAN_DEATH_DT"] < tx_cutoff)
    cand["WL_EVENT"] = died_before_tx.astype(int)

    # WL_CENSOR_DT: Date where patient got censored/died
    cand["WL_CENSOR_DT"] = np.where(
        died_before_tx,
        cand["CAN_DEATH_DT"],
        np.where(
            cand["REC_TX_DT"].notna(),
            cand["REC_TX_DT"],
            cand["LAST_FU_DT"]
        )
    )

    # WL_TIME
    cand["WL_TIME"] = (cand["WL_CENSOR_DT"] - cand["CAN_LISTING_DT"]).dt.days

    # T: treatment indicator (ever transplanted?)
    transplanted_ids = pd.read_parquet(CLEAN_IN/"tx_ki_subset_ver3.parquet")["PX_ID"].unique()
    cand["T"] = cand["PX_ID"].isin(transplanted_ids).astype(int)

    # drop dates ->no use after this step
    drop_dates = [
        "CAN_LISTING_DT","CAN_DEATH_DT","REC_TX_DT","CANHX_END_DT",
        "LAST_FU_DT","WL_CENSOR_DT","CAN_ACTIVATE_DT"
    ]
    cand = cand.drop(columns=[c for c in drop_dates if c in cand], errors="ignore")

    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(CLEAN_OUT/"cand_kipa_subset_ver4.parquet", index=False)
    print("-> cand_kipa_subset_ver4.parquet written with WL_EVENT, WL_TIME, T")

def define_post_tx_and_graft():
    tx = pd.read_parquet(CLEAN_IN/"tx_ki_subset_ver3.parquet")
    follow = pd.read_parquet(CLEAN_IN/"txf_ki_subset_ver3.parquet")

    for col in ("REC_TX_DT","TFL_PX_STAT_DT","REC_FAIL_DT","TFL_FAIL_DT"):
        if col in tx: tx[col] = pd.to_datetime(tx[col], errors="coerce")
        if col in follow: follow[col] = pd.to_datetime(follow[col], errors="coerce")

    # Post‐TX survival
    if "TFL_PX_STAT" in follow:
        death_rows = follow.loc[
            follow["TFL_PX_STAT"]=="D",
            ["PX_ID","TFL_PX_STAT_DT"]
        ]
    else:
        death_rows = pd.DataFrame(columns=["PX_ID","TFL_PX_STAT_DT"])
    death_min = death_rows.groupby("PX_ID")["TFL_PX_STAT_DT"] \
                         .min().rename("DEATH_DT")
    tx = tx.merge(death_min, on="PX_ID", how="left")

    last_fu = follow.groupby("PX_ID")["TFL_PX_STAT_DT"] \
                    .max().rename("LAST_FU_DT")
    tx = tx.merge(last_fu, on="PX_ID", how="left")

    tx["TX_EVENT"] = tx["DEATH_DT"].notna().astype(int)
    tx["TX_TIME"] = (
        np.where(tx["DEATH_DT"].notna(), tx["DEATH_DT"], tx["LAST_FU_DT"])
        - tx["REC_TX_DT"]
    ).dt.days

    # Graft Failure
    graft_rows = follow.loc[follow["TFL_FAIL_DT"].notna(), ["PX_ID","TFL_FAIL_DT"]]
    graft_min = graft_rows.groupby("PX_ID")["TFL_FAIL_DT"] \
                          .min().rename("GF_FU_DT")
    tx = tx.merge(graft_min, on="PX_ID", how="left")

    tx["GRAFT_DT"] = tx[["REC_FAIL_DT","GF_FU_DT"]].min(axis=1)
    tx["GRAFT_EVENT"] = tx["GRAFT_DT"].notna().astype(int)
    tx["GRAFT_TIME"] = (
        np.where(tx["GRAFT_EVENT"]==1, tx["GRAFT_DT"], tx["LAST_FU_DT"])
        - tx["REC_TX_DT"]
    ).dt.days

    drop_dates = [
        "REC_TX_DT","REC_FAIL_DT","TFL_FAIL_DT",
        "LAST_FU_DT","DEATH_DT","GF_FU_DT","GRAFT_DT","TFL_DEATH_DT"
    ]
    tx = tx.drop(columns=[c for c in drop_dates if c in tx], errors="ignore")

    tx.to_parquet(CLEAN_OUT/"tx_ki_subset_ver4.parquet", index=False)
    print("-> tx_ki_subset_ver4.parquet written with TX_EVENT, TX_TIME, GRAFT_EVENT, GRAFT_TIME")

def copy_others():
    for name in ("stathist_kipa","txf_ki"):
        inp = CLEAN_IN/f"{name}_subset_ver3.parquet"
        out = CLEAN_OUT/f"{name}_subset_ver4.parquet"
        if inp.exists():
            pd.read_parquet(inp).to_parquet(out, index=False)
            print(f"-> {out.name} copied unchanged")

def update_meta():
    # raw date fields to drop from metadata
    date_vars = {
        "CAN_LISTING_DT","CAN_DEATH_DT","REC_TX_DT",
        "CANHX_END_DT","REC_FAIL_DT",
        "TFL_FAIL_DT","LAST_FU_DT","CAN_ACTIVATE_DT","TFL_DEATH_DT"
    }
    meta = pd.read_csv(META_IN, dtype=str)

    # drop any date rows
    meta = meta[~meta["Variable"].isin(date_vars)]

    # append the seven new variables
    new = [
      {"Variable":"WL_EVENT",    "group":"c_flag_YNU",           "subset":"cand_kipa", "coerce_done":"1"},
      {"Variable":"WL_TIME",     "group":"a_continuous_numeric", "subset":"cand_kipa", "coerce_done":"1"},
      {"Variable":"T",           "group":"c_flag_YNU",           "subset":"cand_kipa", "coerce_done":"1"},
      {"Variable":"TX_EVENT",    "group":"c_flag_YNU",           "subset":"tx_ki",     "coerce_done":"1"},
      {"Variable":"TX_TIME",     "group":"a_continuous_numeric", "subset":"tx_ki",     "coerce_done":"1"},
      {"Variable":"GRAFT_EVENT", "group":"c_flag_YNU",           "subset":"tx_ki",     "coerce_done":"1"},
      {"Variable":"GRAFT_TIME",  "group":"a_continuous_numeric", "subset":"tx_ki",     "coerce_done":"1"},
    ]
    new_df = pd.DataFrame(new)

    for col in meta.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[meta.columns]

    meta = pd.concat([meta, new_df], ignore_index=True)
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META_OUT, index=False)
    print(f"-> Meta written to {META_OUT.name}")

if __name__=="__main__":
    os.makedirs(CLEAN_OUT, exist_ok=True)
    define_waitlist_and_treatment()
    define_post_tx_and_graft()
    copy_others()
    update_meta()
    print("All subsets (ver4) & meta updated.")
