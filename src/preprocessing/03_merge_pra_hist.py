# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) Merge CAND_KIPA + PRA_HIST -> cand_kipa (add MAX_CPRA)
2) Copy STATHIST_KIPA, TX_KI, TXF_KI unchanged
3) Drop PRA_HIST subset
4) Update meta: remove pra_hist rows, append entry for MAX_CPRA
"""
import os
import pandas as pd
from pathlib import Path

DATA_DIR = os.path.join("Data_Pipeline")
META_DIR = os.path.join(DATA_DIR, "meta_data")

META_IN = os.path.join(META_DIR, "meta_data_ver1.csv")
META_OUT = os.path.join(META_DIR, "meta_data_ver2.csv")

CLEAN_IN = os.path.join(DATA_DIR, "clean_subsets_ver1")
CLEAN_OUT = os.path.join(DATA_DIR, "clean_subsets_ver2")

os.makedirs(CLEAN_OUT, exist_ok=True)

def main():
    os.makedirs(CLEAN_OUT, exist_ok=True)

    print("Loading cand_kipa and pra_hist...")
    cand = pd.read_parquet(os.path.join(CLEAN_IN, "cand_kipa_subset_ver1.parquet"))
    pra = pd.read_parquet(os.path.join(CLEAN_IN, "pra_hist_subset_ver1.parquet"))

    # coerce PRA -> numeric, take max per PX_ID
    pra["CANHX_CPRA"] = pd.to_numeric(pra["CANHX_CPRA"], errors="coerce")
    pra_max = (
        pra.groupby("PX_ID")["CANHX_CPRA"]
           .max()
           .rename("MAX_CPRA")
    )

    # merge back into cand_kipa
    print("Merging MAX_CPRA into cand_kipa…")
    cand = cand.merge(pra_max, on="PX_ID", how="left")

    # write out new cand_kipa
    out_cand = os.path.join(CLEAN_OUT, "cand_kipa_subset_ver2.parquet")
    cand.to_parquet(out_cand, index=False)
    print(f"Wrote merged CAND_KIPA -> {out_cand}")

    # copy the other three tables unchanged
    for subset in ("stathist_kipa","tx_ki","txf_ki"):
        inp = os.path.join(CLEAN_IN,  f"{subset}_subset_ver1.parquet")
        out = os.path.join(CLEAN_OUT, f"{subset}_subset_ver2.parquet")
        if not os.path.exists(inp):
            print(f"!!! Missing input: {inp.name}, skipping")
            continue
        print(f"Copying {inp} -> {out}")
        pd.read_parquet(inp).to_parquet(out, index=False)

    #  Important: it does NOT write pra_hist to ver2!

    print("Updating meta-dataset...")
    meta = pd.read_csv(META_IN, dtype=str)

    # drop all pra_hist rows
    meta = meta[meta.subset != "pra_hist"]

    # append new MAX_CPRA entry
    new = {
        "Variable": "MAX_CPRA",
        "Type": "num",
        "group": "a_continuous_numeric",
        "subset": "cand_kipa",
        "coerce_done": "1"
    }
    meta = pd.concat([meta, pd.DataFrame([new])], ignore_index=True)

    os.makedirs(os.path.dirname(META_OUT), exist_ok=True)
    meta.to_csv(META_OUT, index=False)
    print(f"Wrote updated meta -> {os.path.basename(META_OUT)}")
    print("Finished merge -> ver2 subsets + meta ready")

if __name__=="__main__":
    main()
