# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) Filter CAND_KIPA to:
     - kidney transplants only (ORG_AR == "KI")
     - adults only (CAN_AGE_AT_LISTING >= 18)
     - first kidney transplant only (CAN_PREV_KI == 0 AND CAN_PREV_TX == 0)
2) Drop ORG_AR, WL_ORG, CAN_PREV_KI, CAN_PREV_TX from cand_kipa
3) Apply the filtered PX_IDs to tx_ki and txf_ki subsets
4) Update meta‐data: remove those four features under subset 'cand_kipa'
"""
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path("Data_Pipeline")
CLEAN_IN = DATA_DIR / "clean_subsets_ver2"          
CLEAN_OUT = DATA_DIR / "clean_subsets_ver3"          
META_IN = DATA_DIR / "meta_data" / "meta_data_ver2.csv"
META_OUT = DATA_DIR / "meta_data" / "meta_data_ver3.csv"

# columns to drop from cand_kipa
DROP_COLS = ["ORG_AR", "WL_ORG", "CAN_PREV_KI", "CAN_PREV_TX"]

def main():
    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    cand_in = CLEAN_IN / "cand_kipa_subset_ver2.parquet"
    print(f"Loading {cand_in}…")
    cand = pd.read_parquet(cand_in)

    # apply inclusion criteria
    print("Applying filters to cand_kipa:")
    initial_n = len(cand)
    cand = cand[cand["ORG_AR"] == "KI"]
    print(f"  kidney only: {len(cand):,d} rows ({len(cand)/initial_n:.1%})")
    cand = cand[cand["CAN_AGE_AT_LISTING"] >= 18]
    print(f"  adults only: {len(cand):,d} rows")
    cand = cand[(cand["CAN_PREV_KI"] == 0) & (cand["CAN_PREV_TX"] == 0)]
    print(f"  first transplant only: {len(cand):,d} rows")

    # drop unused columns
    cand = cand.drop(columns=DROP_COLS, errors="ignore")

    # write filtered cand_kipa
    cand_out = CLEAN_OUT / "cand_kipa_subset_ver3.parquet"
    cand.to_parquet(cand_out, index=False)
    print(f"Wrote filtered cand_kipa -> {cand_out.name}")

    # propagate to tx_ki and txf_ki
    keep_ids = cand["PX_ID"].unique()
    for subset in ("tx_ki", "txf_ki"):
        inp = CLEAN_IN / f"{subset}_subset_ver2.parquet"
        outp = CLEAN_OUT / f"{subset}_subset_ver3.parquet"
        if not inp.exists():
            print(f"!!! Missing {inp.name}, skipping")
            continue

        df = pd.read_parquet(inp)
        df = df[df["PX_ID"].isin(keep_ids)]
        df.to_parquet(outp, index=False)
        print(f"Wrote filtered {subset} -> {len(df):,d} rows")

    # copy stathist_kipa unchanged
    inp_st = CLEAN_IN / "stathist_kipa_subset_ver2.parquet"
    out_st = CLEAN_OUT / "stathist_kipa_subset_ver3.parquet"
    if inp_st.exists():
        pd.read_parquet(inp_st).to_parquet(out_st, index=False)
        print(f"Copied stathist_kipa -> {out_st.name}")

    # update meta
    print("Updating meta data…")
    meta = pd.read_csv(META_IN, dtype=str)
    # drop the four variables in cand_kipa
    mask = ~(
        (meta["subset"] == "cand_kipa") &
        (meta["Variable"].isin(DROP_COLS))
    )
    meta = meta[mask].reset_index(drop=True)
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META_OUT, index=False)
    print(f"Wrote updated meta -> {META_OUT.name}")

if __name__ == "__main__":
    main()
