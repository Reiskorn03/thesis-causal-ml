# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
For each raw Parquet, look up its subset in the meta CSV and select exactly those columns.  
Write out a Parquet per subset.
"""

import os
import pandas as pd

def main():
    ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
    META_PATH = os.path.join(ROOT, "Data_Pipeline", "meta_data", "meta_data.csv")
    RAW_DIR = os.path.join(ROOT, "Data_Pipeline", "raw_data")
    OUT_DIR = os.path.join(ROOT, "Data_Pipeline", "clean_subsets")
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = pd.read_csv(META_PATH, dtype=str)

    # process for each parquet (five selected SAFs)
    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith(".parquet"):
            continue

        subset_key = os.path.splitext(fname)[0]             
        subset_key = subset_key.lower()                     
        raw_path = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(OUT_DIR, f"{subset_key}_subset.parquet")

        print(f"\nLoading raw data for subset '{subset_key}' from {raw_path}")
        df = pd.read_parquet(raw_path)
        print(f"raw shape: {df.shape}")

        # select all Variables in meta dataset
        vars_meta = meta.loc[meta["subset"] == subset_key, "Variable"].tolist()
        if not vars_meta:
            print(f"!!! no meta rows found for subset '{subset_key}', skipping.")
            continue

        keep_cols = [c for c in vars_meta if c in df.columns]
        missing = set(vars_meta) - set(keep_cols)
        if missing:
            print(f"!!! {len(missing)} meta-vars not in {subset_key}:")
            print(f"{sorted(missing)}")

        df_sel = df[keep_cols]
        print(f"keeping {len(keep_cols)} cols -> new shape: {df_sel.shape}")

        df_sel.to_parquet(out_path, index=False)
        print(f"wrote clean subset to {out_path}")

if __name__ == "__main__":
    main()
