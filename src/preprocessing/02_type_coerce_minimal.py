# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Take each subset, coerce easy-to-convert columns (Group A, date-like B, Y/N flags from Group C) and flag them in the meta.
Leave everything else as it is.
"""

import os
import pandas as pd

DATA_DIR = os.path.join("Data_Pipeline")
META_DIR = os.path.join(DATA_DIR, "meta_data")

META_IN = os.path.join(META_DIR, "meta_data.csv")
META_OUT = os.path.join(META_DIR, "meta_data_ver1.csv")

CLEAN_V1 = os.path.join(DATA_DIR, "clean_subsets")
CLEAN_V2 = os.path.join(DATA_DIR, "clean_subsets_ver1")

os.makedirs(CLEAN_V2, exist_ok=True)

# Load meta and initialize coercion flag (2 = not yet coerced)
meta = pd.read_csv(META_IN, dtype=str)
meta["coerce_done"] = "2"

# Iterate each subset
for subset in meta["subset"].unique():
    infile = os.path.join(CLEAN_V1, f"{subset}_subset.parquet")
    outfile = os.path.join(CLEAN_V2, f"{subset}_subset_ver1.parquet")
    if not os.path.isfile(infile):
        print(f"-> Skipping {subset}, no file at {infile}")
        continue

    print(f"Processing subset: {subset}")
    df = pd.read_parquet(infile)
    submeta = meta[meta["subset"] == subset]

    for idx, row in submeta.iterrows():
        col = row["Variable"]
        grp = row["group"]
        fmt = str(row.get("Format", "") or "").strip().upper()

        if col not in df.columns:
            # column dropped or absent
            continue

        # make a mask of non-null entries so it never touches original NaNs (NO IMPUTING)
        not_null = df[col].notna()

        # Group A: pure continuous numeric -> cast to float only where not null 
        if grp == "a_continuous_numeric":
            df.loc[not_null, col] = pd.to_numeric(
                df.loc[not_null, col],
                errors="coerce"
            )
            meta.at[idx, "coerce_done"] = "1"

        # Group B: only parse true dates (MMDDYY or DATE) where not null
        elif grp == "b_numeric_with_format" and fmt in ("MMDDYY", "DATE"):
            df.loc[not_null, col] = pd.to_datetime(
                df.loc[not_null, col],
                errors="coerce",
            )
            meta.at[idx, "coerce_done"] = "1"

        # Group C (Y/N flags only): map Y->1, N->0, U->-1, else leave NaN
        elif grp == "c_flag_YNU":
            df.loc[not_null, col] = df.loc[not_null, col].map(
                {"Y": 1.0, "N": 0.0, "U": -1.0}
            )
            meta.at[idx, "coerce_done"] = "1"

    os.makedirs(CLEAN_V2, exist_ok=True)
    df.to_parquet(outfile)
    print(f"-> Wrote coerced subset to {outfile}")

os.makedirs(os.path.dirname(META_OUT), exist_ok=True)
meta.to_csv(META_OUT, index=False)
print(f"Wrote updated meta with coercion flags to {META_OUT}")


