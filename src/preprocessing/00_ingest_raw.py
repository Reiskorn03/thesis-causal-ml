# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Ingest raw selected .SAS7BDAT files into Parquet files under data/raw_data.
"""

import os
import pandas as pd

def main():
    input_dir = os.path.join("Data_Pipeline", "raw_sas")
    output_dir = os.path.join("Data_Pipeline", "raw_data")
    os.makedirs(output_dir, exist_ok=True)

    datasets = ["CAND_KIPA", "PRA_HIST", "STATHIST_KIPA", "TX_KI", "TXF_KI"]

    for ds in datasets:
        in_path = os.path.join(input_dir,  f"{ds}.sas7bdat")
        out_path = os.path.join(output_dir, f"{ds}.parquet")
        print(f"Reading  {in_path} …", end=" ")
        df = pd.read_sas(
            in_path,
            format="sas7bdat",
           encoding="latin1"
        )
        print(f"(shape={df.shape})")
        df.to_parquet(out_path, index=False)
        print(f"Wrote    {out_path}\n")

if __name__ == "__main__":
    main()
