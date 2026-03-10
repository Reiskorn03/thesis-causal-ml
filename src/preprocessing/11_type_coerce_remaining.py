# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
One hot encode remaining categorical-encoded features according to mappings.py,
flag as coerce_done=1, and update meta.
"""
import os
from pathlib import Path
import pandas as pd
from mappings import *  

DATA_DIR = Path("Data_Pipeline")
SPLIT_IN  = DATA_DIR / "splits_ver8"
SPLIT_OUT = DATA_DIR / "splits_ver9"
META_IN   = DATA_DIR / "meta_data" / "meta_data_ver8.csv"
META_OUT  = DATA_DIR / "meta_data" / "meta_data_ver9.csv"

# definitions
to_coerce = pd.DataFrame([
    {"Variable": "CAN_RACE", "subset": "cand_kipa", "mapping": "RACE_MAP"},
    {"Variable": "CAN_ABO", "subset": "cand_kipa", "mapping": "ABO_MAP"},
    {"Variable": "CAN_DIAL", "subset": "cand_kipa", "mapping": "DIALTYLI_MAP"},
    {"Variable": "CAN_FUNCTN_STAT","subset": "cand_kipa", "mapping": "FUNCSTAT_MAP"},
    {"Variable": "CAN_GENDER", "subset": "cand_kipa", "mapping": "GENDER_MAP"},
    {"Variable": "CAN_INIT_STAT", "subset": "cand_kipa", "mapping": "CANDSTAT_MAP"},
    {"Variable": "CAN_MED_COND", "subset": "cand_kipa", "mapping": "MEDCOND_MAP"},
    {"Variable": "CAN_DIAB_TY", "subset": "cand_kipa", "mapping": "DIABTY_MAP"},
    {"Variable": "CAN_DGN", "subset": "cand_kipa", "mapping": "DGN_MAP"},
    {"Variable": "DON_ABO", "subset": "tx_ki", "mapping": "ABO_MAP"},
    {"Variable": "DON_ANTI_CMV", "subset": "tx_ki", "mapping": "SRLSTT_MAP"},
    {"Variable": "DON_ANTI_HCV", "subset": "tx_ki", "mapping": "SRLSTT_MAP"},
    {"Variable": "DON_GENDER", "subset": "tx_ki", "mapping": "GENDER_MAP"},
    {"Variable": "DON_HIST_CANCER","subset": "tx_ki", "mapping": "HSTSTST_MAP"},
    {"Variable": "DON_HIST_DIAB", "subset": "tx_ki", "mapping": "HSTDBDR_MAP"},
    {"Variable": "DON_RACE", "subset": "tx_ki", "mapping": "RACE_MAP"},
    {"Variable": "REC_CMV_IGG", "subset": "tx_ki", "mapping": "STAT_D_MAP"},
    {"Variable": "REC_CMV_IGM", "subset": "tx_ki", "mapping": "STAT_D_MAP"},
    {"Variable": "REC_CMV_STAT", "subset": "tx_ki", "mapping": "STAT_D_MAP"},
    {"Variable": "REC_DGN", "subset": "tx_ki", "mapping": "DGN_MAP"},
    {"Variable": "REC_EBV_STAT", "subset": "tx_ki", "mapping": "STAT_D_MAP"},
    {"Variable": "REC_FUNCTN_STAT","subset": "tx_ki", "mapping": "FUNCSTAT_MAP"},
    {"Variable": "REC_HBV_ANTIBODY","subset": "tx_ki", "mapping": "STAT_D_MAP"},
    {"Variable": "REC_HBV_SURF_ANTIGEN","subset": "tx_ki","mapping": "STAT_D_MAP"},
    {"Variable": "REC_HCV_STAT", "subset": "tx_ki", "mapping": "STAT_D_MAP"},
    {"Variable": "REC_MED_COND", "subset": "tx_ki", "mapping": "MEDCOND_MAP"},
    {"Variable": "TFL_PX_STAT_LAST","subset": "tx_ki", "mapping": "PXSTATB_MAP"},
], columns=["Variable","subset","mapping"])


def main():
    os.makedirs(SPLIT_OUT, exist_ok=True)

    meta = pd.read_csv(META_IN, dtype=str)

    for subset_name in ["cand_kipa", "tx_ki"]:
        train = pd.read_parquet(SPLIT_IN / f"{subset_name}_train_ver8.parquet")
        test = pd.read_parquet(SPLIT_IN / f"{subset_name}_test_ver8.parquet")

        subset_map = to_coerce[to_coerce['subset'] == subset_name]
        for _, row in subset_map.iterrows():
            var = row['Variable']
            map_name = row['mapping']
            mapping = globals()[map_name]

            # map codes to labels
            train[var] = train[var].map(mapping)
            test [var] = test [var].map(mapping)

            assert train[var].isna().sum() == 0, f"Unmapped values in train[{var}]"
            assert test [var].isna().sum() == 0, f"Unmapped values in test[{var}]"

            # one-hot encode
            train = pd.get_dummies(train, columns=[var], prefix=[var], dummy_na=False)
            test = pd.get_dummies(test,  columns=[var], prefix=[var], dummy_na=False)

            # update meta 
            meta.loc[
                (meta['Variable'] == var) & (meta['subset'] == subset_name),
                'coerce_done'
            ] = '1'

        train.to_parquet(SPLIT_OUT / f"{subset_name}_train_ver9.parquet", index=False)
        test .to_parquet(SPLIT_OUT / f"{subset_name}_test_ver9.parquet",  index=False)

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META_OUT, index=False)

    print(f"-> Coercion complete; splits in {SPLIT_OUT}, meta in {META_OUT}")

if __name__ == '__main__':
    main()
