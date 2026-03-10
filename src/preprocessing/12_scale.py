# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
1) Clip continuous features to their plausible ranges (outside -> NaN)
2) Median-impute those NaNs (train-only median)
3) Min–Max scale into [0,1] (fit on train, apply to both)
4) Leave ID and outcome columns untouched
5) Update meta_data impute_done for all a_continuous_numeric features
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path("Data_Pipeline")
SPLIT_IN = DATA_DIR / "splits_ver9"
SPLIT_OUT = DATA_DIR / "splits_ver10"
META_IN = DATA_DIR / "meta_data" / "meta_data_ver9.csv"
META_OUT = DATA_DIR / "meta_data" / "meta_data_ver10.csv"

# plausible ranges
PLAUSIBLE_RANGES_CAND_KIPA = {
    "CAN_AGE_AT_LISTING":     (0.0,  90.0),
    "CAN_BMI":                (8.0,  70.0),
    "CAN_MAX_PCT_SCLER_LT10": (0.0, 100.0),
    "CAN_MIN_PEAK_CREAT":     (0.2,  25.0),
    "CAN_MIN_FINAL_CREAT":    (0.2,  25.0),
    "MAX_CPRA":               (0.0,   1.0),
    "CAN_TOT_ALBUMIN":        (1.0,   6.0),
}

PLAUSIBLE_RANGES_TX_KI = {
    "CAN_LAST_SRTR_PEAK_PRA": (0.0, 100.0),
    "CAN_TOT_ALBUMIN":        (1.0,   6.0),
    "DON_AGE":                (0.0,  90.0),
    "DON_CREAT":              (0.2,  25.0),
    "DON_HIGH_CREAT":         (0.0,   1.0),
    "DON_EXPAND_DON_KI":      (0.0,   1.0),
    "REC_AGE_AT_TX":          (0.0,  90.0),
    "REC_A_MM_EQUIV_CUR":     (0.0,   3.0),
    "REC_A_MM_EQUIV_TX":      (0.0,   3.0),
    "REC_B_MM_EQUIV_CUR":     (0.0,   3.0),
    "REC_B_MM_EQUIV_TX":      (0.0,   3.0),
    "REC_BMI":                (8.0,  70.0),
    "REC_CREAT":              (0.2,  25.0),
    "REC_DISCHRG_CREAT":      (0.2,  25.0),
    "REC_DR_MM_EQUIV_CUR":    (0.0,   3.0),
    "REC_DR_MM_EQUIV_TX":     (0.0,   3.0),
    "REC_MM_EQUIV_CUR":       (0.0,  10.0),
    "REC_MM_EQUIV_TX":        (0.0,  10.0),
    "REC_PRA_MOST_RECENT":    (0.0, 100.0),
    "TFL_CREAT_MAX":          (0.2,  25.0),
    "TFL_CREAT_LAST":         (0.2,  25.0),
}

# IDs and outcomes ->exclude
EXCLUDE = {
    "PX_ID",
    "WL_TIME",
    "TX_TIME",
    "GRAFT_TIME",
    "Y", 
    "Delta",
    "T_365", "T"
}

def main():
    os.makedirs(SPLIT_OUT, exist_ok=True)
    meta = pd.read_csv(META_IN, dtype=str)
    # ensure impute_done column exists
    if "impute_done" not in meta.columns:
        meta["impute_done"] = ""

    for subset, ranges in [
        ("cand_kipa", PLAUSIBLE_RANGES_CAND_KIPA),
        ("tx_ki", PLAUSIBLE_RANGES_TX_KI),
    ]:
        train_fp = SPLIT_IN / f"{subset}_train_ver9.parquet"
        test_fp = SPLIT_IN / f"{subset}_test_ver9.parquet"
        train = pd.read_parquet(train_fp)
        test = pd.read_parquet(test_fp)

        # select continuous features
        cont_feats = (
            meta.query("subset == @subset and group == 'a_continuous_numeric'")
                ["Variable"].tolist()
        )
        # filter to those present and not excluded and present in plausible ranges
        cont_feats = [f for f in cont_feats
                      if f in train.columns and f not in EXCLUDE and f in ranges]

        # Clip outside plausible -> NaN
        for feat in cont_feats:
            lo, hi = ranges[feat]
            for df in (train, test):
                mask = (df[feat] < lo) | (df[feat] > hi)
                df.loc[mask, feat] = pd.NA

        # Median-impute using train medians
        medians = train[cont_feats].median()
        for feat, m in medians.items():
            train[feat].fillna(m, inplace=True)
            test [feat].fillna(m, inplace=True)
        
        # Min–Max scale (fit on train, apply to both)
        scaler = MinMaxScaler()
        train[cont_feats] = scaler.fit_transform(train[cont_feats])
        test [cont_feats] = scaler.transform(test[cont_feats])

        train.to_parquet(SPLIT_OUT / f"{subset}_train_ver10.parquet", index=False)
        test .to_parquet(SPLIT_OUT / f"{subset}_test_ver10.parquet",  index=False)
        print(f"Wrote splits for {subset} -> ver10")

    # save updated meta
    meta.to_csv(META_OUT, index=False)
    print(f"Updated meta -> {META_OUT}")


if __name__ == "__main__":
    main()
