# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Impute missing values and one‐hot encode categorical features applying strategy, using the train/test splits in splits_ver7.

Strategy:
1. a_continuous_numeric (true continuous features): <5%: median, 5–50%: MICE, >50% missing: Drop feature 
2. b_numeric_with_format (categorial features): <10%: mode-fill missing rows, 10–50% missing: fill Missings with MISSING, then handle it afterwards while One Hot Encoding 
3.1 c_flag_YNU (Flags with 1=True,0=False,-1=Unknown): <5%: mode (flags), 5–50%: fill -1 
3.2 c_flag_other (Categorials with a single char, e.g. Gender): <10%: mode-fill missing rows, 10–50% missing: fill Missings with MISSING, then handle it afterwards while OneHot Encoding 
4. d_char_free_text (also categorial features): <10%: mode-fill missing rows, 10–50% missing: fill Missings with MISSING, then handle it afterwards while One Hot Encoding 
(Checked the meta data there are NO features in this group so skip) 
5. e_char_with_encoding (also categorial features): <10%: mode-fill missing rows, 10–50% missing: fill Missings with MISSING, then handle it afterwards while One Hot Encoding.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

DATA_DIR = Path("Data_Pipeline")
SPLIT_IN = DATA_DIR / "splits_ver7"
SPLIT_OUT = DATA_DIR / "splits_ver8"
META_IN = DATA_DIR / "meta_data" / "meta_data_ver7.csv"
META_OUT = DATA_DIR / "meta_data" / "meta_data_ver8.csv"

def main():
    os.makedirs(SPLIT_OUT, exist_ok=True)

    cand_train = pd.read_parquet(SPLIT_IN / "cand_kipa_train_ver7.parquet")
    cand_test = pd.read_parquet(SPLIT_IN / "cand_kipa_test_ver7.parquet")
    tx_train = pd.read_parquet(SPLIT_IN / "tx_ki_train_ver7.parquet")
    tx_test = pd.read_parquet(SPLIT_IN / "tx_ki_test_ver7.parquet")

    meta = pd.read_csv(META_IN, dtype=str)
    meta["impute_done"] = "0"

    dropped_feats = []
    imputed_feats = []

    # set up MICE
    mice = IterativeImputer(random_state=0)

    for subset_name, train, test in [
        ("cand_kipa", cand_train, cand_test),
        ("tx_ki",     tx_train,  tx_test),
    ]:
        sub_meta = meta[meta["subset"] == subset_name]

        # true continous features
        cont_feats = sub_meta.query("group=='a_continuous_numeric'")["Variable"].tolist()
        miss_cont = train[cont_feats].isna().mean()
        small_cont = miss_cont[miss_cont < 0.05].index.tolist()
        mid_cont = miss_cont[(miss_cont >= 0.05)&(miss_cont <= 0.5)].index.tolist()
        large_cont = miss_cont[miss_cont > 0.5].index.tolist()
        dropped_feats += large_cont

        # <5%: median
        for f in small_cont:
            m = train[f].median()
            train[f].fillna(m, inplace=True)
            test[f].fillna(m, inplace=True)
            imputed_feats.append(f)

        # 5–50%: MICE
        if mid_cont:
            train[mid_cont] = mice.fit_transform(train[mid_cont])
            test[mid_cont] = mice.transform(test[mid_cont])
            imputed_feats += mid_cont

        # categorial features
        cat_groups = ["b_numeric_with_format","c_flag_other","e_char_with_encoding"]
        cat_feats = sub_meta.query("group in @cat_groups")["Variable"].tolist()
        miss_cat = train[cat_feats].isna().mean()
        small_cat = miss_cat[miss_cat < 0.10].index.tolist()
        mid_cat = miss_cat[(miss_cat >= 0.10)&(miss_cat <= 0.5)].index.tolist()
        large_cat = miss_cat[miss_cat > 0.5].index.tolist()
        dropped_feats += large_cat

        #  <10%: mode
        for f in small_cat:
            mode = train[f].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else -9999
            train[f].fillna(fill, inplace=True)
            test[f].fillna(fill,  inplace=True)
            imputed_feats.append(f)

        # 10–50%: sentinel
        for f in mid_cat:
            train[f].fillna(-9999, inplace=True)
            test[f].fillna(-9999,  inplace=True)
            imputed_feats.append(f)

        # flags Y/N/U
        flag_feats = sub_meta.query("group=='c_flag_YNU'")["Variable"].tolist()
        miss_flag = train[flag_feats].isna().mean()
        small_flag = miss_flag[miss_flag < 0.05].index.tolist()
        mid_flag = miss_flag[(miss_flag >= 0.05)&(miss_flag <= 0.5)].index.tolist()
        large_flag = miss_flag[miss_flag > 0.5].index.tolist()
        dropped_feats += large_flag

        # <5%: mode
        for f in small_flag:
            mode = train[f].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else 0
            train[f].fillna(fill, inplace=True)
            test[f].fillna(fill,  inplace=True)
            imputed_feats.append(f)

        # 5–50%: fill -1
        for f in mid_flag:
            train[f].fillna(-1, inplace=True)
            test[f].fillna(-1,  inplace=True)
            imputed_feats.append(f)

        # drop the missing >50% features
        to_drop = large_cont + large_cat + large_flag
        train.drop(columns=to_drop, errors="ignore", inplace=True)
        test .drop(columns=to_drop, errors="ignore", inplace=True)

        for df in (train, test):
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype("string")

        train.to_parquet(SPLIT_OUT/f"{subset_name}_train_ver8.parquet", index=False)
        test .to_parquet(SPLIT_OUT/f"{subset_name}_test_ver8.parquet",  index=False)

    dropped_feats = list(set(dropped_feats))
    imputed_feats = list(set(imputed_feats))

    # remove dropped columns
    meta = meta[~meta["Variable"].isin(dropped_feats)].copy()

    # flag imputed (including 0% missingness features)
    meta.loc[meta["Variable"].isin(imputed_feats), "impute_done"] = "1"
    meta.to_csv(META_OUT, index=False)

    print("Imputation complete.")
    print(f"   - dropped {len(dropped_feats)} features:", dropped_feats)
    print(f"   - flagged {len(imputed_feats)} features as imputed (impute_done=1)")

if __name__ == "__main__":
    main()
