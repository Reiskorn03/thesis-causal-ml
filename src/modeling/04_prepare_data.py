# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Merge covariate data, propensity scores, and censoring weights (for 3 horizons), trim extreme PS, run sanity checks, and write out a single Parquet ready for the Causal Survival Forest.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

def main():
    DATA_DIR = Path("ATE_CATE")
    SPLITS_DIR = DATA_DIR / "FINAL_SPLITS"
    PS_DIR = DATA_DIR / "PS_validation"
    IPC_DIR = DATA_DIR / "Censoring_KM"  
    OUT_DIR = DATA_DIR / "CSF_data_NEW2"

    cov_fp = SPLITS_DIR / "cand_kipa_train_ver10.parquet"
    ps_fp = PS_DIR / "ps_cand_kipa_train.parquet"

    cw_365_fp = IPC_DIR / "censoring_weights_horizon_365.parquet"
    cw_730_fp = IPC_DIR / "censoring_weights_horizon_730.parquet"
    cw_1095_fp = IPC_DIR / "censoring_weights_horizon_1095.parquet"

    df_cov = pd.read_parquet(cov_fp)
    df_ps  = pd.read_parquet(ps_fp)

    df_cw_365  = pd.read_parquet(cw_365_fp)
    df_cw_730  = pd.read_parquet(cw_730_fp)
    df_cw_1095 = pd.read_parquet(cw_1095_fp)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"  covariates: {len(df_cov):,} rows")
    print(f"  prop-scores: {len(df_ps):,} rows")
    print(f"  censor-wt's 365d : {len(df_cw_365):,} rows")
    print(f"  censor-wt's 730d : {len(df_cw_730):,} rows")
    print(f"  censor-wt's 1095d: {len(df_cw_1095):,} rows")

    # Merge all dataframes on PX_ID -> inner join
    df = (
        df_cov
        .merge(df_ps, on="PX_ID", how="inner")
        .merge(df_cw_365[['PX_ID', 'ipc_weight']], on="PX_ID", how="inner", suffixes=('', '_365'))
        .merge(df_cw_730[['PX_ID', 'ipc_weight']], on="PX_ID", how="inner", suffixes=('', '_730'))
        .merge(df_cw_1095[['PX_ID', 'ipc_weight']], on="PX_ID", how="inner", suffixes=('', '_1095'))
    )
    
    # Rename the weight columns for clarity
    df.rename(columns={
        'ipc_weight': 'ipc_weight_365',
        'ipc_weight_730': 'ipc_weight_730',
        'ipc_weight_1095': 'ipc_weight_1095'
    }, inplace=True)
    
    print(f"After merge: {len(df):,} rows")

    # Trim PS to [0.05, 0.95]
    before = len(df)
    df = df[(df.ps >= 0.05) & (df.ps <= 0.95)]
    after = len(df)
    print(f"Dropped {before-after:,} rows outside PS [0.05,0.95] "
          f"({100*(before-after)/before:.2f}%) → {after:,} rows remain")

    # Check for issues in PS and all weight columns
    weight_cols = ['ipc_weight_365', 'ipc_weight_730', 'ipc_weight_1095']
    for col in ['ps'] + weight_cols:
        n_miss = df[col].isna().sum()
        n_inf  = (~np.isfinite(df[col])).sum()
        if n_miss:
            print(f"!!!  Warning: {n_miss} missing in {col}")
        if n_inf:
            print(f"!!!  Warning: {n_inf} non-finite values in {col}")

    # Print summaries
    print("\nPropensity (ps) summary:")
    print(df.ps.describe(percentiles=[.01, .05, .5, .95, .99]))
    
    for horizon in [365, 730, 1095]:
        print(f"\nIPC weights (ipc_weight_{horizon}) summary:")
        print(df[f'ipc_weight_{horizon}'].describe(percentiles=[.01, .05, .5, .95, .99]))

    # Group analysis
    df['group'] = 'Never_TX'
    df.loc[(df['T'] == 1) & (df['T_365'] == 1), 'group'] = 'Early_TX'
    df.loc[(df['T'] == 1) & (df['T_365'] == 0), 'group'] = 'Late_TX'
    
    print("\nWeight means by group:")
    print(df.groupby('group')[weight_cols].mean())

    df = df.drop('group', axis=1)

    # Save the prepared data
    out_fp = OUT_DIR / "cand_kipa_csf_data_train.parquet"
    df.to_parquet(out_fp, index=False)
    print(f"\nWrote prepared data to: {out_fp}")
    print(f"Final dataset has {len(df):,} rows and {len(df.columns)} columns")
    
    print("\nFinal columns in dataset:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()