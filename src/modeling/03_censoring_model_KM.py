# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Fit KM censoring model for a specific horizon
    
Parameters:
 - df: DataFrame with columns T_365, T, Y, Delta, PX_ID
 - horizon: int, days (365, 730, or 1095)
    
Returns:
 - result_df: DataFrame with PX_ID, Ghat_censor, ipc_weight
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def fit_censoring_model_for_horizon(df, horizon):
    df_work = df.copy()
    df_work['group'] = 'Never_TX' 
    df_work.loc[(df_work['T'] == 1) & (df_work['T_365'] == 1), 'group'] = 'Early_TX'
    df_work.loc[(df_work['T'] == 1) & (df_work['T_365'] == 0), 'group'] = 'Late_TX'
    
    # Create censoring indicator for horizon x
    df_work['time_to_horizon'] = np.minimum(df_work['Y'], horizon)
    df_work['censored'] = (df_work['Y'] < horizon) & (df_work['Delta'] == 0)
    
    result_df = pd.DataFrame()
    result_df['PX_ID'] = df_work['PX_ID']
    result_df['group'] = df_work['group']
    
    # Fit KM model ->exclude Late TX from censoring model
    mask_for_km = df_work['group'] != 'Late_TX'
    
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df_work.loc[mask_for_km, 'time_to_horizon'],
        event_observed=df_work.loc[mask_for_km, 'censored'],
        label='Censoring KM'
    )
    
    ghat_censor = []
    ipc_weights = []
    
    for idx in df_work.index:
        if df_work.loc[idx, 'group'] == 'Late_TX':
            # Late TX get Ghat = 1 and weight = 1 (artificial censoring ->Landmark design)
            ghat = 1.0
            weight = 1.0
        else:
            # For Early TX and Never TX
            time = df_work.loc[idx, 'time_to_horizon']
            
            # Get probability of NOT being censored by this time (survival probability)
            # This is Ghat_censor(t) = P(C > t)
            ghat = kmf.predict(time)
            
            # IPCW weight = 1 / Ghat_censor(t)
            if ghat > 0.1:  
                weight = 1.0 / ghat
            else:
                weight = 10.0 
                # Set minimum Ghat 
                ghat = 0.1  
        
        ghat_censor.append(float(ghat))
        ipc_weights.append(float(weight))
    
    result_df['Ghat_censor'] = ghat_censor
    result_df['ipc_weight'] = ipc_weights
    
    # Cap extreme weights
    result_df['ipc_weight'] = np.clip(result_df['ipc_weight'], 1.0, 10.0)
    
    # summary stats
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon} days")
    print(f"{'='*60}")
    
    print(f"\nOverall weight statistics:")
    print(result_df['ipc_weight'].describe())
    
    print(f"\nWeights by group:")
    group_stats = result_df.groupby('group')['ipc_weight'].agg([
        'count', 'mean', 'std', 'min', 'max',
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ])
    print(group_stats)
    
    print(f"\nGhat_censor by group:")
    ghat_stats = result_df.groupby('group')['Ghat_censor'].agg([
        'mean', 'std', 'min', 'max'
    ])
    print(ghat_stats)
    
    result_df = result_df[['PX_ID', 'Ghat_censor', 'ipc_weight']]
    
    return result_df

def main():
    BASE_DIR = Path("ATE_CATE")
    DATA_DIR = BASE_DIR / "FINAL_SPLITS"
    df = pd.read_parquet(DATA_DIR / "cand_kipa_train_ver10.parquet")
    
    print(f"Data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Basic data validation
    print(f"\nData validation:")
    print(f"Unique PX_IDs: {df['PX_ID'].nunique()}")
    print(f"T_365 distribution: {df['T_365'].value_counts()}")
    print(f"T distribution: {df['T'].value_counts()}")
    print(f"Delta distribution: {df['Delta'].value_counts()}")
    
    # Identify groups
    df_temp = df.copy()
    df_temp['group'] = 'Never_TX'
    df_temp.loc[(df_temp['T'] == 1) & (df_temp['T_365'] == 1), 'group'] = 'Early_TX'
    df_temp.loc[(df_temp['T'] == 1) & (df_temp['T_365'] == 0), 'group'] = 'Late_TX'
    
    print(f"\nGroup distribution:")
    print(df_temp['group'].value_counts())
    print(f"\nDelta by group:")
    print(pd.crosstab(df_temp['group'], df_temp['Delta'], normalize='index'))
    
    # Process each horizon
    horizons = [365, 730, 1095]
    all_results = {}
    
    for horizon in horizons:
        print(f"\n{'#'*60}")
        print(f"Processing horizon: {horizon} days")
        print(f"{'#'*60}")
        
        # Fit censoring model
        result_df = fit_censoring_model_for_horizon(df, horizon)
        
        all_results[horizon] = result_df
        
        output_filename = f'censoring_weights_horizon_{horizon}.parquet'
        result_df.to_parquet(output_filename, index=False)
        print(f"\nSaved weights to: {output_filename}")
    
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL HORIZONS")
    print(f"{'='*60}")
    
    combined_df = df[['PX_ID', 'T', 'T_365', 'Y', 'Delta']].copy()
    combined_df['group'] = 'Never_TX'
    combined_df.loc[(combined_df['T'] == 1) & (combined_df['T_365'] == 1), 'group'] = 'Early_TX'
    combined_df.loc[(combined_df['T'] == 1) & (combined_df['T_365'] == 0), 'group'] = 'Late_TX'
    
    for horizon in horizons:
        combined_df = combined_df.merge(
            all_results[horizon],
            on='PX_ID',
            suffixes=('', f'_{horizon}')
        )
        combined_df.rename(columns={
            'Ghat_censor': f'Ghat_{horizon}',
            'ipc_weight': f'weight_{horizon}'
        }, inplace=True)
    
    # Print comparison
    print("\nMean weights by group and horizon:")
    weight_cols = [f'weight_{h}' for h in horizons]
    summary_table = combined_df.groupby('group')[weight_cols].mean()
    print(summary_table)
    
    print("\nWeight correlation across horizons:")
    print(combined_df[weight_cols].corr())
    
    # Save combined results
    combined_df.to_parquet('censoring_weights_all_horizons.parquet', index=False)
    print(f"\nSaved combined results to: censoring_weights_all_horizons.parquet")
    
    print("\nDone.")
if __name__ == "__main__":
    main()