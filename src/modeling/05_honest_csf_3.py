# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Fit an honest causal survival forest via R grf::causal_survival_forest,

Outputs:
  - csf_ate_train.csv: ATE estimate ± SE/CI
  - csf_cate_train.csv: Individual CATE estimates (τ̂_i)
  - csf_variable_importance.csv: Variable importances
  - csf_forest_train.rds: Saved forest object
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")
os.environ.pop("BASH_FUNC_which%%", None)
os.environ.pop("R_SESSION_TMPDIR", None)

import rpy2.robjects as ro
from rpy2.robjects import default_converter, conversion
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.pandas2ri import converter as pandas2ri_converter

BASE_DIR = os.path.expanduser("~/Thesis")
EC2_DIR = os.path.expanduser("~/ThesisCode")

DATA_DIR = os.path.join(BASE_DIR, "ATE_CATE", "CSF_data_NEW2")
MODEL_DIR  = os.path.join(EC2_DIR, "ATE_CATE", "Model_New_horizon_3")
os.makedirs(MODEL_DIR, exist_ok=True)

FINAL_FP   = os.path.join(DATA_DIR, "cand_kipa_csf_data_train.parquet")

N_SAMPLE = None   # None for full scale run
NUM_TREES = 5000
SAMPLE_FRAC = 0.5
HONEST_FRAC = 0.5
HORIZON_DAYS = 3 * 365

def main():

    print(f"Loading {FINAL_FP}")
    df = pd.read_parquet(FINAL_FP)
    print("-> rows:", len(df))

    # compute IPTW
    df["iptw"] = np.where(
        df["T_365"] == 1,
        1.0 / df["ps"],
        1.0 / (1.0 - df["ps"])
    )
    print(f"-> IPTW: mean={df['iptw'].mean():.3f}, min={df['iptw'].min():.3f}, max={df['iptw'].max():.3f}")

    # combine with IPCW to get final weights
    if HORIZON_DAYS == 365:
        df["w_final"] = df["iptw"] * df["ipc_weight_365"]
    elif HORIZON_DAYS == 730:
        df["w_final"] = df["iptw"] * df["ipc_weight_730"]
    elif HORIZON_DAYS == 1095:
        df["w_final"] = df["iptw"] * df["ipc_weight_1095"]
    else:
        raise ValueError(f"Unexpected horizon: {HORIZON_DAYS}")

    print(f"-> W_final: mean={df['w_final'].mean():.3f}, min={df['w_final'].min():.3f}, max={df['w_final'].max():.3f}")

    if N_SAMPLE is not None:
        df = df.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
        print(f"Smoke test: subsampled -> {len(df)} rows")

    # reserved non-Input columns (do not include as X)
    reserved = {
      "PX_ID","WL_EVENT","WL_TIME",
      "T","T_365","Y","Delta",
      "ps","grp","Ghat_censor",
      "ipc_weight_365","ipc_weight_730","ipc_weight_1095",  
      "iptw","w_final"
    }
    covars = [c for c in df.columns if c not in reserved]
    print(f"-> using {len(covars)} covariates")  # should be 72

    X_df = df[covars]
    W = df["T_365"].astype(int)
    Y = df["Y"].astype(float)
    D = df["Delta"].astype(int)
    ps_hat = df["ps"].astype(float)
    sw = df["w_final"].astype(float)

    # prepare for R-call
    with localconverter(default_converter + pandas2ri_converter):
        ro.globalenv["X"] = conversion.py2rpy(X_df)
        ro.globalenv["W"] = conversion.py2rpy(W)
        ro.globalenv["Y"] = conversion.py2rpy(Y)
        ro.globalenv["D"] = conversion.py2rpy(D)
        ro.globalenv["ps_hat"] = conversion.py2rpy(ps_hat)
        ro.globalenv["sw"] = conversion.py2rpy(sw)
        ro.globalenv["PX_ID"] = conversion.py2rpy(df["PX_ID"])

    ro.globalenv["num_trees"] = ro.IntVector([NUM_TREES])
    ro.globalenv["sample_frac"] = ro.FloatVector([SAMPLE_FRAC])
    ro.globalenv["honesty_frac"] = ro.FloatVector([HONEST_FRAC])
    ro.globalenv["horizon"] = ro.FloatVector([HORIZON_DAYS])

    r_code = f'''
    library(grf)

    # prepare matrices
    X_mat <- as.matrix(X)
    # T_365 is separate, so covariates = X_mat
    X0 <- X_mat

    failure.times <- seq(0, horizon, by=30)  

    # fit honest causal survival forest
    csf <- causal_survival_forest(
      X = X0,
      Y = Y,
      D = D,
      W = W,
      W.hat = ps_hat,
      sample.weights = sw,
      horizon = horizon,
      failure.times = failure.times,
      num.trees = num_trees,
      sample.fraction = sample_frac,
      honesty.fraction = honesty_frac,
      honesty = TRUE,
      num.threads = parallel::detectCores()
    )

    # ATE + SE/CI
    ate_res <- average_treatment_effect(csf, target.sample="all")
    write.csv(
      data.frame(
        estimate = ate_res[1],
        se = ate_res[2],
        lower = ate_res[1] - 1.96*ate_res[2],
        upper = ate_res[1] + 1.96*ate_res[2]
      ),
      file="{MODEL_DIR}/csf_ate_train.csv",
      row.names=FALSE
    )

    cat(sprintf(" -> ATE=%.4f, SE=%.4f, 95%%CI=[%.4f,%.4f]\n",
        ate_res[1], ate_res[2],
        ate_res[1]-1.96*ate_res[2],
        ate_res[1]+1.96*ate_res[2]
    )); flush.console()

    # Individual CATEs
    tau_hat <- predict(csf)$predictions
    write.csv(
      data.frame(
        PX_ID = PX_ID,
        tau = tau_hat
      ),
      file="{MODEL_DIR}/csf_cate_train.csv",
      row.names=FALSE
    )

    # Variable importance
    vi <- variable_importance(csf)
    write.csv(
      data.frame(
        feature = colnames(X0),
        importance = vi
      ),
      file="{MODEL_DIR}/csf_variable_importance.csv",
      row.names=FALSE
    )

    # Save forest object
    saveRDS(csf, file="{MODEL_DIR}/csf_forest_train.rds")
    '''
    print("->running causal_survival_forest in R...")
    ro.r(r_code)
    print("Done. All outputs under", MODEL_DIR)


if __name__ == "__main__":
    main()
