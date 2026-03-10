# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Load a fitted honest causal survival forest (RDS), apply it on held-out test data, compute test‐set ATE ± CI and individual CATEs, saving all key outputs.
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
MODEL_DIR = os.path.join(EC2_DIR, "ATE_CATE", "Model_New_horizon_3")
TEST_FP = os.path.join(DATA_DIR, "cand_kipa_test_ver10.parquet")
FOREST_RDS = os.path.join(MODEL_DIR, "csf_forest_train.rds")

OUT_ATE = os.path.join(MODEL_DIR, "csf_ate_test.csv")
OUT_CATE = os.path.join(MODEL_DIR, "csf_cate_test.csv")

SMOKE_TEST = False   # True -> subsample 
SMOKE_N = 2000

def main():
    print(f"Loading {TEST_FP}")
    df = pd.read_parquet(TEST_FP)
    print("-> rows:", len(df))

    if SMOKE_TEST:
        df = df.sample(n=min(SMOKE_N, len(df)), random_state=42).reset_index(drop=True)
        print(f"Smoke test: subsampled -> {len(df)} rows")

    # drop non‐Input features
    reserved = {
      "PX_ID","WL_EVENT","WL_TIME",
      "T","T_365","Y","Delta","CAN_ABO_Unkown"
    }
    covars = [c for c in df.columns if c not in reserved]
    print(f"-> using {len(covars)} covariates")

    X_df = df[covars]
    W = df["T_365"].astype(int)
    Y = df["Y"].astype(float)
    D = df["Delta"].astype(int)

    with localconverter(default_converter + pandas2ri_converter):
        ro.globalenv["X"]      = conversion.py2rpy(X_df)
        ro.globalenv["PX_ID"]  = conversion.py2rpy(df["PX_ID"])

    r_code = f'''
    library(grf)

    # load saved forest
    csf <- readRDS("{FOREST_RDS}")

    # matrix of covariates
    X_mat <- as.matrix(X)

    # Test‐set ATE ± CI
    pred <- predict(csf, newdata = X_mat)
    tau_hat <- pred$predictions

    # simple plug‐in standard error
    ate <- mean(tau_hat)
    se <- sd(tau_hat) / sqrt(length(tau_hat))
    lower <- ate - 1.96 * se
    upper <- ate + 1.96 * se

    # write out ATE ± SE/CI
    write.csv(
      data.frame(
        estimate = ate,
        se = se,
        lower = lower,
        upper = upper
      ),
      file = "{OUT_ATE}",
      row.names = FALSE
    )
    cat(sprintf("-> TEST ATE = %.4f, SE = %.4f, 95%%CI = [%.4f, %.4f]\\n",
        ate, se, lower, upper
    )); flush.console()

    # Individual CATEI tried s on Test
    tau_hat <- predict(csf, newdata = X_mat)$predictions
    write.csv(
      data.frame(
        PX_ID = PX_ID,
        tau = tau_hat
      ),
      file = "{OUT_CATE}",
      row.names = FALSE
    )
    cat("-> Wrote test CATEs to {OUT_CATE}\\n")
    '''
    print("-> running evaluation in R...")
    ro.r(r_code)
    print("Done: evaluation results in", MODEL_DIR)


if __name__ == "__main__":
    main()
