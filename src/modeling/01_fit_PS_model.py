# Note: This script retains the original thesis development paths and is not intended
# to be executed in the public repository without access to the private data environment.

"""
Fit propensity scores via grf::probability_forest on cand_kipa_train_ver10, save both the PS vector and the fitted forest object for later scoring.
"""
import os
import subprocess
import pandas as pd
import tempfile
import textwrap
from pathlib import Path

BASE_DIR = os.path.expanduser("~/Thesis")
EC2_DIR = os.path.expanduser("~/ThesisCode")
FEATURE_DIR = os.path.join(BASE_DIR, "ATE_CATE", "FINAL_SPLITS")
NUISANCE_DIR = os.path.join(EC2_DIR, "ATE_CATE", "PS_validation")

TRAIN_FP = os.path.join(FEATURE_DIR, "cand_kipa_train_ver10.parquet")
OUTPUT_PS_PARQ = os.path.join(NUISANCE_DIR, "ps_cand_kipa_train.parquet")
OUTPUT_MODEL_RDS= os.path.join(NUISANCE_DIR, "ps_forest_train.rds")

# cexclude non cavariate columns
EXCLUDE_COLS = {"PX_ID", "WL_TIME", "WL_EVENT", "Y", "T", "Delta"}

def main():
    os.makedirs(NUISANCE_DIR, exist_ok=True)

    df = pd.read_parquet(TRAIN_FP)
    features = [c for c in df.columns if c not in EXCLUDE_COLS and c != "T_365"]

    if "T_365" not in df.columns:
        raise KeyError("T_365 must be present in the training set")

    # coerce any booleans
    for col in features:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'object' and set(df[col].dropna().unique()) <= {"True","False"}:
            df[col] = df[col].map({'True':1,'False':0}).astype(int)

    # CSV for R to read
    tmp_dir = Path(tempfile.mkdtemp())
    csv_in = tmp_dir / "ps_input.csv"
    csv_out = tmp_dir / "ps_output.csv"
    df[features + ["T_365"]].to_csv(csv_in, index=False)

    # write out a R script that fits and saves both PS and forest
    r_script = tmp_dir / "fit_ps.R"
    r_script.write_text(textwrap.dedent(f"""
        args <- commandArgs(trailingOnly=TRUE)
        infile <- args[1]
        ps_out <- args[2]
        model_out <- args[3]

        library(grf)
        library(parallel)

        df <- read.csv(infile, stringsAsFactors=FALSE)
        X <- as.matrix(df[ , setdiff(names(df),"T_365") ])
        Y <- as.factor(df$T_365)

        # honest probability forest
        ps_forest <- probability_forest(
          X, Y,
          num.trees = 2000,
          honesty = TRUE,
          num.threads = detectCores()
        )

        # OOB propensities
        ps <- predict(ps_forest, estimate.oob = TRUE)$predictions

        # write out PS vector
        write.csv(
          data.frame(ps = ps),
          file = ps_out,
          row.names = FALSE,
          quote = FALSE
        )

        # save the forest object for later
        saveRDS(ps_forest, file = model_out)
    """).strip())

    # call Rscript to fit
    subprocess.run(
        ["Rscript", str(r_script),
         str(csv_in), str(csv_out), str(OUTPUT_MODEL_RDS)],
        check=True
    )

    # read back the PS, attach PX_ID, and write to parquet
    ps_df = pd.read_csv(csv_out)
    if "ps" not in ps_df.columns:
        ps_df = ps_df.rename(columns={"ps.1": "ps"})
    print(ps_df.columns)
    result = pd.DataFrame({
        "PX_ID": df["PX_ID"].values,
        "ps"   : ps_df["ps"].values
    })
    result.to_parquet(OUTPUT_PS_PARQ, index=False)
    print(f"Wrote propensity scores -> {OUTPUT_PS_PARQ}")
    print(f"Wrote fitted forest object -> {OUTPUT_MODEL_RDS}")

if __name__ == "__main__":
    main()


