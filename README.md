# Assessing the Impact of Organ Transplantation on Health Outcomes Using Machine Learning

This repository is a cleaned public showcase version of the codebase used for my bachelor thesis on causal machine learning and transplantation outcomes.

## Purpose of this repository

The goal of this repository is to present the structure and methodology of the thesis workflow in a clear and professional way.

It contains:
- preprocessing scripts
- modeling scripts
- evaluation scripts
- result-oriented notebooks
- public-safe metadata

It does **not** contain:
- raw patient-level data
- processed restricted datasets
- outcome parameter artifacts that should remain private

## Important note on executability

This public repository is not intended to be fully runnable by external users.  
Some scripts intentionally retain the original thesis development paths and environment assumptions. This is because the underlying raw data is restricted and not part of the repository.

The repository should therefore be understood as a **code showcase and thesis portfolio artifact**, not as a fully reproducible public package.

## Structure

- `src/preprocessing/` — preprocessing pipeline scripts
- `src/modeling/` — causal modeling scripts
- `src/evaluation/` — evaluation scripts
- `results/` — result-oriented evaluation notebooks
- `data/metadata/` — public-safe metadata files
- `docs/` — supplementary project documentation

## Data Folder

This public repository does not include the original raw data used in the thesis.

Included:
- public-safe metadata files in `data/metadata/`

Not included:
- raw patient-level data
- processed restricted datasets
- intermediate data products
- outcome parameter artifacts