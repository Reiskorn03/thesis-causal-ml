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

## Technical Highlights

This repository documents a causal machine learning workflow built for observational transplant data.

Key technical elements include:
- metadata-driven preprocessing for multi-table clinical registry data
- structured separation of preprocessing, modeling, and evaluation code
- causal modeling for observational health outcome analysis
- evaluation of Causal Survival Forests across multiple time horizons
- handling of restricted clinical data in a public-safe showcase format

## Methodological Overview

The overall workflow represented in this repository includes:

1. preprocessing and harmonization of source data
2. metadata-based feature handling and variable organization
3. cohort construction and study-specific filtering
4. preparation of modeling inputs for causal analysis
5. causal modeling with a focus on **Causal Survival Forests**
6. horizon-based evaluation of treatment effect estimates

## Important note on executability

This public repository is not intended to be fully runnable by external users.  
Some scripts intentionally retain the original thesis development paths and environment assumptions. This is because the underlying raw data is restricted and not part of the repository.

The repository should therefore be understood as a **code showcase and thesis portfolio artifact**, not as a fully reproducible public package.

## Repository Structure

- `src/preprocessing/` — preprocessing pipeline scripts
- `src/modeling/` — causal modeling scripts
- `src/evaluation/` — evaluation scripts
- `results/` — result-oriented evaluation notebooks
- `data/metadata/` — public-safe metadata files
- `docs/` — supplementary project documentation

## Data Access

This public repository does not include the original raw data used in the thesis.

Included:
- public-safe metadata files in `data/metadata/`

Not included:
- raw patient-level data
- processed restricted datasets
- intermediate data products
- outcome parameter artifacts

## What this repository demonstrates

This project is intended to show more than isolated model code. It reflects work on:
- structuring a multi-stage thesis pipeline
- working with restricted observational health data
- organizing preprocessing and modeling logic across separate pipeline components
- applying causal machine learning methods to a clinically relevant problem
- presenting a research codebase in a public portfolio format

