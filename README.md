# Multimodal Survival Prediction

# Multimodal Survival Fusion

## Overview
Short description of the project: multimodal survival prediction using RNA, clinical, pathology, and omics/pathway representations.

## Modalities
List the input types:
- RNA expression
- Clinical tabular data
- Clinical text embeddings
- Pathology tile embeddings
- scFoundation omics embeddings
- Biological category/pathway RNA tokens

## Models
Briefly list implemented model families:
- RNA unimodal Cox
- Clinical unimodal Cox
- Pathology unimodal Cox
- Concat multimodal fusion
- Gated concat multimodal fusion
- Low-rank bilinear multimodal fusion
- SurvPGC-style multimodal model

## Repository Structure
Explain folders:
- `configs/data/`
- `configs/experiments/`
- `scripts/`
- `src/mm_survival/`
- `legacy/`
- `resources/`

## Configuration
Explain:
- `configs/data/local.yaml`
- `configs/data/example.yaml`
- experiment YAMLs
- how `label_name`, `clinical_embedding_name`, `pathology_feature_name`, and `omics_source` work

## Running Experiments
Show commands for:
- unimodal models
- concat/gated/lowrank models
- SurvPGC models

Example:

```bash
python3 scripts/train_unimodal.py \
  --experiment configs/experiments/rna_unimodal.yaml \
  --data configs/data/local.yaml
