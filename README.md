# Multimodal Survival Prediction

# Multimodal Survival Fusion

## Overview
Short description of the project: multimodal survival prediction using RNA, clinical, pathology, and omics/pathway representations.

## Modalities

### Pathology (WSI)

| Encoder | Representation | Shape |
|----------|---------------|--------|
| UNI | Tile embeddings | `[N_tiles, 1024]` |

### Bulk RNA-seq

| Encoder | Representation | Shape |
|----------|---------------|--------|
| scFoundation | Gene expression embeddings | `[4, 768]` |
| — | Raw gene expression | `[1, 19359]` |
| — | Pathway tokens | `[N_pathways, N_genes_per_pathway]` |
| — | Biological category tokens | `[6, N_genes_per_category]` |


### Clinical

| Encoder | Representation | Shape |
|----------|---------------|--------|
| — | Raw tabular | `[1, N_vars]` |
| BioClinical ModernBERT | ModernBERT text embeddings | `[6, 768]` or `[13, 768]` |
| CONCH text encoder | Text embeddings | `[6, 512]` or `[13, 512]` |


### Foundation Model Embeddings

#### RNA-seq Embeddings ([scFoundation](https://github.com/biomap-research/scFoundation))

RNA-seq data were reduced in dimensionality by filtering genes using Hallmark and Reactome pathway collections. Only pathways with at least 90% gene coverage were retained, and the union of their genes was used as input to the scFoundation model. The resulting embeddings were used as RNA representations for downstream multimodal learning.

#### Clinical Text Embeddings ([BioClinical ModernBERT](https://github.com/lindvalllab/BioClinical-ModernBERT/tree/main) and [CONCH](https://github.com/mahmoodlab/CONCH))

Clinical variables were converted into natural language sentences before encoding. Each variable (e.g., age, sex, smoking status, tumor stage) was expressed as a descriptive sentence and encoded using BioClinical ModernBERT or the CONCH text encoder to generate contextual embeddings.

**Example:**

Structured input:

```
Age: 67
Sex: Male
Smoking: Former smoker
```

Converted text:

```
The patient is 67 years old.
The patient is male.
The patient is a former smoker.
```

See [`resources/templates/clinical_embedding_sentence_templates.csv`](resources/templates/clinical_embedding_sentence_templates.csv) for the full clinical text templates.
## Models

This repository implements both unimodal and multimodal survival prediction models using pathology, RNA-seq, and clinical data.

### Unimodal Models

| Modality | Representation |
|-----------|-----------|
| Pathology | UNI tile embeddings |
| RNA-seq | Variance-filtered gene expression |
| Clinical | BioClinical ModernBERT or CONCH text embeddings |

#### RNA-only Cox Model

Gene expression profiles are filtered using coefficient-of-variation feature selection and processed by an MLP encoder. The resulting patient representation is used for Cox survival prediction.

#### Clinical-only Cox Model

Structured clinical variables are converted into natural language sentences and encoded using BioClinical ModernBERT or CONCH. The resulting embeddings are used for Cox survival prediction.

#### Pathology-only Cox Model

UNI tile embeddings extracted from whole-slide images are aggregated using gated attention multiple-instance learning (MIL) to generate slide-level representations for Cox survival prediction.

---

### Multimodal Models

The repository includes multiple fusion strategies for integrating pathology, RNA-seq, and clinical information.

| Fusion Strategy | Input | Modalities |
|----------------|--------|------------|
| Concatenation Fusion | Embeddings | P+R+C |
| Gated Fusion | Embeddings | P+R+C |
| Low-Rank Bilinear Fusion | Embeddings | P+R+C |
| SurvPGC-style Co-Attention Fusion | Tokens | P+R+C |

#### Concatenation Fusion

Pathology, RNA-seq, and clinical embeddings are concatenated into a single feature vector and used for survival prediction.

#### Gated Fusion

Pathology, RNA-seq, and clinical embeddings are weighted by learnable gates before fusion, allowing the model to adaptively determine the contribution of each modality.

#### Low-Rank Bilinear Fusion

Interactions between pathology, RNA-seq, and clinical representations are modeled using low-rank bilinear projections, capturing pairwise cross-modal relationships before survival prediction.

#### ASurvPGC-style Co-Attention Fusion

Pathology, RNA-seq, and clinical tokens are projected into a shared feature space and fused through bidirectional co-attention. Separate co-attention modules model interactions between pathology-RNA and pathology-clinical representations, enabling cross-modal information exchange before survival prediction.

For implementation details, see the corresponding fusion modules in the fusion/ directory.

## Repository Structure


```text
multimodal-survival-fusion/
├── configs/
├── docs/
├── resources/
├── scripts/
├── src/
│   └── mm_survival/
│       ├── data/
│       ├── models/
│       │   ├── encoders/
│       │   └── fusion/
│       ├── training/
│       └── utils/
└── tests/
```

### Folder Overview

* `configs/data/`
  Dataset configuration files defining data locations, labels, modality inputs, embeddings, and external resources.

* `configs/experiments/`
  Experiment-specific configurations defining model architectures, fusion strategies, hyperparameters, and cross-validation settings.

* `scripts/`
  Entry-point scripts for training unimodal models, multimodal fusion models, SurvPGC-style models, and generating cross-validation folds.

* `src/mm_survival/`
  Core Python package containing data loaders, modality encoders, fusion modules, survival models, training pipelines, evaluation metrics, and utility functions.

* `resources/`
  Reusable resources including pathway databases, biological category definitions, and clinical text templates.


```
```


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
