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

#### SurvPGC-style Co-Attention Fusion

Pathology, RNA-seq, and clinical tokens are projected into a shared feature space and fused through bidirectional co-attention. Separate co-attention modules model interactions between pathology-RNA and pathology-clinical representations, enabling cross-modal information exchange before survival prediction.

For implementation details, see the fusion modules in `src/mm_survival/models/fusion/`.

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

## Installation
## Dataset Format
## Configuration

Data locations, model architectures, fusion strategies, and training settings are controlled through YAML configuration files.

### How Configs Work

The repository separates dataset configuration from experiment configuration:

* `configs/data/` contains dataset-specific settings, including data locations, labels, embeddings, pathology features, and external resources.
* `configs/experiments/` contains experiment-specific settings, including model architecture, modality selection, fusion strategy, hyperparameters, and cross-validation settings.

This separation allows the same experiment configuration to be reused across different datasets and environments.

### Data Selectors

Several fields in the experiment configurations select among options defined in the data configuration:

* `label_name` selects the label file used for training and evaluation.
* `clinical_embedding_name` selects the clinical text embedding source (e.g., BioClinical ModernBERT or CONCH).
* `pathology_feature_name` selects the pathology feature set (e.g., UNI or PRISM).
* `omics_source` selects the omics representation used by SurvPGC-style models:

  * `pathway` – pathway-based RNA tokens
  * `category` – biological-category RNA tokens
  * `scfoundation` – precomputed scFoundation embeddings

### Experiment Configurations

| Config file                                | Model / fusion                           | Clinical input             | Omics / RNA input                           | Pathology input     |
| ------------------------------------------ | ---------------------------------------- | -------------------------- | ------------------------------------------- | ------------------- |
| `rna_unimodal.yaml`                        | RNA-only Cox                             | Not used                   | RNA expression with variance-filtered genes | Not used            |
| `clinical_unimodal.yaml`                   | Clinical-only Cox                        | Clinical text embeddings   | Not used                                    | Not used            |
| `pathology_unimodal.yaml`                  | Pathology-only Cox                       | Not used                   | Not used                                    | UNI tile embeddings |
| `concat.yaml`                              | Concatenation fusion                     | Tabular clinical variables | RNA expression with variance-filtered genes | UNI tile embeddings |
| `concat_clinical_embedding.yaml`           | Concatenation fusion                     | Clinical text embeddings   | RNA expression with variance-filtered genes | UNI tile embeddings |
| `gated_concat.yaml`                        | Gated concatenation fusion               | Tabular clinical variables | RNA expression with variance-filtered genes | UNI tile embeddings |
| `gated_concat_clinical_embedding.yaml`     | Gated concatenation fusion               | Clinical text embeddings   | RNA expression with variance-filtered genes | UNI tile embeddings |
| `lowrank_bilinear.yaml`                    | Low-rank bilinear fusion                 | Tabular clinical variables | RNA expression with variance-filtered genes | UNI tile embeddings |
| `lowrank_bilinear_clinical_embedding.yaml` | Low-rank bilinear fusion                 | Clinical text embeddings   | RNA expression with variance-filtered genes | UNI tile embeddings |
| `survpgc.yaml`                             | SurvPGC-style bidirectional co-attention | Clinical text embeddings   | Pathway-based RNA tokens                    | UNI tile embeddings |
| `survpgc_category.yaml`                    | SurvPGC-style bidirectional co-attention | Clinical text embeddings   | Biological-category RNA tokens              | UNI tile embeddings |
| `survpgc_scfoundation.yaml`                | SurvPGC-style bidirectional co-attention | Clinical text embeddings   | scFoundation RNA-seq embeddings             | UNI tile embeddings |


## Running Experiments

All experiments are run from the repository root using a data configuration file and an experiment configuration file.

### Unimodal Models

```bash
python3 scripts/train_unimodal.py \
  --experiment configs/experiments/rna_unimodal.yaml \
  --data configs/data/local.yaml
```

### Multimodal Fusion Models

```bash
python3 scripts/train_cv.py \
  --experiment configs/experiments/concat.yaml \
  --data configs/data/local.yaml
```

### SurvPGC-style Co-Attention Models

```bash
python3 scripts/train_survpgc.py \
  --experiment configs/experiments/survpgc.yaml \
  --data configs/data/local.yaml
```

To run a different experiment, replace the experiment configuration file with any configuration listed in the Configuration section.

## Results
## Citations
