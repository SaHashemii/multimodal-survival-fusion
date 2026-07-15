# Multimodal Survival Fusion

## Overview

A configurable PyTorch framework for multimodal survival prediction using histopathology, RNA-seq, and clinical data, with a systematic comparison of fusion strategies and robustness to missing RNA-seq.

<p align="center">
  <img src="Overview.png" alt="Overview of the multimodal survival prediction framework">
</p>

This repository provides a configurable benchmark for multimodal survival prediction of BCG progression in high-risk non-muscle-invasive bladder cancer (HR-NMIBC). It compares unimodal, bimodal, and trimodal models using histopathology whole-slide images, bulk RNA-seq, and clinical data.

The framework implements both embedding-based and token-based multimodal fusion strategies, including concatenation, scalar-gated fusion, low-rank bilinear fusion, and SurvPGC-style co-attention. It also supports multiple modality representations, including UNI pathology features, tabular and text-embedded clinical data, variance-filtered RNA, pathway and biological-category RNA tokens, and scFoundation embeddings.

To reflect clinical practice where RNA-seq is often unavailable, the framework includes RNA-dropout training and evaluates robustness under missing-RNA inference.

## Key Features

- Benchmark of unimodal, bimodal, and trimodal survival prediction models
- Comparison of embedding-based and token-based multimodal fusion strategies
- Support for pathology, bulk RNA-seq, and clinical data
- Multiple modality representations, including UNI, BioClinical ModernBERT, CONCH, and scFoundation
- RNA-dropout training for robustness to missing RNA-seq
- YAML-based experiment configuration for reproducible benchmarking
- Five-fold cross-validation with configurable training pipelines

## Supported Modalities
- WSI
- Bulk RNA-Seq
- Clinical data

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
| BioClinical ModernBERT | ModernBERT text embeddings | `[5, 768]`  |
| CONCH text encoder | Text embeddings | `[5, 512]`  |


## Modality Representations

#### RNA-seq Embeddings ([scFoundation](https://github.com/biomap-research/scFoundation))

RNA-seq data were reduced in dimensionality by filtering genes using Hallmark and Reactome pathway collections. Only pathways with at least 90% gene coverage were retained, and the union of their genes was used as input to the scFoundation model. The resulting embeddings were used as RNA representations for downstream multimodal learning.

#### Clinical Text Embeddings ([BioClinical ModernBERT](https://github.com/lindvalllab/BioClinical-ModernBERT/tree/main) and [CONCH](https://github.com/mahmoodlab/CONCH))

Clinical variables were converted into natural language sentences before encoding. Each variable (e.g., age, sex, smoking status, number of BCG instillations,...) was expressed as a descriptive sentence and encoded using BioClinical ModernBERT or the CONCH text encoder to generate contextual embeddings.

**Example:**

Structured input:

```
Age: 67
Sex: Male
Smoking status: yes/no
```

Converted text:

```
The patient is 67 years old.
The patient is male.
The patient has a history of smoking.
```

See [`resources/templates/clinical_embedding_sentence_templates.csv`](resources/templates/clinical_embedding_sentence_templates.csv) for the full clinical text templates.

## Models

The repository implements Cox proportional hazards models for unimodal, bimodal, and trimodal survival prediction using histopathology (P), bulk RNA-seq (R), and clinical (C) data.

| Configuration | Modalities | Supported fusion strategies |
|--------------|------------|-----------------------------|
| Unimodal | P, R, or C | Cox survival model |
| Bimodal | P+R, P+C, R+C | Concatenation, Scalar-Gated, Low-Rank Bilinear |
| Trimodal | P+R+C | Concatenation, Scalar-Gated, Low-Rank Bilinear, SurvPGC-style Co-Attention |

Unimodal models evaluate the predictive value of each modality independently. Bimodal models assess complementary information between modality pairs, while trimodal models integrate all three modalities for multimodal survival prediction.


## Fusion Strategies

| Fusion Strategy | Input | Description |
|-----------------|-------|-------------|
| Concatenation | Embeddings | Concatenates modality embeddings before Cox prediction. |
| Scalar-Gated | Embeddings | Learns patient-specific weights for each modality prior to fusion. |
| Low-Rank Bilinear | Embeddings | Models compact pairwise interactions between modalities. |
| SurvPGC-style Co-Attention | Tokens | Performs token-level cross-modal attention before survival prediction. |

<p align="center">
  <img src="fusion_comparison.png" alt="Fusion Comparison" width="1000">
</p>

<p align="center">
  <em>Comparison of the four implemented multimodal fusion strategies.</em>
</p>

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
Clone the repository and install the package:

```
git clone https://github.com/SaHashemii/multimodal-survival-fusion.git
cd multimodal-survival-fusion
pip install -e .
````

> The training pipelines use precomputed modality features. Generating foundation-model embeddings may require additional dependencies.

## Dataset Preparation

This repository does not include the CHIMERA dataset or precomputed modality features.

The training pipeline expects the following inputs:

- Histopathology whole-slide image (WSI) features (e.g., UNI embeddings)
- Bulk RNA-seq data or precomputed RNA representations
- Clinical variables or precomputed clinical embeddings
- Survival labels and patient metadata

Configure dataset paths in a YAML file under [`configs/data/`](configs/data/) before running experiments.
## Configuration

Experiments are configured using two YAML files:

- [`configs/data/`](configs/data/) defines dataset paths and modality resources.
- [`configs/experiments/`](configs/experiments/) defines the model architecture, fusion strategy, and training hyperparameters.

Running an experiment requires specifying one data configuration and one experiment configuration.
For a complete list of experiment configurations, see [`configs/`](configs/).


## Running Experiments

All experiments are run from the repository root by specifying a dataset configuration and an experiment configuration.

### Unimodal

```bash
python3 scripts/train_unimodal.py \
  --experiment configs/experiments/unimodal/rna_unimodal.yaml \
  --data configs/data/local.yaml \
  --fold-assignments /path/to/fold_assignments.csv \
  --output-dir /path/to/output_dir
```

### Bimodal

```bash
python3 scripts/train_two_modality.py \
  --experiment configs/experiments/bimodal/concat_rna_clinical.yaml \
  --data configs/data/local.yaml \
  --fold-assignments /path/to/fold_assignments.csv \
  --output-dir /path/to/output_dir
```

### Trimodal

```bash
python3 scripts/train_cv.py \
  --experiment configs/experiments/trimodal/concat.yaml \
  --data configs/data/local.yaml \
  --fold-assignments /path/to/fold_assignments.csv \
  --output-dir /path/to/output_dir
```

### SurvPGC-style Co-Attention

```bash
python3 scripts/train_survpgc.py \
  --experiment configs/experiments/survpgc/survpgc_category.yaml \
  --data configs/data/local.yaml \
  --fold-assignments /path/to/fold_assignments.csv \
  --output-dir /path/to/output_dir
```

> **Note:** Replace `/path/to/fold_assignments.csv` with a CSV containing `sample_id` and `fold` columns. Replace `/path/to/output_dir` with the directory where you want to save the training outputs. If `--fold-assignments` is omitted, the training scripts will create or reuse fold assignments in the output directory.

For additional experiment configurations, see [`configs/`](configs/).

## Results

| Configuration | Best C-index |
|---------------|-------------:|
| Unimodal | 0.704 ± 0.051 |
| Bimodal | 0.698 ± 0.030 |
| Trimodal | **0.725 ± 0.083** |
| Best missing-RNA | **0.720 ± 0.036** |

## Acknowledgements

This repository builds upon several publicly available resources:

- [UNI](https://github.com/mahmoodlab/UNI)
- [CONCH](https://github.com/mahmoodlab/CONCH)
- [BioClinical ModernBERT](https://github.com/lindvalllab/BioClinical-ModernBERT)
- [scFoundation](https://github.com/biomap-research/scFoundation)
- [SurvPGC](https://github.com/Houjiaxin123/SurvPGC)
