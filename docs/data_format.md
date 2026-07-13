# Data Format

This repository does not assume a fixed data directory name. All paths are supplied through a data config file such as `configs/data/local.yaml`.

## Example Layout

```text
data_root/
├── labels.csv
├── rna.csv
├── clinical.csv
├── clinical_embeddings/
│   ├── SAMPLE_001.pt
│   └── SAMPLE_002.pt
├── omics_embeddings/
│   ├── SAMPLE_001.pt
│   └── SAMPLE_002.pt
├── pathology_index.csv
└── pathology_features/
    ├── SAMPLE_001.pt
    └── SAMPLE_002.pt
```

## Path Resolution

Paths in the data config may be absolute or relative.

- Absolute paths are used as-is.
- Relative paths are resolved against `data.root`.
- `pathology_index.csv` may contain absolute feature paths or paths relative to `pathology_features_root`.

## Required Files

### `labels.csv`

Required columns:

| Column | Description |
|---|---|
| `sample_id` | Patient/sample identifier used to join modalities. |
| `Time` | Survival or progression time. |
| `Event` | Event indicator, where `1` means observed event and `0` means censored. |

### `rna.csv`

RNA expression matrix. The expected public format is genes as rows and samples as columns:

```text
Gene,SAMPLE_001,SAMPLE_002
GENE_A,1.2,0.7
GENE_B,3.4,2.1
```

During training, RNA preprocessing must fit imputation and feature-selection statistics on the training fold only.

### `clinical.csv`

Tabular clinical file with one row per sample. Required column:

| Column | Description |
|---|---|
| `sample_id` | Patient/sample identifier. |

All other columns are treated as clinical covariates.

### `pathology_index.csv`

Required columns:

| Column | Description |
|---|---|
| `sample_id` | Patient/sample identifier. |
| `feature_path` | Path to the saved `.pt` pathology feature file for that sample. |

`feature_path` can be absolute or relative to `pathology_features_root`. The CSV is the source of truth for matching each sample to its pathology features.

### Clinical Embeddings

Clinical embeddings should contain one `.pt` file per sample:

```text
SAMPLE_001.pt
SAMPLE_002.pt
```

The filename stem must match `sample_id`. These files are also used as clinical tokens in token-based models.

### Omics Embeddings

Omics embedding folders should contain one `.pt` file per sample with filename stems matching `sample_id`.

Supported tensor payloads:

- a tensor directly
- a dictionary containing `feats`
- a dictionary containing `embeddings`

Expected tensor shape is usually `[tokens, dim]`. One-dimensional tensors may be interpreted as a single token depending on the loader.

## Private Local Configs

Do not commit real dataset paths or patient data. Create a private config such as:

```text
configs/data/local.yaml
```

This file is ignored by git through `.gitignore`.
