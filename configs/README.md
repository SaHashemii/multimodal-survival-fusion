## Configuration

The repository uses YAML configuration files to separate **dataset settings** from **experiment settings**, making it easy to reuse the same experiments across different datasets and environments.

### Data Configuration

Files in `configs/data/` define dataset-specific settings, including:

- Data locations
- Survival labels
- Histopathology features
- RNA data and embeddings
- Clinical data and embeddings
- External resources

### Experiment Configuration

Files in `configs/experiments/` define the model architecture and training setup. Experiment configurations are organized by model type:

```text
configs/experiments/
├── unimodal/
├── bimodal/
├── trimodal/
├── survpgc/
└── late_fusion/
```

### Data Selectors

Experiment configurations reference options defined in the data configuration through the following fields:

- `label_name` – survival label file
- `clinical_embedding_name` – clinical embedding source (e.g., BioClinical ModernBERT or CONCH)
- `pathology_feature_name` – pathology feature set (e.g., UNI or PRISM)
- `omics_source` – RNA representation used by SurvPGC models:
  - `pathway`
  - `category`
  - `scfoundation`

### Available Experiment Configurations

| Config | Modalities | Fusion / Model | Clinical input | RNA / Omics input | Pathology input |
|---|---|---|---|---|---|
| `unimodal/rna_unimodal.yaml` | RNA | Unimodal Cox | - | Variance-filtered RNA | - |
| `unimodal/clinical_unimodal.yaml` | Clinical | Unimodal Cox | Clinical text embeddings | - | - |
| `unimodal/clinical_unimodal_tabular.yaml` | Clinical | Unimodal Cox | Tabular clinical | - | - |
| `unimodal/pathology_unimodal.yaml` | Pathology | Unimodal Cox | - | - | UNI tile embeddings |
| `unimodal/pathology_unimodal_prism.yaml` | Pathology | Unimodal Cox | - | - | PRISM slide embedding |
| `bimodal/concat_rna_clinical.yaml` | RNA + Clinical | Concatenation | Tabular clinical | Variance-filtered RNA | - |
| `bimodal/concat_rna_clinical_embedding.yaml` | RNA + Clinical | Concatenation | Clinical text embeddings | Variance-filtered RNA | - |
| `bimodal/concat_rna_pathology.yaml` | RNA + Pathology | Concatenation | - | Variance-filtered RNA | UNI tile embeddings |
| `bimodal/concat_pathology_clinical.yaml` | Pathology + Clinical | Concatenation | Tabular clinical | - | UNI tile embeddings |
| `bimodal/concat_pathology_clinical_embedding.yaml` | Pathology + Clinical | Concatenation | Clinical text embeddings | - | UNI tile embeddings |
| `bimodal/gated_rna_clinical.yaml` | RNA + Clinical | Scalar-gated fusion | Tabular clinical | Variance-filtered RNA | - |
| `bimodal/gated_rna_clinical_embedding.yaml` | RNA + Clinical | Scalar-gated fusion | Clinical text embeddings | Variance-filtered RNA | - |
| `bimodal/gated_rna_pathology.yaml` | RNA + Pathology | Scalar-gated fusion | - | Variance-filtered RNA | UNI tile embeddings |
| `bimodal/gated_pathology_clinical.yaml` | Pathology + Clinical | Scalar-gated fusion | Tabular clinical | - | UNI tile embeddings |
| `bimodal/gated_pathology_clinical_embedding.yaml` | Pathology + Clinical | Scalar-gated fusion | Clinical text embeddings | - | UNI tile embeddings |
| `bimodal/lowrank_rna_clinical.yaml` | RNA + Clinical | Low-rank bilinear | Tabular clinical | Variance-filtered RNA | - |
| `bimodal/lowrank_rna_clinical_embedding.yaml` | RNA + Clinical | Low-rank bilinear | Clinical text embeddings | Variance-filtered RNA | - |
| `bimodal/lowrank_rna_pathology.yaml` | RNA + Pathology | Low-rank bilinear | - | Variance-filtered RNA | UNI tile embeddings |
| `bimodal/lowrank_pathology_clinical.yaml` | Pathology + Clinical | Low-rank bilinear | Tabular clinical | - | UNI tile embeddings |
| `bimodal/lowrank_pathology_clinical_embedding.yaml` | Pathology + Clinical | Low-rank bilinear | Clinical text embeddings | - | UNI tile embeddings |
| `trimodal/concat.yaml` | RNA + Pathology + Clinical | Concatenation | Tabular clinical | Variance-filtered RNA | UNI tile embeddings |
| `trimodal/concat_clinical_embedding.yaml` | RNA + Pathology + Clinical | Concatenation | Clinical text embeddings | Variance-filtered RNA | UNI tile embeddings |
| `trimodal/gated_concat.yaml` | RNA + Pathology + Clinical | Scalar-gated fusion | Tabular clinical | Variance-filtered RNA | UNI tile embeddings |
| `trimodal/gated_concat_clinical_embedding.yaml` | RNA + Pathology + Clinical | Scalar-gated fusion | Clinical text embeddings | Variance-filtered RNA | UNI tile embeddings |
| `trimodal/lowrank_bilinear.yaml` | RNA + Pathology + Clinical | Low-rank bilinear | Tabular clinical | Variance-filtered RNA | UNI tile embeddings |
| `trimodal/lowrank_bilinear_clinical_embedding.yaml` | RNA + Pathology + Clinical | Low-rank bilinear | Clinical text embeddings | Variance-filtered RNA | UNI tile embeddings |
| `survpgc/survpgc.yaml` | RNA + Pathology + Clinical | SurvPGC-style co-attention | Clinical text tokens | Pathway RNA tokens | UNI pathology tokens |
| `survpgc/survpgc_category.yaml` | RNA + Pathology + Clinical | SurvPGC-style co-attention | Clinical text tokens | Biological-category RNA tokens | UNI pathology tokens |
| `survpgc/survpgc_scfoundation.yaml` | RNA + Pathology + Clinical | SurvPGC-style co-attention | Clinical text tokens | scFoundation embeddings | UNI pathology tokens |
| `late_fusion/late_fusion.yaml` | RNA + Pathology + Clinical | Late fusion | Unimodal clinical risk | Unimodal RNA risk | Unimodal pathology risk |
