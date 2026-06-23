# Unified SurvPGC Fusion

This folder combines the three current SurvPGC variants into one runner:

- `--omics-source scfoundation`: load one `.pt` scFoundation embedding file per patient.
- `--omics-source category`: build RNA tokens from a biological category CSV.
- `--omics-source pathway`: build RNA tokens from one or more GMT pathway files.

Pathology uses UNI-style `.pt` tile features through `--pathology-index`.
Clinical data uses CONCH/BERT token `.pt` files through `--clinical-token-dir`.
Clinical shapes such as `[6, 512]`, `[6, 768]`, `[13, 512]`, and `[13, 768]`
are inferred automatically, and other fixed shapes work as long as every retained
patient has the same shape.

Examples:

```bash
python 5_CV/model/survpgc_unified/train_survpgc.py \
  --omics-source scfoundation \
  --clinical-token-dir path/to/clinical_tokens \
  --omics-token-dir path/to/scfoundation_tokens
```

```bash
python 5_CV/model/survpgc_unified/train_survpgc.py \
  --omics-source category \
  --clinical-token-dir path/to/clinical_tokens \
  --biological-categories path/to/biological_categories.csv \
  --rna-mode top_cv
```

```bash
python 5_CV/model/survpgc_unified/train_survpgc.py \
  --omics-source pathway \
  --clinical-token-dir path/to/clinical_tokens \
  --pathway-gmt path/to/pathways.gmt \
  --rna-mode all_genes
```

Use explicit paths when running from a clean repository:

```bash
python 5_CV/model/survpgc_unified/train_survpgc.py \
  --input-dir 5_CV \
  --labels-master 5_CV/task3_combined_labels.csv \
  --pathology-index 5_CV/he_paths_all_splits_minimal.csv \
  --clinical-token-dir 5_CV/clinical_tokens/conch \
  --omics-source pathway \
  --rna-master 5_CV/all_samples_RNA_matrix.csv \
  --pathway-gmt resources/h.all.v2026.1.Hs.symbols.gmt.txt \
  --out-dir 5_CV/outputs/survpgc_pathway_conch
```
