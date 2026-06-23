# Multimodal Survival Prediction

# Multimodal Survival Fusion

## Overview
Short description of the project: multimodal survival prediction using RNA, clinical, pathology, and omics/pathway representations.

## Modalities
List the input types:
```html
<table>
  <thead>
    <tr>
      <th>Modality</th>
      <th>Encoder</th>
      <th>Representation</th>
      <th>Shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pathology (WSI)</td>
      <td>UNI</td>
      <td>Tile embeddings</td>
      <td><code>[N_tiles, 1024]</code></td>
    </tr>
    <tr>
      <td rowspan="4">Bulk RNA-seq</td>
      <td>scFoundation</td>
      <td>Gene expression embeddings</td>
      <td><code>[4, 768]</code></td>
    </tr>
    <tr>
      <td>—</td>
      <td>Raw gene expression</td>
      <td><code>[1, 19359]</code></td>
    </tr>
    <tr>
      <td>—</td>
      <td>Pathway tokens</td>
      <td><code>[N_pathways, N_genes_per_pathway]</code></td>
    </tr>
    <tr>
      <td>—</td>
      <td>Biological category tokens</td>
      <td><code>[6, N_genes_per_category]</code></td>
    </tr>
    <tr>
      <td rowspan="3">Clinical</td>
      <td>—</td>
      <td>Raw tabular</td>
      <td><code>[1, N_vars]</code></td>
    </tr>
    <tr>
      <td>BioClinical ModernBERT</td>
      <td>Text embeddings</td>
      <td><code>[6, 768]</code> or <code>[13, 768]</code></td>
    </tr>
    <tr>
      <td>CONCH text encoder</td>
      <td>Text embeddings</td>
      <td><code>[6, 512]</code> or <code>[13, 512]</code></td>
    </tr>
  </tbody>
</table>
```

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
