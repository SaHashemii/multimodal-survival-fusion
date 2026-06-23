# rna_extractors

Modular RNA-seq feature extractor package for multimodal survival/classification models on TCGA cohorts. All three strategies share a unified `BaseRNAExtractor` interface and are drop-in interchangeable within any PyTorch multimodal pipeline.

---

## Package structure

```
rna_extractors/
├── __init__.py               # Public API exports
├── base.py                   # Abstract BaseRNAExtractor (nn.Module)
├── mlp_utils.py              # Shared MLP builder + L1 utility
├── factory.py                # build_rna_extractor() factory
├── all_features.py           # Strategy 1: All genes + L1-regularised MLP
├── pathway.py                # Strategy 2: Pathway aggregation → MLP
├── variance_filter.py        # Strategy 3: CV-based gene filtering → MLP
└── example_integration.py    # End-to-end multimodal usage demo
```

---

## Strategies at a glance

| | Strategy | Input dim | Requires `fit()` | Regularization |
|---|---|---|---|---|
| 1 | `all_features` | All genes (~20k) | No | L1 on gate layer |
| 2 | `pathway` | Pathway scores (~50–500) | No | None (optional attention) |
| 3 | `variance_filter` | Top-K CV genes | **Yes** | None |

---

## Installation

No extra dependencies beyond your existing environment:

```bash
pip install torch numpy
```

For pathway aggregation, download a GMT file from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb):

```bash
# Example: MSigDB Hallmark gene sets
wget https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.1.Hs/h.all.v2023.1.Hs.symbols.gmt
```

---

## Quick start

```python
from rna_extractors import build_rna_extractor

# Pick a strategy
extractor = build_rna_extractor(
    strategy="variance_filter",   # "all_features" | "pathway" | "variance_filter"
    input_dim=20531,              # number of genes in your expression matrix
    config={"top_k": 2000, "out_dim": 256}
)

# Fit on training data (no-op for strategies 1 & 2)
extractor.fit(x_train)           # x_train: (N, G) log1p-TPM tensor

# Extract features
feat = extractor(x_batch)        # (B, out_dim)

# Add regularization to your training loss
loss = task_loss + extractor.regularization_loss()
```

---

## Strategy reference

### Strategy 1 — All features + L1 MLP (`all_features`)

Takes the full expression vector and passes it through a trainable MLP. A dedicated `gate` linear layer (genes → first hidden dim) receives an explicit L1 weight penalty, acting as a soft gene-selection mechanism: unused genes are progressively pushed toward zero weight without hard thresholding.

**Pipeline:**
```
(B, G) → BatchNorm → Linear [gate, L1 penalty] → MLP body → (B, out_dim)
```

**Config:**

| Key | Type | Default | Description |
|---|---|---|---|
| `hidden_dims` | `list[int]` | `[512, 256]` | MLP hidden layer widths |
| `out_dim` | `int` | `256` | Output embedding dimension |
| `l1_lambda` | `float` | `1e-4` | L1 penalty weight on gate layer |
| `input_norm` | `bool` | `True` | Apply BatchNorm to raw input |
| `dropout` | `float` | `0.25` | Dropout rate |
| `activation` | `str` | `"relu"` | `"relu"` \| `"gelu"` \| `"silu"` |
| `norm` | `str` | `"layer"` | `"layer"` \| `"batch"` \| `"none"` |

**Example:**
```python
extractor = build_rna_extractor(
    "all_features",
    input_dim=20531,
    config={
        "hidden_dims": [512, 256],
        "out_dim": 256,
        "l1_lambda": 1e-4,
    }
)
feat = extractor(x_batch)                        # no fit() needed
loss = cox_loss + extractor.regularization_loss()
```

---

### Strategy 2 — Pathway aggregation (`pathway`)

Compresses ~20k genes into biologically meaningful pathway scores using a GMT gene set file (MSigDB Hallmark, KEGG, Reactome, etc.), then feeds the compact score vector through an MLP. Dramatically reduces input dimensionality before any learning occurs.

**Pipeline:**
```
(B, G) → BatchNorm → membership matrix (G, P) → pathway scores (B, P) → MLP → (B, out_dim)
```

Three aggregation modes are supported:
- `"mean"` — mean expression of member genes per pathway (default)
- `"sum"` — sum of member gene expression
- `"attention"` — learnable per-gene soft attention weights within each pathway

**Config:**

| Key | Type | Default | Description |
|---|---|---|---|
| `gmt_file` | `str` | `None` | Path to `.gmt` gene set file |
| `gene_names` | `list[str]` | `None` | Ordered gene names matching input columns |
| `agg_method` | `str` | `"mean"` | `"mean"` \| `"sum"` \| `"attention"` |
| `learn_weights` | `bool` | `False` | Learnable per-gene weights per pathway |
| `min_genes` | `int` | `5` | Minimum gene overlap to retain a pathway |
| `out_dim` | `int` | `256` | Output embedding dimension |
| `hidden_dims` | `list[int]` | `[256]` | MLP hidden dims after pathway scores |
| `dropout` | `float` | `0.25` | Dropout rate |
| `input_norm` | `bool` | `True` | Apply BatchNorm to raw genes before aggregation |

**Example:**
```python
extractor = build_rna_extractor(
    "pathway",
    input_dim=20531,
    config={
        "gmt_file": "h.all.v2023.1.Hs.symbols.gmt",
        "gene_names": my_gene_list,       # list[str] matching column order
        "agg_method": "mean",
        "out_dim": 256,
        "hidden_dims": [256],
    }
)
feat = extractor(x_batch)    # no fit() needed

# Inspect pathway scores (useful for visualisation / interpretation)
scores, pathway_names = extractor.get_pathway_scores(x_batch)
# scores: (B, P),  pathway_names: list[str]
```

**GMT format note:** Each row is tab-separated as `PATHWAY_NAME  DESCRIPTION  GENE1  GENE2 ...`. Gene names are matched case-insensitively against `gene_names`. Pathways with fewer than `min_genes` overlapping genes are dropped silently.

---

### Strategy 3 — Variance filter (`variance_filter`)

Selects the top-K most variable genes measured by **coefficient of variation (CV = σ / μ)** computed on the training cohort, then feeds the reduced expression vector through an MLP. CV normalises variance by mean expression, preventing highly expressed housekeeping genes from dominating the selection.

**Pipeline:**
```
fit(x_train): compute per-gene CV → rank → store top-K indices

forward(x):
(B, G) → index select top-K → BatchNorm → MLP → (B, out_dim)
```

**Config:**

| Key | Type | Default | Description |
|---|---|---|---|
| `top_k` | `int` | `2000` | Number of genes to retain |
| `use_cv` | `bool` | `True` | Use CV (σ/μ); `False` uses raw variance |
| `cv_percentile` | `float` | `None` | Keep genes above this CV percentile; overrides `top_k` |
| `out_dim` | `int` | `256` | Output embedding dimension |
| `hidden_dims` | `list[int]` | `[512, 256]` | MLP hidden dims |
| `dropout` | `float` | `0.25` | Dropout rate |
| `input_norm` | `bool` | `True` | Apply BatchNorm to selected genes |
| `gene_names` | `list[str]` | `None` | Gene names for reporting utilities |
| `eps` | `float` | `1e-8` | Stability constant for CV denominator |

**Example:**
```python
extractor = build_rna_extractor(
    "variance_filter",
    input_dim=20531,
    config={
        "top_k": 2000,
        "use_cv": True,
        "out_dim": 256,
        "gene_names": my_gene_list,
    }
)

# Must call fit() before forward()
extractor.fit(x_train)

feat = extractor(x_batch)

# Inspect selected genes
summary = extractor.get_score_summary(top_n=20)
print(summary["top_genes"])      # list of gene names by CV rank
print(summary["num_selected"])   # actual K (may differ if cv_percentile used)

selected = extractor.get_selected_genes()  # full list of selected gene names
```

> **Important:** `fit()` must be called on training-set data only. The selected gene indices are fixed after fitting and saved as a buffer, so the model is serialisable with `torch.save` / `torch.load` without re-fitting.

---

## Shared configuration options

All strategies accept these keys in `config`:

| Key | Type | Default | Description |
|---|---|---|---|
| `out_dim` | `int` | `256` | Output embedding dimension |
| `hidden_dims` | `list[int]` | strategy-specific | MLP hidden layer widths |
| `dropout` | `float` | `0.25` | Dropout probability |
| `activation` | `str` | `"relu"` | `"relu"` \| `"gelu"` \| `"silu"` |
| `norm` | `str` | `"layer"` | LayerNorm / BatchNorm / none inside MLP |

---

## Integration into a multimodal model

```python
import torch
import torch.nn as nn
from rna_extractors import build_rna_extractor

class MultimodalSurvivalModel(nn.Module):
    def __init__(self, rna_extractor, wsi_dim=1024, clinical_dim=32):
        super().__init__()
        self.rna_extractor  = rna_extractor
        self.wsi_proj       = nn.Linear(wsi_dim, 256)
        self.clinical_proj  = nn.Linear(clinical_dim, 64)

        fused_dim = rna_extractor.output_dim + 256 + 64
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),   # log-risk score for Cox
        )

    def forward(self, rna, wsi, clinical):
        rna_feat      = self.rna_extractor(rna)
        wsi_feat      = self.wsi_proj(wsi)
        clinical_feat = self.clinical_proj(clinical)
        fused         = torch.cat([rna_feat, wsi_feat, clinical_feat], dim=1)
        return self.head(fused)


# --- Build ---
extractor = build_rna_extractor("variance_filter", input_dim=20531,
                                config={"top_k": 2000, "out_dim": 256})
extractor.fit(x_train)

model = MultimodalSurvivalModel(extractor)

# --- Training loop ---
for rna, wsi, clinical, events, durations in train_loader:
    log_risk  = model(rna, wsi, clinical)
    task_loss = cox_loss(log_risk, events, durations)
    reg_loss  = extractor.regularization_loss()   # non-zero only for all_features
    loss      = task_loss + reg_loss
    loss.backward()
    optimizer.step()
```

---

## Switching strategies

Because all extractors share the same interface, swapping is a one-line change:

```python
# Ablation: compare all three in the same training loop
for strategy in ["all_features", "pathway", "variance_filter"]:
    extractor = build_rna_extractor(strategy, input_dim=20531, config=base_cfg)
    extractor.fit(x_train)                    # safe no-op for strategies 1 & 2
    model = MultimodalSurvivalModel(extractor)
    train_and_evaluate(model, strategy)
```

---

## Saving and loading

```python
# Save
torch.save({
    "model_state": model.state_dict(),
    "extractor_config": extractor.get_config(),
    "strategy": "variance_filter",
}, "checkpoint.pt")

# Load
ckpt      = torch.load("checkpoint.pt")
extractor = build_rna_extractor(
    ckpt["strategy"],
    input_dim=ckpt["extractor_config"]["input_dim"],
    config=ckpt["extractor_config"],
)
model     = MultimodalSurvivalModel(extractor)
model.load_state_dict(ckpt["model_state"])
# No re-fitting needed: selected_idx buffer is restored from state_dict
```

---

## Design notes

**Why CV instead of raw variance?**  
Raw variance is dominated by highly expressed genes. CV = σ/μ normalises by mean expression, giving biologically low-abundance but highly variable genes (e.g. transcription factors, signalling molecules) a fair chance of selection — these are often more prognostically informative than housekeeping genes.

**Why a separate `regularization_loss()` method?**  
Baking L1 into `forward()` means the penalty accumulates during validation and inference, inflating reported loss numbers. Keeping it separate lets you call `extractor.regularization_loss()` exclusively in the training branch.

**Why `fit()` as an explicit step?**  
Gene selection statistics must be computed on training data only, never on the full cohort (data leakage). The explicit `fit(x_train)` call mirrors sklearn's `fit/transform` convention and makes the leakage boundary clear. For strategies 1 and 2, `fit()` is a no-op, so the calling code is always the same.

**Pathway debug mode:**  
If no `gmt_file` is provided to `PathwayAggregationExtractor`, it falls back to equal-sized pseudo-pathways (1 gene per 50) for unit-testing without a GMT file.
