"""
rna_extractors/example_integration.py
======================================
Shows how to plug any RNA extractor into a multimodal survival model.
Demonstrates all three strategies end-to-end.

Run:
    python -m rna_extractors.example_integration
"""

import torch
import torch.nn as nn
from pathlib import Path

from rna_extractors import build_rna_extractor

# ─────────────────────────────────────────────────────────────────────────────
# Toy multimodal model skeleton
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalSurvivalModel(nn.Module):
    """
    Skeleton multimodal model combining:
      - RNA extractor (any strategy)
      - Placeholder WSI encoder (GigaPath / UNI bag-of-patches)
      - Placeholder clinical encoder
    → fused embedding → Cox/DSS risk head
    """

    def __init__(self, rna_extractor, wsi_dim: int = 1024, clinical_dim: int = 32):
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
        rna_feat      = self.rna_extractor(rna)           # (B, D_rna)
        wsi_feat      = self.wsi_proj(wsi)                # (B, 256)
        clinical_feat = self.clinical_proj(clinical)      # (B, 64)

        fused    = torch.cat([rna_feat, wsi_feat, clinical_feat], dim=1)
        log_risk = self.head(fused)                       # (B, 1)
        return log_risk

    def total_loss(self, log_risk, events, durations, task_loss_fn):
        task_loss = task_loss_fn(log_risk, events, durations)
        reg_loss  = self.rna_extractor.regularization_loss()
        return task_loss + reg_loss


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    torch.manual_seed(42)

    NUM_GENES     = 20531   # TCGA pan-cancer gene count
    BATCH_SIZE    = 8
    WSI_DIM       = 1024
    CLINICAL_DIM  = 32
    N_TRAIN       = 64

    # Synthetic data
    x_train   = torch.randn(N_TRAIN, NUM_GENES).abs()   # expression (N, G)
    x_batch   = torch.randn(BATCH_SIZE, NUM_GENES).abs()
    wsi_batch = torch.randn(BATCH_SIZE, WSI_DIM)
    clin_batch= torch.randn(BATCH_SIZE, CLINICAL_DIM)

    strategies = [
        (
            "all_features",
            {
                "hidden_dims": [512, 256],
                "out_dim": 256,
                "l1_lambda": 1e-4,
                "input_norm": True,
            },
        ),
        (
            "pathway",
            {
                # gmt_file: "/path/to/h.all.v2023.1.Hs.symbols.gmt",  # MSigDB Hallmark
                # gene_names: [...],                                    # your gene list
                "agg_method": "mean",
                "learn_weights": False,
                "out_dim": 256,
                "hidden_dims": [256],
            },
        ),
        (
            "variance_filter",
            {
                "top_k": 2000,
                "use_cv": True,
                "out_dim": 256,
                "hidden_dims": [512, 256],
            },
        ),
    ]

    for strategy, cfg in strategies:
        print(f"\n{'─'*60}")
        print(f"  Strategy: {strategy}")
        print(f"{'─'*60}")

        extractor = build_rna_extractor(strategy, input_dim=NUM_GENES, config=cfg)

        # fit() is needed only for variance_filter; others are no-ops
        extractor.fit(x_train)

        model = MultimodalSurvivalModel(extractor, wsi_dim=WSI_DIM, clinical_dim=CLINICAL_DIM)
        model.train()

        with torch.no_grad():
            log_risk = model(x_batch, wsi_batch, clin_batch)
            reg_loss = extractor.regularization_loss()

        print(f"  Config:          {extractor.get_config()}")
        print(f"  Output dim:      {extractor.output_dim}")
        print(f"  log_risk shape:  {log_risk.shape}")
        print(f"  reg_loss:        {reg_loss.item():.6f}")

        # Variance filter extras
        if strategy == "variance_filter":
            summary = extractor.get_score_summary(top_n=5)
            print(f"  Top-5 CV genes:  {summary['top_genes']}")
            print(f"  Selected genes:  {summary['num_selected']}")

        # Pathway extras
        if strategy == "pathway":
            scores, names = extractor.get_pathway_scores(x_batch)
            print(f"  Pathway scores:  {scores.shape}  ({len(names)} pathways)")

    print(f"\n{'─'*60}")
    print("  All strategies OK.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    demo()
