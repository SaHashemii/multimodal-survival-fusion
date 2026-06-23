"""
Option 2 – Pathway Aggregation Extractor  (multi-GMT edition)
=============================================================
Accepts one OR multiple .gmt files — all pathway gene sets are merged
into a single universe before the membership matrix is built.

Config keys (additions over single-GMT version)
------------------------------------------------
  gmt_files : list[str]  – list of .gmt file paths (replaces gmt_file)
  gmt_file  : str        – single .gmt path; still accepted for back-compat
  deduplicate_pathways : bool – drop duplicate pathway names (keep first)
                                 default True
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from .base import BaseRNAExtractor
from .mlp_utils import build_mlp


# ─────────────────────────────────────────────────────────────────────────────
# GMT parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """Return {pathway_name: [gene, …]} from a single .gmt file."""
    pathways: Dict[str, List[str]] = {}
    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name, _desc, *genes = parts
            pathways[name] = [g.upper() for g in genes if g]
    return pathways


def load_gmt_files(
    gmt_paths: List[str],
    deduplicate: bool = True,
) -> Dict[str, List[str]]:
    """
    Load and merge one or more .gmt files.

    Args:
        gmt_paths:   list of paths to .gmt files
        deduplicate: if True, keep the first occurrence of each pathway name
                     (later files' duplicates are silently skipped)

    Returns:
        merged {pathway_name: [gene, …]} dict
    """
    merged: Dict[str, List[str]] = {}
    for path in gmt_paths:
        pw = parse_gmt(path)
        n_added = n_dup = 0
        for name, genes in pw.items():
            if name in merged:
                if deduplicate:
                    n_dup += 1
                    continue
                else:
                    # Rename duplicate: append source file stem
                    name = f"{name}__{Path(path).stem}"
            merged[name] = genes
            n_added += 1
        import logging
        logging.getLogger(__name__).info(
            f"  GMT {Path(path).name}: +{n_added} pathways  "
            f"({n_dup} duplicates {'skipped' if deduplicate else 'renamed'})"
        )
    return merged


def build_membership_matrix(
    gene_names: List[str],
    pathways: Dict[str, List[str]],
    min_genes: int = 5,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Build binary membership matrix M of shape (G, P).
    M[g, p] = 1  iff gene g is a member of pathway p.
    """
    gene_index = {g.upper(): i for i, g in enumerate(gene_names)}
    G = len(gene_names)
    kept_names: List[str] = []
    columns: List[np.ndarray] = []

    for name, genes in pathways.items():
        idx = [gene_index[g] for g in genes if g in gene_index]
        if len(idx) < min_genes:
            continue
        col = np.zeros(G, dtype=np.float32)
        col[idx] = 1.0
        columns.append(col)
        kept_names.append(name)

    if not columns:
        raise ValueError(
            f"No pathways survived min_genes={min_genes} filter. "
            "Check gene name matching between GMT and your expression matrix."
        )

    M = torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)  # (G, P)
    return M, kept_names


# ─────────────────────────────────────────────────────────────────────────────
# Learnable attention aggregation
# ─────────────────────────────────────────────────────────────────────────────

class GeneAttentionAggregator(nn.Module):
    def __init__(self, G: int, P: int, membership: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(G, P))
        self.register_buffer("mask", (membership == 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked = self.logits.clone()
        masked[self.mask] = float("-inf")
        weights = torch.softmax(masked, dim=0)
        return x @ weights


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class PathwayAggregationExtractor(BaseRNAExtractor):
    """
    RNA → pathway scores → MLP → embedding.

    Accepts one or multiple .gmt files via config["gmt_files"] (list)
    or legacy config["gmt_file"] (single string).
    """

    def __init__(self, input_dim: int, config: Dict[str, Any] = {}):
        super().__init__(input_dim, config)

        gene_names: List[str]  = config.get("gene_names", [f"G{i}" for i in range(input_dim)])
        agg_method: str        = config.get("agg_method", "mean")
        learn_weights: bool    = config.get("learn_weights", False)
        min_genes: int         = config.get("min_genes", 5)
        out_dim: int           = config.get("out_dim", 256)
        hidden_dims: List[int] = config.get("hidden_dims", [256])
        dropout: float         = config.get("dropout", 0.25)
        activation: str        = config.get("activation", "relu")
        norm: str              = config.get("norm", "layer")
        use_input_norm: bool   = config.get("input_norm", True)
        deduplicate: bool      = config.get("deduplicate_pathways", True)

        # Resolve GMT file(s) — support both list and single string
        gmt_files: List[str] = []
        if "gmt_files" in config and config["gmt_files"]:
            raw = config["gmt_files"]
            gmt_files = [raw] if isinstance(raw, str) else list(raw)
        elif "gmt_file" in config and config["gmt_file"]:
            gmt_files = [config["gmt_file"]]

        self._out_dim  = out_dim
        self.agg_method = agg_method
        self.learn_weights = learn_weights
        self.gmt_files = gmt_files

        # Input normalisation
        self.input_norm = nn.BatchNorm1d(input_dim) if use_input_norm else nn.Identity()

        # Build membership matrix
        if gmt_files:
            pathways = load_gmt_files(gmt_files, deduplicate=deduplicate)
            M, self.pathway_names = build_membership_matrix(
                gene_names, pathways, min_genes=min_genes
            )
        else:
            # Debug fallback: pseudo-pathways
            G, P = input_dim, max(1, input_dim // 50)
            M = torch.zeros(G, P)
            for g in range(G):
                M[g, g % P] = 1.0
            self.pathway_names = [f"pseudo_pathway_{p}" for p in range(P)]
            import logging
            logging.getLogger(__name__).warning(
                "No GMT file(s) provided — using pseudo-pathways. "
                "Set gmt_files in config for biological meaning."
            )

        self.register_buffer("membership", M)
        P = M.shape[1]
        self.num_pathways = P

        # Aggregation
        if learn_weights or agg_method == "attention":
            self.aggregator = GeneAttentionAggregator(input_dim, P, M)
        else:
            self.aggregator = None

        # MLP
        self.mlp = build_mlp(
            in_dim=P, hidden_dims=hidden_dims, out_dim=out_dim,
            dropout=dropout, activation=activation, norm=norm,
        )
        self._is_fitted = True

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        if self.aggregator is not None:
            return self.aggregator(x)
        M      = self.membership
        counts = M.sum(dim=0).clamp(min=1)
        pathway_sum = x @ M
        if self.agg_method == "mean":
            return pathway_sum / counts
        return pathway_sum   # "sum"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x     = self.input_norm(x)
        scores = self.aggregate(x)
        return self.mlp(scores)

    def get_pathway_scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        with torch.no_grad():
            x      = self.input_norm(x)
            scores = self.aggregate(x)
        return scores, self.pathway_names