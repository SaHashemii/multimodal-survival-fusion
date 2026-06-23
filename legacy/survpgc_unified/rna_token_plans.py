from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RNATokenPlan:
    names: list[str]
    gene_names: list[list[str]]
    gene_indices: list[list[int]]
    stats: pd.DataFrame


def load_rna_matrix(path: Path) -> pd.DataFrame:
    """Load gene x sample RNA CSV as sample x gene dataframe."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    rna = df.T
    rna.index.name = "sample_id"
    rna.columns = rna.columns.astype(str)
    return rna.apply(pd.to_numeric, errors="coerce")


def load_biological_categories(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    categories: dict[str, list[str]] = {}
    for col in df.columns:
        genes = [str(g).strip() for g in df[col].dropna().tolist() if str(g).strip()]
        categories[str(col).strip()] = list(dict.fromkeys(genes))
    return categories


def load_gmt_pathways(paths: list[Path]) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    for path in paths:
        source = path.stem
        with open(path) as handle:
            for line_no, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0].strip()
                genes = [gene.strip() for gene in parts[2:] if gene.strip()]
                if not name or not genes:
                    continue
                key = name if name not in pathways else f"{source}:{name}"
                if key in pathways:
                    raise ValueError(f"Duplicate pathway name after source prefix at {path}:{line_no}: {key}")
                pathways[key] = list(dict.fromkeys(genes))
    if not pathways:
        raise ValueError(f"No pathways loaded from GMT files: {[str(path) for path in paths]}")
    return pathways


def impute_rna_with_medians(data: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return data.fillna(medians).astype(np.float32)


def select_top_cv_genes(train_rna: pd.DataFrame, top_k: int = 2000, eps: float = 1e-8) -> list[str]:
    means = train_rna.mean(axis=0).abs()
    stds = train_rna.std(axis=0)
    cv = stds / (means + eps)
    cv = cv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cv.sort_values(ascending=False).head(top_k).index.astype(str).tolist()


def build_category_plan(
    categories: dict[str, list[str]],
    rna_gene_names: list[str],
    selected_genes: list[str] | None,
    min_genes_per_group: int,
    fold: int,
) -> RNATokenPlan:
    rna_gene_set = set(rna_gene_names)
    selected_set = set(selected_genes) if selected_genes is not None else None
    gene_to_idx = {gene: idx for idx, gene in enumerate(rna_gene_names)}
    rows = []
    kept_names: list[str] = []
    kept_gene_names: list[list[str]] = []
    kept_gene_indices: list[list[int]] = []

    for category, definition_genes in categories.items():
        found = [gene for gene in definition_genes if gene in rna_gene_set]
        final = found if selected_set is None else [gene for gene in found if gene in selected_set]
        status = "KEPT" if len(final) >= min_genes_per_group else "REMOVED_TOO_FEW_GENES"
        rows.append(
            {
                "fold": fold,
                "token_source": "category",
                "token_name": category,
                "definition_genes": len(set(definition_genes)),
                "rna_genes_found": len(set(found)),
                "coverage": len(set(found)) / max(len(set(definition_genes)), 1),
                "selected_genes": len(set(final)),
                "status": status,
            }
        )
        if status == "KEPT":
            kept_names.append(category)
            kept_gene_names.append(final)
            kept_gene_indices.append([gene_to_idx[gene] for gene in final])

    return RNATokenPlan(kept_names, kept_gene_names, kept_gene_indices, pd.DataFrame(rows))


def build_pathway_plan(
    pathways: dict[str, list[str]],
    rna_gene_names: list[str],
    selected_genes: list[str] | None,
    min_genes_per_group: int,
    min_pathway_coverage: float,
    fold: int,
) -> RNATokenPlan:
    rna_gene_set = set(rna_gene_names)
    selected_set = set(selected_genes) if selected_genes is not None else None
    gene_to_idx = {gene: idx for idx, gene in enumerate(rna_gene_names)}
    rows = []
    kept_names: list[str] = []
    kept_gene_names: list[list[str]] = []
    kept_gene_indices: list[list[int]] = []

    for pathway, definition_genes in pathways.items():
        definition_unique = list(dict.fromkeys(definition_genes))
        found = [gene for gene in definition_unique if gene in rna_gene_set]
        coverage = len(found) / max(len(definition_unique), 1)
        final = found if selected_set is None else [gene for gene in found if gene in selected_set]
        if coverage < min_pathway_coverage:
            status = "REMOVED_LOW_COVERAGE"
        elif len(final) < min_genes_per_group:
            status = "REMOVED_TOO_FEW_GENES"
        else:
            status = "KEPT"
        rows.append(
            {
                "fold": fold,
                "token_source": "pathway",
                "token_name": pathway,
                "definition_genes": len(definition_unique),
                "rna_genes_found": len(found),
                "coverage": coverage,
                "selected_genes": len(final),
                "status": status,
            }
        )
        if status == "KEPT":
            kept_names.append(pathway)
            kept_gene_names.append(final)
            kept_gene_indices.append([gene_to_idx[gene] for gene in final])

    return RNATokenPlan(kept_names, kept_gene_names, kept_gene_indices, pd.DataFrame(rows))


def log_token_plan(plan: RNATokenPlan, fold: int) -> None:
    kept = plan.stats[plan.stats["status"] == "KEPT"]
    print(f"\nFold {fold}")
    print(f"Total RNA token definitions evaluated: {len(plan.stats)}")
    print(f"Total RNA tokens kept: {len(plan.names)}")
    if not kept.empty:
        print(
            "Kept token gene counts: "
            f"min={int(kept['selected_genes'].min())} "
            f"median={float(kept['selected_genes'].median()):.1f} "
            f"max={int(kept['selected_genes'].max())}"
        )
    print("Token status counts:", plan.stats["status"].value_counts().to_dict())
    print("First kept tokens:", ", ".join(plan.names[:20]))
