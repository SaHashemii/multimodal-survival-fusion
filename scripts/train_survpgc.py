#!/usr/bin/env python3
"""Train SurvPGC-style multimodal Cox models with cross-validation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

from mm_survival.data.labels import load_labels
from mm_survival.data.pathology import load_pathology_index
from mm_survival.data.rna_token_plans import (
    build_category_plan,
    build_pathway_plan,
    impute_rna_with_medians,
    load_biological_categories,
    load_gmt_pathways,
    load_rna_matrix,
    select_top_cv_genes,
)
from mm_survival.data.splits import prepare_fold_split, summarize_fold_split
from mm_survival.models.survpgc import SurvPGCUnifiedCox
from mm_survival.training.artifacts import ensure_dir, save_checkpoint, write_history, write_json
from mm_survival.training.cross_validation import make_fold_assignments
from mm_survival.training.plots import write_kaplan_meier_plot
from mm_survival.training.survpgc_trainer import evaluate_survpgc, train_survpgc
from mm_survival.utils.config import load_yaml, materialize_data_config, resolve_path, resolve_repo_path
from mm_survival.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SurvPGC-style Cox model.")
    parser.add_argument("--experiment", type=Path, required=True, help="SurvPGC experiment config YAML.")
    parser.add_argument("--data", type=Path, required=True, help="Data config YAML.")
    parser.add_argument("--fold-assignments", type=Path, default=None, help="Optional existing fold_assignments.csv.")
    parser.add_argument("--fold", type=int, default=None, help="Train only one fold.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    return parser.parse_args()


def load_feature_tensor(path: Path) -> torch.Tensor | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except OSError as exc:
        print(f"[Feature] WARN: cannot load {path}: {exc}")
        return None
    feats = obj.get("feats") if isinstance(obj, dict) and "feats" in obj else obj
    feats = obj.get("embeddings") if isinstance(obj, dict) and "embeddings" in obj else feats
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if not torch.is_tensor(feats):
        print(f"[Feature] WARN: unsupported object in {path}")
        return None
    feats = feats.float()
    if feats.ndim == 1:
        feats = feats.unsqueeze(0)
    if feats.ndim != 2 or feats.shape[0] == 0:
        print(f"[Feature] WARN: invalid tensor shape in {path}: {tuple(feats.shape)}")
        return None
    return feats.contiguous()


def sample_and_pad_pathology(feats: torch.Tensor, sample_id: str, seed: int, token_cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    if feats.shape[0] > token_cap:
        digest = hashlib.sha256(f"{sample_id}:{seed}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))
        idx = rng.choice(feats.shape[0], size=token_cap, replace=False)
        idx.sort()
        feats = feats[idx]
    mask = torch.zeros(token_cap, dtype=torch.bool)
    n_tokens = min(feats.shape[0], token_cap)
    out = torch.zeros(token_cap, feats.shape[1], dtype=torch.float32)
    out[:n_tokens] = feats[:n_tokens].float()
    mask[:n_tokens] = True
    return out, mask


def stack_by_id(data: dict[str, torch.Tensor], sample_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.stack([data[sample_id] for sample_id in sample_ids], dim=0).to(device)


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


def load_survpgc_inputs(data_cfg: dict, exp_cfg: dict, seed: int):
    data_root = Path(data_cfg["root"]).expanduser()
    labels = load_labels(resolve_path(data_root, data_cfg["labels"]))
    pathology_index_path = resolve_path(data_root, data_cfg["pathology_index"])
    pathology_features_root = resolve_path(data_root, data_cfg.get("pathology_features_root")) or pathology_index_path.parent
    pathology_index = load_pathology_index(pathology_index_path)
    clinical_dir = resolve_path(data_root, data_cfg["clinical_embeddings"])

    surv_data_cfg = exp_cfg.get("data", {})
    omics_source = surv_data_cfg.get("omics_source", "pathway")
    rna = None
    omics_dir = None
    if omics_source == "scfoundation":
        omics_dir = resolve_path(data_root, data_cfg["omics_embeddings"])
    else:
        rna = load_rna_matrix(resolve_path(data_root, data_cfg["rna"]))

    label_ids = set(labels.index.astype(str))
    pathology_ids = set(pathology_index.index.astype(str))
    clinical_ids = {sample_id for sample_id in label_ids | pathology_ids if (clinical_dir / f"{sample_id}.pt").is_file()}
    if omics_source == "scfoundation":
        omics_ids = {sample_id for sample_id in label_ids | pathology_ids if (omics_dir / f"{sample_id}.pt").is_file()}
        common = sorted(label_ids & pathology_ids & clinical_ids & omics_ids)
    else:
        common = sorted(label_ids & pathology_ids & clinical_ids & set(rna.index.astype(str)))

    model_cfg = exp_cfg.get("model", {})
    pathology_token_cap = int(model_cfg.get("pathology_token_cap", 4096))
    pathology_tokens: dict[str, torch.Tensor] = {}
    pathology_masks: dict[str, torch.Tensor] = {}
    clinical_tokens: dict[str, torch.Tensor] = {}
    omics_tokens: dict[str, torch.Tensor] = {}
    dropped = []

    for sample_id in common:
        feature_path = Path(pathology_index.loc[sample_id, "feature_path"])
        if not feature_path.is_absolute():
            feature_path = pathology_features_root / feature_path
        path_feats = load_feature_tensor(feature_path)
        clinic = load_feature_tensor(clinical_dir / f"{sample_id}.pt")
        omics = load_feature_tensor(omics_dir / f"{sample_id}.pt") if omics_source == "scfoundation" else None
        if path_feats is None or clinic is None or (omics_source == "scfoundation" and omics is None):
            dropped.append(sample_id)
            continue
        path_tok, path_mask = sample_and_pad_pathology(path_feats, sample_id, seed, pathology_token_cap)
        pathology_tokens[sample_id] = path_tok
        pathology_masks[sample_id] = path_mask
        clinical_tokens[sample_id] = clinic
        if omics is not None:
            omics_tokens[sample_id] = omics

    sample_ids = sorted(set(common) - set(dropped))
    missing_summary = {
        "labels": len(label_ids),
        "pathology_index": len(pathology_ids),
        "clinical_embeddings": len(clinical_ids),
        "common_before_feature_load": len(common),
        "dropped_feature_load": dropped,
        "retained_samples": len(sample_ids),
    }
    return sample_ids, labels.loc[sample_ids], rna.loc[sample_ids] if rna is not None else None, pathology_tokens, pathology_masks, clinical_tokens, omics_tokens, missing_summary


def prepare_fold_omics(exp_cfg: dict, data_cfg: dict, resources_cfg: dict, rna: pd.DataFrame | None, split):
    surv_data_cfg = exp_cfg.get("data", {})
    model_cfg = exp_cfg.get("model", {})
    rna_cfg = model_cfg.get("rna", {})
    omics_source = surv_data_cfg.get("omics_source", "pathway")
    if omics_source == "scfoundation":
        return None, None, None, None, None, None, None

    data_root = Path(data_cfg["root"]).expanduser()
    gene_names = rna.columns.astype(str).tolist()
    fit_raw = rna.loc[split.fit_ids, gene_names]
    medians = fit_raw.median(axis=0, skipna=True).fillna(0.0)
    fit_rna = impute_rna_with_medians(fit_raw, medians)
    train_rna = impute_rna_with_medians(rna.loc[split.train_ids, gene_names], medians)
    val_rna = impute_rna_with_medians(rna.loc[split.val_ids, gene_names], medians)
    test_rna = impute_rna_with_medians(rna.loc[split.test_ids, gene_names], medians)
    rna_mode = surv_data_cfg.get("rna_mode", "all_genes")
    selected_genes = select_top_cv_genes(fit_rna, top_k=int(rna_cfg.get("top_k", 2000))) if rna_mode == "top_cv" else None

    if omics_source == "category":
        definitions = load_biological_categories(resolve_repo_path(REPO_ROOT, resources_cfg["biological_categories"]))
        plan = build_category_plan(
            definitions,
            gene_names,
            selected_genes,
            int(rna_cfg.get("min_genes_per_token", 10)),
            split.fold,
        )
    else:
        gmt_paths = [resolve_repo_path(REPO_ROOT, path) for path in resources_cfg["pathway_gmt"]]
        definitions = load_gmt_pathways(gmt_paths)
        plan = build_pathway_plan(
            definitions,
            gene_names,
            selected_genes,
            int(rna_cfg.get("min_genes_per_token", 10)),
            float(rna_cfg.get("min_pathway_coverage", 0.90)),
            split.fold,
        )
    if not plan.names:
        raise ValueError(f"Fold {split.fold} has no retained RNA tokens.")
    return train_rna, fit_rna, val_rna, test_rna, selected_genes, plan, {str(k): float(v) for k, v in medians.items()}


def tensor_pack(sample_ids, labels, omics_data, pathology_tokens, pathology_masks, clinical_tokens, device, omics_source):
    if omics_source == "scfoundation":
        omics_tensor = stack_by_id(omics_data, sample_ids, device)
    else:
        omics_tensor = to_tensor(omics_data.loc[sample_ids].values.astype(np.float32), device)
    return (
        stack_by_id(pathology_tokens, sample_ids, device),
        omics_tensor,
        stack_by_id(clinical_tokens, sample_ids, device),
        stack_by_id(pathology_masks, sample_ids, device),
        to_tensor(labels.loc[sample_ids, "Time"].values.astype(np.float32), device),
        to_tensor(labels.loc[sample_ids, "Event"].values.astype(np.float32), device),
    )


def build_model(exp_cfg: dict, args_shape: dict, rna_gene_indices, device: torch.device) -> SurvPGCUnifiedCox:
    model_cfg = exp_cfg.get("model", {})
    rna_cfg = model_cfg.get("rna", {})
    omics_in_dim = int(rna_cfg.get("token_dim", 256)) if rna_gene_indices is not None else args_shape["omics_in_dim"]
    omics_token_count = len(rna_gene_indices) if rna_gene_indices is not None else args_shape["omics_token_count"]
    return SurvPGCUnifiedCox(
        pathology_in_dim=args_shape["pathology_in_dim"],
        clinical_in_dim=args_shape["clinical_in_dim"],
        clinical_token_count=args_shape["clinical_token_count"],
        omics_in_dim=omics_in_dim,
        omics_token_count=omics_token_count,
        projection_dim=int(model_cfg.get("projection_dim", 256)),
        attention_dim_head=int(model_cfg.get("attention_dim_head", 128)),
        fusion_hidden_dims=model_cfg.get("fusion_hidden_dims", [256, 64]),
        fusion_dropout=float(model_cfg.get("fusion_dropout", 0.30)),
        rna_gene_indices=rna_gene_indices,
        rna_hidden_dim=int(rna_cfg.get("hidden_dim", 256)),
        rna_token_dim=int(rna_cfg.get("token_dim", 256)),
        rna_dropout=float(rna_cfg.get("dropout", 0.25)),
    ).to(device)


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.experiment)
    data_cfg_full = load_yaml(args.data)
    data_cfg_raw = data_cfg_full.get("data", {})
    resources_cfg = data_cfg_full.get("resources", {})
    exp_info = exp_cfg.get("experiment", {})
    surv_data_cfg = exp_cfg.get("data", {})
    data_cfg = materialize_data_config(data_cfg_raw, surv_data_cfg)
    cv_cfg = exp_cfg.get("cv", {})
    train_cfg = exp_cfg.get("training", {})
    omics_source = surv_data_cfg.get("omics_source", "pathway")

    if exp_info.get("model_type") != "survpgc":
        raise ValueError("scripts/train_survpgc.py requires model_type=survpgc.")

    output_dir = args.output_dir or resolve_path(REPO_ROOT, exp_info.get("output_dir", "outputs/survpgc_multimodal"))
    output_dir = ensure_dir(output_dir)
    seed = int(cv_cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(train_cfg.get("device", "cpu"))

    sample_ids, labels, rna, pathology_tokens, pathology_masks, clinical_tokens, scfoundation_tokens, missing_summary = load_survpgc_inputs(data_cfg, exp_cfg, seed)
    write_json(missing_summary, output_dir / "missing_summary.json")

    if args.fold_assignments is not None:
        fold_assignments = pd.read_csv(args.fold_assignments)
    else:
        fold_path = output_dir / "fold_assignments.csv"
        if fold_path.is_file():
            fold_assignments = pd.read_csv(fold_path)
        else:
            fold_assignments = make_fold_assignments(sample_ids, labels, int(cv_cfg.get("n_splits", 5)), seed)
            fold_assignments.to_csv(fold_path, index=False)

    folds_to_run = [args.fold] if args.fold is not None else list(range(int(cv_cfg.get("n_splits", 5))))
    results = []
    test_risk_tables = []
    for fold in folds_to_run:
        split = prepare_fold_split(sample_ids, labels, fold_assignments, fold, seed=seed + fold, val_size=float(cv_cfg.get("val_size", 0.20)))
        train_omics = fit_omics = val_omics = test_omics = scfoundation_tokens
        selected_genes = None
        plan = None
        rna_medians = None
        if omics_source != "scfoundation":
            train_omics, fit_omics, val_omics, test_omics, selected_genes, plan, rna_medians = prepare_fold_omics(exp_cfg, data_cfg, resources_cfg, rna, split)

        train_tensors = tensor_pack(split.train_ids, labels, train_omics, pathology_tokens, pathology_masks, clinical_tokens, device, omics_source)
        fit_tensors = tensor_pack(split.fit_ids, labels, fit_omics, pathology_tokens, pathology_masks, clinical_tokens, device, omics_source)
        val_tensors = tensor_pack(split.val_ids, labels, val_omics, pathology_tokens, pathology_masks, clinical_tokens, device, omics_source)
        test_tensors = tensor_pack(split.test_ids, labels, test_omics, pathology_tokens, pathology_masks, clinical_tokens, device, omics_source)
        shape = {
            "pathology_in_dim": int(train_tensors[0].shape[-1]),
            "clinical_token_count": int(train_tensors[2].shape[1]),
            "clinical_in_dim": int(train_tensors[2].shape[2]),
            "omics_token_count": int(train_tensors[1].shape[1]) if omics_source == "scfoundation" else (len(plan.names) if plan else 0),
            "omics_in_dim": int(train_tensors[1].shape[2]) if omics_source == "scfoundation" else int(exp_cfg.get("model", {}).get("rna", {}).get("token_dim", 256)),
        }
        set_seed(seed + fold)
        model = build_model(exp_cfg, shape, plan.gene_indices if plan is not None else None, device)
        model, history = train_survpgc(
            model,
            fit_tensors,
            device=device,
            val_tensors=val_tensors,
            epochs=int(train_cfg.get("epochs", 300)),
            patience=int(train_cfg.get("patience", 40)),
            batch_size=int(train_cfg.get("batch_size", 64)),
            min_events_per_batch=int(train_cfg.get("min_events_per_batch", 3)),
            training_style=train_cfg.get("training_style", "rna_clinical_batch"),
            lr=float(train_cfg.get("lr", 2e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
            grad_clip=float(train_cfg.get("grad_clip", 5.0)),
        )

        fold_dir = ensure_dir(output_dir / f"fold_{fold}")
        write_history(history, fold_dir / "history.csv")
        train_ci, train_risk = evaluate_survpgc(model, train_tensors, split.train_ids)
        test_ci, test_risk = evaluate_survpgc(model, test_tensors, split.test_ids)
        train_risk.to_csv(fold_dir / "train_risk_scores.csv", index=False)
        test_risk.to_csv(fold_dir / "test_risk_scores.csv", index=False)
        test_risk_tables.append(test_risk.assign(fold=fold))
        if plan is not None:
            plan.stats.to_csv(fold_dir / "rna_token_stats.csv", index=False)
            pd.Series(plan.names).to_csv(fold_dir / "rna_token_names.txt", index=False, header=False)
        if selected_genes is not None:
            pd.Series(selected_genes).to_csv(fold_dir / "selected_top_cv_genes.txt", index=False, header=False)
        if rna_medians is not None:
            write_json(rna_medians, fold_dir / "rna_imputation_medians.json")

        summary = {
            **summarize_fold_split(labels, split),
            "c_index": test_ci,
            "train_c_index": train_ci,
            "omics_source": omics_source,
            "rna_mode": surv_data_cfg.get("rna_mode") if omics_source != "scfoundation" else None,
            "omics_tokens": shape["omics_token_count"],
            "pathology_in_dim": shape["pathology_in_dim"],
            "clinical_token_count": shape["clinical_token_count"],
            "clinical_in_dim": shape["clinical_in_dim"],
        }
        write_json(summary, fold_dir / "summary.json")
        save_checkpoint(
            fold_dir / "model.pt",
            model,
            summary,
            extra={"experiment_config": exp_cfg},
        )
        print(f"[fold {fold}] test_c_index={test_ci:.4f} train_c_index={train_ci:.4f}")
        results.append(summary)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results_per_fold.csv", index=False)
    aggregate = {
        "experiment": exp_info.get("name", args.experiment.stem),
        "folds": len(results),
        "mean_c_index": float(results_df["c_index"].mean()) if not results_df.empty else None,
        "std_c_index": float(results_df["c_index"].std(ddof=0)) if not results_df.empty else None,
    }
    if test_risk_tables:
        all_test_risks = pd.concat(test_risk_tables, ignore_index=True)
        all_test_risks.to_csv(output_dir / "test_risk_scores_all_folds.csv", index=False)
        aggregate["kaplan_meier"] = write_kaplan_meier_plot(
            all_test_risks,
            output_dir / "kaplan_meier_by_risk.png",
            title=str(aggregate["experiment"]),
        )
    write_json(aggregate, output_dir / "summary.json")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
