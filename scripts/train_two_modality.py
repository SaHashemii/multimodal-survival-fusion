#!/usr/bin/env python3
"""
Train two-modality Cox ablations with cross-validation
======================================================

Runs bimodal experiments for modality-pair ablations:

  RNA + clinical
  RNA + pathology
  pathology + clinical

Supported fusion modules follow the experiment config and include concat,
scalar-gated, and low-rank bilinear fusion.

Pipeline
--------
  load configs → select modality pair → build/read outer folds
  prepare shared fold tensors → fit RNA extractor when RNA is used
  train Cox model → save fold metrics, risk scores, summaries, and KM plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch

from mm_survival.data.embedding_fold import prepare_embedding_fold_data
from mm_survival.data.multimodal import load_common_multimodal_dataset
from mm_survival.data.splits import prepare_fold_split, summarize_fold_split
from mm_survival.models.encoders.rna import build_rna_extractor
from mm_survival.models.two_modality_cox import (
    TwoModalityConcatCoxModel,
    TwoModalityGatedCoxModel,
    TwoModalityLowRankBilinearCoxModel,
)
from mm_survival.training.artifacts import ensure_dir, save_checkpoint, write_history, write_json
from mm_survival.training.cross_validation import make_fold_assignments
from mm_survival.training.embedding_trainer import evaluate_embedding_multimodal, train_embedding_multimodal
from mm_survival.training.plots import write_kaplan_meier_plot
from mm_survival.utils.config import load_yaml, materialize_data_config, resolve_path
from mm_survival.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-modality concat models with 5-fold CV.")
    parser.add_argument("--experiment", type=Path, required=True, help="Two-modality experiment config YAML.")
    parser.add_argument("--data", type=Path, required=True, help="Data config YAML.")
    parser.add_argument("--fold-assignments", type=Path, default=None, help="Optional existing fold_assignments.csv.")
    parser.add_argument("--fold", type=int, default=None, help="Train only one fold.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override experiment output directory.")
    return parser.parse_args()


def _build_rna_extractor(model_cfg: dict, gene_names: list[str], x_fit: torch.Tensor):
    rna_cfg = model_cfg.get("rna", {})
    extractor = build_rna_extractor(
        strategy=rna_cfg.get("extractor", "variance_filter"),
        input_dim=len(gene_names),
        config={
            "top_k": rna_cfg.get("top_k", 2000),
            "use_cv": rna_cfg.get("use_cv", True),
            "cv_percentile": rna_cfg.get("cv_percentile"),
            "eps": rna_cfg.get("eps", 1e-8),
            "hidden_dims": rna_cfg.get("hidden_dims", [512, 256]),
            "out_dim": rna_cfg.get("out_dim", 256),
            "dropout": rna_cfg.get("dropout", 0.25),
            "activation": rna_cfg.get("activation", "selu"),
            "norm": rna_cfg.get("norm", "layer"),
            "input_norm": rna_cfg.get("input_norm", True),
            "gene_names": gene_names,
        },
    )
    extractor.fit(x_fit.detach().cpu())
    return extractor


def _build_model(exp_cfg: dict, fold_data, pathology_in_dim: int, clinical_source: str, modalities: list[str], device: torch.device):
    model_cfg = exp_cfg.get("model", {})
    clinical_cfg = model_cfg.get("clinical", {})
    pathology_cfg = model_cfg.get("pathology", {})
    head_cfg = model_cfg.get("head", {})
    fusion_cfg = model_cfg.get("fusion", {})
    rna_extractor = None
    if "rna" in modalities:
        rna_extractor = _build_rna_extractor(model_cfg, fold_data.rna_gene_names, fold_data.fit_tensors[0])

    common_kwargs = {
        "modalities": modalities,
        "rna_extractor": rna_extractor,
        "clinical_source": clinical_source,
        "clinical_dim": fold_data.clinical_dim,
        "clinical_token_count": fold_data.clinical_token_count,
        "pathology_in_dim": pathology_in_dim,
        "clinical_hidden_dims": clinical_cfg.get("hidden_dims", [256, 128]),
        "clinical_emb_dim": clinical_cfg.get("emb_dim", 128),
        "clinical_token_hidden_dim": clinical_cfg.get("token_hidden_dim", 256),
        "clinical_token_out_dim": clinical_cfg.get("token_out_dim", 128),
        "clinical_dropout": clinical_cfg.get("dropout", 0.20),
        "clinical_activation": clinical_cfg.get("activation", "selu"),
        "pathology_emb_dim": pathology_cfg.get("emb_dim", 256),
        "pathology_aggregator": pathology_cfg.get("aggregator", "gated"),
        "pathology_attn_dim": pathology_cfg.get("attn_dim", 128),
        "pathology_dropout": pathology_cfg.get("dropout", 0.20),
        "head_hidden_dims": head_cfg.get("hidden_dims", [256, 64]),
        "head_dropout": head_cfg.get("dropout", 0.30),
        "head_activation": head_cfg.get("activation", "selu"),
    }
    model_type = exp_cfg.get("experiment", {}).get("model_type")
    if model_type == "two_modality_concat":
        model = TwoModalityConcatCoxModel(**common_kwargs)
    elif model_type == "two_modality_gated":
        model = TwoModalityGatedCoxModel(**common_kwargs)
    elif model_type == "two_modality_lowrank":
        model = TwoModalityLowRankBilinearCoxModel(
            fusion_rank=int(fusion_cfg.get("rank", 64)),
            fusion_out_dim=int(fusion_cfg.get("out_dim", 64)),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported two-modality model_type: {model_type}")
    return model.to(device)


def _configured_fold_assignments(data_cfg: dict) -> Path | None:
    return resolve_path(data_cfg["root"], data_cfg.get("fold_assignments"))


def _load_or_make_folds(dataset, cv_cfg: dict, output_dir: Path, fold_assignments_path: Path | None) -> pd.DataFrame:
    if fold_assignments_path is not None:
        return pd.read_csv(fold_assignments_path)

    output_path = output_dir / "fold_assignments.csv"
    if output_path.is_file():
        return pd.read_csv(output_path)

    folds = make_fold_assignments(
        dataset.sample_ids,
        dataset.labels,
        n_splits=int(cv_cfg.get("n_splits", 5)),
        seed=int(cv_cfg.get("seed", 42)),
    )
    folds.to_csv(output_path, index=False)
    return folds


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.experiment)
    data_cfg_full = load_yaml(args.data)
    data_cfg_raw = data_cfg_full.get("data", {})
    output_cfg = data_cfg_full.get("outputs", {})
    exp_info = exp_cfg.get("experiment", {})
    exp_data_cfg = exp_cfg.get("data", {})
    data_cfg = materialize_data_config(data_cfg_raw, exp_data_cfg)
    cv_cfg = exp_cfg.get("cv", {})
    train_cfg = exp_cfg.get("training", {})
    model_cfg = exp_cfg.get("model", {})

    allowed_model_types = {"two_modality_concat", "two_modality_gated", "two_modality_lowrank"}
    if exp_info.get("model_type") not in allowed_model_types:
        raise ValueError(f"scripts/train_two_modality.py requires model_type in {sorted(allowed_model_types)}.")
    modalities = list(exp_data_cfg.get("modalities", []))
    if len(modalities) != 2:
        raise ValueError("Two-modality configs must define data.modalities with exactly two entries.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = resolve_path(REPO_ROOT, exp_info.get("output_dir", output_cfg.get("root", "outputs")))
    output_dir = ensure_dir(output_dir)

    seed = int(cv_cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(train_cfg.get("device", "cpu"))
    clinical_source = exp_data_cfg.get("clinical_source", "tabular")
    pathology_tile_cap = model_cfg.get("pathology", {}).get("tile_cap")

    dataset = load_common_multimodal_dataset(
        data_cfg,
        clinical_source=clinical_source,
        seed=seed,
        pathology_tile_cap=pathology_tile_cap,
    )
    write_json(dataset.missing_summary, output_dir / "missing_summary.json")

    fold_assignments_path = args.fold_assignments or _configured_fold_assignments(data_cfg)
    fold_assignments = _load_or_make_folds(dataset, cv_cfg, output_dir, fold_assignments_path)
    n_splits = int(cv_cfg.get("n_splits", 5))
    folds_to_run = [args.fold] if args.fold is not None else list(range(n_splits))
    results = []
    test_risk_tables = []

    for fold in folds_to_run:
        split = prepare_fold_split(
            dataset.sample_ids,
            dataset.labels,
            fold_assignments,
            fold,
            seed=seed + fold,
            val_size=float(cv_cfg.get("val_size", 0.20)),
        )
        fold_data = prepare_embedding_fold_data(
            dataset,
            split,
            clinical_source=clinical_source,
            device=device,
        )
        pathology_in_dim = int(fold_data.pathology_train[0].shape[1])
        set_seed(seed + fold)
        model = _build_model(exp_cfg, fold_data, pathology_in_dim, clinical_source, modalities, device)

        model, history = train_embedding_multimodal(
            model,
            fold_data.fit_tensors,
            fold_data.pathology_fit,
            device=device,
            val_tensors=fold_data.val_tensors,
            pathology_val=fold_data.pathology_val,
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
        train_ci, train_risk = evaluate_embedding_multimodal(
            model,
            fold_data.train_tensors,
            fold_data.pathology_train,
            split.train_ids,
            device,
        )
        test_ci, test_risk = evaluate_embedding_multimodal(
            model,
            fold_data.test_tensors,
            fold_data.pathology_test,
            split.test_ids,
            device,
        )
        train_risk.to_csv(fold_dir / "train_risk_scores.csv", index=False)
        test_risk.to_csv(fold_dir / "test_risk_scores.csv", index=False)
        test_risk_tables.append(test_risk.assign(fold=fold))

        selected_genes = []
        if model.rna_extractor is not None and hasattr(model.rna_extractor, "get_selected_genes"):
            selected_genes = [str(gene) for gene in model.rna_extractor.get_selected_genes()]
            pd.Series(selected_genes).to_csv(fold_dir / "selected_genes.txt", index=False, header=False)

        summary = {
            **summarize_fold_split(dataset.labels, split),
            "c_index": test_ci,
            "train_c_index": train_ci,
            "modalities": modalities,
            "clinical_source": clinical_source,
            "clinical_dim": fold_data.clinical_dim,
            "clinical_token_count": fold_data.clinical_token_count,
            "pathology_in_dim": pathology_in_dim,
            "selected_gene_count": len(selected_genes),
            "fusion": exp_info.get("model_type", "").replace("two_modality_", ""),
            "fusion_input_dim": model.output_dim,
        }
        write_json(summary, fold_dir / "summary.json")
        write_json(fold_data.rna_medians, fold_dir / "rna_imputation_medians.json")
        write_json(fold_data.clinical_metadata, fold_dir / "clinical_metadata.json")
        save_checkpoint(
            fold_dir / "model.pt",
            model,
            summary,
            extra={
                "selected_genes": selected_genes,
                "experiment_config": exp_cfg,
            },
        )
        print(f"[fold {fold}] modalities={modalities} test_c_index={test_ci:.4f} train_c_index={train_ci:.4f}")
        results.append(summary)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results_per_fold.csv", index=False)
    aggregate = {
        "experiment": exp_info.get("name", args.experiment.stem),
        "modalities": modalities,
        "fusion": exp_info.get("model_type", "").replace("two_modality_", ""),
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
