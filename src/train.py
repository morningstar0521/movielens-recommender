"""
End-to-end training: load -> split -> fit baseline + 4 ALS variants ->
evaluate -> log to MLflow -> save best model.

Run:
    python -m src.train --data-dir data/ml-25m --out-dir models

WHY a single CLI entry point:
- Reproducibility. One command gets the same numbers every time.
- Interview defence. 'How do you train?' -> 'python -m src.train'.

WHY MLflow over W&B / TensorBoard for this project:
- Free, local-only, no account needed.
- Stores artefacts (the .joblib model) alongside metrics so the API
  could in principle pull `models:/movielens-als/Production` from the
  registry. For the portfolio we keep it simple and just save to disk.

Memory note:
- Peak memory is the moment we fit ALS with factors=128, which holds
  two factor matrices of size (n_users, 128) and (n_items, 128) in
  float32. On ML-25M that is ~150 MB - small. The blocker is loading
  ratings.csv, not training.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List

import mlflow
import numpy as np

from .data_loader import load_movielens, save_artifacts
from .popularity import fit_popularity
from .als_model import ALSConfig, ALSModel
from .evaluation import evaluate, format_comparison_table


# Four ALS configs that sweep the three knobs that matter most:
# factors (model capacity), regularization (overfitting control),
# iterations (convergence). All else stays at sensible defaults.
ALS_CONFIGS: List[ALSConfig] = [
    ALSConfig(factors=32,  regularization=0.05, iterations=10),
    ALSConfig(factors=64,  regularization=0.05, iterations=15),
    ALSConfig(factors=64,  regularization=0.10, iterations=15),
    ALSConfig(factors=128, regularization=0.05, iterations=20),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder with ratings.csv and movies.csv")
    parser.add_argument("--out-dir", default="models", help="where to write artefacts")
    parser.add_argument("--mlflow-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="movielens-als")
    parser.add_argument("--eval-users", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # ------------------------------------------------------------------
    # Load + split
    # ------------------------------------------------------------------
    data = load_movielens(
        data_dir=args.data_dir,
        min_user_ratings=5,
        min_item_ratings=5,
        train_frac=0.8,
        alpha=40.0,
        positive_threshold=3.5,
    )

    # Persist the train matrix + mappings so the API can load them.
    save_artifacts(data, args.out_dir)

    # We also need the raw train ratings for the popularity baseline.
    # Reconstruct an approximate train_df from the sparse matrix is messy,
    # so we re-derive it from movies + the kept items. For simplicity we
    # re-read ratings here and reuse the same split logic. For a portfolio
    # this is fine; in prod we would refactor to share state.
    import pandas as pd
    ratings_path = os.path.join(args.data_dir, "ratings.csv")
    raw = pd.read_csv(
        ratings_path,
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
    )
    raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    train_df = raw.iloc[: int(len(raw) * 0.8)]

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print("\n=== Popularity baseline ===")
    with mlflow.start_run(run_name="popularity"):
        mlflow.log_param("model", "popularity_bayesian")
        mlflow.log_param("prior_quantile", 0.90)
        mlflow.log_param("k", args.k)

        pop = fit_popularity(train_df, data.item_id_to_idx)

        def pop_recommend(u: int) -> np.ndarray:
            return pop.recommend(u, data.train_ui, n=args.k)

        result = evaluate(
            pop_recommend,
            data.test_df,
            data.user_id_to_idx,
            data.item_id_to_idx,
            k=args.k,
            sample_users=args.eval_users,
        )
        for name, val in result.as_dict().items():
            mlflow.log_metric(name, val)
        print(result)
        all_results: List[Dict[str, object]] = [
            {"run_name": "popularity", **result.as_dict()}
        ]

    # ------------------------------------------------------------------
    # ALS sweep
    # ------------------------------------------------------------------
    best_run = None
    best_ndcg = -1.0
    best_model: ALSModel | None = None

    for i, cfg in enumerate(ALS_CONFIGS):
        run_name = f"als_f{cfg.factors}_r{cfg.regularization}_it{cfg.iterations}"
        print(f"\n=== {run_name} ===")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cfg.as_dict())
            mlflow.log_param("model", "als_implicit")
            mlflow.log_param("k", args.k)

            model = ALSModel(config=cfg).fit(data.train_ui)

            def als_recommend(u: int, m=model) -> np.ndarray:
                return m.recommend(u, data.train_ui, n=args.k)

            result = evaluate(
                als_recommend,
                data.test_df,
                data.user_id_to_idx,
                data.item_id_to_idx,
                k=args.k,
                sample_users=args.eval_users,
            )
            for name, val in result.as_dict().items():
                mlflow.log_metric(name, val)

            # Save artefact for this run so MLflow tracks it.
            tmp_path = os.path.join(args.out_dir, f"{run_name}.joblib")
            model.save(tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path="model")
            print(result)
            all_results.append({"run_name": run_name, **result.as_dict()})

            if result.ndcg_at_k > best_ndcg:
                best_ndcg = result.ndcg_at_k
                best_run = run_name
                best_model = model

    # ------------------------------------------------------------------
    # Promote best model to a stable path the API can read.
    # ------------------------------------------------------------------
    assert best_model is not None, "No ALS run succeeded"
    best_path = os.path.join(args.out_dir, "best_als.joblib")
    best_model.save(best_path)
    print(f"\n[train] best run: {best_run} (ndcg@{args.k} = {best_ndcg:.4f})")
    print(f"[train] saved -> {best_path}")

    print("\n=== Comparison ===")
    print(format_comparison_table(all_results))


if __name__ == "__main__":
    main()
