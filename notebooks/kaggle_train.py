"""
Kaggle-runnable training script for the MovieLens recommender.

HOW TO USE
----------
1. Go to https://www.kaggle.com/code and click "New Notebook".
2. In the right sidebar -> "+ Add Input" -> search "movielens 25m" and add
   the dataset by GroupLens (typically at path /kaggle/input/movielens-25m/
   or /kaggle/input/movielens-25m-dataset/).
3. In Notebook settings (right sidebar):
     - Accelerator: None (CPU is enough; GPU does not help ALS here).
     - Internet: On (needed for pip install implicit / mlflow).
4. Paste this ENTIRE file into ONE code cell.
5. Adjust DATA_DIR below if the input path is different.
6. Run All. Total time: ~10-15 min on Kaggle CPU.
7. When done, the right sidebar shows a "models" folder under /kaggle/working/.
   Click the three dots -> Download to get models.zip.
8. Unzip into your local repo at movie-recommender/models/ and start the
   API + Streamlit locally. You never trained on your Mac.

WHY this file is self-contained (not imports from src/):
- Kaggle notebooks are one-file environments. Uploading the whole repo
  as a Kaggle Dataset just to import 5 modules is more friction than
  a 400-line script.
- Every function below is a copy of the one in src/. Behaviour is
  identical, so metrics logged here match what src/train.py would
  produce locally.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 0. Install runtime deps that Kaggle does not preinstall.
# ---------------------------------------------------------------------------
import subprocess
import sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# Kaggle's default image already has numpy, pandas, scipy, scikit-learn,
# joblib, and pyarrow 21+. It does NOT have `implicit`.
#
# WHY no MLflow here: MLflow 2.19 caps pyarrow<19, and downgrading
# pyarrow breaks Kaggle's pre-built C extensions (PyExtensionType /
# IpcReadOptions ABI mismatch). We log the same records to a plain
# JSONL file instead. `src/train.py` still uses MLflow for local runs.
_pip("implicit==0.7.2")

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Tiny stand-in for mlflow.start_run / log_param / log_metric that writes
# to a JSONL file. Keeps the training loop below unchanged in spirit.
# ---------------------------------------------------------------------------
class RunLogger:
    """Append-only experiment tracker. One JSON object per run."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Truncate on start so re-runs don't accumulate.
        open(self.path, "w").close()
        self._current: Optional[dict] = None

    def start(self, run_name: str) -> None:
        self._current = {
            "run_name": run_name,
            "start_ts": time.time(),
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

    def log_params(self, params: dict) -> None:
        assert self._current is not None
        self._current["params"].update({k: _jsonable(v) for k, v in params.items()})

    def log_param(self, k: str, v) -> None:
        self.log_params({k: v})

    def log_metric(self, k: str, v: float) -> None:
        assert self._current is not None
        self._current["metrics"][k] = float(v)

    def log_artifact(self, path: str) -> None:
        assert self._current is not None
        self._current["artifacts"].append(path)

    def end(self) -> None:
        assert self._current is not None
        self._current["end_ts"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(self._current) + "\n")
        self._current = None


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


# ---------------------------------------------------------------------------
# CONFIG - the only thing you may need to change.
# ---------------------------------------------------------------------------
# The exact Kaggle input path depends on which MovieLens 25M dataset
# you attached. Common ones:
#   /kaggle/input/movielens-25m-dataset/
#   /kaggle/input/movielens-25m/
# Print os.listdir("/kaggle/input") in a fresh cell if unsure.
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/movielens-25m-dataset",
    "/kaggle/input/movielens-25m",
    "/kaggle/input/movielens-25m-dataset/ml-25m",
    "/kaggle/input/movielens-25m/ml-25m",
]
OUT_DIR = "/kaggle/working/models"
RUNS_PATH = "/kaggle/working/experiment_runs.jsonl"
EVAL_USERS = 10_000
K = 10


def _find_data_dir() -> str:
    """Walk the candidates and pick the first one that has ratings.csv."""
    for c in CANDIDATE_DATA_DIRS:
        if os.path.exists(os.path.join(c, "ratings.csv")):
            return c
    # Last resort: walk /kaggle/input to locate ratings.csv.
    for root, _, files in os.walk("/kaggle/input"):
        if "ratings.csv" in files and "movies.csv" in files:
            return root
    raise FileNotFoundError(
        "Could not find ratings.csv. Attach MovieLens 25M as a notebook input."
    )


# ===========================================================================
# 1. DATA LOADER (mirrors src/data_loader.py)
# ===========================================================================
@dataclass
class MovieLensData:
    train_ui: sp.csr_matrix
    test_df: pd.DataFrame
    user_id_to_idx: Dict[int, int]
    item_id_to_idx: Dict[int, int]
    idx_to_item_id: np.ndarray
    movies: pd.DataFrame
    split_timestamp: int


def load_movielens(
    data_dir: str,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    train_frac: float = 0.8,
    alpha: float = 40.0,
    positive_threshold: float = 3.5,
) -> Tuple[MovieLensData, pd.DataFrame]:
    """
    Returns (MovieLensData, raw_train_df). raw_train_df is needed for the
    popularity baseline which wants actual rating values, not the
    confidence-weighted matrix.

    WHY chronological split: production always trains on the past and
    predicts the future. A random split leaks future preferences and
    inflates offline metrics.
    """
    print("[data] reading csv...")
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.csv"),
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.csv"),
        dtype={"movieId": np.int32, "title": "string", "genres": "string"},
    )
    print(f"[data] raw ratings: {len(ratings):,}")

    # k-core filter: drop very-sparse users/items. Two passes because
    # filtering items can drop users below threshold and vice versa.
    for _ in range(2):
        ic = ratings["movieId"].value_counts()
        ratings = ratings[ratings["movieId"].isin(ic[ic >= min_item_ratings].index)]
        uc = ratings["userId"].value_counts()
        ratings = ratings[ratings["userId"].isin(uc[uc >= min_user_ratings].index)]
    print(f"[data] after k-core filter: {len(ratings):,}")

    # Chronological 80/20 split on GLOBAL timestamp.
    ratings = ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cutoff = int(len(ratings) * train_frac)
    split_ts = int(ratings.iloc[cutoff]["timestamp"])
    train_df = ratings.iloc[:cutoff].copy()
    test_df = ratings.iloc[cutoff:].copy()
    print(f"[data] train {len(train_df):,}  test {len(test_df):,}  cutoff_ts {split_ts}")

    # Drop cold-start rows from test (users/items not present in train).
    tu = set(train_df["userId"].unique())
    ti = set(train_df["movieId"].unique())
    test_df = test_df[test_df["userId"].isin(tu) & test_df["movieId"].isin(ti)].reset_index(drop=True)
    print(f"[data] test after cold-start drop: {len(test_df):,}")

    # Build confidence matrix. Rating threshold 3.5 = 'liked'.
    pos = train_df[train_df["rating"] >= positive_threshold]
    users = np.sort(pos["userId"].unique())
    items = np.sort(pos["movieId"].unique())
    u2i = {int(u): i for i, u in enumerate(users)}
    i2i = {int(m): i for i, m in enumerate(items)}
    idx_to_item = items.astype(np.int32)

    rows = pos["userId"].map(u2i).to_numpy()
    cols = pos["movieId"].map(i2i).to_numpy()
    conf = (1.0 + alpha * (pos["rating"].to_numpy() - positive_threshold)).astype(np.float32)
    train_ui = sp.csr_matrix((conf, (rows, cols)), shape=(len(users), len(items)))
    print(f"[data] train matrix {train_ui.shape}  nnz={train_ui.nnz:,}")

    data = MovieLensData(
        train_ui=train_ui,
        test_df=test_df,
        user_id_to_idx=u2i,
        item_id_to_idx=i2i,
        idx_to_item_id=idx_to_item,
        movies=movies,
        split_timestamp=split_ts,
    )
    return data, train_df


def save_artifacts(data: MovieLensData, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sp.save_npz(os.path.join(out_dir, "train_ui.npz"), data.train_ui)
    joblib.dump(
        {
            "user_id_to_idx": data.user_id_to_idx,
            "item_id_to_idx": data.item_id_to_idx,
            "idx_to_item_id": data.idx_to_item_id,
            "split_timestamp": data.split_timestamp,
        },
        os.path.join(out_dir, "mappings.joblib"),
    )
    data.movies.to_parquet(os.path.join(out_dir, "movies.parquet"))


# ===========================================================================
# 2. POPULARITY BASELINE (mirrors src/popularity.py)
# ===========================================================================
@dataclass
class PopularityModel:
    ranked_item_idx: np.ndarray

    def recommend(self, u_idx: int, train_ui: sp.csr_matrix, n: int = 10) -> np.ndarray:
        seen = set(train_ui[u_idx].indices.tolist())
        out = []
        for i in self.ranked_item_idx:
            if i in seen:
                continue
            out.append(i)
            if len(out) >= n:
                break
        return np.asarray(out, dtype=np.int32)


def fit_popularity(train_df: pd.DataFrame, item_id_to_idx: dict) -> PopularityModel:
    """IMDb-style Bayesian mean. Prevents 1-rating items dominating."""
    tdf = train_df[train_df["movieId"].isin(item_id_to_idx.keys())]
    stats = tdf.groupby("movieId")["rating"].agg(["mean", "count"])
    C = stats["mean"].mean()
    m = stats["count"].quantile(0.90)
    score = (stats["count"] / (stats["count"] + m)) * stats["mean"] + (
        m / (stats["count"] + m)
    ) * C
    stats["score"] = score
    stats = stats.reset_index()
    stats["idx"] = stats["movieId"].map(item_id_to_idx)
    stats = stats.sort_values("score", ascending=False)
    return PopularityModel(ranked_item_idx=stats["idx"].to_numpy(dtype=np.int32))


# ===========================================================================
# 3. ALS (mirrors src/als_model.py)
# ===========================================================================
@dataclass
class ALSConfig:
    factors: int = 64
    regularization: float = 0.05
    iterations: int = 15
    alpha: float = 40.0
    random_state: int = 42

    def as_dict(self) -> dict:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }


@dataclass
class ALSModel:
    config: ALSConfig
    user_factors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    item_factors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def fit(self, train_ui: sp.csr_matrix) -> "ALSModel":
        from implicit.als import AlternatingLeastSquares
        m = AlternatingLeastSquares(
            factors=self.config.factors,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            random_state=self.config.random_state,
        )
        m.fit(train_ui, show_progress=True)
        self.user_factors = np.asarray(m.user_factors)
        self.item_factors = np.asarray(m.item_factors)
        return self

    def recommend(self, u_idx: int, train_ui: sp.csr_matrix, n: int = 10) -> np.ndarray:
        if u_idx >= self.user_factors.shape[0]:
            return np.empty(0, dtype=np.int32)
        scores = self.item_factors @ self.user_factors[u_idx]
        seen = train_ui[u_idx].indices
        scores[seen] = -np.inf
        if n >= scores.size:
            return np.argsort(-scores).astype(np.int32)
        top = np.argpartition(-scores, n)[:n]
        return top[np.argsort(-scores[top])].astype(np.int32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "config": self.config.as_dict(),
                "user_factors": self.user_factors,
                "item_factors": self.item_factors,
            },
            path,
        )


# ===========================================================================
# 4. EVAL (mirrors src/evaluation.py)
# ===========================================================================
def _dcg(rel: List[int]) -> float:
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(rel)))


def evaluate(
    rec_fn: Callable[[int], np.ndarray],
    test_df: pd.DataFrame,
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    k: int = 10,
    positive_threshold: float = 3.5,
    sample_users: int = 10_000,
    random_state: int = 42,
) -> Dict[str, float]:
    pos = test_df[test_df["rating"] >= positive_threshold]
    user_rel: Dict[int, set] = {}
    for u, i in zip(pos["userId"].to_numpy(), pos["movieId"].to_numpy()):
        ui = user_id_to_idx.get(int(u))
        ii = item_id_to_idx.get(int(i))
        if ui is None or ii is None:
            continue
        user_rel.setdefault(ui, set()).add(ii)

    rng = np.random.default_rng(random_state)
    eligible = np.array(list(user_rel.keys()))
    if sample_users and sample_users < len(eligible):
        eligible = rng.choice(eligible, size=sample_users, replace=False)

    P, R, N = [], [], []
    for u in eligible:
        recs = rec_fn(int(u))[:k]
        if len(recs) == 0:
            continue
        rel = user_rel[u]
        if not rel:
            continue
        hits = [1 if int(r) in rel else 0 for r in recs]
        P.append(sum(hits) / k)
        R.append(sum(hits) / len(rel))
        idcg = _dcg([1] * min(k, len(rel)))
        N.append(_dcg(hits) / idcg if idcg > 0 else 0.0)

    return {
        "precision_at_k": float(np.mean(P)) if P else 0.0,
        "recall_at_k": float(np.mean(R)) if R else 0.0,
        "ndcg_at_k": float(np.mean(N)) if N else 0.0,
        "n_users_evaluated": float(len(P)),
    }


# ===========================================================================
# 5. RUN
# ===========================================================================
DATA_DIR = _find_data_dir()
print(f"[main] using data at {DATA_DIR}")

runs = RunLogger(RUNS_PATH)

data, raw_train_df = load_movielens(DATA_DIR)
save_artifacts(data, OUT_DIR)

# --- baseline ---
print("\n=== Popularity baseline ===")
runs.start("popularity")
runs.log_param("model", "popularity_bayesian")
runs.log_param("prior_quantile", 0.90)
runs.log_param("k", K)
pop = fit_popularity(raw_train_df, data.item_id_to_idx)
m_pop = evaluate(
    lambda u: pop.recommend(u, data.train_ui, n=K),
    data.test_df, data.user_id_to_idx, data.item_id_to_idx,
    k=K, sample_users=EVAL_USERS,
)
for name, v in m_pop.items():
    runs.log_metric(name, v)
print(m_pop)
runs.end()
all_rows = [{"run_name": "popularity", **m_pop}]

# --- ALS sweep ---
ALS_CONFIGS: List[ALSConfig] = [
    ALSConfig(factors=32,  regularization=0.05, iterations=10),
    ALSConfig(factors=64,  regularization=0.05, iterations=15),
    ALSConfig(factors=64,  regularization=0.10, iterations=15),
    ALSConfig(factors=128, regularization=0.05, iterations=20),
]

best_ndcg = -1.0
best_name = ""
best_model: Optional[ALSModel] = None

for cfg in ALS_CONFIGS:
    name = f"als_f{cfg.factors}_r{cfg.regularization}_it{cfg.iterations}"
    print(f"\n=== {name} ===")
    runs.start(name)
    runs.log_params(cfg.as_dict())
    runs.log_param("model", "als_implicit")
    runs.log_param("k", K)
    model = ALSModel(cfg).fit(data.train_ui)
    m_als = evaluate(
        lambda u, mdl=model: mdl.recommend(u, data.train_ui, n=K),
        data.test_df, data.user_id_to_idx, data.item_id_to_idx,
        k=K, sample_users=EVAL_USERS,
    )
    for k_, v in m_als.items():
        runs.log_metric(k_, v)
    art_path = os.path.join(OUT_DIR, f"{name}.joblib")
    model.save(art_path)
    runs.log_artifact(art_path)
    print(m_als)
    runs.end()
    all_rows.append({"run_name": name, **m_als})

    if m_als["ndcg_at_k"] > best_ndcg:
        best_ndcg = m_als["ndcg_at_k"]
        best_name = name
        best_model = model

assert best_model is not None
best_model.save(os.path.join(OUT_DIR, "best_als.joblib"))
print(f"\n[main] best = {best_name}  ndcg@{K}={best_ndcg:.4f}")

# --- comparison table ---
df = pd.DataFrame(all_rows)[
    ["run_name", "precision_at_k", "recall_at_k", "ndcg_at_k", "n_users_evaluated"]
]
print("\n=== Comparison ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

# --- copy run log next to artefacts so it ends up in the zip ---
import shutil
shutil.copy(RUNS_PATH, os.path.join(OUT_DIR, "experiment_runs.jsonl"))

zip_path = "/kaggle/working/models_bundle"
shutil.make_archive(zip_path, "zip", OUT_DIR)
print(f"\n[main] artefacts zipped -> {zip_path}.zip")
print("Open the right sidebar 'Output' tab and download models_bundle.zip.")
print(f"Experiment log: {RUNS_PATH} (also inside the zip).")
