"""
Offline ranking metrics: Precision@K, Recall@K, NDCG@K.

WHY these three metrics together:
- Precision@K: of the K items I recommended, how many did the user
  actually engage with? Measures *quality*.
- Recall@K: of the items the user engaged with after the cutoff, how
  many appeared in my top-K? Measures *coverage*.
- NDCG@K: like recall, but rewards putting relevant items near the top
  of the list. The most commonly reported metric in recsys papers.

WHY 'relevant' = rating >= 3.5:
- We trained on implicit feedback with the same threshold. Keeping the
  definition consistent across train and eval is critical.

WHY we sample users for evaluation:
- ML-25M has ~280K users. Scoring all of them takes ~30 min on a
  laptop. Sampling 10K users gives metrics with tiny variance and
  evaluates in ~1 minute. This is a standard trick (see Hu et al. 2008
  and Spotify's open-source recsys docs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class EvalResult:
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    n_users_evaluated: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "n_users_evaluated": float(self.n_users_evaluated),
        }


def _build_user_relevant_items(
    test_df: pd.DataFrame,
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    positive_threshold: float,
) -> Dict[int, set]:
    """Map user_idx -> set of relevant item_idx in test."""
    pos = test_df[test_df["rating"] >= positive_threshold]
    out: Dict[int, set] = {}
    for u, i in zip(pos["userId"].to_numpy(), pos["movieId"].to_numpy()):
        ui = user_id_to_idx.get(int(u))
        ii = item_id_to_idx.get(int(i))
        if ui is None or ii is None:
            continue
        out.setdefault(ui, set()).add(ii)
    return out


def _dcg(relevances: List[int]) -> float:
    """Standard DCG with log2 discount."""
    return float(sum(r / np.log2(idx + 2) for idx, r in enumerate(relevances)))


def evaluate(
    recommend_fn: Callable[[int], np.ndarray],
    test_df: pd.DataFrame,
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    k: int = 10,
    positive_threshold: float = 3.5,
    sample_users: int = 10_000,
    random_state: int = 42,
) -> EvalResult:
    """
    `recommend_fn(user_idx) -> np.ndarray of item_idx (length k)`.

    The callable lets us reuse this for both popularity and ALS without
    each model knowing about the eval harness.
    """
    user_relevant = _build_user_relevant_items(
        test_df, user_id_to_idx, item_id_to_idx, positive_threshold
    )

    rng = np.random.default_rng(random_state)
    eligible_users = np.array(list(user_relevant.keys()))
    if sample_users and sample_users < len(eligible_users):
        eligible_users = rng.choice(eligible_users, size=sample_users, replace=False)

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []

    for user_idx in eligible_users:
        recs = recommend_fn(int(user_idx))
        if len(recs) == 0:
            continue
        recs = recs[:k]

        relevant = user_relevant[user_idx]
        n_relevant = len(relevant)
        if n_relevant == 0:
            continue

        hits = [1 if int(r) in relevant else 0 for r in recs]
        n_hits = sum(hits)

        precisions.append(n_hits / k)
        recalls.append(n_hits / n_relevant)

        # Ideal DCG: as many 1s as possible, capped at min(k, n_relevant).
        ideal_hits = [1] * min(k, n_relevant)
        idcg = _dcg(ideal_hits)
        ndcg = _dcg(hits) / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return EvalResult(
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        n_users_evaluated=len(precisions),
    )


def format_comparison_table(rows: List[Dict[str, object]]) -> str:
    """Pretty-print results for the README / stdout."""
    df = pd.DataFrame(rows)
    cols = ["run_name", "precision_at_k", "recall_at_k", "ndcg_at_k", "n_users_evaluated"]
    df = df[cols]
    return df.to_string(index=False, float_format=lambda x: f"{x:.4f}")
