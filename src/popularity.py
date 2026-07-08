"""
Popularity baseline.

WHY a baseline matters:
- Recommender papers without a baseline are interview red flags.
- Popularity is the cheapest possible recommender: it ignores the user
  entirely and returns the globally most popular items. If your CF
  model cannot beat it, the CF model is broken.
- It also gives a floor for engineering metrics: latency, memory,
  and infra cost should all be lower than CF. Anything worse than
  popularity on those axes is not worth deploying.

Scoring rule: Bayesian-adjusted mean rating.
    score = (v / (v + m)) * R + (m / (v + m)) * C
where R = mean rating of the item, v = number of ratings,
      C = global mean rating, m = prior weight (here, 90th percentile
      of rating counts).

This is the IMDb-style "Top 250" formula. It prevents a movie with
2 ratings of 5.0 from outranking a movie with 50,000 ratings of 4.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class PopularityModel:
    # Item indices sorted from most-popular to least-popular.
    ranked_item_idx: np.ndarray

    def recommend(
        self,
        user_idx: int,
        train_ui: sp.csr_matrix,
        n: int = 10,
    ) -> np.ndarray:
        """
        Return top-N item indices the user has NOT already interacted with.

        WHY filter seen items: a recommender that suggests a movie the
        user already rated 5 stars is useless. Standard practice for
        offline eval (and prod) is to drop seen items before truncating
        at K.
        """
        # Items the user already touched in train.
        seen = set(train_ui[user_idx].indices.tolist())

        recs = []
        for item_idx in self.ranked_item_idx:
            if item_idx in seen:
                continue
            recs.append(item_idx)
            if len(recs) >= n:
                break
        return np.asarray(recs, dtype=np.int32)


def fit_popularity(train_df: pd.DataFrame, item_id_to_idx: dict) -> PopularityModel:
    """
    Fit popularity scores from the TRAIN ratings dataframe.

    We accept the dataframe (not the sparse matrix) because we want the
    raw rating values for the Bayesian mean. The sparse matrix has
    confidence values, not ratings.
    """
    # Restrict to items the index knows about (drops any items missing
    # from the implicit-feedback matrix).
    train_df = train_df[train_df["movieId"].isin(item_id_to_idx.keys())]

    stats = train_df.groupby("movieId")["rating"].agg(["mean", "count"])
    C = stats["mean"].mean()
    m = stats["count"].quantile(0.90)  # prior strength

    score = (stats["count"] / (stats["count"] + m)) * stats["mean"] + (
        m / (stats["count"] + m)
    ) * C
    stats["score"] = score

    # Convert movieId -> internal idx so the API can use the same indexing.
    stats = stats.reset_index()
    stats["idx"] = stats["movieId"].map(item_id_to_idx)
    stats = stats.sort_values("score", ascending=False)

    return PopularityModel(ranked_item_idx=stats["idx"].to_numpy(dtype=np.int32))


def batch_recommend(
    model: PopularityModel,
    user_indices: Sequence[int],
    train_ui: sp.csr_matrix,
    n: int = 10,
) -> List[np.ndarray]:
    return [model.recommend(u, train_ui, n=n) for u in user_indices]
