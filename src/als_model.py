"""
Collaborative filtering via Alternating Least Squares on implicit feedback.

WHY ALS (and not deep learning, not item-kNN):
- Deep learning recommenders (NCF, two-tower, transformers) need a lot
  more data and compute to outperform ALS, and on MovieLens 25M the
  gap is small. For an interview portfolio that emphasises 'production
  thinking', ALS gives strong metrics with a 10-line training loop and
  ~5 minutes of training on a laptop.
- Item-kNN is fine for very small catalogues but blows up memory on
  ~50K items (the item-item similarity matrix is 50K * 50K dense
  floats = 10 GB).
- ALS is what Spotify, Etsy, and YouTube's early stacks used. It is
  the canonical baseline and easy to defend.

Library: `implicit` (Ben Frederickson). Uses Cython + BLAS, GPU optional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import joblib
import numpy as np
import scipy.sparse as sp


# Lazy import: `implicit` triggers a BLAS thread-count warning on import
# that clutters CI logs. We only import inside the trainer.
def _import_als():
    from implicit.als import AlternatingLeastSquares  # type: ignore
    return AlternatingLeastSquares


@dataclass
class ALSConfig:
    factors: int = 64
    regularization: float = 0.05
    iterations: int = 15
    alpha: float = 40.0          # used in the data loader, kept here for logging
    use_gpu: bool = False
    random_state: int = 42

    def as_dict(self) -> dict:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "use_gpu": self.use_gpu,
            "random_state": self.random_state,
        }


@dataclass
class ALSModel:
    """
    Thin wrapper so we can pickle the trained factors without dragging
    the whole implicit package into the runtime image.
    """
    config: ALSConfig
    user_factors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    item_factors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # We keep a reference to the underlying implicit object only during
    # training, because implicit's recommend() uses an internal index
    # that is faster than our hand-rolled top-K. At save time we drop it.
    _impl: Optional[object] = None

    # ------------------------------------------------------------------
    def fit(self, train_ui: sp.csr_matrix) -> "ALSModel":
        AlternatingLeastSquares = _import_als()
        self._impl = AlternatingLeastSquares(
            factors=self.config.factors,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            use_gpu=self.config.use_gpu,
            random_state=self.config.random_state,
        )
        # implicit>=0.6 expects (user, item) csr.
        self._impl.fit(train_ui, show_progress=True)

        # Cache the learned factors as numpy so we can save/load without
        # depending on the implicit version that trained them.
        self.user_factors = np.asarray(self._impl.user_factors)
        self.item_factors = np.asarray(self._impl.item_factors)
        return self

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_idx: int,
        train_ui: sp.csr_matrix,
        n: int = 10,
    ) -> np.ndarray:
        """
        Score = user_factors[user] @ item_factors.T, masking seen items.

        WHY a hand-rolled scorer instead of self._impl.recommend():
        - It works after the model is loaded from disk (no implicit dep).
        - It is also the version we will deploy.
        """
        if user_idx >= self.user_factors.shape[0]:
            return np.empty(0, dtype=np.int32)

        scores = self.item_factors @ self.user_factors[user_idx]
        # Mask items already seen in train.
        seen = train_ui[user_idx].indices
        scores[seen] = -np.inf

        # argpartition is O(n) vs argsort O(n log n). Matters for ~50K items.
        if n >= scores.size:
            return np.argsort(-scores).astype(np.int32)
        top = np.argpartition(-scores, n)[:n]
        return top[np.argsort(-scores[top])].astype(np.int32)

    def explain(
        self,
        rec_idx: int,
        liked_indices: Sequence[int],
    ) -> Optional[int]:
        """
        Return the liked item that is closest to `rec_idx` in latent
        factor space, or None if the input is empty.

        WHY item-item cosine over factors: ALS produces a dense vector
        per item. Similar factor vectors -> similar rating patterns
        across users -> canonical "people who liked X also liked Y"
        signal. Cosine (not dot product) so magnitude doesn't dominate.

        The API uses this to attach a human-readable reason to each
        recommendation ("Because you liked The Matrix").
        """
        if not liked_indices or rec_idx >= self.item_factors.shape[0]:
            return None
        liked = [i for i in liked_indices if 0 <= i < self.item_factors.shape[0]]
        if not liked:
            return None

        rec_vec = self.item_factors[rec_idx]
        rec_norm = rec_vec / (np.linalg.norm(rec_vec) + 1e-9)
        liked_vecs = self.item_factors[liked]
        liked_norms = liked_vecs / (
            np.linalg.norm(liked_vecs, axis=1, keepdims=True) + 1e-9
        )
        sims = liked_norms @ rec_norm
        return int(liked[int(np.argmax(sims))])

    def recommend_from_items(
        self,
        item_indices: Sequence[int],
        n: int = 10,
        exclude: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Recommend for a 'pseudo-user' built from a list of liked items.

        WHY this exists: the API needs to handle anonymous visitors who
        do not have a user_id. We project them into latent space by
        averaging the item factors of movies they liked. This is the
        standard 'fold-in' trick and approximates running one ALS
        update for that user. It will not match a fully-trained user
        embedding but it is good enough for cold-start.
        """
        if not item_indices:
            return np.empty(0, dtype=np.int32)

        pseudo_user = self.item_factors[list(item_indices)].mean(axis=0)
        scores = self.item_factors @ pseudo_user

        exclude_set = set(item_indices)
        if exclude is not None:
            exclude_set.update(exclude)
        for i in exclude_set:
            if 0 <= i < scores.size:
                scores[i] = -np.inf

        if n >= scores.size:
            return np.argsort(-scores).astype(np.int32)
        top = np.argpartition(-scores, n)[:n]
        return top[np.argsort(-scores[top])].astype(np.int32)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Drop the implicit ref before pickling.
        impl = self._impl
        self._impl = None
        try:
            joblib.dump(
                {
                    "config": self.config.as_dict(),
                    "user_factors": self.user_factors,
                    "item_factors": self.item_factors,
                },
                path,
            )
        finally:
            self._impl = impl

    @staticmethod
    def load(path: str) -> "ALSModel":
        blob = joblib.load(path)
        cfg = ALSConfig(**blob["config"])
        model = ALSModel(config=cfg)
        model.user_factors = blob["user_factors"]
        model.item_factors = blob["item_factors"]
        return model


def batch_recommend(
    model: ALSModel,
    user_indices: Sequence[int],
    train_ui: sp.csr_matrix,
    n: int = 10,
) -> List[np.ndarray]:
    return [model.recommend(u, train_ui, n=n) for u in user_indices]
