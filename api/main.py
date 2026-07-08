"""
FastAPI service for the movie recommender.

Design:
- Model + mappings are loaded ONCE at import time, not per request.
  Cold-start latency is ~2 seconds; warm latency is sub-10ms.
- The /recommend endpoint accepts either a known user_id (uses the
  trained user factor) or a list of liked titles (uses the fold-in
  trick from als_model.recommend_from_items).
- Titles are matched with a simple case-insensitive substring match.
  For production we would use a proper search index (Whoosh / Tantivy),
  but for this portfolio a linear scan over 60K titles is fast enough.

WHY everything in one file:
- The API is intentionally small. Splitting it into 'routers',
  'services', 'repositories' for a single endpoint is over-engineering
  and is something interviewers actively penalise.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import MovieOut, RecommendRequest, RecommendResponse

# Imported relative to the repo root, so run uvicorn from the repo root:
#   uvicorn api.main:app --reload
from src.als_model import ALSModel
from src.data_loader import load_artifacts


MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "best_als.joblib"))


# ---------------------------------------------------------------------------
# Load once at startup.
# ---------------------------------------------------------------------------
print(f"[api] loading model from {MODEL_PATH}")
_model = ALSModel.load(MODEL_PATH)
_train_ui, _mappings, _movies = load_artifacts(MODELS_DIR)
_user_id_to_idx: Dict[int, int] = _mappings["user_id_to_idx"]
_item_id_to_idx: Dict[int, int] = _mappings["item_id_to_idx"]
_idx_to_item_id: np.ndarray = _mappings["idx_to_item_id"]
print(f"[api] loaded {len(_user_id_to_idx):,} users, {len(_item_id_to_idx):,} items")


# Build a lowercase-title -> movieId index once.
_movies = _movies.set_index("movieId")
_title_lookup: List[Tuple[str, int]] = [
    (str(t).lower(), int(mid)) for mid, t in _movies["title"].items()
]


def _find_movie_id_by_title(query: str) -> Optional[int]:
    """
    Substring match, prefers exact match. Strips the trailing '(YYYY)'.
    """
    q = re.sub(r"\s*\(\d{4}\)\s*$", "", query).strip().lower()
    if not q:
        return None
    # 1. exact match
    for title, mid in _title_lookup:
        if title.startswith(q + " (") or title == q:
            return mid
    # 2. substring
    for title, mid in _title_lookup:
        if q in title:
            return mid
    return None


def _to_movie_out(
    movie_id: int,
    score: Optional[float] = None,
    because_of: Optional[str] = None,
) -> MovieOut:
    row = _movies.loc[movie_id]
    return MovieOut(
        movie_id=int(movie_id),
        title=str(row["title"]),
        genres=str(row["genres"]).split("|") if row["genres"] else [],
        score=score,
        because_of=because_of,
    )


def _title_for_idx(item_idx: int) -> Optional[str]:
    if item_idx is None or not (0 <= item_idx < len(_idx_to_item_id)):
        return None
    mid = int(_idx_to_item_id[item_idx])
    try:
        return str(_movies.loc[mid, "title"])
    except KeyError:
        return None


def _user_top_liked_indices(u_idx: int, top_k: int = 10) -> list:
    """
    For an existing MovieLens user, return the item indices they
    liked most (by confidence weight) in train. Used as the pool for
    the recommendation explanation lookup.
    """
    row = _train_ui[u_idx]
    if row.nnz == 0:
        return []
    order = np.argsort(-row.data)[:top_k]
    return row.indices[order].tolist()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MovieLens Recommender",
    description="ALS-based movie recommender. POST /recommend with user_id or liked_titles.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "n_users": len(_user_id_to_idx),
        "n_items": len(_item_id_to_idx),
        "model_factors": int(_model.config.factors),
    }


@app.get("/movies/search")
def search_movies(q: str, limit: int = 20) -> List[MovieOut]:
    """Used by the frontend to populate the multi-select."""
    q_low = q.lower().strip()
    if not q_low:
        return []
    out: List[MovieOut] = []
    for title, mid in _title_lookup:
        if q_low in title:
            out.append(_to_movie_out(mid))
            if len(out) >= limit:
                break
    return out


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if (req.user_id is None) == (req.liked_titles is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of user_id or liked_titles.",
        )

    if req.user_id is not None:
        u_idx = _user_id_to_idx.get(int(req.user_id))
        if u_idx is None:
            raise HTTPException(404, f"user_id {req.user_id} not in training set")
        rec_idx = _model.recommend(u_idx, _train_ui, n=req.n)
        # Explanation pool = the user's own top-liked items in train.
        liked_indices = _user_top_liked_indices(u_idx, top_k=10)
        matched: List[MovieOut] = []
        source = "user_id"
    else:
        item_ids: List[int] = []
        matched = []
        for title in (req.liked_titles or []):
            mid = _find_movie_id_by_title(title)
            if mid is None:
                continue
            i_idx = _item_id_to_idx.get(int(mid))
            if i_idx is None:
                continue
            item_ids.append(i_idx)
            matched.append(_to_movie_out(mid))
        if not item_ids:
            raise HTTPException(404, "None of the liked_titles matched the catalogue")
        rec_idx = _model.recommend_from_items(item_ids, n=req.n)
        liked_indices = item_ids
        source = "liked_titles"

    # Compute "because you liked X" for every recommendation.
    recs: List[MovieOut] = []
    for i in rec_idx:
        i = int(i)
        mid = int(_idx_to_item_id[i])
        because_idx = _model.explain(i, liked_indices)
        because_title = _title_for_idx(because_idx) if because_idx is not None else None
        recs.append(_to_movie_out(mid, because_of=because_title))

    return RecommendResponse(
        source=source,
        matched_input=matched,
        recommendations=recs,
    )
