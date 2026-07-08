"""Pydantic request/response models for the API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    # Exactly one of these must be supplied. Validated in the endpoint.
    user_id: Optional[int] = Field(
        default=None,
        description="A MovieLens userId present in the training set.",
    )
    liked_titles: Optional[List[str]] = Field(
        default=None,
        description="Free-text movie titles the user has liked. "
                    "Fuzzy-matched against the catalogue.",
    )
    n: int = Field(default=10, ge=1, le=100, description="How many recommendations to return.")


class MovieOut(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    score: Optional[float] = None
    # When present on a recommendation, this is the title of the user's
    # own liked movie that is most similar to the recommendation in the
    # ALS factor space. Frontend uses it to render "Because you liked X".
    because_of: Optional[str] = None


class RecommendResponse(BaseModel):
    source: str  # 'user_id' or 'liked_titles'
    matched_input: List[MovieOut]  # what we matched in the catalogue
    recommendations: List[MovieOut]
