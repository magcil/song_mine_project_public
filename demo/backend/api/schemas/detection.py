from typing import Dict, List

from pydantic import BaseModel


# Pydantic Models
class Result(BaseModel):
    winner: str
    score: float
    datetime: str
    device: str


class PaginatedResult(BaseModel):
    totalCount: int
    data: List[Result]
    page: int
    page_size: int


class PopularWinner(BaseModel):
    winner: str
    occurrences: int


class TopWinnersByDevice(BaseModel):
    device: str
    winners: Dict[str, int]

