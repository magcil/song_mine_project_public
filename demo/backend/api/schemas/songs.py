from pydantic import BaseModel
from typing import List

class Song(BaseModel):
    id: int
    title: str
    artist: str
    # Add other fields as needed

class TotalTracksResponse(BaseModel):
    total_tracks: int

class AllTracksResponse(BaseModel):
    totalCount: int
    skip: int
    limit: int
    songs: List[Song]
