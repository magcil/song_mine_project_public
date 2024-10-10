import os
import pathlib
import sys
from typing import Dict, List

from api.schemas.songs import AllTracksResponse, Song, TotalTracksResponse
from fastapi import APIRouter, File, Request, UploadFile

DIR_PATH = pathlib.Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, DIR_PATH)

songs_router = r = APIRouter()


# @r.post("/insertTrack")
# async def insert_new_track(request: Request, file: UploadFile = File(...)):
#     """
#     Insert new track to database
#     """
#     return {"status": "ok"}


@r.get("/totalTracks")
def get_num_tracks(request: Request):
    """
    Total number of tracks in database
    """
    num_songs = request.app.fm.get_total_songs()
    return {
        "total_tracks": num_songs,
    }


@r.get("/getAllTracks")
def get_all_tracks(request: Request, skip: int = 0, limit: int = 100):
    """
    Total number of tracks in database
    """
    songs, total_songs = request.app.fm.get_all_tracks(skip, limit)

    return {"totalCount": total_songs, "skip": skip, "limit": limit, "songs": songs}
