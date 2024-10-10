import os
import pathlib
import sys

from fastapi import APIRouter, Request


fingerprints_router = r = APIRouter()


@r.get("/totalFingerprints")
def get_all_fingerprints(request: Request):
    """
    Total number of fingerprints in database
    """
    fingerprints = request.app.fm.get_total_fingerprints()
    return {"totalCount": fingerprints}
