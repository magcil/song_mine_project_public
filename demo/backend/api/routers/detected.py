import os
import sys
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.schemas.users import User, UserInDB
from api.routers.auth import get_current_user
from db.database import db
from api.schemas.detection import Result, PaginatedResult, PopularWinner, TopWinnersByDevice


# FastAPI Router
detected_router = r = APIRouter()


@r.get(
    "/getDetected",
    response_model=PaginatedResult,
)
async def getDetected(
    request: Request,
    device: str = None,
    page: int = 1,
    page_size: int = 15,
    current_user: UserInDB = Depends(get_current_user),
):
    query = {}
    if device:
        query["device"] = device

    skip = (page - 1) * page_size
    cursor = db["results"].find(query).skip(skip).limit(page_size).sort("datetime", -1)

    results = []
    async for result in cursor:
        results.append(Result(**result))

    total_count = await db["results"].count_documents(query)

    return {
        "totalCount": total_count,
        "data": results,
        "page": page,
        "page_size": page_size,
    }


@r.get("/getMostPopularWinner", response_model=PopularWinner)
async def getMostPopularWinner(
    request: Request, current_user: UserInDB = Depends(get_current_user)
):
    pipeline = [
        {"$group": {"_id": "$winner", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 1},
    ]
    cursor = db["results"].aggregate(pipeline)
    popular_winner = await cursor.to_list(length=1)

    if not popular_winner:
        return {"winner": "None", "occurrences": 0}

    return {
        "winner": popular_winner[0]["_id"],
        "occurrences": popular_winner[0]["count"],
    }


# @r.get("/getTopWinnersByDevice", response_model=List[TopWinnersByDevice])
# async def getTopWinnersByDevice(
#     request: Request,
#     top_n: int = 10,
#     current_user: UserInDB = Depends(get_current_user),
# ):
#     pipeline = [
#         {
#             "$group": {
#                 "_id": {"device": "$device", "winner": "$winner"},
#                 "count": {"$sum": 1},
#             }
#         },
#         {"$sort": {"count": -1}},
#         {
#             "$group": {
#                 "_id": "$_id.device",
#                 "winners": {"$push": {"winner": "$_id.winner", "count": "$count"}},
#             }
#         },
#         {"$project": {"device": "$_id", "winners": {"$slice": ["$winners", top_n]}}},
#     ]
#     cursor = db["results"].aggregate(pipeline)
#     top_winners_by_device = []
#     async for record in cursor:
#         top_winners_by_device.append(TopWinnersByDevice(**record))

#     return top_winners_by_device
