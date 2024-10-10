from datetime import datetime
from typing import Optional
from bson import ObjectId
from app.api.models.pyobjectid_model import PydanticObjectId
from pydantic import BaseModel, Field


class Result(BaseModel):
    id: Optional[PydanticObjectId] = Field(alias="_id")
    winner: str
    score: float
    datetime: datetime
    device: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
