import asyncio
import io
import os
import struct
import sys

import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import uvicorn
from fastapi import APIRouter, FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.routers.database import database_router
from api.routers.fingerprints import fingerprints_router
from api.routers.songs import songs_router
from api.routers.transcription import transcription_router
from api.routers.detected import detected_router
from api.routers.auth import auth_router
from fingerprinting.faissMind import FaissMind

DIR_PATH = os.path.dirname(__file__)


app = FastAPI(
    title="Museek Server",
    description="Museek FastAPI Server",
    version="1.0.0",
    max_upload_size=100 * 1024 * 1024,
    root_path="/api",
)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    fm = FaissMind(
        index_path=os.path.join(DIR_PATH, "data/IVF50_PQ32.index"),
        songs_path=os.path.join(DIR_PATH, "data/IVF50_PQ32.json"),
        k=4,
    )
    app.fm = fm


@app.exception_handler(Exception)
def validation_exception_handler(request, err):
    base_error_message = f"Failed to execute: {request.method}: {request.url}"
    return JSONResponse(
        status_code=400, content={"message": f"{base_error_message}. Detail: {err}"}
    )


# Routers
app.include_router(
    songs_router,
    prefix="/songs",
    tags=["Songs"],
)

app.include_router(
    fingerprints_router,
    prefix="/fingerprints",
    tags=["Fingerprints"],
)

app.include_router(
    transcription_router,
    prefix="/transcription",
    tags=["Transcription"],
)
# app.include_router(
#     database_router,
#     prefix="/api/database",
#     tags=["Database Operations"],
# )
app.include_router(
    detected_router,
    prefix="/detected",
    tags=["Historical Results"],
)
app.include_router(
    auth_router,
    prefix="/auth",
    tags=["Authorization"],
)


@app.get("/")
def status():
    return {"Status": "Up!"}
