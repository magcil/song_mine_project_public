import io
import os
import pathlib
import sys
import time

import librosa
import numpy as np
import torch
from fastapi import APIRouter, File, Query, Request, UploadFile
from pydub import AudioSegment
from scipy.io.wavfile import write
from typing_extensions import Annotated

DIR_PATH = pathlib.Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, DIR_PATH)

from models.neural_fingerprinter import Neural_Fingerprinter

from utils import utils

model = Neural_Fingerprinter()
model.load_state_dict(
    torch.load(
        os.path.join(DIR_PATH, "pre_trained_models/museeker.pt"),
        map_location=torch.device("cpu"),
    )
)

transcription_router = r = APIRouter()
THRESHOLD = 0.2
NOT_IN_DATABASE = "Not in database."
F = 8000
H = 4000


@r.post("/recording")
async def transcript_recording(
    request: Request,
    file: UploadFile = File(...),
    seg_dur: Annotated[
        int,
        Query(
            title="Segment duration",
            description="The duration (in secs) corresponding to each segment query.",
            ge=1,
            le=30,
        ),
    ] = 10,
):
    """
    Transcript an audio recording from audio file on queries of fixed duration (seg_dur)
    """
    audio_bytes = await file.read()

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=8000, mono=True)

    total_dur = y.size // sr
    total_segs, remainders = divmod(total_dur, seg_dur)

    response = {
        "filename": file.filename,
        "Total duration": time.strftime("%H:%M:%S", time.gmtime(total_dur)),
        "Duration query": str(seg_dur) + " seconds",
    }

    model.eval()
    with torch.no_grad():
        pred_results = {}
        for seg in range(total_segs):
            y_slice = y[seg * seg_dur * F : (seg + 1) * seg_dur * F]
            J = int(np.floor((y_slice.size - F) / H)) + 1
            batch = np.stack(
                [
                    utils.extract_mel_spectrogram(y_slice[j * H : j * H + F]).reshape(
                        1, 256, 32
                    )
                    for j in range(J)
                ]
            )
            query = model(torch.from_numpy(batch))
            winner, score, offset = request.app.fm.predict_query_offset(query.numpy())
            record_offset = time.strftime("%H:%M:%S", time.gmtime(seg * seg_dur))

            m_start, s_start = divmod(offset, 60)
            m_end, s_end = divmod(offset + seg_dur, 60)
            pred_results[record_offset] = {
                "winner": winner,
                "score": float(score),
                "offset": f"{int(m_start)}:{int(s_start):02d}-{int(m_end)}:{int(s_end):02d}",
            }
        # Are there any secs left?
        if remainders:
            y_slice = y[
                (seg + 1) * seg_dur * F : (seg + 1) * seg_dur * F + remainders * F
            ]
            J = int(np.floor((y_slice.size - F) / H)) + 1
            batch = np.stack(
                [
                    utils.extract_mel_spectrogram(y_slice[j * H : j * H + F]).reshape(
                        1, 256, 32
                    )
                    for j in range(J)
                ]
            )
            query = model(torch.from_numpy(batch))
            winner, score, offset = request.app.fm.predict_query_offset(query.numpy())
            record_offset = time.strftime("%H:%M:%S", time.gmtime((seg + 1) * seg_dur))
            m_start, s_start = divmod(offset, 60)
            m_end, s_end = divmod(offset + remainders, 60)
            pred_results[record_offset] = {
                "winner": winner,
                "score": float(score),
                "offset": f"{int(m_start)}:{int(s_start):02d}-{int(m_end)}:{int(s_end):02d}",
            }

    response["results"] = pred_results
    return response


@r.post("/blob")
async def blob(request: Request, file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    audio_segment = audio_segment.set_channels(1)
    audio_segment = audio_segment.set_frame_rate(8000)

    sr = audio_segment.frame_rate

    y = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / (
        1 << (8 * audio_segment.sample_width - 1)
    )
    y = y.astype("float32")

    dur = y.size // sr
    y_trimmed = y[: sr * dur]

    model.eval()
    with torch.no_grad():
        J = int(np.floor((y_trimmed.size - F) / H)) + 1
        batch = np.stack(
            [
                utils.extract_mel_spectrogram(y_trimmed[j * H : j * H + F]).reshape(
                    1, 256, 32
                )
                for j in range(J)
            ]
        )
        query = model(torch.from_numpy(batch))
        winner, score = request.app.fm.get_query_results(query.numpy())
        result = {"winner": winner, "score": score}

    return result
