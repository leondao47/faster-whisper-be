import asyncio
import io
import json
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from faster_whisper import WhisperModel

from .settings import settings
from .streaming import StreamingWhisper
from .vad import VADGate

app = FastAPI(title="Whisper Fast Transcription API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily initialized singletons
_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.model_name,
            device=("cuda" if settings.device == "auto" else settings.device),
            compute_type=settings.compute_type,
        )
    return _model


def _decode_audio_bytes(raw: bytes, content_type: Optional[str]) -> tuple[np.ndarray, int]:
    """
    Decode incoming audio bytes into float32 PCM using either soundfile or ffmpeg.
    Returns (audio_float32, sample_rate).
    """
    # Try direct decode first (works for wav/flac/ogg if libsndfile supports it)
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype='float32')
        return data, sr
    except Exception:
        pass  # fall through to ffmpeg

    # Fallback: use ffmpeg to transcode to 16k mono wav
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as src:
        src_path = src.name
        src.write(raw)
    dst_path = src_path + ".wav"
    try:
        # Requires ffmpeg installed on the system
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ac", "1", "-ar", str(settings.sample_rate),
            "-f", "wav", dst_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data, sr = sf.read(dst_path, dtype='float32')
        return data, sr
    finally:
        try: os.remove(src_path)
        except: pass
        try: os.remove(dst_path)
        except: pass

@app.get("/ping")
async def ping():
    return JSONResponse({"status": "ok"})

@app.post("/v1/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Offline file transcription (fast).
    Accepts wav/mp3/m4a/webm/ogg... (ffmpeg fallback handles most formats)
    """
    raw = await file.read()
    data, sr = _decode_audio_bytes(raw, file.content_type)

    # Resample to model rate if needed (quick & good enough)
    if sr != settings.sample_rate:
        x = np.arange(len(data), dtype=np.float32)
        factor = settings.sample_rate / float(sr)
        xi = np.arange(0, len(data), 1.0 / factor, dtype=np.float32)
        data = np.interp(xi, x, data).astype(np.float32)
        sr = settings.sample_rate

    segments, info = get_model().transcribe(
        data,
        language=language or settings.whisper_lang,
        beam_size=settings.beam_size,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    out_segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments
    ]
    return JSONResponse({
        "language": info.language,
        "duration": info.duration,
        "segments": out_segments,
        "text": " ".join(s["text"] for s in out_segments).strip(),
    })


@app.websocket("/v1/stream")
async def stream(ws: WebSocket):
    """
    WebSocket streaming transcription.
    Expected client messages:
      - Binary: PCM16 little-endian mono @ 16kHz frames (10/20/30ms per frame)
      - Text: control JSON, e.g. {"event":"start", "lang":"en"} or {"event":"flush"}

    Server sends JSON text frames:
      {"type":"partial"|"final", "start":sec, "end":sec, "text":"..."}
      {"type":"info", "message":"..."}
    """
    await ws.accept()
    sw = StreamingWhisper()
    vad = VADGate(sample_rate=settings.sample_rate,
                  frame_ms=settings.vad_frame_ms,
                  aggressiveness=settings.vad_aggressiveness)

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                frame = msg["bytes"]
                # Route through VAD gate
                _started, stopped = vad.update(frame)
                if vad.triggered:
                    sw.append_audio(frame)
                if stopped:
                    # force a decode at speech end
                    results = await sw.maybe_decode()
                    if results:
                        for r in results:
                            await ws.send_text(json.dumps({
                                "type": "final",
                                "start": r.start,
                                "end": r.end,
                                "text": r.text,
                            }))
                else:
                    # periodic decode for partials/finals
                    results = await sw.maybe_decode()
                    if results:
                        for r in results:
                            await ws.send_text(json.dumps({
                                "type": "final",
                                "start": r.start,
                                "end": r.end,
                                "text": r.text,
                            }))

            elif "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    await ws.send_text(json.dumps({"type": "info", "message": "bad control frame"}))
                    continue
                if data.get("event") == "flush":
                    results = await sw.maybe_decode()
                    if results:
                        for r in results:
                            await ws.send_text(json.dumps({
                                "type": "final",
                                "start": r.start,
                                "end": r.end,
                                "text": r.text,
                            }))
                elif data.get("event") == "reset":
                    sw = StreamingWhisper()
                    await ws.send_text(json.dumps({"type": "info", "message": "state reset"}))
                elif data.get("event") == "start":
                    if lang := data.get("lang"):
                        settings.lang = lang
                        await ws.send_text(json.dumps({"type": "info", "message": f"lang={lang}"}))
                else:
                    await ws.send_text(json.dumps({"type": "info", "message": "ok"}))
    except WebSocketDisconnect:
        return
