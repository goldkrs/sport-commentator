import base64
import asyncio
import io
import logging
import os
import tempfile
import time
import wave
from typing import Optional

import pyttsx3
from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyttsx3_tts_api")

app = FastAPI(
    title="pyttsx3 TTS server",
    description="Streams text via the local pyttsx3 speech engine and returns WAV audio.",
)

_TTS_RATE = int(os.getenv("TTS_RATE", "100"))
_TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
_TTS_PITCH = int(os.getenv("TTS_PITCH", "125"))
_tts_lock = asyncio.Lock()
_VOICE_RATE_MIN = int(os.getenv("VOICE_RATE_MIN", "150"))
_VOICE_RATE_MAX = int(os.getenv("VOICE_RATE_MAX", "240"))


def _encode_audio_to_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("ascii")


def _synthesize_text_with_pyttsx3(text: str, rate_override: Optional[int] = None) -> tuple[bytes, int, float]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate_override if rate_override is not None else _TTS_RATE)
        engine.setProperty("volume", _TTS_VOLUME)
        try:
            engine.setProperty("pitch", _TTS_PITCH)
        except Exception:
            logger.debug("Pitch adjustment not supported by pyttsx3 driver")
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with open(tmp_path, "rb") as fh:
            audio_bytes = fh.read()
        with wave.open(tmp_path, "rb") as wf:
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
        duration_ms = (frame_count / sample_rate) * 1000
        return audio_bytes, sample_rate, duration_ms
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def compute_dynamic_rate(text: str, duration_ms: float) -> int:
    duration_s = max(duration_ms, 1) / 1000.0
    words = len(text.strip().split())
    words = max(words, 1)
    words_per_minute = (words / duration_s) * 60.0
    rate = int(words_per_minute)
    if rate < _VOICE_RATE_MIN:
        rate = _VOICE_RATE_MIN
    elif rate > _VOICE_RATE_MAX:
        rate = _VOICE_RATE_MAX
    return rate


@app.post("/synthesize", summary="Convert text to streaming WAV using pyttsx3")
async def synthesize(
    text: str = Form(..., description="Chunk of text to render."),
    language: str = Form("English", description="Language hint."),
    instruct: Optional[str] = Form(None, description="Optional instruction for the voice."),
):
    trim_text = text.strip()
    if not trim_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    start = time.perf_counter()
    # pyttsx3 is not thread-safe, so protect each inference run.
    async with _tts_lock:
        audio_bytes, sample_rate, duration_ms = await asyncio.to_thread(_synthesize_text_with_pyttsx3, trim_text)
    latency_ms = (time.perf_counter() - start) * 1000

    headers = {
        "X-Sample-Rate": str(sample_rate),
        "X-Duration-Ms": f"{duration_ms:.0f}",
        "X-Latency-Ms": f"{latency_ms:.0f}",
        "X-Language": language,
        "X-Instruct": instruct or "",
    }
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers=headers)


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type in {"reference_audio", "ref_text", "language", "instruct"}:
                await websocket.send_json({"type": "ack", "detail": msg_type})
                continue
            if msg_type == "text_chunk":
                text = (message.get("text") or "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "detail": "text chunk cannot be empty"})
                    continue

                target_duration_ms = float(message.get("duration_ms") or 5000.0)
                rate_to_use = compute_dynamic_rate(text, target_duration_ms)
                logger.info(
                    "Synthesizing text chunk (%d chars) duration_ms=%.0f rate=%d",
                    len(text),
                    target_duration_ms,
                    rate_to_use,
                )
                await websocket.send_json({"type": "generation_start", "text": text})
                start = time.perf_counter()
                async with _tts_lock:
                    audio_bytes, sample_rate, duration_ms = await asyncio.to_thread(
                        _synthesize_text_with_pyttsx3, text, rate_override=rate_to_use
                    )
                latency_ms = (time.perf_counter() - start) * 1000

                encoded = _encode_audio_to_base64(audio_bytes)
                await websocket.send_json(
                    {
                        "type": "audio_chunk",
                        "audio": encoded,
                        "sample_rate": sample_rate,
                        "duration_ms": duration_ms,
                        "latency_ms": latency_ms,
                    }
                )
                continue
            if msg_type == "reset":
                await websocket.send_json({"type": "reset_ack"})
                continue

            await websocket.send_json({"type": "error", "detail": f"unknown msg type: {msg_type}"})
    except WebSocketDisconnect:
        logger.info("TTS WebSocket disconnected")


@app.get("/health", summary="Check that pyttsx3 TTS is ready")
async def health_check():
    return {"status": "ready", "engine": "pyttsx3", "rate": _TTS_RATE, "volume": _TTS_VOLUME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.text_to_speech:app", host="0.0.0.0", port=8001, log_level="info")
