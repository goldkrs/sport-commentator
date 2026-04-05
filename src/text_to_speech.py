import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from kokoro import KPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyttsx3_tts_api")

app = FastAPI(
    title="pyttsx3 TTS server",
    description="Streams text via the local Kokoro speech engine and returns WAV audio.",
)

# Preserve the same environment knobs where possible so the rest of the service contract stays familiar.
_TTS_RATE = int(os.getenv("TTS_RATE", "250"))
_TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
_TTS_PITCH = int(os.getenv("TTS_PITCH", "125"))  # kept for API/config compatibility; Kokoro does not use pitch directly
_tts_lock = asyncio.Lock()
_VOICE_RATE_MIN = int(os.getenv("VOICE_RATE_MIN", "150"))
_VOICE_RATE_MAX = int(os.getenv("VOICE_RATE_MAX", "500"))

_KOKORO_DEFAULT_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")
_KOKORO_DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
_KOKORO_SPLIT_PATTERN = os.getenv("KOKORO_SPLIT_PATTERN", r"\n+")
_KOKORO_SAMPLE_RATE = int(os.getenv("KOKORO_SAMPLE_RATE", "24000"))

_pipeline_cache: dict[str, KPipeline] = {}


def _encode_audio_to_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("ascii")


def _normalize_language_hint(language: Optional[str]) -> str:
    value = (language or "").strip().lower()
    if not value:
        return _KOKORO_DEFAULT_LANG_CODE

    mapping = {
        "a": "a",
        "american": "a",
        "american english": "a",
        "english": "a",
        "en": "a",
        "en-us": "a",
        "b": "b",
        "british": "b",
        "british english": "b",
        "en-gb": "b",
        "e": "e",
        "spanish": "e",
        "es": "e",
        "f": "f",
        "french": "f",
        "fr": "f",
        "h": "h",
        "hindi": "h",
        "hi": "h",
        "i": "i",
        "italian": "i",
        "it": "i",
        "j": "j",
        "japanese": "j",
        "ja": "j",
        "p": "p",
        "portuguese": "p",
        "pt": "p",
        "pt-br": "p",
        "brazilian portuguese": "p",
        "z": "z",
        "mandarin": "z",
        "chinese": "z",
        "zh": "z",
        "zh-cn": "z",
    }
    return mapping.get(value, _KOKORO_DEFAULT_LANG_CODE)


def _voice_matches_lang_code(voice: str, lang_code: str) -> bool:
    if not voice or len(voice) < 1:
        return False
    voice_prefix = voice.split("_")[0].strip().lower()
    return voice_prefix.startswith(lang_code.lower())


def _resolve_voice(language: Optional[str], instruct: Optional[str]) -> tuple[str, str]:
    lang_code = _normalize_language_hint(language)

    # Preserve the same API contract. `instruct` is still accepted; if caller passes `voice=<kokoro_voice>`
    # we use it, otherwise it remains informational metadata like before.
    requested_voice = None
    if instruct:
        raw = instruct.strip()
        if raw.lower().startswith("voice="):
            requested_voice = raw.split("=", 1)[1].strip()

    voice = requested_voice or _KOKORO_DEFAULT_VOICE
    if not _voice_matches_lang_code(voice, lang_code):
        logger.warning(
            "Voice %s does not match lang_code=%s; falling back to default voice %s",
            voice,
            lang_code,
            _KOKORO_DEFAULT_VOICE,
        )
        voice = _KOKORO_DEFAULT_VOICE

    return lang_code, voice


def _get_pipeline(lang_code: str) -> KPipeline:
    pipeline = _pipeline_cache.get(lang_code)
    if pipeline is None:
        logger.info("Loading Kokoro pipeline for lang_code=%s", lang_code)
        pipeline = KPipeline(lang_code=lang_code)
        _pipeline_cache[lang_code] = pipeline
    return pipeline


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


def _rate_to_kokoro_speed(rate: int) -> float:
    # Map the existing "speech rate" semantics onto Kokoro's `speed` parameter.
    # 180-200 WPM ~= around 1.0x. Keep the range bounded for intelligibility.
    speed = rate / 180.0
    if speed < 0.7:
        speed = 0.7
    elif speed > 2.5:
        speed = 2.5
    return speed


def _audio_array_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if _TTS_VOLUME != 1.0:
        audio = np.clip(audio * _TTS_VOLUME, -1.0, 1.0)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _synthesize_text_with_kokoro(
    text: str,
    language: Optional[str] = None,
    instruct: Optional[str] = None,
    rate_override: Optional[int] = None,
) -> tuple[bytes, int, float]:
    lang_code, voice = _resolve_voice(language, instruct)
    pipeline = _get_pipeline(lang_code)
    effective_rate = rate_override if rate_override is not None else _TTS_RATE
    speed = _rate_to_kokoro_speed(effective_rate)

    logger.info(
        "Kokoro synthesizing %d chars lang_code=%s voice=%s rate=%s speed=%.2f",
        len(text),
        lang_code,
        voice,
        effective_rate,
        speed,
    )

    chunks: list[np.ndarray] = []
    generator = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=_KOKORO_SPLIT_PATTERN,
    )

    for idx, (_gs, _ps, audio) in enumerate(generator):
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.size == 0:
            continue
        chunks.append(audio_np)
        logger.debug("Kokoro produced chunk %d with %d samples", idx, audio_np.size)

    if not chunks:
        raise RuntimeError("Kokoro returned no audio for the provided text")

    merged_audio = np.concatenate(chunks)
    wav_bytes = _audio_array_to_wav_bytes(merged_audio, _KOKORO_SAMPLE_RATE)
    duration_ms = (len(merged_audio) / _KOKORO_SAMPLE_RATE) * 1000.0
    return wav_bytes, _KOKORO_SAMPLE_RATE, duration_ms


@app.post("/synthesize", summary="Convert text to streaming WAV using Kokoro")
async def synthesize(
    text: str = Form(..., description="Chunk of text to render."),
    language: str = Form("English", description="Language hint."),
    instruct: Optional[str] = Form(None, description="Optional instruction for the voice."),
):
    trim_text = text.strip()
    print(trim_text)
    if not trim_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    start = time.perf_counter()
    async with _tts_lock:
        try:
            audio_bytes, sample_rate, duration_ms = await asyncio.to_thread(
                _synthesize_text_with_kokoro,
                trim_text,
                language,
                instruct,
            )
        except Exception as exc:
            logger.exception("Kokoro synthesis failed")
            raise HTTPException(status_code=500, detail=f"kokoro synthesis failed: {exc}") from exc
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
    session_language = "English"
    session_instruct: Optional[str] = None

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            logger.debug("TTS WS incoming | type=%s | payload=%s", msg_type, {k:v for k,v in message.items() if k != "audio"})

            if msg_type == "reference_audio":
                await websocket.send_json({"type": "ack", "detail": msg_type})
                continue
            if msg_type == "ref_text":
                await websocket.send_json({"type": "ack", "detail": msg_type})
                continue
            if msg_type == "language":
                session_language = (message.get("value") or message.get("language") or "English").strip() or "English"
                await websocket.send_json({"type": "ack", "detail": msg_type})
                continue
            if msg_type == "instruct":
                session_instruct = (message.get("value") or message.get("instruct") or "").strip() or None
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
                try:
                    async with _tts_lock:
                        audio_bytes, sample_rate, duration_ms = await asyncio.to_thread(
                            _synthesize_text_with_kokoro,
                            text,
                            session_language,
                            session_instruct,
                            rate_to_use,
                        )
                except Exception as exc:
                    logger.exception("Kokoro WebSocket synthesis failed")
                    await websocket.send_json({"type": "error", "detail": f"kokoro synthesis failed: {exc}"})
                    continue
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
                session_language = "English"
                session_instruct = None
                await websocket.send_json({"type": "reset_ack"})
                continue

            await websocket.send_json({"type": "error", "detail": f"unknown msg type: {msg_type}"})
    except WebSocketDisconnect:
        logger.info("TTS WebSocket disconnected")


@app.get("/health", summary="Check that Kokoro TTS is ready")
async def health_check():
    return {
        "status": "ready",
        "engine": "kokoro",
        "rate": _TTS_RATE,
        "volume": _TTS_VOLUME,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.text_to_speech:app", host="0.0.0.0", port=8001, log_level="debug")
