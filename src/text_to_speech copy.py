import base64
import io
import asyncio
import logging
import time
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3_tts_api")

app = FastAPI(
    title="Qwen3-TTS Streaming",
    description="Streams text through the Qwen3-TTS-12Hz-0.6B-Base voice-clone model and returns WAV audio.",
)

_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
_tts_kwargs = {"device_map": "auto", "dtype": _dtype}
if torch.cuda.is_available():
    _tts_kwargs["attn_implementation"] = "flash_attention_2"

logger.info("Loading TTS model %s", MODEL_ID)
tts_model = Qwen3TTSModel.from_pretrained(MODEL_ID, **_tts_kwargs)
logger.info("Model %s ready (dtype=%s)", MODEL_ID, _dtype)


def _decode_audio_bytes(contents: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(contents) as buffer:
        waveform, sr = sf.read(buffer, dtype="float32")
    return waveform, sr


def _write_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


async def _load_reference_audio(upload: UploadFile) -> Tuple[np.ndarray, int]:
    contents = await upload.read()
    await upload.close()
    return await asyncio.to_thread(_decode_audio_bytes, contents)


def _base64_to_audio(payload: str) -> bytes:
    if "," in payload and payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


async def _load_reference_audio_from_base64(payload: str) -> Tuple[np.ndarray, int]:
    data = _base64_to_audio(payload)
    return await asyncio.to_thread(_decode_audio_bytes, data)


def _encode_audio_to_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("ascii")


@app.post("/synthesize", summary="Convert text to streaming WAV via the base voice clone model")
async def synthesize(
    text: str = Form(..., description="The chunk of text to render; send sequential chunks for streaming."),
    language: str = Form("English", description="Language hint, e.g., English or Chinese."),
    ref_text: Optional[str] = Form(
        None, description="Transcription of the reference audio (required unless using x_vector_only_mode)."
    ),
    instruct: Optional[str] = Form(
        None, description="Optional stylistic instruction for the generated speech."
    ),
    x_vector_only_mode: bool = Form(
        False,
        description=(
            "If true, only the speaker embedding is reused and `ref_text` can be omitted. "
            "Still requires `reference_audio`."
        ),
    ),
    reference_audio: Optional[UploadFile] = File(
        None, description="Upload a short (<5s) WAV/MP3 reference clip to clone the voice."
    ),
    max_new_tokens: int = Form(1024, description="Maximum tokens ahead of the prompt."),
    temperature: float = Form(0.7, description="Sampling temperature for the speech LM."),
    top_p: float = Form(0.9, description="Top-p filtering."),
):
    trim_text = text.strip()
    if not trim_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    reference_input: Optional[Tuple[np.ndarray, int]] = None
    if reference_audio:
        reference_input = await _load_reference_audio(reference_audio)
        if not (ref_text or x_vector_only_mode):
            raise HTTPException(
                status_code=400,
                detail="Provide ref_text unless x_vector_only_mode is true when uploading reference_audio.",
            )

    call_kwargs = {
        "text": trim_text,
        "language": language,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "x_vector_only_mode": x_vector_only_mode,
    }
    if instruct:
        call_kwargs["instruct"] = instruct
    if reference_input:
        call_kwargs["ref_audio"] = reference_input
        if ref_text:
            call_kwargs["ref_text"] = ref_text

    start = time.perf_counter()
    wavs, sample_rate = await asyncio.to_thread(tts_model.generate_voice_clone, **call_kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    if not wavs:
        raise HTTPException(status_code=500, detail="TTS model returned no audio")

    output_audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    sample_count = int(output_audio.shape[0] if hasattr(output_audio, "shape") else len(output_audio))
    audio_bytes = await asyncio.to_thread(_write_wav_bytes, output_audio, sample_rate)

    headers = {
        "X-Sample-Rate": str(sample_rate),
        "X-Duration-Ms": f"{(sample_count / sample_rate) * 1000:.0f}",
        "X-Latency-Ms": f"{latency_ms:.0f}",
    }
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers=headers)


@app.get("/health", summary="Check that the TTS model is loaded")
async def health_check():
    return {
        "status": "ready",
        "model": MODEL_ID,
        "dtype": str(_dtype),
        "device_map": "auto",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.text_to_speech:app", host="0.0.0.0", port=8001, log_level="info")


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    logger.info("Accepted new TTS WebSocket connection from %s", websocket.client)
    reference_input: Optional[Tuple[np.ndarray, int]] = None
    ref_text: Optional[str] = None
    x_vector_only_mode = False
    language = "English"
    instruct: Optional[str] = None

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")
            logger.debug("TTS socket received message type=%s", msg_type)

            if msg_type == "reference_audio":
                payload = message.get("data")
                if not payload:
                    logger.warning("reference_audio message received with empty payload")
                    await websocket.send_json({"type": "error", "detail": "reference audio payload required"})
                    continue
                reference_input = await _load_reference_audio_from_base64(payload)
                await websocket.send_json({"type": "reference_ack"})
                logger.info(
                    "Stored reference audio (duration=%.1fs, sr=%s)",
                    reference_input[0].shape[0] / reference_input[1],
                    reference_input[1],
                )
            elif msg_type == "ref_text":
                ref_text = message.get("text")
                await websocket.send_json({"type": "ref_text_ack", "text": ref_text})
                logger.info("Set reference text length=%d", len(ref_text or ""))
            elif msg_type == "language":
                language = message.get("value", language)
                logger.info("Language overridden to %s", language)
            elif msg_type == "instruct":
                instruct = message.get("text")
                logger.info("Instruction updated length=%d", len(instruct or ""))
            elif msg_type == "x_vector_only_mode":
                x_vector_only_mode = bool(message.get("value", False))
                await websocket.send_json({"type": "x_vector_ack", "value": x_vector_only_mode})
                logger.info("x_vector_only_mode set to %s", x_vector_only_mode)
            elif msg_type == "text_chunk":
                text = (message.get("text") or "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "detail": "text chunk cannot be empty"})
                    continue

                call_kwargs = {
                    "language": message.get("language", language),
                    "max_new_tokens": message.get("max_new_tokens", 1024),
                }
                if instruct:
                    call_kwargs["instruct"] = instruct

                generation_type = "voice_clone" if reference_input else "voice_design"
                if generation_type == "voice_clone":
                    call_kwargs.update(
                        {
                            "text": text,
                            "temperature": message.get("temperature", 0.7),
                            "top_p": message.get("top_p", 0.9),
                            "x_vector_only_mode": message.get("x_vector_only_mode", x_vector_only_mode),
                            "ref_audio": reference_input,
                        }
                    )
                    if ref_text:
                        call_kwargs["ref_text"] = ref_text
                    logger.info(
                        "TTS voice_clone request text_len=%d language=%s temp=%.2f top_p=%.2f ref_text=%s",
                        len(text),
                        call_kwargs.get("language"),
                        call_kwargs.get("temperature"),
                        call_kwargs.get("top_p"),
                        bool(call_kwargs.get("ref_text")),
                    )
                    tts_fn = tts_model.generate_voice_clone
                else:
                    call_kwargs = {
                        "text": text,
                        "language": message.get("language", language),
                        "instruct": instruct or "Generate a clear, neutral streaming voice.",
                        "temperature": message.get("temperature", 0.7),
                        "top_p": message.get("top_p", 0.9),
                        "non_streaming_mode": False,
                    }
                    logger.info(
                        "TTS voice_design request text_len=%d language=%s temp=%.2f top_p=%.2f",
                        len(text),
                        call_kwargs.get("language"),
                        call_kwargs.get("temperature"),
                        call_kwargs.get("top_p"),
                    )
                    tts_fn = tts_model.generate_voice_design

                await websocket.send_json({"type": "generation_start", "text": text})
                start = time.perf_counter()
                wavs, sample_rate = await asyncio.to_thread(tts_fn, **call_kwargs)
                latency_ms = (time.perf_counter() - start) * 1000

                if not wavs:
                    logger.error("TTS model returned no audio for text_len=%d", len(text))
                    await websocket.send_json({"type": "error", "detail": "TTS model returned no audio"})
                    continue

                output_audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
                audio_bytes = await asyncio.to_thread(_write_wav_bytes, output_audio, sample_rate)
                encoded = _encode_audio_to_base64(audio_bytes)
                logger.info(
                    "Sending audio chunk (sample_rate=%d duration=%.1fms latency=%.1fms bytes=%d base64_len=%d)",
                    sample_rate,
                    len(output_audio) / sample_rate * 1000,
                    latency_ms,
                    len(audio_bytes),
                    len(encoded),
                )
                await websocket.send_json(
                    {
                        "type": "audio_chunk",
                        "audio": encoded,
                        "sample_rate": sample_rate,
                        "duration_ms": (len(output_audio) / sample_rate) * 1000,
                        "latency_ms": latency_ms,
                    }
                )
            elif msg_type == "reset":
                reference_input = None
                ref_text = None
                x_vector_only_mode = False
                instruct = None
                await websocket.send_json({"type": "reset_ack"})
            else:
                await websocket.send_json({"type": "error", "detail": f"unknown msg type: {msg_type}"})
    except WebSocketDisconnect:
        logger.info("TTS WebSocket disconnected")
