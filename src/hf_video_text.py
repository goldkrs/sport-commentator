import base64
import io
import json
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# IMPORTANT:
# 1) Ensure your token has "Inference Providers" permission.
# 2) Use provider-qualified model id for reliable routing.
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_API_KEY = os.getenv("HF_API_KEY", "YOUR_HF_API_KEY")
MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct:novita")
HF_CHUNK_SIZE = int(os.getenv("HF_CHUNK_SIZE", "30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hf_qwen3_vl_api")

COMPANY_STYLE_PROMPT = (
    "You are a professional football commentator covering the live action. "
    "Keep the narration concise, vivid, and suitable for speaking; never describe "
    "the scene as frames or metadata. Start sentences with energetic verbs, "
    "avoid repetition, and keep the output short enough that it could be voiced "
    "within the buffered segment (max 20 tokens). You will receive brief sections "
    "of frames in sequence, so treat any previously generated narration as context "
    "and continue building the story from that history. Make sure every response "
    "completely describes the current action before handing off to the next chunk "
    "so the transcript does not stop mid-thought."
)

app = FastAPI(
    title="HF Qwen3-VL WebSocket API",
    description="Video/frame to text using Hugging Face hosted Qwen/Qwen3-VL-8B-Instruct",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VARIATION_VERBS = ['passes', 'drives', 'shoots', 'threads', 'launches', 'sweeps']
SAMPLE_SETS = [
    [
        'Now threads a quick ball through midfield—low and laser-focused.',
        'Now bursts past the marker and drifts good weight into the box.',
        'Now lofts a delivery over the defense—waiting for the run to meet it.',
    ],
    [
        'Now steps up, nicks the ball, and shepherds play wide.',
        'Now unloads a sharp shot—danger, danger in the six-yard box.',
        'Now slams a driven cross into the mixer.',
    ],
    [
        'Now slices a pinpoint through pass into the channel.',
        'Now spins a curling effort toward the far post—keepers scrambling.',
        'Now sweeps the ball into space for the runner to latch onto.',
    ],
    [
        'Now pings a diagonal ball—and the striker is onside.',
        'Now thunders a half-volley that rattles the bar.',
        'Now nudges a cheeky flick behind the defender for the overlap.',
    ],
    [
        'Now drifts a teasing cross to the back stick for a tap-in chance.',
        'Now darts inside, shields three defenders, and lays it off cleanly.',
        'Now threads one more pass to keep the tempo electric.',
    ],
]
DEFAULT_COMMENTARY = "Short football commentary."
CLEAN_RE = re.compile(r"^This sequence of [^,]+,?\s*(captures|documents)", re.IGNORECASE)


# -----------------------------
# Utils
# -----------------------------
def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def decode_ws_frame(encoded: str) -> Image.Image:
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    payload = base64.b64decode(encoded)
    return Image.open(io.BytesIO(payload)).convert("RGB")


def build_messages(prompt: str, frames: List[Image.Image]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    # Keep same order as original: images first, then text
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame)},
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_payload(
    messages: List[Dict[str, Any]],
    stream: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    # Some providers ignore these; kept for contract parity
    payload["extra_body"] = {
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    return payload


def get_headers() -> Dict[str, str]:
    if not HF_API_KEY or HF_API_KEY == "YOUR_HF_API_KEY":
        logger.error("HF_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="HF_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }


def clean_transcript_text(text: str) -> str:
    if not text:
        return ""
    cleaned = CLEAN_RE.sub("", text).strip()
    cleaned = cleaned.lstrip(", ").strip()
    return cleaned


def get_sample_commentary(state: Dict[str, int]) -> str:
    if not SAMPLE_SETS:
        return DEFAULT_COMMENTARY
    sample_set_index = state.get("sample_set_index", 0) % len(SAMPLE_SETS)
    sample_lines = SAMPLE_SETS[sample_set_index] or []
    if not sample_lines:
        return DEFAULT_COMMENTARY
    line_index = state.get("sample_line_index", 0) % len(sample_lines)
    line = sample_lines[line_index]
    line_index += 1
    if line_index >= len(sample_lines):
        line_index = 0
        state["sample_set_index"] = (sample_set_index + 1) % len(SAMPLE_SETS)
    state["sample_line_index"] = line_index
    return line


def apply_variation(text: str, state: Dict[str, int]) -> str:
    trimmed = text.strip()
    if not trimmed:
        return ""
    variation_index = state.get("variation_index", 0)
    verb = VARIATION_VERBS[variation_index % len(VARIATION_VERBS)]
    state["variation_index"] = variation_index + 1
    if trimmed.lower().startswith(verb):
        return trimmed
    return f"Now {verb} {trimmed}"


def format_commentary(raw_text: str, state: Dict[str, int]) -> str:
    '''
    cleaned = clean_transcript_text(raw_text)
    if not cleaned or len(cleaned.split()) < 3:
        return get_sample_commentary(state)
    return apply_variation(cleaned, state)
    '''

    return raw_text


def chunk_frames(frames: List[Image.Image], size: int) -> List[List[Image.Image]]:
    if size <= 0:
        return [frames]
    return [frames[i : i + size] for i in range(0, len(frames), size)]


# -----------------------------
# HF Calls
# -----------------------------
async def call_hf_non_streaming(
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    payload = build_payload(
        messages=messages,
        stream=False,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    headers = get_headers()

    logger.info(
        "HF non-stream request | model=%s | frames=%d | max_new_tokens=%s | temp=%.2f | top_p=%.2f | top_k=%s | rep_pen=%.2f",
        MODEL_ID,
        max(0, len(messages[0].get("content", [])) - 1) if messages else 0,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    )

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(HF_API_URL, headers=headers, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "HF non-stream response | status=%s | duration_ms=%.2f | body_preview=%s",
            response.status_code,
            elapsed_ms,
            (response.text or "")[:1000],
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = (response.text or "")[:2000]
            logger.exception("HF non-stream HTTP error")
            raise HTTPException(status_code=502, detail=f"Hugging Face error: {detail}") from exc

        return response.json()


async def call_hf_streaming(
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> AsyncIterator[str]:
    payload = build_payload(
        messages=messages,
        stream=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    headers = get_headers()

    logger.info(
        "HF stream request | model=%s | frames=%d | max_new_tokens=%s | temp=%.2f | top_p=%.2f | top_k=%s | rep_pen=%.2f",
        MODEL_ID,
        max(0, len(messages[0].get("content", [])) - 1) if messages else 0,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    )

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", HF_API_URL, headers=headers, json=payload) as response:
            logger.info("HF stream headers received | status=%s", response.status_code)

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = await response.aread()
                logger.exception("HF stream HTTP error")
                raise HTTPException(
                    status_code=502,
                    detail=f"Hugging Face error: {detail[:2000].decode(errors='ignore')}",
                ) from exc

            async for line in response.aiter_lines():
                if not line:
                    continue

                logger.info("HF raw SSE line: %s", line[:500])

                if not line.startswith("data:"):
                    continue

                yield line

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("HF stream completed | duration_ms=%.2f", elapsed_ms)


# -----------------------------
# Load image
# -----------------------------
async def load_image(upload: UploadFile) -> Image.Image:
    try:
        contents = await upload.read()
        logger.debug("Load frame | filename=%s | bytes=%d", upload.filename, len(contents))
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        logger.exception("Invalid image upload")
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode {upload.filename or 'frame'}: {exc}",
        ) from exc
    finally:
        await upload.close()


# -----------------------------
# REST Endpoint
# -----------------------------
@app.post("/generate")
async def generate_from_frames(
    prompt: str = Form(COMPANY_STYLE_PROMPT),
    frames: List[UploadFile] = File(..., description="Frame images in chronological order"),
    max_new_tokens: int = Form(30, description="Tokens to produce"),
    temperature: float = Form(0.7, description="Sampling temperature"),
    top_p: float = Form(0.9, description="Top-p sampling"),
    top_k: int = Form(40, description="Top-k sampling"),
    repetition_penalty: float = Form(1.2, description="Penalize repeated tokens"),
):
    if not frames:
        logger.warning("/generate called without frames")
        raise HTTPException(status_code=400, detail="At least one frame is required")

    frame_names = [upload.filename or f"frame_{idx}" for idx, upload in enumerate(frames, 1)]
    logger.info(
        "/generate called | frame_count=%d | prompt_len=%d | frame_names=%s",
        len(frames),
        len(prompt or ""),
        frame_names,
    )

    images = [await load_image(f) for f in frames]
    image_chunks = chunk_frames(images, HF_CHUNK_SIZE)
    if not image_chunks:
        logger.error("No visible chunks could be built")
        raise HTTPException(status_code=400, detail="Unable to build frame chunks")

    history = []
    total_tokens = 0
    total_duration_ms = 0.0
    start = time.perf_counter()
    final_chunk_text = ""

    for idx, chunk in enumerate(image_chunks):
        running_context = " ".join(history[-2:])
        chunk_prompt = (
            prompt
            if idx == 0
            else f"{prompt} Previous narration: {running_context}. Continue with the next batch."
        )
        print('chunk size', len(chunk))
        chunk_messages = build_messages(chunk_prompt, chunk)
        chunk_start = time.perf_counter()
        chunk_response = await call_hf_non_streaming(
            chunk_messages,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        )
        chunk_duration = (time.perf_counter() - chunk_start) * 1000
        total_duration_ms += chunk_duration

        try:
            chunk_text = chunk_response["choices"][0]["message"]["content"]
            usage = chunk_response.get("usage", {})
        except Exception:
            logger.exception("Bad HF response shape for chunk %d", idx + 1)
            raise HTTPException(
                status_code=502,
                detail=f"Unexpected HF response for chunk {idx + 1}: {chunk_response}",
            )

        if not chunk_text:
            logger.warning("Chunk %d returned empty text, skipping", idx + 1)
            continue

        history.append(chunk_text.strip())
        final_chunk_text = chunk_text.strip()
        total_tokens += usage.get("completion_tokens", 0) or 0
        logger.info(
            "/generate chunk %d/%d completed | duration_ms=%.2f | output_len=%d | completion_tokens=%s",
            idx + 1,
            len(image_chunks),
            chunk_duration,
            len(chunk_text),
            usage.get("completion_tokens"),
        )

    duration_ms = (time.perf_counter() - start) * 1000
    if not segments:
        logger.error("All HF chunk responses were empty")
        raise HTTPException(status_code=500, detail="Model returned no text for any chunk")

    combined_text = "\n".join(segments)
    logger.info(
        "/generate completed | total_duration_ms=%.2f | aggregated_len=%d | total_completion_tokens=%s",
        duration_ms,
        len(combined_text),
        total_tokens,
    )

    return {
        "prompt": prompt,
        "description": combined_text,
        "frame_names": frame_names,
        "tokens_generated": total_tokens,
        "duration_ms": total_duration_ms,
        "generation_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
    }


@app.post("/generate_paragraph")
async def generate_paragraph_from_frames(
    prompt: str = Form(COMPANY_STYLE_PROMPT),
    frames: List[UploadFile] = File(..., description="Frame images in chronological order"),
    max_new_tokens: int = Form(30, description="Tokens to produce"),
    temperature: float = Form(0.7, description="Sampling temperature"),
    top_p: float = Form(0.9, description="Top-p sampling"),
    top_k: int = Form(40, description="Top-k sampling"),
    repetition_penalty: float = Form(1.2, description="Penalize repeated tokens"),
):
    if not frames:
        logger.warning("/generate_paragraph called without frames")
        raise HTTPException(status_code=400, detail="At least one frame is required")

    frame_names = [upload.filename or f"frame_{idx}" for idx, upload in enumerate(frames, 1)]
    logger.info(
        "/generate_paragraph called | frame_count=%d | prompt_len=%d | frame_names=%s",
        len(frames),
        len(prompt or ""),
        frame_names,
    )

    images = [await load_image(f) for f in frames]
    image_chunks = chunk_frames(images, HF_CHUNK_SIZE)
    if not image_chunks:
        logger.error("No chunks generated for /generate_paragraph")
        raise HTTPException(status_code=400, detail="Unable to create frame chunks")
    client_state: Dict[str, int] = {
        "variation_index": 0,
        "sample_set_index": 0,
        "sample_line_index": 0,
    }

    history: List[str] = []
    total_tokens = 0
    total_duration_ms = 0.0
    start = time.perf_counter()
    final_chunk_text = ""

    for idx, chunk in enumerate(image_chunks):
        print('chunk size', len(chunk))
        running_context = " ".join(history[-2:])
        chunk_prompt = (
            prompt
            if idx == 0
            else f"{prompt} Previous narration: {running_context}. Continue with the next batch."
        )
        chunk_messages = build_messages(chunk_prompt, chunk)
        chunk_start = time.perf_counter()
        chunk_response = await call_hf_non_streaming(
            chunk_messages,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        )
        chunk_duration = (time.perf_counter() - chunk_start) * 1000
        total_duration_ms += chunk_duration

        try:
            chunk_text = chunk_response["choices"][0]["message"]["content"]
            usage = chunk_response.get("usage", {})
        except Exception:
            logger.exception("Bad HF response shape for /generate_paragraph chunk %d", idx + 1)
            raise HTTPException(
                status_code=502,
                detail=f"Unexpected HF response for chunk {idx + 1}: {chunk_response}",
            )

        if not chunk_text:
            logger.warning("Chunk %d returned empty text during /generate_paragraph", idx + 1)
            continue

        history.append(chunk_text.strip())
        final_chunk_text = chunk_text.strip()
        total_tokens += usage.get("completion_tokens", 0) or 0
        logger.info(
            "/generate_paragraph chunk %d/%d completed | duration_ms=%.2f | len=%d | tokens=%s",
            idx + 1,
            len(image_chunks),
            chunk_duration,
            len(chunk_text),
            usage.get("completion_tokens"),
        )

    duration_ms = (time.perf_counter() - start) * 1000
    if not final_chunk_text:
        logger.error("All /generate_paragraph chunks returned empty text")
        raise HTTPException(status_code=500, detail="Model returned no text in any chunk")

    formatted = format_commentary(final_chunk_text, client_state)

    logger.info(
        "/generate_paragraph completed | total_duration_ms=%.2f | aggregated_len=%d | tokens=%s",
        duration_ms,
        len(formatted),
        total_tokens,
    )

    return {
        "prompt": prompt,
        "text": formatted,
        "frame_names": frame_names,
        "tokens_generated": total_tokens,
        "duration_ms": total_duration_ms,
        "generation_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
    }


# -----------------------------
# WebSocket Endpoint
# -----------------------------
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    frames: List[Image.Image] = []
    prompt = COMPANY_STYLE_PROMPT
    client_state: Dict[str, int] = {
        "variation_index": 0,
        "sample_set_index": 0,
        "sample_line_index": 0,
    }

    logger.info("WebSocket connected")

    try:
        while True:
            logger.info("Waiting for next WS JSON message...")
            msg = await websocket.receive_json()
            logger.info("Raw WS JSON received: %s", msg)

            t = msg.get("type")
            logger.info("WS msg | type=%s", t)

            if t == "frame":
                payload = msg.get("data")
                if not payload:
                    logger.warning("Frame message received without data")
                    await websocket.send_json({"type": "error", "detail": "frame payload required"})
                    continue

                try:
                    img = decode_ws_frame(payload)
                except Exception as exc:
                    logger.exception("Failed to decode websocket frame")
                    await websocket.send_json({"type": "error", "detail": f"invalid frame payload: {exc}"})
                    continue

                frames.append(img)
                logger.info("Frame stored | count=%d", len(frames))
                await websocket.send_json({"type": "ack", "index": len(frames)})

            elif t == "prompt":
                prompt = msg.get("text", prompt)
                logger.info("Prompt updated | len=%d", len(prompt or ""))
                await websocket.send_json({"type": "prompt_ack", "prompt": prompt})

            elif t == "generate":
                logger.info("Generate received | frame_count=%d", len(frames))
                if not frames:
                    await websocket.send_json({"type": "error", "detail": "send at least one frame first"})
                    continue

                run_prompt = msg.get("prompt", prompt)
                messages = build_messages(run_prompt, frames)

                await websocket.send_json(
                    {"type": "generation_start", "frame_count": len(frames), "prompt": run_prompt}
                )

                chunks: List[str] = []
                chunk_count = 0

                async for raw in call_hf_streaming(
                    messages,
                    msg.get("max_new_tokens", 30),
                    msg.get("temperature", 0.7),
                    msg.get("top_p", 0.9),
                    msg.get("top_k", 40),
                    msg.get("repetition_penalty", 1.2),
                ):
                    logger.info("HF raw event: %s", raw[:300])

                    if raw.strip() == "data: [DONE]":
                        logger.info("HF DONE received")
                        break

                    try:
                        payload = json.loads(raw.replace("data:", "", 1).strip())
                    except json.JSONDecodeError as exc:
                        logger.exception("Failed parsing HF chunk", exc_info=exc)
                        continue

                    choices = payload.get("choices") or []
                    if not choices:
                        logger.warning("HF chunk has no choices: %s", payload)
                        continue

                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")

                    if text:
                        chunk_count += 1
                        chunks.append(text)
                        await websocket.send_json({"type": "chunk", "text": text})

                if chunk_count == 0:
                    logger.warning("NO CHUNKS RECEIVED FROM HF")

                final = "".join(chunks).strip()
                formatted_text = format_commentary(final, client_state)
                logger.info(
                    "Final generated | raw_chars=%d | formatted_chars=%d | chunks=%d",
                    len(final),
                    len(formatted_text),
                    chunk_count,
                )

                await websocket.send_json(
                    {
                        "type": "final",
                        "text": formatted_text,
                        "frame_count": len(frames),
                    }
                )

            elif t == "reset":
                frames.clear()
                logger.info("Frames reset")
                await websocket.send_json({"type": "reset_ack"})

            else:
                logger.warning("Unknown WS type: %s", t)
                await websocket.send_json({"type": "error", "detail": f"unknown msg type: {t}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("Unhandled WebSocket error")
        raise


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
async def health():
    logger.debug("/health called")
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.hf_video_text:app", host="0.0.0.0", port=8000, log_level="info")
