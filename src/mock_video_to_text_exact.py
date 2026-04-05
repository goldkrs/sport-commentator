import asyncio
import base64
import io
import logging
import random
import threading
import time
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3_vl_api")

app = FastAPI(
    title="Qwen3-VL Frame Describer",
    description="Serves Qwen3-VL-2B-Instruct over HTTP for text generation that consumes frames.",
)

logger.info("Mock mode enabled for model %s", MODEL_ID)
logger.info("Skipping model and processor loading")


class MockStreamer:
    def __init__(self) -> None:
        self._queue: List[str] = []
        self._finished = False
        self._condition = threading.Condition()

    def push(self, text: str) -> None:
        with self._condition:
            self._queue.append(text)
            self._condition.notify_all()

    def end(self) -> None:
        with self._condition:
            self._finished = True
            self._condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        with self._condition:
            while not self._queue and not self._finished:
                self._condition.wait()
            if self._queue:
                return self._queue.pop(0)
            raise StopIteration


def _mock_text(prompt: str, frame_count: int) -> str:
    openings = [
        "The frames appear to show",
        "This sequence seems to contain",
        "I can observe",
        "The uploaded frames suggest",
        "These visuals look like",
    ]
    subjects = [
        "a person moving through the scene",
        "objects changing position across frames",
        "a short action sequence with visible motion",
        "a scene with gradual visual transitions",
        "several elements shifting over time",
    ]
    details = [
        "lighting remains fairly consistent",
        "there are small frame-to-frame changes",
        "the overall composition stays stable",
        "motion appears limited but noticeable",
        "the scene contains visible temporal progression",
    ]
    closings = [
        "This is a mock response and not real model inference.",
        "This output is randomly generated for testing.",
        "No Qwen3 model was used to create this description.",
        "This description is intentionally synthetic for fast API testing.",
        "This mock keeps the API contract but skips actual inference.",
    ]

    return (
        f"{random.choice(openings)} {random.choice(subjects)}. "
        f"Frame count received: {frame_count}. "
        f"Prompt considered: {prompt!r}. "
        f"{random.choice(details)}. "
        f"{random.choice(closings)}"
    )


def _generate_text_from_images(
    messages: List[Dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict:
    prompt = "Describe the following frames in detail."
    frame_count = 0

    if messages:
        content = messages[0].get("content", [])
        frame_count = sum(1 for block in content if block.get("type") == "image")
        for block in content:
            if block.get("type") == "text":
                prompt = block.get("text", prompt)
                break

    text = _mock_text(prompt, frame_count)
    tokens = max(1, len(text.split()))
    return {
        "text": text,
        "tokens": tokens,
    }


def _decode_ws_frame(encoded: str) -> Image.Image:
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    payload = base64.b64decode(encoded)
    return Image.open(io.BytesIO(payload)).convert("RGB")


def _build_messages(prompt: str, frames: List[Image.Image]) -> List[Dict]:
    blocks = [{"type": "image", "image": frame} for frame in frames]
    return [
        {
            "role": "user",
            "content": [*blocks, {"type": "text", "text": prompt}],
        }
    ]


def _prepare_generation_inputs(messages: List[Dict]) -> Dict[str, int]:
    frame_count = 0
    prompt = "Describe the following frames in detail."

    if messages:
        content = messages[0].get("content", [])
        frame_count = sum(1 for block in content if block.get("type") == "image")
        for block in content:
            if block.get("type") == "text":
                prompt = block.get("text", prompt)
                break

    return {
        "frame_count": frame_count,
        "prompt": prompt,
    }


async def _stream_generation_to_websocket(
    websocket: WebSocket,
    frames: List[Image.Image],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> None:
    print("Receieved Frames Count", len(frames))
    messages = _build_messages(prompt, frames)
    batch_inputs = _prepare_generation_inputs(messages)
    streamer = MockStreamer()

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
    }

    def run_generation() -> None:
        _ = generation_kwargs
        text = _mock_text(batch_inputs["prompt"], batch_inputs["frame_count"])
        pieces = text.split(" ")
        for piece in pieces:
            chunk = piece + " "
            time.sleep(0.02)
            streamer.push(chunk)
        streamer.end()

    await websocket.send_json(
        {"type": "generation_start", "frame_count": len(frames), "prompt": prompt}
    )
    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    def _next_stream_chunk() -> Dict[str, str]:
        try:
            return {"done": "false", "chunk": next(streamer)}
        except StopIteration:
            return {"done": "true", "chunk": ""}

    loop = asyncio.get_running_loop()
    chunks: List[str] = []
    try:
        while True:
            result = await loop.run_in_executor(None, _next_stream_chunk)
            if result["done"] == "true":
                break
            chunk = result["chunk"]
            if chunk:
                chunks.append(chunk)
                print(f"Streaming chunk ({len(chunk)} bytes) {chunk}")
                await websocket.send_json({"type": "chunk", "text": chunk})
    finally:
        thread.join()

    final_text = "".join(chunks).strip()
    preview = final_text[:150].replace("\n", " ")
    logger.info("Streamed VL final text (%d chunks): %s", len(chunks), preview or "<empty>")
    await websocket.send_json({"type": "final", "text": final_text, "frame_count": len(frames)})


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    frames: List[Image.Image] = []
    prompt = "Describe the following frames in detail."
    generation_lock = asyncio.Lock()
    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")
            logger.debug("WebSocket received msg type=%s", msg_type)
            if msg_type == "frame":
                payload = message.get("data")
                if not payload:
                    await websocket.send_json({"type": "error", "detail": "frame payload required"})
                    continue
                image = await asyncio.to_thread(_decode_ws_frame, payload)
                frames.append(image)
                await websocket.send_json({"type": "ack", "index": len(frames)})
                logger.debug("Stored frame %d (payload size=%d bytes)", len(frames), len(payload))
            elif msg_type == "prompt":
                prompt = message.get("text", prompt)
                await websocket.send_json({"type": "prompt_ack", "prompt": prompt})
            elif msg_type == "generate":
                if not frames:
                    await websocket.send_json({"type": "error", "detail": "send at least one frame first"})
                    continue
                logger.info(
                    "Triggering generation for %d frame(s) prompt=%s max_new_tokens=%s",
                    len(frames),
                    message.get("prompt", prompt),
                    message.get("max_new_tokens", 512),
                )
                async with generation_lock:
                    await _stream_generation_to_websocket(
                        websocket,
                        frames,
                        message.get("prompt", prompt),
                        message.get("max_new_tokens", 512),
                        message.get("temperature", 0.7),
                        message.get("top_p", 0.9),
                        message.get("top_k", 40),
                        message.get("repetition_penalty", 1.2),
                    )
            elif msg_type == "reset":
                frames.clear()
                await websocket.send_json({"type": "reset_ack"})
            else:
                await websocket.send_json({"type": "error", "detail": f"unknown msg type: {msg_type}"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


async def _load_frame_image(upload: UploadFile) -> Image.Image:
    try:
        contents = await upload.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode {upload.filename or 'frame'}: {exc}",
        )
    finally:
        await upload.close()


@app.post("/generate")
async def generate_from_frames(
    prompt: str = Form("Describe the following frames in detail."),
    frames: List[UploadFile] = File(..., description="Frame images in chronological order"),
    max_new_tokens: int = Form(512, description="Tokens to produce"),
    temperature: float = Form(0.7, description="Sampling temperature"),
    top_p: float = Form(0.9, description="Top-p sampling"),
    top_k: int = Form(40, description="Top-k sampling"),
    repetition_penalty: float = Form(1.2, description="Penalize repeated tokens"),
):
    if not frames:
        raise HTTPException(status_code=400, detail="At least one frame is required")

    frame_names = [upload.filename or f"frame_{idx}" for idx, upload in enumerate(frames, 1)]
    image_blocks = [
        {"type": "image", "image": await _load_frame_image(upload)}
        for upload in frames
    ]

    messages = [
        {
            "role": "user",
            "content": [*image_blocks, {"type": "text", "text": prompt}],
        }
    ]

    start = time.perf_counter()
    generation = await asyncio.to_thread(
        _generate_text_from_images,
        messages,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    )
    duration_ms = (time.perf_counter() - start) * 1000

    if not generation["text"]:
        raise HTTPException(status_code=500, detail="Model returned an empty response")

    return {
        "prompt": prompt,
        "description": generation["text"],
        "frame_names": frame_names,
        "tokens_generated": generation["tokens"],
        "duration_ms": duration_ms,
        "generation_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "ready", "model": MODEL_ID}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
