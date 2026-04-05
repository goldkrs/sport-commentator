import asyncio
import base64
import io
import logging
import threading
import time
from typing import Dict, List, Tuple

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
COMMENTARY_INSTRUCTION = (
    "You are providing short football play-by-play commentary. "
    "Stay focused on the action, keep it crisp, and do not describe the scene or frame itself."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3_vl_api")

app = FastAPI(
    title="Qwen3-VL Frame Describer",
    description="Serves Qwen3-VL-2B-Instruct over HTTP for text generation that consumes frames.",
)

logger.info("Loading model %s", MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
model.eval()
logger.info("Model %s ready", MODEL_ID)


def _generate_text_from_images(
    messages: List[Dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict:
    do_sample = temperature > 0.0 or top_p < 1.0
    with torch.inference_mode():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    trimmed_ids = [
        out_ids[len(in_ids) :].cpu()
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    total_tokens = sum(int(seq.shape[0]) for seq in trimmed_ids)
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return {
        "text": "\n".join(filter(None, (text.strip() for text in decoded))).strip(),
        "tokens": total_tokens,
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


def _prepare_generation_inputs(messages: List[Dict]) -> Dict[str, torch.Tensor]:
    encoding = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(model.device) for key, value in encoding.items()}


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
    
    print('Receieved Frames Count', len(frames))
    messages = _build_messages(prompt, frames)
    batch_inputs = _prepare_generation_inputs(messages)
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        decode_kwargs={"clean_up_tokenization_spaces": True},
    )

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
    }

    def run_generation() -> None:
        with torch.inference_mode():
            print('Inside Inference mode')
            #model.generate(**batch_inputs, **generation_kwargs)
            streamer.on_finalized_text("hello ")

    
    await websocket.send_json(
        {"type": "generation_start", "frame_count": len(frames), "prompt": prompt}
    )
    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    loop = asyncio.get_running_loop()
    chunks: List[str] = []
    try:
        while True:
            chunk = await loop.run_in_executor(None, next, streamer)
            if chunk:
                chunks.append(chunk)
                print(f"Streaming chunk (%d bytes)", chunk)
                await websocket.send_json({"type": "chunk", "text": chunk})
    except StopIteration:
        pass
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
    prompt = COMMENTARY_INSTRUCTION
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
            elif msg_type == "generate":
                if not frames:
                    await websocket.send_json({"type": "error", "detail": "send at least one frame first"})
                    continue
                logger.info(
                    "Triggering generation for %d frame(s) prompt=%s max_new_tokens=%s",
                    len(frames),
                    message.get("prompt", prompt),
                    min(20, message.get("max_new_tokens", 20)),
                )
                async with generation_lock:
                    await _stream_generation_to_websocket(
                        websocket,
                        frames,
                        message.get("prompt", prompt),
                        min(20, message.get("max_new_tokens", 20)),
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

    uvicorn.run("src.video_to_text:app", host="0.0.0.0", port=8000, log_level="info")
