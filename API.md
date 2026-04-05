# Frame description API

This service loads `Qwen/Qwen3-VL-2B-Instruct` via Hugging Face Transformers and exposes a single `/generate` endpoint that takes one or more frame images plus an optional prompt. The model is eagerly cached, uses `device_map="auto"` to span available devices, and produces text via the standard multi-image chat template described on the official model card. citeturn0search5

## Local setup

1. `python -m pip install -r requirements.txt`
   * `requirements.txt` now includes `torchvision`, which `Qwen3VLVideoProcessor` depends on for image preprocessing, so install it with the same CUDA/CPU build as `torch`.
2. (Optional but recommended) log into the Hub with `huggingface-cli login` if you need access to gated checkpoints.
3. Ensure the Hugging Face cache directory has enough disk space for the ~19 GB Qwen3-VL weights.

## Running

```bash
uvicorn src.video_to_text:app --host 0.0.0.0 --port 8000 --log-level info
```

Add `--reload` if you want uvicorn to watch file changes.

## Calling `/generate`

POST `multipart/form-data` with at least one `frames` file and an optional `prompt`. Parameters such as `max_new_tokens`, `temperature`, `top_p`, `top_k`, and `repetition_penalty` are exposed as form fields and default to conservative values (512 tokens, 0.7 temperature, 0.9 top-p, 40 top-k, 1.2 repetition penalty).

Example `curl`:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "prompt=Summarize these frames." \
  -F "frames=@frame1.png" \
  -F "frames=@frame2.png" \
  -F "max_new_tokens=256"
```

The response includes the generated description, a list of frame names, tokens produced, latency in milliseconds, and the generation parameters treasured by the client.

### WebSocket streaming (`/ws/stream`)

If you need to send frames incrementally and receive streaming text updates, connect to `ws://127.0.0.1:8000/ws/stream` (or the public host/port). The WebSocket accepts JSON messages and replies with chunked text as tokens stream out:

- `{"type": "frame", "data": "<base64 PNG or JPEG>", "frame_id": "optional id"}` — pushes another frame inside the current batch; the server responds with `{"type": "ack", "index": ...}`.
- `{"type": "prompt", "text": "..."} `— updates the prompt applied to the next generation and returns `{"type": "prompt_ack", "prompt": "..."}`.
- `{"type": "generate", "prompt": "optional override", "max_new_tokens": 512, "temperature": 0.7, ...}` — runs `Qwen3-VL` against the frames collected so far and streams tokens back via `{"type": "chunk", "text": "..."}` messages; the server wraps up with `{"type": "final", "text": "...", "frame_count": N}` once the generation completes.
- `{"type": "reset"}` — clears any buffered frames so you can start a new clip.

Every token chunk is emitted as soon as it is produced (using `TextIteratorStreamer`), so you can display partial transcripts while you keep sending more frames. Send `generate` again if you append more frames later; the service uses the same cached model instance as the REST endpoint.

## Health

`GET /health` confirms that the frame-describer model is loaded and ready.

## Text-to-Speech API

The text-to-speech service exposes `Qwen/Qwen3-TTS-12Hz-0.6B-Base` through the `qwen-tts` package, which handles the tokenizer and Base weights, and supports the low-latency voice-clone flow that outputs the first audio packet immediately. citeturn3view0

### Local setup

1. `python -m pip install -r requirements.txt`
   * `requirements.txt` now also installs `qwen-tts`, `soundfile`, and `numpy`, so keep the same CUDA/CPU build for `torch`/`torchvision` as mentioned earlier.
2. The Base model clones a voice from a short reference clip (≈3 seconds), so record and upload a clean sample and keep its transcript alongside each chunk you send.

### Running

```bash
uvicorn src.text_to_speech:app --host 0.0.0.0 --port 8001 --log-level info
```

### `/synthesize`

Send `multipart/form-data` per chunk of text you want to render. Fields:

- `text` (required): the narrative or dialogue to speak. Trim whitespace before sending.
- `language`: hint such as `English` or `Chinese`.
- `reference_audio`: upload a short WAV/MP3 clip; the Base model uses it to derive the speaker embedding.
- `ref_text`: transcript of the reference clip (omit only when `x_vector_only_mode` is true).
- `x_vector_only_mode`: boolean that lets you reuse only the speaker embedding so you can skip the transcript.
- `instruct`: optional style/instruction string forwarded to `generate_voice_clone`.
- `max_new_tokens`, `temperature`, `top_p`: sampling controls passed through to the generator.

Every response is sent as `audio/wav` with headers `X-Sample-Rate`, `X-Duration-Ms`, and `X-Latency-Ms`. Play the stream or save the file; chain sequential POSTs (with the same reference clip) to simulate a continuous text stream.

Example `curl`:

```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "text=Hello, this is chunk one." \
  -F "language=English" \
  -F "ref_text=Hello, this is a reference sentence." \
  -F "reference_audio=@ref.wav"
```

### Health

`GET /health` returns `status`, `model`, `dtype`, and `device_map`, so you can check that the voice clone service is initialized.

### WebSocket TTS (`/ws/tts`)

Connect to `ws://127.0.0.1:8001/ws/tts` (or your host/port) to stream chunks of text and receive base64-encoded WAV audio per chunk. The WebSocket understands the following JSON messages:

- `{"type": "reference_audio", "data": "data:audio/wav;base64,..."}` — upload the reference clip once; reply with `{"type": "reference_ack"}`.
- `{"type": "ref_text", "text": "Reference transcription"}` — send the transcription; reply `{"type": "ref_text_ack", "text": "..."}`.
- `{"type": "x_vector_only_mode", "value": true}` — toggle to reuse only the speaker embedding and skip the transcript.
- `{"type": "instruct", "text": "friendly tone"}` or `{"type": "language", "value": "English"}` — set generation hints for future chunks.
- `{"type": "text_chunk", "text": "...", "max_new_tokens": 512, "temperature": 0.7}` — renders the chunk; you first receive `{"type": "generation_start"}` followed by `{"type": "audio_chunk", "audio": "<base64>", "sample_rate": 48000, "duration_ms": 180, "latency_ms": 200}` once the waveform is ready.
- `{"type": "reset"}` — clears cached references and responds with `{"type": "reset_ack"}`.

Use successive `text_chunk` messages to stream multiple phrases while reusing the same cloned voice (just upload the reference clip once and keep the same `ref_text`). Decode the base64 `audio_chunk` payload to play or save each snippet as needed.

## Frontend orchestrator

A Node-based UI under `frontend/` hosts a single page that:

- uploads the silent video, plays it below the buttons, and captures frames every 500 ms;
- streams those frames to `ws://localhost:8000/ws/stream`, listens for `chunk` text events, and pipes each chunk into `ws://localhost:8001/ws/tts`;
- decodes the returned base64 WAV payloads and plays them through the Web Audio API while the video keeps running.

### Local setup

1. Install Node dependencies (`cd frontend && npm install`).
2. Start the static server (`npm start`). It listens on port 3000 by default.
3. Visit `http://localhost:3000`, upload your silent video and reference clip, then hit **Start streaming**. The UI handles resetting, stopping, and showing live transcripts/status logs.

The orchestrator keeps both WebSockets open, automatically schedules `generate` messages every ~2 s, and reuses the same voice reference clip until you hit **Reset session**.
