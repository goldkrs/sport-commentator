import os
from pathlib import Path
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_SYMLINKS", "1")
from huggingface_hub import snapshot_download

cache_dir = Path.home() / ".cache" / "huggingface"
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    cache_dir=str(cache_dir),
    allow_patterns="*",
    max_workers=10,
    repo_type="model",
)
