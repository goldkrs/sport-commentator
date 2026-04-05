from pathlib import Path
from huggingface_hub import snapshot_download

cache_dir = Path.home() / ".cache" / "huggingface"
snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    cache_dir=str(cache_dir),
    allow_patterns="*",
    max_workers=10,
    repo_type="model",
)
