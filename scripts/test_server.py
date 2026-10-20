#!/usr/bin/env python3
"""Run Qwen2.5 3B Instruct behind an OpenAI-compatible llama.cpp server.

Install the runtime once with:
    python3 -m pip install "llama-cpp-python[server]" huggingface-hub

Then start this server with:
    python3 scripts/test_server.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# This GGUF quantization is maintained by bartowski.
MODEL_REPO = "bartowski/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
GGML_TYPE_IDS = {"f16": "1", "q4_0": "2", "q8_0": "8"}


def model_path() -> Path:
    configured = os.environ.get("LOCAL_LLM_MODEL_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_DIR / MODEL_FILE


def ensure_model(path: Path) -> None:
    if path.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(
            "Missing dependencies. Install with: "
            'python3 -m pip install "llama-cpp-python[server]" huggingface-hub'
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=path.parent,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != path:
        downloaded_path.replace(path)


def main() -> int:
    path = model_path()
    ensure_model(path)
    host = os.environ.get("LOCAL_LLM_HOST", "127.0.0.1")
    port = os.environ.get("LOCAL_LLM_PORT", "8000")
    # TODO: Revisit these once we get on the dedicated server.
    # Qwen2.5 3B has only 2 KV heads, so its context is cheap: measured on an 8 GB M2, the full
    # native 32k context costs about 612 MiB of KV cache at q8_0 (1152 MiB at f16), plus ~90 MiB of
    # compute buffers, on top of the ~1.8 GiB of weights.
    context = os.environ.get("LOCAL_LLM_CONTEXT", "32768")
    # q8_0 halves the KV cache against f16 with negligible quality loss; q4_0 halves it again
    # but degrades long-context recall noticeably. Quantized V cache requires flash attention.
    kv_type = os.environ.get("LOCAL_LLM_KV_TYPE", "q8_0")
    threads = os.environ.get("LOCAL_LLM_THREADS", "4")

    llama_server = shutil.which("llama-server")
    if not llama_server and kv_type not in GGML_TYPE_IDS:
        raise SystemExit(f"LOCAL_LLM_KV_TYPE must be one of {', '.join(GGML_TYPE_IDS)}.")
    if llama_server:
        command = [
            llama_server,
            "--model",
            str(path),
            "--host",
            host,
            "--port",
            port,
            "--ctx-size",
            context,
            "--threads",
            threads,
            # One slot gets the whole context, and keeps the manuscript prefix cached between
            # cards: the app sends the same manuscript first in every request of a question set,
            # so only the first card pays to process it.
            "--parallel",
            "1",
            "--flash-attn",
            "on",
            "--cache-type-k",
            kv_type,
            "--cache-type-v",
            kv_type,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            str(path),
            "--host",
            host,
            "--port",
            port,
            "--n_ctx",
            context,
            "--n_threads",
            threads,
            "--chat_format",
            "qwen",
            "--flash_attn",
            "true",
            # llama-cpp-python takes ggml type IDs rather than names.
            "--type_k",
            GGML_TYPE_IDS[kv_type],
            "--type_v",
            GGML_TYPE_IDS[kv_type],
        ]
    print(f"Serving {path} at http://{host}:{port}/v1", flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
