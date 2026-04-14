from __future__ import annotations

import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def prepare_hf_env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("LD_LIBRARY_PATH", "/opt/rocm/lib:/opt/rocm/lib64")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


@dataclass
class EventLogger:
    log_path: Path
    events: list[dict[str, Any]] = field(default_factory=list)

    def log(self, step: str, message: str, **extra: Any) -> None:
        event = {"ts": now_utc(), "step": step, "message": message, **extra}
        self.events.append(event)
        line = json.dumps(event, sort_keys=True)
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@contextmanager
def time_limit(seconds: int, step: str):
    if seconds <= 0:
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"Step '{step}' exceeded timeout of {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def timed_step(logger: EventLogger, step: str, timeout_sec: int):
    start = time.perf_counter()
    logger.log(step, "start")
    try:
        with time_limit(timeout_sec, step):
            yield
    except Exception as exc:
        logger.log(step, "error", runtime_sec=round(time.perf_counter() - start, 3), error=repr(exc))
        raise
    else:
        logger.log(step, "end", runtime_sec=round(time.perf_counter() - start, 3))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_env_dump(path: str | Path, model_id: str | None = None) -> None:
    prepare_hf_env()
    lines = []
    for key in [
        "LD_LIBRARY_PATH",
        "HIP_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "HF_HOME",
        "HF_HUB_DISABLE_XET",
        "TRANSFORMERS_CACHE",
    ]:
        lines.append(f"{key}={os.environ.get(key, '')}")

    lines.append(f"torch={torch.__version__}")
    lines.append(f"transformers={transformers.__version__}")
    lines.append(f"cuda_available={torch.cuda.is_available()}")
    lines.append(f"torch_hip={getattr(torch.version, 'hip', None)}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        lines.append(f"device_name={props.name}")
        lines.append(f"device_total_memory={props.total_memory}")
        lines.append(f"device_capability={getattr(props, 'gcnArchName', '')}")
        lines.append(f"memory_allocated={torch.cuda.memory_allocated(idx)}")
        lines.append(f"memory_reserved={torch.cuda.memory_reserved(idx)}")
    if model_id:
        try:
            snapshot = snapshot_download(repo_id=model_id, allow_patterns=["config.json"])
            lines.append(f"hf_snapshot={snapshot}")
        except Exception as exc:
            lines.append(f"hf_snapshot_error={exc!r}")

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reopen_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
