from __future__ import annotations

import base64
import sys
from pathlib import Path


MODEL_NAME = "digit_cnn.tflite"
PARTS_DIR = "model_parts"


def resource_path(relative_path: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative_path


def ensure_model_file(target_path: Path | None = None) -> Path:
    target = target_path or resource_path(MODEL_NAME)
    if target.exists():
        return target

    parts_dir = resource_path(PARTS_DIR)
    parts = sorted(parts_dir.glob("*.b64.*"))
    if not parts:
        raise FileNotFoundError(f"找不到模型文件或模型分片：{target}")

    encoded = "".join(part.read_text(encoding="ascii").strip() for part in parts)
    target.write_bytes(base64.b64decode(encoded))
    return target
