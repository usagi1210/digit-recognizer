from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import ai_edge_litert

from model_assets import ensure_model_file


ROOT = Path(__file__).resolve().parent
APP = ROOT / "digit_gui.py"
MODEL = ROOT / "digit_cnn.tflite"
NAME = "DigitRecognizer"


def main() -> None:
    if not APP.exists():
        raise FileNotFoundError(APP)
    ensure_model_file(MODEL)

    separator = ";" if os.name == "nt" else ":"
    package_dir = Path(ai_edge_litert.__file__).resolve().parent
    cache_dir = ROOT / ".pyinstaller_cache"
    cache_dir.mkdir(exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name",
        NAME,
        "--add-data",
        f"{MODEL}{separator}.",
        "--hidden-import",
        "ai_edge_litert.interpreter",
        "--hidden-import",
        "ai_edge_litert.metrics_portable",
    ]
    parts_dir = ROOT / "model_parts"
    if parts_dir.exists():
        command.extend(["--add-data", f"{parts_dir}{separator}model_parts"])
    binary_exts = {".so", ".pyd", ".dll", ".dylib"}
    for binary in sorted(package_dir.iterdir()):
        if binary.suffix.lower() in binary_exts:
            command.extend(["--add-binary", f"{binary}{separator}ai_edge_litert"])
    command.append(str(APP))
    print(" ".join(command))
    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(cache_dir)
    subprocess.run(command, cwd=ROOT, env=env, check=True)

    suffix = ".exe" if os.name == "nt" else ""
    output = ROOT / "dist" / f"{NAME}{suffix}"
    if output.exists():
        size_mb = output.stat().st_size / 1024 / 1024
        print(f"输出文件：{output}")
        print(f"文件大小：{size_mb:.1f} MB")
    else:
        print(f"打包命令完成，请检查 dist 目录：{ROOT / 'dist'}")


if __name__ == "__main__":
    main()
