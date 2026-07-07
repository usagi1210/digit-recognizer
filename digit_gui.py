from __future__ import annotations

import csv
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover - useful when running outside the bundled app env.
    from tflite_runtime.interpreter import Interpreter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODEL_FILE = "digit_cnn.tflite"


def resource_path(relative_path: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative_path


def preprocess_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = 1.0 - pixels
    return pixels[None, ..., None].astype(np.float32)


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def label_from_path(path: Path, root: Path) -> int | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) >= 2 and relative.parts[0].isdigit():
        value = int(relative.parts[0])
        if 0 <= value <= 9:
            return value
    stem = path.stem
    if stem and stem[0].isdigit():
        value = int(stem[0])
        if 0 <= value <= 9:
            return value
    return None


def run_self_test(folder: Path) -> int:
    classifier = DigitClassifier(resource_path(MODEL_FILE))
    images = list(iter_images(folder))
    if not images:
        print(f"no images found: {folder}")
        return 2

    total = len(images)
    labeled = 0
    correct = 0
    for path in images:
        digit, confidence = classifier.predict(path)
        truth = label_from_path(path, folder)
        if truth is None:
            print(f"{path}\tpred={digit}\tconfidence={confidence:.6f}")
        else:
            labeled += 1
            correct += int(digit == truth)
    if labeled:
        print(f"images={total} labeled={labeled} correct={correct} accuracy={correct / labeled:.6f}")
    else:
        print(f"images={total} labeled=0")
    return 0


@dataclass
class Prediction:
    path: Path
    digit: int
    confidence: float
    truth: int | None = None

    @property
    def correct(self) -> bool | None:
        if self.truth is None:
            return None
        return self.truth == self.digit


class DigitClassifier:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件：{model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

    def predict(self, path: Path) -> tuple[int, float]:
        sample = preprocess_image(path)
        self.interpreter.set_tensor(self.input_detail["index"], sample)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_detail["index"])[0]
        digit = int(np.argmax(scores))
        confidence = float(scores[digit])
        return digit, confidence


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("手写数字识别")
        self.geometry("820x560")
        self.minsize(720, 480)

        self.classifier = DigitClassifier(resource_path(MODEL_FILE))
        self.last_results: list[Prediction] = []

        self._build_widgets()

    def _build_widgets(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        title = ttk.Label(self, text="手写数字识别", font=("Arial", 20, "bold"))
        title.grid(row=0, column=0, sticky="w", padx=18, pady=(16, 6))

        toolbar = ttk.Frame(self)
        toolbar.grid(row=1, column=0, sticky="ew", padx=18, pady=8)
        toolbar.columnconfigure(3, weight=1)

        ttk.Button(toolbar, text="选择图片", command=self.choose_image).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(toolbar, text="选择文件夹", command=self.choose_folder).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(toolbar, text="导出CSV", command=self.export_csv).grid(row=0, column=2, padx=(0, 8))

        self.summary_var = tk.StringVar(value="请选择单张数字图片，或选择包含数字图片的文件夹。")
        ttk.Label(toolbar, textvariable=self.summary_var, anchor="e").grid(row=0, column=3, sticky="ew")

        table_frame = ttk.Frame(self)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=18, pady=(8, 16))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("file", "prediction", "confidence", "truth", "correct")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=16)
        self.table.heading("file", text="文件")
        self.table.heading("prediction", text="预测")
        self.table.heading("confidence", text="置信度")
        self.table.heading("truth", text="真实标签")
        self.table.heading("correct", text="是否正确")
        self.table.column("file", width=430, anchor="w")
        self.table.column("prediction", width=80, anchor="center")
        self.table.column("confidence", width=100, anchor="center")
        self.table.column("truth", width=100, anchor="center")
        self.table.column("correct", width=100, anchor="center")
        self.table.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)

    def choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择数字图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not filename:
            return
        path = Path(filename)
        self.run_predictions([path], path.parent)

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        root = Path(folder)
        images = list(iter_images(root))
        if not images:
            messagebox.showwarning("没有图片", "这个文件夹里没有可识别的图片文件。")
            return
        self.run_predictions(images, root)

    def run_predictions(self, paths: list[Path], root: Path) -> None:
        try:
            results: list[Prediction] = []
            for path in paths:
                digit, confidence = self.classifier.predict(path)
                results.append(Prediction(path=path, digit=digit, confidence=confidence, truth=label_from_path(path, root)))
            self.last_results = results
            self.render_results(root)
        except Exception as exc:  # pragma: no cover - GUI safety net.
            traceback.print_exc()
            messagebox.showerror("识别失败", str(exc))

    def render_results(self, root: Path) -> None:
        self.table.delete(*self.table.get_children())
        labeled = 0
        correct = 0

        for item in self.last_results:
            truth = "" if item.truth is None else str(item.truth)
            if item.correct is None:
                correct_text = ""
            else:
                labeled += 1
                if item.correct:
                    correct += 1
                correct_text = "是" if item.correct else "否"
            try:
                display_path = str(item.path.relative_to(root))
            except ValueError:
                display_path = item.path.name
            self.table.insert(
                "",
                "end",
                values=(display_path, item.digit, f"{item.confidence:.2%}", truth, correct_text),
            )

        total = len(self.last_results)
        if labeled:
            self.summary_var.set(f"共识别 {total} 张；有标签 {labeled} 张；准确率 {correct / labeled:.2%}")
        elif total == 1:
            item = self.last_results[0]
            self.summary_var.set(f"预测数字：{item.digit}，置信度：{item.confidence:.2%}")
        else:
            self.summary_var.set(f"共识别 {total} 张；未发现真实标签，已显示预测结果。")

    def export_csv(self) -> None:
        if not self.last_results:
            messagebox.showinfo("没有结果", "请先选择图片或文件夹进行识别。")
            return
        filename = filedialog.asksaveasfilename(
            title="保存预测结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filename:
            return

        with Path(filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "prediction", "confidence", "truth", "correct"])
            for item in self.last_results:
                writer.writerow(
                    [
                        str(item.path),
                        item.digit,
                        f"{item.confidence:.6f}",
                        "" if item.truth is None else item.truth,
                        "" if item.correct is None else int(item.correct),
                    ]
                )
        messagebox.showinfo("已导出", f"预测结果已保存到：\n{filename}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        raise SystemExit(run_self_test(Path(sys.argv[2])))
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
