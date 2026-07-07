from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

import train_cnn_tensorflow as trainer


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tf_digit_cnn_best.keras"


def preprocess_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return arr[None, ..., None]


def evaluate_validation(model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    (_, _, _), (x_val, y_val, _) = trainer.load_split()
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    cm = trainer.confusion_matrix(y_val, pred)
    print(json.dumps({"model": str(model_path), "val_loss": float(val_loss), "val_accuracy": float(val_acc)}, indent=2))
    print("confusion_matrix_rows=true_cols=pred")
    print(cm)


def predict_images(paths, model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    for path in paths:
        x = preprocess_image(path)
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        print(json.dumps({"path": str(path), "prediction": pred, "probability": float(probs[pred])}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        evaluate_validation()
    else:
        predict_images([Path(arg) for arg in sys.argv[1:]])
