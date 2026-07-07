from pathlib import Path
import json

import numpy as np
import tensorflow as tf

import train_cnn_tensorflow as trainer


ROOT = Path(__file__).resolve().parent
KERAS_MODEL = ROOT / "tf_digit_cnn_best.keras"
INFERENCE_KERAS = ROOT / "digit_cnn_inference.keras"
TFLITE_MODEL = ROOT / "digit_cnn.tflite"
TFLITE_REPORT = ROOT / "tflite_report.json"


def strip_augmentation_layers(model):
    """Create an inference-only model by skipping Keras random augmentation."""
    inputs = tf.keras.Input(shape=(28, 28, 1), name="image")
    x = inputs
    start = False
    for layer in model.layers:
        if layer.name == "block1_conv1":
            start = True
        if not start:
            continue
        x = layer(x, training=False)
    return tf.keras.Model(inputs, x, name="digit_cnn_inference")


def representative_dataset():
    (x_train, _, _), _ = trainer.load_split()
    for sample in x_train[:300]:
        yield [sample[None, ...].astype(np.float32)]


def main():
    model = tf.keras.models.load_model(KERAS_MODEL)
    inference_model = strip_augmentation_layers(model)
    inference_model.save(INFERENCE_KERAS)

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    TFLITE_MODEL.write_bytes(tflite_model)

    # Verify TFLite accuracy using TensorFlow's built-in interpreter.
    _, (x_val, y_val, _) = trainer.load_split()
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = []
    for sample in x_val:
        interpreter.set_tensor(input_details["index"], sample[None, ...].astype(np.float32))
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details["index"])[0]
        preds.append(int(np.argmax(scores)))
    preds = np.asarray(preds)
    acc = float((preds == y_val).mean())

    report = {
        "source_model": str(KERAS_MODEL),
        "inference_keras": str(INFERENCE_KERAS),
        "tflite_model": str(TFLITE_MODEL),
        "tflite_size_bytes": TFLITE_MODEL.stat().st_size,
        "validation_accuracy": acc,
        "input": "float32 shape [1,28,28,1], pixel = 1 - normalized_gray",
        "output": "float32 probabilities for digits 0-9",
    }
    TFLITE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
