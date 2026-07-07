from pathlib import Path
import json
import os
import random

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
DATA_NPZ = ROOT / "mydata_augmented_normalized.npz"
MODEL_PATH = ROOT / "tf_digit_cnn.keras"
BEST_MODEL_PATH = ROOT / "tf_digit_cnn_best.keras"
LOG_CSV_PATH = ROOT / "tf_training_log.csv"
REPORT_PATH = ROOT / "tf_training_report.json"
CONFUSION_PATH = ROOT / "tf_confusion_matrix.npy"

SEED = 20260706
TRAIN_PER_CLASS = 400
VAL_PER_CLASS = 100
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split():
    data = np.load(DATA_NPZ, allow_pickle=True)
    x = data["x"].astype("float32")
    y = data["y"].astype("int64")
    names = data["names"].astype(str)

    # Saved preprocessing is white background / dark ink. The model sees
    # background=0 and ink=1, which is easier for handwritten-digit CNNs.
    x = 1.0 - x
    x = x[..., None]

    rng = np.random.default_rng(SEED)
    train_idx = []
    val_idx = []
    for digit in range(10):
        idx = np.where(y == digit)[0].copy()
        rng.shuffle(idx)
        if len(idx) < TRAIN_PER_CLASS + VAL_PER_CLASS:
            raise ValueError(f"class {digit} has only {len(idx)} samples")
        train_idx.extend(idx[:TRAIN_PER_CLASS])
        val_idx.extend(idx[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS])

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (x[train_idx], y[train_idx], names[train_idx]), (x[val_idx], y[val_idx], names[val_idx])


def make_datasets(x_train, y_train, x_val, y_val):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train), seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def conv_block(x, filters, dropout_rate, name):
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name=f"{name}_conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name=f"{name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu2")(x)
    x = tf.keras.layers.MaxPooling2D(name=f"{name}_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
    return x


def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="image")

    # Mild augmentation at train time only. These transforms avoid flips because
    # flipping digits can change the label semantics.
    x = tf.keras.layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0, name="aug_rotation")(inputs)
    x = tf.keras.layers.RandomTranslation(
        0.08,
        0.08,
        fill_mode="constant",
        fill_value=0.0,
        name="aug_translation",
    )(x)
    x = tf.keras.layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0, name="aug_zoom")(x)
    x = tf.keras.layers.RandomContrast(0.15, name="aug_contrast")(x)

    x = conv_block(x, 32, 0.15, "block1")
    x = conv_block(x, 64, 0.25, "block2")

    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    x = tf.keras.layers.Conv2D(
        128,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name="head_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Activation("relu", name="head_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer, name="dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="digit")(x)

    model = tf.keras.Model(inputs, outputs, name="generalized_digit_cnn")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((10, 10), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def main():
    set_seed()
    (x_train, y_train, _), (x_val, y_val, _) = load_split()
    train_ds, val_ds = make_datasets(x_train, y_train, x_val, y_val)
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=14,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(LOG_CSV_PATH),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    pred = np.argmax(model.predict(x_val, batch_size=BATCH_SIZE, verbose=0), axis=1)
    cm = confusion_matrix(y_val, pred)
    np.save(CONFUSION_PATH, cm)
    model.save(MODEL_PATH)

    report = {
        "seed": SEED,
        "framework": "tensorflow",
        "tensorflow_version": tf.__version__,
        "dataset": str(DATA_NPZ),
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "train_per_class": TRAIN_PER_CLASS,
        "val_per_class": VAL_PER_CLASS,
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "model_path": str(MODEL_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "confusion_matrix_path": str(CONFUSION_PATH),
        "notes": "Input pixels are 1 - normalized grayscale, so background=0 and ink=1.",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("confusion_matrix_rows=true_cols=pred")
    print(cm)


if __name__ == "__main__":
    main()
