import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import json

# =========================
# CONFIG
# =========================
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 5   # number of additional epochs
DATASET_DIR = "flower_photos"  # same dataset as before
MODEL_PATH = "image_classifier.keras"
CLASS_NAMES_PATH = "class_names.json"

# =========================
# LOAD DATASET
# =========================
data_dir = pathlib.Path(DATASET_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names

# Save class names again (safe overwrite)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = keras.models.load_model(MODEL_PATH)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

model.save(MODEL_PATH)

print("Training complete. Model updated and saved.")
