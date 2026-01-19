import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import json
from configparser import ConfigParser
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential

cfg = ConfigParser()
cfg.read("settings.ini")

# =========================
# CONFIG
# =========================
IMG_SIZE = cfg["model"]["image_size"]
BATCH_SIZE = cfg["model"]["batch_size"]
EPOCHS = cfg["model"]["initial"]
DATASET_DIR = cfg["paths"]["dataset_path"]
MODEL_PATH = cfg["model"]["model_path"]
CLASS_NAMES_PATH = cfg["model"]["class_names_path"]
VALIDATION_SPLIT = cfg["model"]["validation_split"]
TRAINING_SEED = cfg["model"]["training_seed"]
ACTIVATION_METHOD = cfg['model']["neuron_activation_method"]
MODEL_OPTIMIZATION = cfg["model"]["model_optimizer"]

import pathlib 
data_dir = pathlib.Path(DATASET_DIR)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found at: {data_dir}")

image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"Total images found: {image_count}")
if image_count == 0:
    print("Warning: No images found. Check your dataset path and format.")
    all_files = list(data_dir.glob('*/*'))
    print(f"Found files (first 5): {[str(f) for f in all_files[:5]]}")


train_ds = tf.keras.utils.image_dataset_from_directory( 
	data_dir, 
	validation_split=VALIDATION_SPLIT, 
	subset="training", 
	seed=TRAINING_SEED, 
	image_size=IMG_SIZE, 
	batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory( 
	data_dir, 
	validation_split=VALIDATION_SPLIT, 
	subset="validation", 
	seed=TRAINING_SEED, 
	image_size=IMG_SIZE, 
	batch_size=BATCH_SIZE)

class_names = train_ds.class_names 
print(class_names)

import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 10)) 
for images, labels in train_ds.take(1): 
	for i in range(25): 
		ax = plt.subplot(5, 5, i + 1) 
		plt.imshow(images[i].numpy().astype("uint8")) 
		plt.title(class_names[labels[i]]) 
		plt.axis("off")

num_classes = len(class_names) 


##############
## Training ##
##############

model = Sequential([ 
	layers.Rescaling(1./255, input_shape=(180,180, 3)), 
	layers.Conv2D(16, 3, padding='same', activation=ACTIVATION_METHOD), 
	layers.MaxPooling2D(), 
	layers.Conv2D(32, 3, padding='same', activation=ACTIVATION_METHOD), 
	layers.MaxPooling2D(), 
	layers.Conv2D(64, 3, padding='same', activation=ACTIVATION_METHOD), 
	layers.MaxPooling2D(), 
	layers.Flatten(), 
	layers.Dense(128, activation=ACTIVATION_METHOD), 
	layers.Dense(num_classes) 
])

model.compile(optimizer=MODEL_OPTIMIZATION, 
	loss=tf.keras.losses.SparseCategoricalCrossentropy( 
	from_logits=True), 
	metrics=['accuracy']) 

model.summary()

epochs=EPOCHS
history = model.fit( 
train_ds, 
validation_data=val_ds, 
epochs=epochs 
)

model.save(MODEL_PATH)
import json

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)


acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 
epochs_range = range(epochs) 
plt.figure(figsize=(8, 8)) 
plt.subplot(1, 2, 1) 
plt.plot(epochs_range, acc, label='Training Accuracy') 
plt.plot(epochs_range, val_acc, label='Validation Accuracy') 
plt.legend(loc='lower right') 
plt.title('Training and Validation Accuracy') 
plt.subplot(1, 2, 2) 
plt.plot(epochs_range, loss, label='Training Loss') 
plt.plot(epochs_range, val_loss, label='Validation Loss') 
plt.legend(loc='upper right') 
plt.title('Training and Validation Loss') 
plt.show()
