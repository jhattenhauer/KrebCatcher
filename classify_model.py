import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import json
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential

model = keras.models.load_model("image_classifier.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

def predict(image_path):
    img = PIL.Image.open(image_path).convert("RGB")
    img = img.resize((180, 180))
    img = np.expand_dims(np.array(img), axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    return class_names[class_id]

