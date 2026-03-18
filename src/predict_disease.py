import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense

# ===============================
# Path to your trained model
# ===============================
# src/predict_disease.py
MODEL_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/src/model/plant_disease_model.h5"
DATASET_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/dataset/plantvillage"

# ===============================
# Monkey-patch Dense to ignore 'quantization_config'
# ===============================
old_dense_init = Dense.__init__

def new_dense_init(self, *args, **kwargs):
    # Remove unsupported kwarg if it exists
    kwargs.pop("quantization_config", None)
    old_dense_init(self, *args, **kwargs)

Dense.__init__ = new_dense_init

# Load the model
model = load_model(MODEL_PATH)

# Restore Dense.__init__ to original (optional)
Dense.__init__ = old_dense_init

# ===============================
# Automatically get class names from dataset
# ===============================
class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
index_to_class = {i: name for i, name in enumerate(class_names)}

IMG_SIZE = 128  # same size as training

# ===============================
# Helper to preprocess image
# ===============================
def _prepare_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# Prediction function
# ===============================
def predict_disease(img_path):
    img_array = _prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = index_to_class[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    return predicted_class, confidence