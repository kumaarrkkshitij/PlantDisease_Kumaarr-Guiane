import os
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

# =========================
# Paths and parameters
# =========================
DATASET_PATH = "dataset/plantvillage"
MODEL_SAVE_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/src/model/plant_disease_model_baseline.h5"
LOGS_PATH = "experiments/logs/baseline_training_log.csv"
PLOT_PATH = "experiments/results/baseline_training_plot.png"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10  # early stopping will reduce actual epochs

# =========================
# Make directories
# =========================
os.makedirs("model", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# Data augmentation
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# Load base model
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # freeze base layers

# =========================
# Add classification head
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Callbacks
# =========================
early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
csv_logger = CSVLogger(LOGS_PATH, append=False)

# =========================
# Train only classification head
# =========================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    callbacks=[early_stop, csv_logger]
)

# =========================
# Optional fine-tuning
# =========================
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_data,
    epochs=EPOCHS,
    callbacks=[early_stop, csv_logger]
)

# =========================
# Save model
# =========================
model.save(MODEL_SAVE_PATH)
print(f"Baseline model saved at {MODEL_SAVE_PATH}")

# =========================
# Plot training curves
# =========================
plt.figure(figsize=(10, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss (head)')
plt.plot(history_ft.history['loss'], label='Loss (fine-tune)')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acc (head)')
plt.plot(history_ft.history['accuracy'], label='Acc (fine-tune)')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()
print(f"Training plots saved at {PLOT_PATH}")