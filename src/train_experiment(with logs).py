import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

DATASET_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/dataset/plantvillage"
MODEL_SAVE_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/src/model/plant_disease_model_baseline.h5"
LOGS_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/experiments/logs/baseline_training_log.csv"
PLOT_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/experiments/results/baseline_training_plot.png"

# Parameters
IMG_SIZE = 128  # reduced for faster CPU training
BATCH_SIZE = 32  # smaller batch size for CPU memory
EPOCHS = 10  # initial max epochs; early stopping will reduce unnecessary epochs

os.makedirs("model", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Data augmentation (minimal to reduce CPU load)
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True  # only horizontal flip for speed
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load pretrained MobileNetV2 without top layers
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all base model layers to reduce CPU computation
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
csv_logger = CSVLogger(CSV_LOG_FILE, append=True)

# Train only the classification head first
model.fit(
    train_data,
    epochs=EPOCHS,
    callbacks=[early_stop, csv_logger]
)

# Optional: fine-tune last few layers for higher accuracy
base_model.trainable = True
for layer in base_model.layers[:-20]:  # freeze all but last 20 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),  # lower learning rate for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    epochs=EPOCHS,
    callbacks=[early_stop, csv_logger]
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"CPU-optimized training complete. Model saved to {MODEL_SAVE_PATH}")