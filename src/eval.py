import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense as KerasDense
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# =========================
# Paths (adjusted for your Mac)
# =========================
DATASET_PATH = "/Users/Work/Downloads/plant-disease-classification/dataset/plantvillage"
MODEL_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/src/model/plant_disease_model.h5"
RESULTS_DIR = "/Users/Work/Downloads/plant-disease-classification/experiments/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Parameters
# =========================
IMG_SIZE = 128
BATCH_SIZE = 32

# =========================
# Monkey-patch Dense to ignore quantization_config if exists
# =========================
old_dense_init = KerasDense.__init__

def new_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    old_dense_init(self, *args, **kwargs)

KerasDense.__init__ = new_dense_init
model = load_model(MODEL_PATH)
KerasDense.__init__ = old_dense_init

# =========================
# Prepare data generator
# =========================
datagen = ImageDataGenerator(rescale=1./255)
data_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# Predict
# =========================
preds = model.predict(data_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = data_gen.classes
class_names = list(data_gen.class_indices.keys())

# =========================
# Metrics
# =========================
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"\nMacro F1-score: {macro_f1:.4f}\n")

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm, "\n")

report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()

# Print classification report in terminal like your last output
report_text = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report_text)

# Save CSV
report_csv_path = os.path.join(RESULTS_DIR, "classification_report.csv")
report_df.to_csv(report_csv_path, index=True)
print(f"Classification report saved at {report_csv_path}")

# =========================
# Plot confusion matrix heatmap
# =========================
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"Confusion matrix heatmap saved at {cm_path}")