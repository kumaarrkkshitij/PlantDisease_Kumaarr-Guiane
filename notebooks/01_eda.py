# 01_eda.py
import os
import matplotlib.pyplot as plt

# ===============================
# Dataset path
# ===============================
DATASET_PATH = "/Users/Work/Downloads/plant-disease-classification/dataset/plantvillage"

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder not found at {DATASET_PATH}")

# ===============================
# Get all classes
# ===============================
classes = sorted([d for d in os.listdir(DATASET_PATH) 
                  if os.path.isdir(os.path.join(DATASET_PATH, d))])

# ===============================
# Count images per class
# ===============================
counts = [len(os.listdir(os.path.join(DATASET_PATH, cls))) for cls in classes]

# ===============================
# Vertical bar plot
# ===============================
plt.figure(figsize=(20, 8))
plt.bar(classes, counts)
plt.xticks(rotation=90, fontsize=10)
plt.ylabel("Number of images")
plt.title("Number of images per class (Vertical)")
plt.tight_layout()
plt.show()

# ===============================
# Horizontal bar plot (better for many classes)
# ===============================
plt.figure(figsize=(12, len(classes) * 0.3))  # adjust height based on number of classes
plt.barh(classes, counts)
plt.xlabel("Number of images")
plt.title("Number of images per class (Horizontal)")
plt.tight_layout()
plt.show()

# ===============================
# Optional: Print summary stats
# ===============================
print(f"Total classes: {len(classes)}")
print(f"Total images: {sum(counts)}")
for cls, count in zip(classes, counts):
    print(f"{cls}: {count} images")