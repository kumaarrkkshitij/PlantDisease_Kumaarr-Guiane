# Before training, verify that the dataset loads correctly.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "/Users/Work/Downloads/6INTELSY FINAL PROJECT/dataset/plantvillage"

# Only rescaling; no validation split
datagen = ImageDataGenerator(rescale=1./255)

# Load all images from the train folder
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32
)

print("\nNumber of Images:", train_data.samples)
print("Number of classes:", train_data.num_classes)
print("Classes:", train_data.class_indices)
