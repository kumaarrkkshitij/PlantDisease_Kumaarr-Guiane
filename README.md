**Verdantis AI – Plant Leaf Disease Classification**
Project Overview

Verdantis AI is a deep learning system designed to identify plant leaf diseases using images. It leverages a fine-tuned MobileNetV2 CNN model trained on the PlantVillage dataset (38 plant classes, ~43,444 images). The system provides accurate predictions to help farmers and plant enthusiasts quickly detect diseases.

Features
- Upload plant leaf images through a web interface.
- Classify leaves into healthy or diseased categories.
- Display disease type with healthy scores.
- Provides metrics and visualizations for model performance (accuracy, macro-F1, confusion matrix).

System/Application Development Workflow
- Install Dependencies (requirments.txt)
Dependencies include: tensorflow, pillow, scipy, flask, scikit-learn, seaborn, matplotlib

Import Dataset
- Dataset link: PlantVillage - https://www.kaggle.com/datasets/emmarex/plantdisease
- Contains 38 plant classes with ~43,444 images.
- Used for training the CNN model.

Verify Dataset Loading
- python test_dataset.py, ensures images load correctly before training.

Train the Model
- python src/train_model.py

MobileNetV2 is fine-tuned/
Classification head trained first, optional fine-tuning afterwards.

Training logs and plots saved to logs/ and results/.

Launch the Application
- python src/app.py
- Access locally at http://127.0.0.1:5000

Predict Plant Leaf Diseases

Upload images through the web interface.

See predictions along with healthy scores.

Authors

Verdantis AI Team:
Kumaarr, Kkshitij
Del Mundo, Guiane Carlo

Notes / Ethics:
- Predictions are assistive, not diagnostic.
- Misclassifications may occur due to lighting, leaf angles, or unseen cultivars.
- Always verify critical decisions with an expert.
