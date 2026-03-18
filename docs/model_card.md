# Model Card: Plant Disease Classification (Vision)

## Model Details
- **Model type:** Convolutional Neural Network (CNN)  
- **Architecture:** MobileNetV2 (pre-trained on ImageNet, fine-tuned for plant leaf diseases)  
- **Task:** Image-based classification of plant leaf diseases  
- **Input:** RGB images of plant leaves  
- **Output:** Disease class label and confidence percentage  

## Dataset
- **Source:** PlantVillage dataset  
- **Size:** ~43,444 images  
- **Classes:** 38 [plant, disease] combinations  
- **Notes:** Images include leaves from various plants in controlled conditions. Some real-field images may be added later.  
  For testing, images can be randomly collected from the real world or sourced from the Internet to evaluate generalization.

## Metrics
- **Accuracy:** Measures overall correct predictions  
- **Macro F1-score:** Evaluates balance across all classes  
- **Confusion Matrix:** Provides insight on per-class performance  

## Limitations
- Model may perform poorly on images with extreme lighting, shadows, or occluded leaves  
- Model trained primarily on healthy and diseased leaves from the dataset; unknown cultivars or new diseases may be misclassified  
- Model performance on real-world images or images collected from the Internet may vary due to differences in background, lighting, or image quality  
