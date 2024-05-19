# Analyzing Transfer Learning Approaches for Melanoma Detection: A Comparative Study

## Overview
This project aims to analyze various transfer learning approaches for melanoma detection using convolutional neural networks (CNNs). The study compares the performance of different CNN models, including AlexNet, ResNet50, ResNet101, VGG16, VGG19, DenseNet201, InceptionResNetV2, InceptionV3, DenseNet121, and EfficientNetB1. The dataset used for training and evaluation contains pre-defined classes for benign and malignant melanoma.

## Dataset
The dataset used in this project can be accessed from [Kaggle](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/code). It consists of images categorized into benign and malignant classes, providing a suitable foundation for training and testing melanoma detection models.

## Methodology
The project utilizes transfer learning, a popular technique in deep learning, where pre-trained models are employed as a starting point for training on a new task. The following CNN architectures are utilized in this study:

- AlexNet
- ResNet50
- ResNet101
- VGG16
- VGG19
- DenseNet201
- InceptionResNetV2
- InceptionV3
- DenseNet121
- EfficientNetB1

Training is conducted using Jupyter Notebooks (.ipynb files) to facilitate experimentation and analysis. Due to significant training times, the training process is divided into four separate notebooks.

## Results
After extensive training and evaluation, the ResNet101 model emerges as the top performer in terms of accuracy for melanoma detection. Detailed analysis and performance metrics can be found in the respective Jupyter Notebooks.


## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/melanoma-detection.git
2. Download the dataset from the provided Kaggle link and place it in the data/ directory.
3. Run the Jupyter Notebooks in the notebooks/ directory sequentially for training and evaluation.
4. Refer to the results for insights into model performance and choose the ResNet101 model for deployment based on its superior accuracy.

## Future Work
1. Investigate further optimization techniques to improve model performance.
2. Explore additional CNN architectures and fine-tuning strategies for enhanced melanoma detection.

## Contributors
1. [Ritika Chandavarkar](https://github.com/ri-chand)
2. [Sohan Nath](https://github.com/snnath)
