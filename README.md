# Plant Disease Prediction Model

This project aims to develop a robust plant disease prediction model using a Convolutional Neural Network (CNN) trained on the PlantVillage dataset. The model can classify images of plant leaves into different disease categories, aiding in early disease detection and prevention.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results](#results)
- [Future Work](#future-work)

## Project Overview

Plant diseases pose a significant threat to global food security. Early and accurate detection of these diseases is crucial for timely intervention and minimizing crop losses. This project leverages the power of deep learning to build a model capable of identifying plant diseases from leaf images.

## Dataset

The PlantVillage dataset, a publicly available collection of plant leaf images with corresponding disease labels, is used to train and evaluate the model. The dataset contains images of various plant species and a wide range of diseases.

## Model Architecture

A CNN architecture is employed for image classification. The model consists of convolutional layers, max pooling layers, a flattening layer, and dense layers. The final layer uses a softmax activation function to output probabilities for each disease class.

## Usage

1. **Install Dependencies:** Ensure you have the necessary libraries installed (see [Dependencies](#dependencies)).
2. **Download Dataset:** Download the PlantVillage dataset using Kaggle API (refer to the code for instructions).
3. **Preprocess Data:** Resize images, normalize pixel values, and split the dataset into training and validation sets.
4. **Train Model:** Train the CNN model using the preprocessed data.
5. **Evaluate Model:** Assess model performance on the validation set using metrics like accuracy and loss.
6. **Predict:** Utilize the trained model to predict the disease class for new plant leaf images.

## Dependencies

- Python
- TensorFlow
- Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- Kaggle API

## Results

The trained model achieves a satisfactory accuracy on the validation set. The specific results, including accuracy and loss values, are available in the code.
