
# MNIST Handwritten Digit Recognition Project

This repository contains the implementation of a handwritten digit recognition system using the MNIST dataset. The project leverages Convolutional Neural Networks (CNNs) to identify and classify digits from 0 to 9 in scanned handwritten images.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Solution Approach](#solution-approach)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Problem Statement

Handwritten digit recognition is a widely used application of machine learning and computer vision, commonly seen in postal automation, bank cheque verification, and more. The goal of this project is to build a system that can accurately recognize digits from images using the MNIST dataset.

---

## Project Overview

The project includes:
- Preprocessing of the MNIST dataset.
- Data augmentation to enhance the robustness of the model.
- Building and training a CNN model to recognize digits.
- Saving and visualizing predictions.

Key Features:
- **Dataset Used**: MNIST (Modified National Institute of Standards and Technology).
- **Input Size**: Images of size 28x28 pixels.
- **Accuracy Achieved**: Training accuracy of **98.77%**.

---

## Solution Approach

1. **Data Preprocessing**:
   - Normalized the pixel values to a range of 0 to 1 for better model performance.
   - Augmented the dataset by generating additional variations of the images.

2. **Model Architecture**:
   - Used TensorFlow to build a **Convolutional Neural Network (CNN)**.
   - The CNN contains **three layers** tailored to handle 28x28 grayscale images.

3. **Training**:
   - **Epochs**: 15
   - **Batch Size**: 128
   - Optimized the learning rate scheduler during training.

4. **Evaluation and Prediction**:
   - Saved the model predictions in a CSV file.
   - Visualized the predictions for validation.

---

## Repository Structure

```plaintext
├── MNIST-CNN-NUMBER-RECOGNITION-ENSEMBLE.csv  # Model predictions
├── NUMBER_RECOGNITION_MNIST_DATASET_SANDEEP MISHRA.ipynb  # Main notebook
├── README.md  # Project description
├── TRAIN&TEST_DATASETS.zip  # Training and testing datasets
├── app.py  # Script for model deployment
├── mnist_cnn_model.h5  # Saved Keras model
├── mnist_cnn_model.json  # Model architecture
├── mnist_cnn_model.pkl  # Model in pickle format
├── my_model.keras  # Saved model in Keras format
└── requirements.txt  # Required libraries
```

---

## Results

- **Training Accuracy**: 98.77%
- Visualized predictions confirm the robustness of the model.
- Predictions saved in `MNIST-CNN-NUMBER-RECOGNITION-ENSEMBLE.csv`.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model or use the pre-trained model to make predictions:
   - To train: Run the notebook `NUMBER_RECOGNITION_MNIST_DATASET_SANDEEP MISHRA.ipynb`.
   - To deploy: Run `app.py`.

---

## Dependencies

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Keras

Install the dependencies using the provided `requirements.txt` file.

---

This project demonstrates the power of CNNs in image classification and offers a great starting point for further research in computer vision tasks. Feel free to explore and contribute to the repository!
