# Digit Recognizer Project

This project implements a Convolutional Neural Network (CNN) for digit recognition using the MNIST dataset.

## Overview

The model achieves an accuracy of 99.28% on the test set from the Kaggle Digit Recognizer competition.

## Files Included

- `Digit_Recognizer.ipynb`: Jupyter Notebook containing the model training code.
- `digit-recognizer/sample_submission.csv`: Sample submission file for the Kaggle competition.
- `digit-recognizer/test.csv`: Dataset file containing test data.
- `digit-recognizer/train.csv`: Dataset file containing training data.

## Model Architecture

The model architecture includes:
- Three sets of convolutional layers with ReLU activation.
- Max pooling layers for downsampling.
- Dropout layers to prevent overfitting.
- Fully connected layers with ReLU activation.
- Softmax output layer for multiclass classification.

## Usage

To train the model:
1. Open and run `Digit_Recognizer.ipynb` in a Jupyter Notebook environment.
2. Ensure the necessary datasets (`test.csv` and `train.csv`) are accessible.

## Dependencies

- Python 3.x
- TensorFlow
- Jupyter Notebook

## Acknowledgments

- Kaggle for providing the MNIST dataset and hosting the competition.
- TensorFlow and Keras for their powerful tools and libraries.

