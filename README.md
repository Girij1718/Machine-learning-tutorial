Machine Learning Tutorial:
  A simple deep learning project that trains a Multi-Layer Perceptron (MLP) to recognize handwritten digits (0-9) using the MNIST dataset. The model is tested with different activation functions to analyze their impact on accuracy and training performance.

Project Overview:
  This project explores how different activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU, and Swish) affect the performance of an MLP neural network trained on the MNIST dataset. The model learns to classify digits from 0 to 9 based on grayscale images (28×28 pixels).

Dataset Information:
-Dataset Name: MNIST (Modified National Institute of Standards and Technology)
-Total Images: 70,000 (60,000 for training, 10,000 for testing)
-Image Size: 28×28 pixels, grayscaleLabels: 10 classes (0-9)

The dataset is preprocessed by normalizing pixel values and converting them into PyTorch tensors for training.

How This Was Built:

1️.Setting Up the Environment
To run this project, you need Python and the following libraries installed:
pip install torch torchvision matplotlib numpy
The code automatically uses CUDA (GPU) if available for faster training.

2️.Loading and Preprocessing the Dataset
Downloaded the MNIST dataset using Torchvision.
Applied normalization to scale pixel values between -1 and 1.
Used batching (batch size: 64) for efficient training.

3️.Building the MLP Model
The Multi-Layer Perceptron (MLP) consists of:
-Input Layer: 784 neurons (28×28 pixels flattened).
-Hidden Layers: Two fully connected layers (128 & 64 neurons).
-Output Layer: 10 neurons (one for each digit).
-Activation Functions: ReLU, Sigmoid, Tanh, Leaky ReLU, Swish.
-Loss Function: CrossEntropyLoss (Softmax included).
-Optimizer: Adam (learning rate: 0.001).

4️.Training the Model
The model is trained for 5 epochs, tracking:
- Training accuracy
- Test accuracy
- Training loss
The performance is compared across different activation functions.

5️.Evaluating and Visualizing Results
