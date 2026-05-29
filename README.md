# MNIST_ConvolutionalNeuralNetwork
Utilizing a Convolutional Neural Network to perform handwritten digit classification on MNIST dataset. The code is designed to make it easy to adjust the model's hyperparameters, architecture, and training process.

## Requirements
The code automatically downloads the MNIST dataset at the start of program execution so a internet connection is required to run it. Required libraries: tensorflow, sklearn, matplotlib

## Preprocessing:
Pixel values are normalized and a training/validation split of 90/10 is used for this model.

## Model Hyperparameters
The models hyperparameters can be easily adjusted, however the best results are with the following parameters:
  - Epochs: 20
  - Learning Rate: 0.001 with reduction by factor of 0.2 if 3 epochs without validation loss (minimum learning rate of 0.00001)
  - Dropout rate: 0.2 (Only on the fully connected layer(s))
  - Batch size: 32
  - Batch normalization after each convolution
  - ReLU activation
  - L1 & L2 regularization turned off
  - Loss: Sparse Categorical Cross Entropy
  - Optimzer: Adam

## Model Architecture
  - Input: (28x28x1)
  - Convultion1: 8 Filters (3x3), "same" padding, (28x28x8)
  - Batch Normalization
  - ReLU Activation
  - Max Pooling (2x2): 14x14x8
  - Convultion2: 16 Filters (3x3), "same" padding, (14x14x8)
  - Batch Normalization
  - ReLU Activation
  - Max Pooling (2x2): 7x7x16
  - Flattening: (7x7x16)-> 784
  - FullyConnectedLayer1: 128
  - Dropout: 0.2 (Can be turned off)
  - FullyConnectedLayer2: 10 (Output) (softmax activation)

## Results
Results on the train, validation, and test sets are provided below. The model hyperparameters and architecture chosen for test evaluation is shown in the "Model Hyperparameters" and "Model Architecture" section of this readme file. 

The model trained for 20 epochs but it restored to epoch 16:
  - Train accuracy: 0.998, loss: 0.0009
  - Validation accuracy: 0.9932 , loss: 0.0340
  - Test accuracy: 0.9910 , loss: 0.0321

Training plots are provided in the repository
